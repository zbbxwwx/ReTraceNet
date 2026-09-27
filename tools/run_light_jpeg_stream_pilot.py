import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import run_e5_paired_chroma_adaptation as e5
import run_e6_multi_quality_chroma as e6
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair

from fair_training_data import FairDataset

from retracenet.light_jpeg_stream import (
    LightweightJPEGEvidenceStream,
    ReliabilityGatedForensicFusion,
)


OUT = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "light_jpeg_stream_pilot_20260924"
)

CKPT = (
    e6.OUT / "epoch02.pt"
)

SEED = 3407
EPOCHS = 10
BATCH = 4


def main():

    OUT.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    parent = torch.load(
        e6.PARENT,
        map_location="cpu",
        weights_only=False
    )

    checkpoint = torch.load(
        CKPT,
        map_location="cpu",
        weights_only=False
    )

    cfg = copy.deepcopy(parent["config"])

    cfg["training"]["dct_sample_dropout"] = 0.0

    model = fair.e.build_final_model(
        cfg["model"]
    ).cuda()

    # Load original network first.
    model.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    channels = tuple(
        cfg["model"].get(
            "forensic_channels",
            (64,128,256,512)
        )
    )

    # ------------------------------------------------------
    # Replace CAT branch.
    # ------------------------------------------------------

    model.dct_branch = LightweightJPEGEvidenceStream(
        channels
    ).cuda()

    model.balanced_forensic_fusions = torch.nn.ModuleList(
        ReliabilityGatedForensicFusion(c)
        for c in channels
    ).cuda()

    # Original DCT adapters are no longer required.
    model.dct_inject_adapters = torch.nn.ModuleList(
        [torch.nn.Identity() for _ in channels]
    )

    model.dct_evidence_head = None
    model.splicing_region_affinity = None

    # ------------------------------------------------------
    # Patch only the DCT injection routine.
    # No changes to QTR/HCV/DHER.
    # ------------------------------------------------------

    def inject(image, detail, qtable, available):

        if available is None:
            available = torch.ones(
                image.shape[0],
                1,
                device=image.device,
                dtype=image.dtype
            )

        freq_features = model.dct_branch(
            image,
            qtable
        )

        fused = []

        for i, (r, f) in enumerate(
            zip(detail, freq_features)
        ):
            out, _, _ = (
                model.balanced_forensic_fusions[i](
                    r,
                    f,
                    available.to(r.dtype)
                )
            )

            fused.append(out)

        return fused

    old_forward = model.forward

    def new_forward(
        image,
        intervention_mask=None,
        dct=None,
        qtable=None,
        dct_available=None
    ):
        # Manually reproduce only the initial feature stage,
        # then temporarily bypass the old DCT injector.

        original_branch = model.dct_branch
        original_adapters = model.dct_inject_adapters

        # Build residual detail.
        content = model.content_encoder(image)
        detail = model.forensic_stream(image)

        detail = inject(
            image,
            detail,
            qtable,
            dct_available
        )

        # Temporarily bypass old injection.
        model.dct_branch = None
        model.dct_inject_adapters = None

        try:
            # We need the rest of the existing forward path,
            # so replace forensic_stream result through a hook.
            original_forensic = model.forensic_stream

            class FixedDetail(torch.nn.Module):
                def __init__(self, feats):
                    super().__init__()
                    self.feats = feats

                def forward(self, x):
                    return self.feats

            model.forensic_stream = FixedDetail(detail)

            result = old_forward(
                image=image,
                intervention_mask=intervention_mask,
                dct=dct,
                qtable=qtable,
                dct_available=dct_available
            )

            model.forensic_stream = original_forensic

        finally:
            model.dct_branch = original_branch
            model.dct_inject_adapters = original_adapters

        return result

    model.forward = new_forward

    # ------------------------------------------------------
    # Training parameters
    # ------------------------------------------------------

    for p in model.parameters():
        p.requires_grad_(False)

    train_names = (
        "forensic_stream.",
        "dct_branch.",
        "balanced_forensic_fusions.",
        "qtr.",
        "hcv2.",
        "decoder.",
        "dher.",
        "mask_head.",
    )

    for n,p in model.named_parameters():
        if n.startswith(train_names):
            p.requires_grad_(True)

    new_params = []
    old_params = []

    for n,p in model.named_parameters():

        if not p.requires_grad:
            continue

        if (
            n.startswith("dct_branch.")
            or n.startswith(
                "balanced_forensic_fusions."
            )
        ):
            new_params.append(p)
        else:
            old_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": new_params,
                "lr": 1e-4,
                "weight_decay": 1e-4
            },
            {
                "params": old_params,
                "lr": 1e-5,
                "weight_decay": 1e-4
            }
        ]
    )

    train_records = fair.e.read_manifest(
        e6.MANIFEST,
        cfg["data"]["root"],
        "train"
    )

    val_records = fair.e.read_manifest(
        e6.MANIFEST,
        cfg["data"]["root"],
        "val"
    )

    train_base = FairDataset(
        train_records,
        OUT/"cache_train",
        True
    )

    val_base = FairDataset(
        val_records,
        OUT/"cache_val",
        False
    )

    train_ds = e6.MixedQuality(
        train_base
    )

    loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    loss_fn = stable.v3.e24.ReTraceLoss.from_config(
        stable.v3.e24.clean_loss_config(
            cfg["loss"]
        )
    ).cuda()

    original_summary = __import__(
        "yaml"
    ).safe_load(
        (e6.OUT/"summary.yaml").read_text()
    )

    locked_threshold = float(
        original_summary["history"][-1]
        ["clean"]["oracle"]["threshold"]
    )

    history=[]

    print(
        "Trainable:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    for epoch in range(1,EPOCHS+1):

        model.train()
        model.content_encoder.eval()

        total=0.
        start=time.time()

        for step,b in enumerate(loader,1):

            b={
                k:(
                    v.cuda(non_blocking=True)
                    if torch.is_tensor(v)
                    else v
                )
                for k,v in b.items()
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):

                out=stable.v3.fwd(
                    model,b,cfg,False
                )

                loss=stable.v3.objective(
                    out,b,loss_fn,cfg
                )["total"]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    p for p in model.parameters()
                    if p.requires_grad
                ],
                5.
            )

            optimizer.step()

            total+=float(loss)

            if step==1 or step%100==0:
                print(
                    f"[E{epoch}] "
                    f"{step}/{len(loader)} "
                    f"loss={total/step:.5f}",
                    flush=True
                )

        model.eval()

        row={
            "epoch":epoch,
            "loss":total/len(loader)
        }

        for name,q in [
            ("clean",None),
            ("q90",90),
            ("q80",80)
        ]:

            ds=e6.FixedQuality(
                val_base,
                quality=q,
                subsampling=2
            )

            result=e5.eval_ds(
                model,
                ds,
                cfg,
                f"ep{epoch:02d}_{name}",
                locked_threshold
            )

            row[name]={
                "auc":float(
                    result["auc"]
                ),
                "oracle_f1":float(
                    result["oracle"]["f1"]
                ),
                "locked_f1":float(
                    result["clean_locked"]["f1"]
                ),
                "precision":float(
                    result["clean_locked"]
                    ["precision"]
                ),
                "recall":float(
                    result["clean_locked"]
                    ["recall"]
                )
            }

        history.append(row)

        print(
            json.dumps(
                row,
                indent=2
            ),
            flush=True
        )

        torch.save(
            {
                "epoch":epoch,
                "model":model.state_dict(),
                "history":history
            },
            OUT/f"epoch{epoch:02d}.pt"
        )

        (OUT/"history.json").write_text(
            json.dumps(
                history,
                indent=2
            )
        )

    print("DONE")


if __name__=="__main__":
    main()
