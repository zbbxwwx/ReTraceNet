import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import run_e5_paired_chroma_adaptation as e5
import run_e6_multi_quality_chroma as e6
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset


CKPT = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "e6_q90_q80_joint_adaptation_20260923/epoch02.pt"
)

OUT = Path(
    "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/"
    "light_frequency_qtr_pilot_20260924"
)

SEED = 3407
EPOCHS = 5
BATCH = 4


def make_dct_filters():
    """
    64 orthonormal 8x8 DCT-II basis filters.
    This is implemented independently and does not use CAT-Net code.
    """
    filters = []
    for u in range(8):
        for v in range(8):
            basis = torch.zeros(8, 8)
            au = math.sqrt(1 / 8) if u == 0 else math.sqrt(2 / 8)
            av = math.sqrt(1 / 8) if v == 0 else math.sqrt(2 / 8)

            for x in range(8):
                for y in range(8):
                    basis[x, y] = (
                        au * av
                        * math.cos((2 * x + 1) * u * math.pi / 16)
                        * math.cos((2 * y + 1) * v * math.pi / 16)
                    )

            filters.append(basis)

    return torch.stack(filters)[:, None]


class LocalFrequencyCue(nn.Module):
    """
    RGB -> local 8x8 DCT -> compact spatial frequency maps.

    Output channels:
      0: low-frequency AC energy
      1: mid-frequency AC energy
      2: high-frequency AC energy
      3: horizontal-oriented AC energy
      4: vertical-oriented AC energy
    """

    def __init__(self):
        super().__init__()

        self.register_buffer("dct_filters", make_dct_filters())

        uv = [(u, v) for u in range(8) for v in range(8)]

        low = []
        mid = []
        high = []
        horizontal = []
        vertical = []

        for i, (u, v) in enumerate(uv):
            if u == 0 and v == 0:
                continue

            s = u + v

            if s <= 2:
                low.append(i)
            elif s <= 5:
                mid.append(i)
            else:
                high.append(i)

            if u <= 1 and v >= 2:
                horizontal.append(i)

            if v <= 1 and u >= 2:
                vertical.append(i)

        self.groups = [
            low,
            mid,
            high,
            horizontal,
            vertical,
        ]

    def forward(self, image):
        r = image[:, 0:1]
        g = image[:, 1:2]
        b = image[:, 2:3]

        gray = 0.299 * r + 0.587 * g + 0.114 * b

        h = (gray.shape[-2] // 8) * 8
        w = (gray.shape[-1] // 8) * 8

        gray = gray[..., :h, :w]

        coeff = F.conv2d(
            gray,
            self.dct_filters.to(dtype=gray.dtype),
            stride=8,
        )

        coeff = torch.log1p(coeff.abs())

        maps = []

        for ids in self.groups:
            maps.append(
                coeff[:, ids].mean(1, keepdim=True)
            )

        x = torch.cat(maps, dim=1)

        # Per-image normalization keeps scale stable across JPEG qualities.
        mean = x.mean((2, 3), keepdim=True)
        std = x.std((2, 3), keepdim=True, unbiased=False)

        return (x - mean) / (std + 1e-5)


class FrequencyAdapter(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        hidden = 32

        self.net = nn.Sequential(
            nn.Conv2d(5, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),

            nn.Conv2d(
                hidden,
                hidden,
                3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(8, hidden),
            nn.GELU(),

            nn.Conv2d(hidden, out_channels, 1),
        )

        # Start from zero contribution.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.alpha = nn.Parameter(torch.tensor(-2.0))

    def forward(self, freq, size):
        x = F.interpolate(
            freq,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        return torch.sigmoid(self.alpha) * self.net(x)


class FrequencyAwareQTR(nn.Module):
    """
    Wrap the existing QTR.

    Frequency evidence is injected only inside QTR.
    HCV / decoder / DHER interfaces remain unchanged.
    """

    def __init__(self, original_qtr, detail_channels=(64, 128, 256)):
        super().__init__()

        self.base_qtr = original_qtr
        self.frequency = LocalFrequencyCue()

        self.adapters = nn.ModuleList(
            FrequencyAdapter(c)
            for c in detail_channels
        )

    @property
    def use_reliability_gate(self):
        return self.base_qtr.use_reliability_gate

    @property
    def residual_refinement(self):
        return self.base_qtr.residual_refinement

    def forward(self, image, content, detail):
        freq = self.frequency(image)

        enriched = []

        for i in range(3):
            extra = self.adapters[i](
                freq,
                detail[i].shape[-2:],
            )

            enriched.append(detail[i] + extra)

        return self.base_qtr(
            image,
            content,
            enriched,
        )


def evaluate(model, val_base, cfg, threshold):
    results = {}

    settings = [
        ("clean", None),
        ("q90_420", 90),
        ("q80_420", 80),
    ]

    for name, quality in settings:

        ds = e6.FixedQuality(
            val_base,
            quality=quality,
            subsampling=2,
        )

        result = e5.eval_ds(
            model,
            ds,
            cfg,
            f"pilot_{name}",
            threshold,
        )

        results[name] = {
            "auc": float(result["auc"]),
            "oracle_f1": float(
                result["oracle"]["f1"]
            ),
            "oracle_threshold": float(
                result["oracle"]["threshold"]
            ),
            "locked_f1": float(
                result["clean_locked"]["f1"]
            ),
            "locked_precision": float(
                result["clean_locked"]["precision"]
            ),
            "locked_recall": float(
                result["clean_locked"]["recall"]
            ),
        }

    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    parent = torch.load(
        e6.PARENT,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint = torch.load(
        CKPT,
        map_location="cpu",
        weights_only=False,
    )

    cfg = copy.deepcopy(parent["config"])
    cfg["training"]["dct_sample_dropout"] = 0.0

    model = fair.e.build_final_model(
        cfg["model"]
    ).cuda()

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    # ---------------------------------------------------------
    # Remove CAT-style DCT path AFTER checkpoint loading.
    # ---------------------------------------------------------

    model.dct_branch = None
    model.dct_inject_adapters = None
    model.balanced_forensic_fusions = None
    model.dct_evidence_head = None
    model.splicing_region_affinity = None

    assert model.qtr is not None

    old_qtr = model.qtr

    model.qtr = FrequencyAwareQTR(
        old_qtr,
        detail_channels=(64, 128, 256),
    ).cuda()

    # ---------------------------------------------------------
    # Train only modules needed to adapt to the new evidence path.
    # Content encoder stays fixed.
    # ---------------------------------------------------------

    for p in model.parameters():
        p.requires_grad_(False)

    train_prefixes = (
        "forensic_stream.",
        "qtr.",
        "hcv2.",
        "decoder.",
        "dher.",
        "mask_head.",
    )

    for name, p in model.named_parameters():
        if name.startswith(train_prefixes):
            p.requires_grad_(True)

    new_params = []
    old_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if (
            "qtr.frequency" in name
            or "qtr.adapters" in name
        ):
            new_params.append(p)
        else:
            old_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": new_params,
                "lr": 1e-4,
                "weight_decay": 1e-4,
            },
            {
                "params": old_params,
                "lr": 1e-5,
                "weight_decay": 1e-4,
            },
        ]
    )

    train_records = fair.e.read_manifest(
        e6.MANIFEST,
        cfg["data"]["root"],
        "train",
    )

    val_records = fair.e.read_manifest(
        e6.MANIFEST,
        cfg["data"]["root"],
        "val",
    )

    train_base = FairDataset(
        train_records,
        OUT / "cache_train",
        True,
    )

    val_base = FairDataset(
        val_records,
        OUT / "cache_val",
        False,
    )

    train_ds = e6.MixedQuality(train_base)

    loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    loss_fn = stable.v3.e24.ReTraceLoss.from_config(
        stable.v3.e24.clean_loss_config(
            cfg["loss"]
        )
    ).cuda()

    original_summary = json.loads(
        json.dumps(
            __import__("yaml").safe_load(
                (e6.OUT / "summary.yaml").read_text()
            )
        )
    )

    baseline = original_summary["history"][-1]

    locked_threshold = float(
        baseline["clean"]["oracle"]["threshold"]
    )

    history = []

    print("CAT DCT branch removed:", model.dct_branch is None)
    print("Locked threshold:", locked_threshold)
    print(
        "Trainable params:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    for epoch in range(1, EPOCHS + 1):

        model.train()

        # Keep frozen content encoder in eval mode.
        model.content_encoder.eval()

        total = 0.0
        started = time.time()

        for step, batch in enumerate(loader, 1):

            batch = {
                k: (
                    v.cuda(non_blocking=True)
                    if torch.is_tensor(v)
                    else v
                )
                for k, v in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                out = stable.v3.fwd(
                    model,
                    batch,
                    cfg,
                    False,
                )

                loss = stable.v3.objective(
                    out,
                    batch,
                    loss_fn,
                    cfg,
                )["total"]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ],
                5.0,
            )

            optimizer.step()

            total += float(loss)

            if (
                step == 1
                or step % 100 == 0
                or step == len(loader)
            ):
                print(
                    f"[pilot] epoch={epoch}"
                    f" step={step}/{len(loader)}"
                    f" loss={total/step:.6f}",
                    flush=True,
                )

        model.eval()

        results = evaluate(
            model,
            val_base,
            cfg,
            locked_threshold,
        )

        row = {
            "epoch": epoch,
            "train_loss": total / len(loader),
            "elapsed_minutes": (
                time.time() - started
            ) / 60,
            **results,
        }

        history.append(row)

        print(
            json.dumps(
                row,
                indent=2,
            ),
            flush=True,
        )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "history": history,
                "experiment":
                    "Light_Frequency_QTR_Pilot",
            },
            OUT / f"epoch{epoch:02d}.pt",
        )

        (OUT / "history.json").write_text(
            json.dumps(
                history,
                indent=2,
            )
        )

    print("\n========== FINAL ==========")

    for row in history:
        print(
            f"E{row['epoch']} | "
            f"Clean={row['clean']['locked_f1']:.4f} | "
            f"Q90={row['q90_420']['locked_f1']:.4f} | "
            f"Q80={row['q80_420']['locked_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
