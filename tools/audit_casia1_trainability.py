"""CASIA1 train/validation failure triage for the V3 in-domain pipeline.

Runs two decisive checks on GPU0:
1. Evaluate the stopped CASIA1 checkpoint on its full fake training split with
   the exact same evaluator used for validation.
2. Rebuild the same public-pretrained V3 initialization and attempt to overfit
   32 fixed fake images without stochastic spatial augmentation.
"""
import copy
import json
import math
import random
import traceback
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from retrace.data import Record
import run_balanced_residual_v3_full30 as v3
import run_cross_dataset_after_chain as cross
from fair_training_data import FairDataset


ROOT = Path("/group4/zouwenxuan/iml_generalization")
RUN = ROOT / "outputs/retracenet_v2/v3_offline_master_20260912/indomain_CASIA1"
OUT = RUN / "trainability_audit"
CKPT = RUN / "last.pt"
GROUPS = ("content_encoder.", "forensic_stream.", "dct_branch.",
          "balanced_forensic_fusions.", "qtr.", "hcv2.", "decoder.",
          "mask_head.", "dher.")


def dump(name, value):
    v3.e.dump(OUT / name, value)


def records(rows, split):
    return [Record(image=Path(r["image"]),
                   mask=Path(r["mask"]) if r.get("mask") else None,
                   split=split, dataset="CASIA1", label=int(r["label"]))
            for r in rows]


def fixed_batch(ds, indices, seed=73001):
    plan = [(idx, seed + j) for j, idx in enumerate(indices)]
    return next(iter(DataLoader(ds, batch_size=len(indices), sampler=plan,
                                num_workers=0, pin_memory=True)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {"status": "starting", "checkpoint": str(CKPT)}
    dump("summary.yaml", state)
    try:
        split = json.loads((RUN / "split_721.json").read_text())
        train_fake = records([r for r in split["train"] if int(r["label"]) == 1], "train")
        checkpoint = torch.load(CKPT, map_location="cpu")
        cfg = copy.deepcopy(checkpoint["config"])

        # Check 1: exact common evaluator on every fake training image.
        model = v3.e.build_final_model(cfg["model"]).cuda().eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        state.update(status="evaluating_full_train_fake", train_fake_count=len(train_fake),
                     checkpoint_epoch=int(checkpoint.get("epoch", -1)))
        dump("summary.yaml", state)
        train_result = cross.evaluate_dataset(
            model, train_fake, OUT / "jpeg_cache_train", None, None,
            "CASIA1_TRAIN_FAKE", OUT / "train_fake_epoch10",
            evaluation_mode="diagnostic_train_same_evaluator",
            threshold_source="CASIA1_train_oracle_diagnostic")
        state["train_fake"] = train_result["pixel_localization_fake_only"]
        dump("summary.yaml", state)
        del model
        torch.cuda.empty_cache()

        # Select deterministic <=512 fake images so training sees every GT pixel
        # and no random crop can miss the manipulation.
        eligible = []
        for i, rec in enumerate(train_fake):
            with Image.open(rec.image) as im:
                w, h = im.size
            if h <= 512 and w <= 512:
                eligible.append((i, rec, h, w))
        if len(eligible) < 32:
            raise RuntimeError(f"Only {len(eligible)} fake train images are <=512")
        rng = random.Random(3407)
        rng.shuffle(eligible)
        chosen = eligible[:32]
        chosen_records = [x[1] for x in chosen]
        (OUT / "overfit32_manifest.json").write_text(json.dumps([
            {"image": str(r.image), "mask": str(r.mask), "height": h, "width": w}
            for _, r, h, w in chosen], indent=2))

        # Check the actual supervision before optimizing.
        train_ds = FairDataset(chosen_records, OUT / "jpeg_cache_overfit", True)
        probe = fixed_batch(train_ds, list(range(4)))
        valid = probe["valid"].float()
        gt = probe["mask"].float()
        gt_positive = ((gt * valid).flatten(1).sum(1) > 0)
        supervision = {
            "batch_fake_fraction": float(probe["original_label"].float().mean()),
            "batch_positive_crop_fraction": float(gt_positive.float().mean()),
            "batch_valid_foreground_ratio": float((gt * valid).sum() / valid.sum().clamp_min(1)),
            "per_image_valid_foreground_ratio": [float(x) for x in
                ((gt * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1))],
            "rgb_dct_same_jpeg_paths": bool(all(str(a) == str(b) for a, b in zip(probe["path"], probe["jpeg_path"]))),
            "crop_origins": probe["crop_origin"].tolist(),
            "all_origins_block_aligned": bool((probe["crop_origin"] % 8 == 0).all()),
        }
        if supervision["batch_positive_crop_fraction"] != 1.0:
            raise RuntimeError(f"Fixed small fake images lost GT foreground: {supervision}")

        # Same initialization policy as run_v3_indomain_721.py: public pretrained
        # encoders via constructor, newly initialized task modules.
        torch.manual_seed(3407)
        np.random.seed(3407)
        random.seed(3407)
        model = v3.e.build_final_model(cfg["model"]).cuda()
        v3.mode_train(model)
        seg = v3.e24.ReTraceLoss.from_config(v3.e24.clean_loss_config(cfg["loss"])).cuda()
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=.9, weight_decay=5e-4)
        state.update(status="overfitting_32", overfit_initialization="same_public_pretrained_constructor",
                     supervision_audit=supervision)
        dump("summary.yaml", state)

        history = []
        max_steps = 1600
        evaluate_every = 100
        # Deterministic cycling; images are <=512, hence seed cannot change crop.
        for step in range(1, max_steps + 1):
            ids = [((step - 1) * 4 + j) % 32 for j in range(4)]
            batch = fixed_batch(train_ds, ids, 900000 + step * 10)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = v3.fwd(model, batch, cfg, True)
                terms = v3.objective(output, batch, seg, cfg)
                loss = terms["total"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}: {float(loss)}")
            loss.backward()
            grad = {}
            if step == 1:
                for prefix in GROUPS:
                    grad[prefix] = math.sqrt(sum(float(p.grad.float().square().sum())
                        for n, p in model.named_parameters()
                        if n.startswith(prefix) and p.grad is not None))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            before = None
            if step == 1:
                before = {n: p.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            opt.step()
            if step == 1:
                delta = {prefix: sum(float((p.detach() - before[n]).abs().sum())
                    for n, p in model.named_parameters() if n in before and n.startswith(prefix))
                    for prefix in GROUPS}
                state["step1_grad_norm"] = grad
                state["step1_parameter_delta_L1"] = delta
                dump("summary.yaml", state)
            if step == 1 or step % 25 == 0:
                dump("progress.yaml", {"status": "overfitting_32", "step": step,
                                       "max_steps": max_steps, "loss": float(loss)})
                print("[overfit32]", step, max_steps, float(loss), flush=True)
            if step % evaluate_every == 0:
                result = cross.evaluate_dataset(
                    model, chosen_records, OUT / "jpeg_cache_overfit", None, None,
                    "CASIA1_OVERFIT32", OUT / f"overfit_eval_step{step:04d}",
                    evaluation_mode="diagnostic_overfit_same_evaluator",
                    threshold_source="overfit32_oracle_diagnostic")
                metric = result["pixel_localization_fake_only"]["test_f1_oracle_diagnostic"]
                row = {"step": step, "loss": float(loss), "f1": float(metric["f1"]),
                       "precision": float(metric["precision"]), "recall": float(metric["recall"]),
                       "threshold": float(metric["threshold"]),
                       "auc": result["pixel_localization_fake_only"]["auc"]}
                history.append(row)
                state["overfit_history"] = history
                dump("summary.yaml", state)
                if row["f1"] >= .90:
                    state.update(status="completed", conclusion="overfit_pass",
                                 stop_reason="32-image train fake-only pooled F1 reached 0.90",
                                 overfit_pass=True, final_step=step)
                    dump("summary.yaml", state)
                    return
                v3.mode_train(model)
        state.update(status="completed", conclusion="overfit_fail",
                     stop_reason="maximum 1600 steps without reaching F1 0.90",
                     overfit_pass=False, final_step=max_steps)
        dump("summary.yaml", state)
    except BaseException:
        state.update(status="failed", stop_reason=traceback.format_exc())
        dump("summary.yaml", state)
        raise


if __name__ == "__main__":
    main()
