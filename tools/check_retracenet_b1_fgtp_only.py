#!/usr/bin/env python
"""Step-0 and smoke checks for B1 FGTP."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.utils import choose_device, load_checkpoint, load_config, set_seed
from retracenet.model import build_model
from train_v2_b0 import build_optimizer, load_initial_weights


def load_checkpoint_to(model, path, device):
    checkpoint = load_checkpoint(path, device)
    state = checkpoint["model"]
    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
    result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "missing": list(result.missing_keys),
        "skipped": len(state) - len(compatible),
        "epoch": checkpoint.get("epoch"),
        "best_metric": checkpoint.get("best_metric"),
    }


def main() -> None:
    b0_config_path = "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/b0_fulltrain_v2_from_e5/config.yaml"
    b1_config_path = "configs/casia2_retracenet_b1_fgtp_only_probe.yaml"
    checkpoint_path = "/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/b0_fulltrain_v2_from_e5/best.pt"
    set_seed(3407)
    device = choose_device("cuda:0" if torch.cuda.is_available() else "cpu")
    b0_config = load_config(b0_config_path)
    b1_config = load_config(b1_config_path)
    b0 = build_model(b0_config["model"]).to(device).eval()
    b1 = build_model(b1_config["model"]).to(device).eval()
    b0_report = load_checkpoint_to(b0, checkpoint_path, device)
    b1_report = load_checkpoint_to(b1, checkpoint_path, device)
    criterion = ReTraceLoss.from_config(b1_config["loss"]).to(device)
    loader = build_loader({**b1_config["data"], "batch_size": 1, "num_workers": 0}, "val", shuffle=False)

    max_diff = 0.0
    max_loss_diff = 0.0
    checked = 0
    first_shapes = {}
    with torch.inference_mode():
        for batch in loader:
            if checked >= 32:
                break
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            out0 = b0(image)
            out1 = b1(image)
            diff = float((out0["mask_logits"] - out1["mask_logits"]).abs().max().cpu())
            loss0 = criterion(out0, mask, label)["total"]
            loss1 = criterion(out1, mask, label)["total"]
            max_diff = max(max_diff, diff)
            max_loss_diff = max(max_loss_diff, float((loss0 - loss1).abs().cpu()))
            if checked == 0:
                first_shapes = {
                    "image": list(image.shape),
                    "C": [list(x.shape) for x in out1["content_features"]],
                    "N1": list(out1["fgtp"]["n1"].shape),
                    "N2": list(out1["fgtp"]["n2"].shape),
                    "T1": list(out1["fgtp"]["t1"].shape),
                    "T2": list(out1["fgtp"]["t2"].shape),
                    "mask_logits": list(out1["mask_logits"].shape),
                }
            checked += 1
    if max_diff > 1e-6 or max_loss_diff > 1e-6:
        raise RuntimeError(f"step-0 mismatch logits={max_diff} loss={max_loss_diff}")

    b1.train()
    for name, parameter in b1.named_parameters():
        parameter.requires_grad_(name.startswith("fgtp."))
    trainable_names = [name for name, parameter in b1.named_parameters() if parameter.requires_grad]
    if not trainable_names or any(not name.startswith("fgtp.") for name in trainable_names):
        raise RuntimeError(f"bad trainable names: {trainable_names}")
    optimizer = torch.optim.AdamW([p for p in b1.parameters() if p.requires_grad], lr=float(b1_config["training"]["learning_rate"]), weight_decay=float(b1_config["training"]["weight_decay"]))
    alpha_start = [float(b1.fgtp.scale1.alpha.detach().cpu()), float(b1.fgtp.scale2.alpha.detach().cpu())]
    smoke = []
    for step, batch in zip(range(5), loader):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        outputs = b1(image)
        losses = criterion(outputs, mask, label)
        losses["total"].backward()
        alpha_grad = [
            0.0 if b1.fgtp.scale1.alpha.grad is None else float(b1.fgtp.scale1.alpha.grad.detach().abs().cpu()),
            0.0 if b1.fgtp.scale2.alpha.grad is None else float(b1.fgtp.scale2.alpha.grad.detach().abs().cpu()),
        ]
        internal_grad = 0.0
        for name, parameter in b1.named_parameters():
            if name.startswith("fgtp.") and "alpha" not in name and parameter.grad is not None:
                internal_grad += float(parameter.grad.detach().abs().sum().cpu())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        smoke.append({
            "step": step + 1,
            "loss": float(losses["total"].detach().cpu()),
            "alpha_grad_abs": alpha_grad,
            "alpha": [float(b1.fgtp.scale1.alpha.detach().cpu()), float(b1.fgtp.scale2.alpha.detach().cpu())],
            "internal_grad_abs_sum": internal_grad,
        })
    alpha_end = [float(b1.fgtp.scale1.alpha.detach().cpu()), float(b1.fgtp.scale2.alpha.detach().cpu())]
    if smoke[0]["alpha_grad_abs"][0] <= 0.0 or smoke[0]["alpha_grad_abs"][1] <= 0.0:
        raise RuntimeError(f"alpha did not receive first-step gradient: {smoke[0]}")
    if abs(alpha_end[0]) <= 0.0 or abs(alpha_end[1]) <= 0.0:
        raise RuntimeError("alpha did not leave zero")
    if smoke[-1]["internal_grad_abs_sum"] <= 0.0:
        raise RuntimeError("FGTP internal parameters did not receive gradients by step 5")

    fgtp_params = sum(p.numel() for n, p in b1.named_parameters() if n.startswith("fgtp."))
    summary = {
        "status": "PASS",
        "device": str(device),
        "b0_load": {"loaded": b0_report["loaded"], "epoch": b0_report["epoch"], "best_metric": b0_report["best_metric"]},
        "b1_load": {"loaded": b1_report["loaded"], "missing_count": len(b1_report["missing"]), "missing_first20": b1_report["missing"][:20], "skipped": b1_report["skipped"]},
        "step0": {"batches": checked, "max_abs_diff": max_diff, "max_loss_diff": max_loss_diff},
        "shapes": first_shapes,
        "fgtp_params": fgtp_params,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": sum(p.numel() for p in b1.parameters() if p.requires_grad),
        "alpha_start": alpha_start,
        "alpha_end": alpha_end,
        "smoke_5step": smoke,
    }
    out = Path("/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/b1_fgtp_only_probe_from_b0_e2/pretrain_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
