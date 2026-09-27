#!/usr/bin/env python
"""Step-0 equivalence and gradient smoke checks for B1 HCV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.utils import choose_device, load_checkpoint, load_config, set_seed
from retracenet.model import build_model
from train_v2_b1_hcv_only import freeze_except_hcv, strict_b0_load_report
from train_v2_b0 import load_initial_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--b0-config", default="/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/b0_fulltrain_v2_from_e5/config.yaml")
    parser.add_argument("--hcv-config", default="configs/casia2_retracenet_b1_hcv_only_probe.yaml")
    parser.add_argument("--checkpoint", default="/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/b0_fulltrain_v2_from_e5/best.pt")
    parser.add_argument("--output", default="/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/b1_hcv_only_probe_from_b0_e2/pretrain_check.json")
    return parser.parse_args()


def load_checkpoint_to(model: torch.nn.Module, path: str, device: torch.device) -> dict[str, object]:
    checkpoint = load_checkpoint(path, device)
    state = checkpoint["model"]
    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
    result = model.load_state_dict(compatible, strict=False)
    skipped_shape = [k for k, v in state.items() if k in model_state and tuple(v.shape) != tuple(model_state[k].shape)]
    unexpected = [k for k in state if k not in model_state]
    return {
        "loaded": len(compatible),
        "missing": list(result.missing_keys),
        "unexpected": unexpected,
        "skipped_shape": skipped_shape,
        "epoch": checkpoint.get("epoch"),
        "best_metric": checkpoint.get("best_metric"),
    }


def sum_named_grad(model: torch.nn.Module, include: str, exclude: str | None = None) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if include not in name:
            continue
        if exclude is not None and exclude in name:
            continue
        if parameter.grad is not None:
            total += float(parameter.grad.detach().abs().sum().cpu())
    return total


def main() -> None:
    args = parse_args()
    set_seed(3407)
    device = choose_device(args.device or ("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")))
    b0_config = load_config(args.b0_config)
    hcv_config = load_config(args.hcv_config)
    b0 = build_model(b0_config["model"]).to(device).eval()
    hcv = build_model(hcv_config["model"]).to(device).eval()
    b0_report = load_checkpoint_to(b0, args.checkpoint, device)
    hcv_report = load_initial_weights(hcv, args.checkpoint, device, [])
    strict_b0_load_report(hcv_report)
    criterion = ReTraceLoss.from_config(hcv_config["loss"]).to(device)
    loader = build_loader({**hcv_config["data"], "batch_size": 1, "num_workers": 0}, "val", shuffle=False)

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
            out1 = hcv(image)
            max_diff = max(max_diff, float((out0["mask_logits"] - out1["mask_logits"]).abs().max().cpu()))
            loss0 = criterion(out0, mask, label)["total"]
            loss1 = criterion(out1, mask, label)["total"]
            max_loss_diff = max(max_loss_diff, float((loss0 - loss1).abs().cpu()))
            if checked == 0:
                diag = out1["hcv"]
                first_shapes = {
                    "image": list(image.shape),
                    "proposals": [list(x.shape) for x in diag["proposals"]],
                    "deltas": [list(x.shape) for x in diag["deltas"]],
                    "energies": [list(x.shape) for x in diag["energies"]],
                    "evidence_features": [list(x.shape) for x in diag["evidence_features"]],
                    "mask_logits": list(out1["mask_logits"].shape),
                }
            checked += 1
    if max_diff > 1e-6 or max_loss_diff > 1e-6:
        raise RuntimeError(f"step-0 mismatch logits={max_diff} loss={max_loss_diff}")

    hcv.train()
    trainable_names = freeze_except_hcv(hcv)
    if not trainable_names or any(not name.startswith("hcv.") for name in trainable_names):
        raise RuntimeError(f"bad trainable names: {trainable_names}")
    optimizer = torch.optim.AdamW([p for p in hcv.parameters() if p.requires_grad], lr=float(hcv_config["training"]["learning_rate"]), weight_decay=float(hcv_config["training"]["weight_decay"]))
    alpha_start = hcv.hcv.alpha.detach().cpu().tolist()
    lambda_start = [float(hcv.hcv.lambda_1.detach().cpu()), float(hcv.hcv.lambda_2.detach().cpu())]
    smoke = []
    for step, batch in zip(range(5), loader):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        outputs = hcv(image)
        losses = criterion(outputs, mask, label)
        losses["total"].backward()
        alpha_grad = hcv.hcv.alpha.grad.detach().abs().cpu().tolist() if hcv.hcv.alpha.grad is not None else [0.0, 0.0, 0.0]
        smoke.append({
            "step": step + 1,
            "loss": float(losses["total"].detach().cpu()),
            "alpha_grad_abs": alpha_grad,
            "alpha": hcv.hcv.alpha.detach().cpu().tolist(),
            "lambda_grad_abs": [
                0.0 if hcv.hcv.lambda_1.grad is None else float(hcv.hcv.lambda_1.grad.detach().abs().cpu()),
                0.0 if hcv.hcv.lambda_2.grad is None else float(hcv.hcv.lambda_2.grad.detach().abs().cpu()),
            ],
            "proposal_grad_abs_sum": sum_named_grad(hcv, "local_proposal.proposal"),
            "reference_grad_abs_sum": sum_named_grad(hcv, "reference."),
            "verifier_grad_abs_sum": sum_named_grad(hcv, "verifier."),
            "adapter_grad_abs_sum": sum_named_grad(hcv, "adapter."),
            "internal_grad_abs_sum": sum_named_grad(hcv, "hcv.", exclude="alpha"),
        })
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    alpha_end = hcv.hcv.alpha.detach().cpu().tolist()
    if max(smoke[0]["alpha_grad_abs"]) <= 0.0:
        raise RuntimeError(f"HCV alpha did not receive first-step gradient: {smoke[0]}")
    if max(abs(x) for x in alpha_end) <= 0.0:
        raise RuntimeError("HCV alpha did not leave zero")
    if smoke[-1]["proposal_grad_abs_sum"] <= 0.0:
        raise RuntimeError("HCV proposal parameters did not receive gradients by step 5")
    if smoke[-1]["internal_grad_abs_sum"] <= 0.0:
        raise RuntimeError("HCV internal parameters did not receive gradients by step 5")

    param_count = sum(p.numel() for n, p in hcv.named_parameters() if n.startswith("hcv."))
    summary = {
        "status": "PASS",
        "device": str(device),
        "b0_load": {"loaded": b0_report["loaded"], "epoch": b0_report["epoch"], "best_metric": b0_report["best_metric"]},
        "hcv_load": {
            "loaded": hcv_report["loaded"],
            "missing_count": len(hcv_report["missing"]),
            "missing_first20": hcv_report["missing"][:20],
            "unexpected_count": len(hcv_report["unexpected"]),
            "skipped_shape_count": len(hcv_report["skipped_shape"]),
        },
        "step0": {"batches": checked, "max_abs_diff": max_diff, "max_loss_diff": max_loss_diff},
        "shapes": first_shapes,
        "alpha_start": alpha_start,
        "alpha_end": alpha_end,
        "lambda_start": lambda_start,
        "hcv_params": param_count,
        "trainable_parameter_count": sum(p.numel() for p in hcv.parameters() if p.requires_grad),
        "trainable_parameter_names": trainable_names,
        "smoke_5step": smoke,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
