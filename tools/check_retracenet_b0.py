#!/usr/bin/env python
"""Pre-train checks for the v2 B0 base."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.utils import choose_device, load_config, set_seed
from retracenet.model import build_model
from train_v2_b0 import build_optimizer, load_initial_weights


def finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def main() -> None:
    config_path = "configs/casia2_retracenet_b0_strong_base.yaml"
    config = load_config(config_path)
    set_seed(int(config.get("seed", 3407)))
    device = choose_device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader = build_loader({**config["data"], "batch_size": config["training"]["batch_size"], "num_workers": 0}, "train")
    val_loader = build_loader({**config["data"], "batch_size": config["evaluation"]["batch_size"], "num_workers": 0}, "val", shuffle=False)
    model = build_model(config["model"]).to(device)
    report = load_initial_weights(
        model,
        config["training"].get("init_checkpoint"),
        device,
        list(config["training"].get("init_load_patterns", []) or []),
    )
    criterion = ReTraceLoss.from_config(config["loss"]).to(device)
    optimizer = build_optimizer(model, config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    batch = next(iter(train_loader))
    image = batch["image"].to(device)
    mask = batch["mask"].to(device)
    label = batch["label"].to(device)
    outputs = model(image)
    losses = criterion(outputs, mask, label)
    losses["total"].backward()
    grad_count = 0
    grad_abs = 0.0
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            grad_count += 1
            grad_abs += float(parameter.grad.detach().abs().sum().cpu())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    first_shapes = {
        "image": list(image.shape),
        "mask": list(mask.shape),
        "mask_logits": list(outputs["mask_logits"].shape),
        "aux_logits": [list(x.shape) for x in outputs["aux_logits"]],
        "content": [list(x.shape) for x in outputs["content_features"]],
    }
    if outputs["mask_logits"].shape[-2:] != mask.shape[-2:]:
        raise RuntimeError(f"prediction/GT size mismatch: {outputs['mask_logits'].shape} vs {mask.shape}")
    if not finite_tensor(outputs["mask_logits"]) or not finite_tensor(losses["total"]):
        raise RuntimeError("non-finite forward/loss detected")
    if grad_count == 0 or grad_abs <= 0.0:
        raise RuntimeError("no nonzero trainable gradients detected")

    step_stats = []
    previous = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    for step in range(5):
        batch = next(iter(train_loader))
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        outputs = model(image)
        losses = criterion(outputs, mask, label)
        losses["total"].backward()
        total_grad = 0.0
        max_grad = 0.0
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                value = float(parameter.grad.detach().abs().max().cpu())
                max_grad = max(max_grad, value)
                total_grad += float(parameter.grad.detach().abs().sum().cpu())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step_stats.append({"step": step + 1, "loss": float(losses["total"].detach().cpu()), "grad_abs_sum": total_grad, "grad_abs_max": max_grad})
    updated = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not torch.equal(previous[name].to(parameter.device), parameter.detach()):
            updated += 1

    val_probs = []
    val_nan = 0
    with torch.inference_mode():
        for index, batch in enumerate(val_loader):
            if index >= 32:
                break
            image = batch["image"].to(device)
            logits = model(image)["mask_logits"]
            if not finite_tensor(logits):
                val_nan += 1
            val_probs.append(logits.sigmoid().detach().float().cpu())
    val_all = torch.cat([x.flatten() for x in val_probs])
    sanity = {
        "val_batches": len(val_probs),
        "prob_mean": float(val_all.mean()),
        "prob_std": float(val_all.std(unbiased=False)),
        "prob_min": float(val_all.min()),
        "prob_max": float(val_all.max()),
        "nonfinite_batches": val_nan,
    }
    if sanity["nonfinite_batches"] or sanity["prob_std"] <= 1e-8:
        raise RuntimeError(f"bad 32-val sanity: {sanity}")

    blocked_names = ("Hierarchical", "Counterfactual", "DualHypothesis", "EvidenceReason", "Conflict", "BoundaryRefinement", "Reexamination")
    blocked_modules = sorted({type(module).__name__ for module in model.modules() if any(token in type(module).__name__ for token in blocked_names)})
    if blocked_modules:
        raise RuntimeError(f"blocked legacy-like modules present: {blocked_modules}")

    summary = {
        "status": "PASS",
        "device": str(device),
        "init_load": {
            "path": report["path"],
            "loaded": report["loaded"],
            "missing_count": len(report["missing"]),
            "missing_first12": report["missing"][:12],
            "unexpected_count": len(report["unexpected"]),
            "unexpected_first12": report["unexpected"][:12],
            "skipped_shape_first12": report["skipped_shape"][:12],
        },
        "params": {"total": total_params, "trainable": trainable_params},
        "shapes": first_shapes,
        "single_batch": {"loss": float(losses["total"].detach().cpu()), "grad_params": grad_count, "grad_abs_sum": grad_abs},
        "smoke_5step": step_stats,
        "updated_trainable_params": updated,
        "val32_sanity": sanity,
        "blocked_modules": blocked_modules,
    }
    out = Path("/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/b0_strong_base_10ep/pretrain_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
