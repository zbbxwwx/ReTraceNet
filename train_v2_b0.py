#!/usr/bin/env python
"""Train the clean v2 B0 base from a YAML configuration."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml

from retrace.data import build_loader
from retrace.engine import evaluate_model, train_one_epoch
from retrace.losses import ReTraceLoss
from retrace.utils import atomic_torch_save, choose_device, load_checkpoint, load_config, set_seed
from retracenet.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init")
    parser.add_argument("--device")
    return parser.parse_args()


def load_initial_weights(model: torch.nn.Module, path: str | None, device: torch.device, patterns: list[str]) -> dict[str, Any]:
    if not path:
        return {"path": "", "loaded": 0, "missing": list(model.state_dict().keys()), "unexpected": [], "skipped_shape": []}
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    filtered = {}
    unexpected = []
    skipped_shape = []
    for key, value in source.items():
        if patterns and not any(pattern in key for pattern in patterns):
            continue
        if key not in target:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            skipped_shape.append((key, tuple(value.shape), tuple(target[key].shape)))
            continue
        filtered[key] = value
    result = model.load_state_dict(filtered, strict=False)
    return {
        "path": path,
        "loaded": len(filtered),
        "missing": list(result.missing_keys),
        "unexpected": unexpected,
        "skipped_shape": skipped_shape,
    }


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    base_lr = float(config["training"]["learning_rate"])
    base_weight_decay = float(config["training"]["weight_decay"])
    group_configs = list(config["training"].get("parameter_groups", []) or [])
    named_parameters = list(model.named_parameters())
    if not group_configs:
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=base_lr, weight_decay=base_weight_decay)
    assigned: set[str] = set()
    groups = []
    for index, group_config in enumerate(group_configs):
        patterns = list(group_config.get("patterns", []) or [])
        params = []
        names = []
        for name, parameter in named_parameters:
            if not parameter.requires_grad or name in assigned:
                continue
            if any(pattern in name for pattern in patterns):
                params.append(parameter)
                names.append(name)
                assigned.add(name)
        if not params:
            print(f"parameter_group[{index}] matched no trainable params: patterns={patterns}")
            continue
        groups.append({"params": params, "lr": float(group_config.get("lr", base_lr)), "weight_decay": float(group_config.get("weight_decay", base_weight_decay))})
        print(f"parameter_group[{index}] params={sum(p.numel() for p in params)} lr={groups[-1]['lr']} weight_decay={groups[-1]['weight_decay']} patterns={patterns} examples={names[:8]}")
    remaining = [p for name, p in named_parameters if p.requires_grad and name not in assigned]
    if remaining:
        groups.append({"params": remaining, "lr": base_lr, "weight_decay": base_weight_decay})
        print(f"parameter_group[default] params={sum(p.numel() for p in remaining)} lr={base_lr} weight_decay={base_weight_decay}")
    if not groups:
        raise ValueError("no trainable parameters matched parameter_groups")
    return torch.optim.AdamW(groups)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 3407)))
    device = choose_device(args.device)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output_dir / "config.yaml")

    train_data = {**config["data"], "batch_size": config["training"]["batch_size"]}
    val_data = {**config["data"], "batch_size": config["evaluation"]["batch_size"]}
    train_loader = build_loader(train_data, "train")
    val_loader = build_loader(val_data, "val", shuffle=False)

    model = build_model(config["model"]).to(device)
    init_path = args.init or config["training"].get("init_checkpoint")
    init_report = load_initial_weights(model, init_path, device, list(config["training"].get("init_load_patterns", []) or []))
    print(yaml.safe_dump({"initial_weight_load": init_report, "optimizer": "reset_from_config"}, sort_keys=False).strip())

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: trainable={trainable_params} total={total_params}")

    criterion = ReTraceLoss.from_config(config["loss"]).to(device)
    optimizer = build_optimizer(model, config)
    epochs = int(config["training"]["epochs"])

    constant_epochs = int(config["training"].get("constant_lr_epochs", 0))
    min_lr_ratio = float(config["training"].get("min_lr_ratio", 0.0))

    def multiplier(epoch: int) -> float:
        # Keep the initial learning rate unchanged for the first
        # `constant_epochs` epochs.
        if epoch < constant_epochs:
            return 1.0

        decay_epochs = max(1, epochs - constant_epochs)
        progress = (epoch - constant_epochs + 1) / decay_epochs
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Cosine decay from 1.0 to min_lr_ratio instead of zero.
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    amp = bool(config["training"].get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_metric = -1.0
    stale = 0
    selection_metric = str(config["training"].get("selection_metric", "pixel_f1_oracle"))
    milestone_offsets = set(int(x) for x in config["training"].get("milestone_offsets", []) or [])

    history_path = output_dir / "history.csv"
    fieldnames = [
        "epoch", "lr", "train_total", "val_total", "val_f1", "val_f1_oracle",
        "val_precision_fixed", "val_recall_fixed", "val_iou_fixed",
        "val_precision_oracle", "val_recall_oracle", "val_iou_oracle",
        "val_auc", "oracle_threshold", "selection_metric", "selection_value", "parameter_group_lrs",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    for epoch in range(epochs):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_losses = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            amp,
            float(config["training"].get("grad_clip", 1.0)),
            float(config["training"].get("consistency_probability", 0.0)),
            float(config["training"].get("intervention_teacher_probability", 0.0)),
            int(config["training"].get("gradient_accumulation_steps", 1)),
            epoch,
        )
        val_losses, metrics = evaluate_model(
            model,
            criterion,
            val_loader,
            device,
            amp,
            float(config["evaluation"].get("threshold", 0.5)),
            int(config["evaluation"].get("thresholds", 101)),
        )
        current = float(metrics.get(selection_metric, metrics["pixel_f1_fixed"]))
        improved = current > best_metric
        if improved:
            best_metric = current
            stale = 0
        else:
            stale += 1
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_metric": best_metric,
            "selection_metric": selection_metric,
            "config": config,
            "initial_weight_load": init_report,
        }
        atomic_torch_save(state, output_dir / "last.pt")
        if improved:
            atomic_torch_save(state, output_dir / "best.pt")
        if (epoch + 1) in milestone_offsets:
            atomic_torch_save(state, output_dir / f"milestone_eplus{epoch + 1:02d}.pt")
        row = {
            "epoch": epoch,
            "lr": epoch_lr,
            "train_total": train_losses["total"],
            "val_total": val_losses["total"],
            "val_f1": metrics["pixel_f1_fixed"],
            "val_f1_oracle": metrics.get("pixel_f1_oracle"),
            "val_precision_fixed": metrics.get("pixel_precision_fixed"),
            "val_recall_fixed": metrics.get("pixel_recall_fixed"),
            "val_iou_fixed": metrics.get("pixel_iou_fixed"),
            "val_precision_oracle": metrics.get("pixel_precision_oracle"),
            "val_recall_oracle": metrics.get("pixel_recall_oracle"),
            "val_iou_oracle": metrics.get("pixel_iou_oracle"),
            "val_auc": metrics["pixel_auc"],
            "oracle_threshold": metrics.get("oracle_threshold"),
            "selection_metric": selection_metric,
            "selection_value": current,
            "parameter_group_lrs": ";".join(str(group["lr"]) for group in optimizer.param_groups),
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        print(yaml.safe_dump(row, sort_keys=False).strip())
        scheduler.step()
        if stale >= int(config["training"].get("early_stopping", epochs)):
            print(f"early stopping at epoch {epoch}")
            break


if __name__ == "__main__":
    main()
