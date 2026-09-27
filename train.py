#!/usr/bin/env python
"""Train ReTraceNet from a YAML configuration."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import torch
import yaml

from retrace.data import build_loader
from retrace.engine import evaluate_model, train_one_epoch
from retrace.losses import ReTraceLoss
from retrace.model import build_model
from retrace.utils import atomic_torch_save, choose_device, load_checkpoint, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device")
    return parser.parse_args()


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
    trainable_patterns = list(config["training"].get("trainable_patterns", []) or [])
    freeze_patterns = list(config["training"].get("freeze_patterns", []) or [])
    if trainable_patterns:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(pattern in name for pattern in trainable_patterns)
    elif freeze_patterns:
        for name, parameter in model.named_parameters():
            if any(pattern in name for pattern in freeze_patterns):
                parameter.requires_grad = False
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_patterns or freeze_patterns:
        print(
            f"parameter freezing: trainable={trainable_params}/{total_params} "
            f"trainable_patterns={trainable_patterns} freeze_patterns={freeze_patterns}"
        )
    criterion = ReTraceLoss.from_config(config["loss"]).to(device)
    base_lr = float(config["training"]["learning_rate"])
    base_weight_decay = float(config["training"]["weight_decay"])
    parameter_group_configs = list(config["training"].get("parameter_groups", []) or [])
    if parameter_group_configs:
        named_parameters = list(model.named_parameters())
        assigned: set[str] = set()
        optimizer_groups = []
        for group_index, group_config in enumerate(parameter_group_configs):
            patterns = list(group_config.get("patterns", []) or [])
            if not patterns:
                continue
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
                print(f"parameter_group[{group_index}] matched no trainable params: patterns={patterns}")
                continue
            optimizer_groups.append({
                "params": params,
                "lr": float(group_config.get("lr", base_lr)),
                "weight_decay": float(group_config.get("weight_decay", base_weight_decay)),
            })
            print(
                f"parameter_group[{group_index}] params={sum(p.numel() for p in params)} "
                f"lr={optimizer_groups[-1]['lr']} weight_decay={optimizer_groups[-1]['weight_decay']} "
                f"patterns={patterns} examples={names[:8]}"
            )
        remaining = [parameter for name, parameter in named_parameters if parameter.requires_grad and name not in assigned]
        if remaining:
            optimizer_groups.append({"params": remaining, "lr": base_lr, "weight_decay": base_weight_decay})
            print(f"parameter_group[default] params={sum(p.numel() for p in remaining)} lr={base_lr} weight_decay={base_weight_decay}")
        if not optimizer_groups:
            raise ValueError("no trainable parameters matched parameter_groups")
        optimizer = torch.optim.AdamW(optimizer_groups)
    else:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=base_lr,
            weight_decay=base_weight_decay,
        )
    epochs = int(config["training"]["epochs"])
    warmup = int(config["training"].get("warmup_epochs", 0))

    def multiplier(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    amp = bool(config["training"].get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch, best_f1, stale = 0, -1.0, 0
    selection_metric = str(config["training"].get("selection_metric", "pixel_f1_oracle"))
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        strict_resume = bool(config["training"].get("strict_resume", True))
        resume_state = checkpoint["model"]
        skipped_shape = []
        if not strict_resume:
            model_state = model.state_dict()
            filtered_state = {}
            for key, value in resume_state.items():
                if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
                    skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                    continue
                filtered_state[key] = value
            resume_state = filtered_state
        load_result = model.load_state_dict(resume_state, strict=strict_resume)
        if not strict_resume:
            missing = list(getattr(load_result, "missing_keys", []))
            unexpected = list(getattr(load_result, "unexpected_keys", []))
            if missing or unexpected or skipped_shape:
                print(
                    f"non-strict resume from {args.resume}: "
                    f"missing={missing[:12]} unexpected={unexpected[:12]} "
                    f"skipped_shape={skipped_shape[:12]}"
                )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint.get("best_metric", checkpoint.get("best_f1", -1.0)))
        reset_optimizer = bool(config["training"].get("reset_optimizer_on_resume", False))
        if reset_optimizer:
            print(
                f"resume model from {args.resume} at epoch {start_epoch}; "
                "optimizer/scheduler/scaler reset from config"
            )
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])

    history_path = output_dir / "history.csv"
    fieldnames = [
        "epoch", "lr", "train_total", "val_total", "val_f1", "val_f1_oracle",
        "val_precision_fixed", "val_recall_fixed", "val_iou_fixed",
        "val_precision_oracle", "val_recall_oracle", "val_iou_oracle",
        "val_auc", "oracle_threshold", "selection_metric", "selection_value",
        "train_counterfactual", "val_counterfactual", "train_causal_correction",
        "val_causal_correction", "train_causal_gate", "val_causal_gate",
        "train_dhel_directional", "val_dhel_directional", "residual_intervention_gain",
        "val_manipulated_causal_drop", "val_background_counterfactual_change",
        "val_intervention_precision", "pos_gate_auc_for_need_up", "neg_gate_auc_for_need_down",
        "gate_fn_to_tp", "gate_fp_to_tn", "gate_tp_to_fn", "gate_tn_to_fp",
        "gate_good_flips", "gate_bad_flips", "dhel_feature_fusion_gain",
        "parameter_group_lrs",
    ]
    if start_epoch == 0 or not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    for epoch in range(start_epoch, epochs):
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
            gate_threshold=float(config["loss"].get("causal_correction_threshold", config["evaluation"].get("threshold", 0.5))),
        )
        current_f1 = metrics["pixel_f1_fixed"]
        selection_value = float(metrics.get(selection_metric, current_f1))
        improved = selection_value > best_f1
        if improved:
            best_f1, stale = selection_value, 0
        else:
            stale += 1
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": best_f1,
            "best_metric": best_f1,
            "selection_metric": selection_metric,
            "config": config,
        }
        atomic_torch_save(state, output_dir / "last.pt")
        if improved:
            atomic_torch_save(state, output_dir / "best.pt")
        residual_gain = ""
        if hasattr(model, "residual_intervention_gain"):
            residual_gain = float(torch.sigmoid(model.residual_intervention_gain.detach()).cpu())
        dhel_feature_fusion_gain = ""
        if hasattr(model, "dhel_feature_fusion_gain"):
            dhel_feature_fusion_gain = float(torch.sigmoid(model.dhel_feature_fusion_gain.detach()).cpu())
        parameter_group_lrs = ";".join(str(group["lr"]) for group in optimizer.param_groups)
        row = {
            "epoch": epoch,
            "lr": epoch_lr,
            "train_total": train_losses["total"],
            "val_total": val_losses["total"],
            "val_f1": current_f1,
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
            "selection_value": selection_value,
            "train_counterfactual": train_losses.get("counterfactual", 0.0),
            "val_counterfactual": val_losses.get("counterfactual", 0.0),
            "train_causal_correction": train_losses.get("causal_correction", 0.0),
            "val_causal_correction": val_losses.get("causal_correction", 0.0),
            "train_causal_gate": train_losses.get("causal_gate", 0.0),
            "val_causal_gate": val_losses.get("causal_gate", 0.0),
            "train_dhel_directional": train_losses.get("dhel_directional", 0.0),
            "val_dhel_directional": val_losses.get("dhel_directional", 0.0),
            "residual_intervention_gain": residual_gain,
            "val_manipulated_causal_drop": metrics.get("manipulated_causal_drop"),
            "val_background_counterfactual_change": metrics.get("background_counterfactual_change"),
            "val_intervention_precision": metrics.get("intervention_precision"),
            "pos_gate_auc_for_need_up": metrics.get("pos_gate_auc_for_need_up"),
            "neg_gate_auc_for_need_down": metrics.get("neg_gate_auc_for_need_down"),
            "gate_fn_to_tp": metrics.get("gate_fn_to_tp"),
            "gate_fp_to_tn": metrics.get("gate_fp_to_tn"),
            "gate_tp_to_fn": metrics.get("gate_tp_to_fn"),
            "gate_tn_to_fp": metrics.get("gate_tn_to_fp"),
            "gate_good_flips": metrics.get("gate_good_flips"),
            "gate_bad_flips": metrics.get("gate_bad_flips"),
            "dhel_feature_fusion_gain": dhel_feature_fusion_gain,
            "parameter_group_lrs": parameter_group_lrs,
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow(row)
        print(yaml.safe_dump(row, sort_keys=False).strip())
        scheduler.step()
        if stale >= int(config["training"].get("early_stopping", epochs)):
            print(f"early stopping at epoch {epoch}")
            break


if __name__ == "__main__":
    main()
