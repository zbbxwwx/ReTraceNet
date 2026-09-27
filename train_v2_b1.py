#!/usr/bin/env python
"""Train the v2 B1 FGTP screening run."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from retrace.data import build_loader
from retrace.engine import evaluate_model, train_one_epoch
from retrace.losses import ReTraceLoss
from retrace.utils import atomic_torch_save, choose_device, load_config, set_seed
from retracenet.model import build_model
from train_v2_b0 import build_optimizer, load_initial_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init")
    parser.add_argument("--device")
    return parser.parse_args()


def lr_multiplier(epoch: int, epochs: int, constant_epochs: int, min_lr_ratio: float) -> float:
    if epoch < constant_epochs:
        return 1.0
    progress = (epoch - constant_epochs) / max(1, epochs - constant_epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def empty_diag() -> dict[str, dict[str, float]]:
    return {
        f"{field}_{scale}_{bucket}": {"sum": 0.0, "count": 0.0}
        for field in ("residual", "n_norm", "t_norm")
        for scale in ("s1", "s2")
        for bucket in ("tiny", "medium", "large", "background")
    }


def add_stat(stats: dict[str, dict[str, float]], key: str, values: torch.Tensor, mask: torch.Tensor) -> None:
    selected = values[mask]
    if selected.numel() == 0:
        return
    stats[key]["sum"] += float(selected.sum().cpu())
    stats[key]["count"] += float(selected.numel())


def bucket_from_ratio(ratio: float) -> str:
    if ratio < 0.01:
        return "tiny"
    if ratio < 0.10:
        return "medium"
    return "large"


@torch.inference_mode()
def fgtp_diagnostics(model: torch.nn.Module, loader, device: torch.device, max_batches: int) -> dict[str, Any]:
    model.eval()
    stats = empty_diag()
    batches = 0
    for batch in loader:
        if batches >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        outputs = model(image)
        diag = outputs.get("fgtp", {})
        if not diag:
            break
        batches += 1
        ratio = float(mask.mean().cpu())
        bucket = bucket_from_ratio(ratio) if ratio > 0 else "background"
        for scale, suffix in (("s1", "1"), ("s2", "2")):
            n = diag[f"n{suffix}"].float()
            t = diag[f"t{suffix}"].float()
            residual = diag[f"t{suffix}_minus_n{suffix}"].float().norm(dim=1)
            n_norm = n.norm(dim=1)
            t_norm = t.norm(dim=1)
            resized = F.interpolate(mask.float(), size=n.shape[-2:], mode="nearest")[:, 0] > 0.5
            background = ~resized
            add_stat(stats, f"residual_{scale}_background", residual, background)
            add_stat(stats, f"n_norm_{scale}_background", n_norm, background)
            add_stat(stats, f"t_norm_{scale}_background", t_norm, background)
            if bucket != "background":
                add_stat(stats, f"residual_{scale}_{bucket}", residual, resized)
                add_stat(stats, f"n_norm_{scale}_{bucket}", n_norm, resized)
                add_stat(stats, f"t_norm_{scale}_{bucket}", t_norm, resized)
    result: dict[str, Any] = {"batches": batches}
    for key, value in stats.items():
        result[key] = value["sum"] / max(1.0, value["count"])
    for scale in ("s1", "s2"):
        for bucket in ("tiny", "medium", "large"):
            result[f"n_contrast_{scale}_{bucket}"] = result[f"n_norm_{scale}_{bucket}"] - result[f"n_norm_{scale}_background"]
            result[f"t_contrast_{scale}_{bucket}"] = result[f"t_norm_{scale}_{bucket}"] - result[f"t_norm_{scale}_background"]
    return result


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
    diag_loader = build_loader({**val_data, "num_workers": 0}, "val", shuffle=False)

    model = build_model(config["model"]).to(device)
    init_path = args.init or config["training"].get("init_checkpoint")
    init_report = load_initial_weights(model, init_path, device, list(config["training"].get("init_load_patterns", []) or []))
    print(yaml.safe_dump({"initial_weight_load": {"path": init_report["path"], "loaded": init_report["loaded"], "missing_count": len(init_report["missing"]), "missing_first12": init_report["missing"][:12], "unexpected_count": len(init_report["unexpected"]), "skipped_shape_first12": init_report["skipped_shape"][:12]}, "optimizer": "reset_from_config"}, sort_keys=False).strip())
    print(f"parameters: trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)} total={sum(p.numel() for p in model.parameters())}")

    criterion = ReTraceLoss.from_config(config["loss"]).to(device)
    optimizer = build_optimizer(model, config)
    epochs = int(config["training"]["epochs"])
    constant_epochs = int(config["training"].get("constant_epochs", 0))
    min_lr_ratio = float(config["training"].get("min_lr_ratio", 0.0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: lr_multiplier(epoch, epochs, constant_epochs, min_lr_ratio),
    )
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
        "val_auc", "oracle_threshold", "selection_metric", "selection_value",
        "fgtp_alpha1", "fgtp_alpha2", "parameter_group_lrs", "fgtp_diag_path",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    for epoch in range(epochs):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_losses = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device, amp,
            float(config["training"].get("grad_clip", 1.0)),
            float(config["training"].get("consistency_probability", 0.0)),
            float(config["training"].get("intervention_teacher_probability", 0.0)),
            int(config["training"].get("gradient_accumulation_steps", 1)), epoch,
        )
        val_losses, metrics = evaluate_model(
            model, criterion, val_loader, device, amp,
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
        diag = fgtp_diagnostics(model, diag_loader, device, int(config["training"].get("fgtp_diagnostic_batches", 128)))
        diag_path = output_dir / f"fgtp_diagnostics_epoch{epoch:03d}.yaml"
        diag_path.write_text(yaml.safe_dump(diag, sort_keys=False), encoding="utf-8")
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
        alpha1 = float(model.fgtp.scale1.alpha.detach().cpu())
        alpha2 = float(model.fgtp.scale2.alpha.detach().cpu())
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
            "fgtp_alpha1": alpha1,
            "fgtp_alpha2": alpha2,
            "parameter_group_lrs": ";".join(str(group["lr"]) for group in optimizer.param_groups),
            "fgtp_diag_path": str(diag_path),
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
