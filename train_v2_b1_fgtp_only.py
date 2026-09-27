#!/usr/bin/env python
"""Train B1 FGTP-only probe with the B0 body frozen."""

from __future__ import annotations

import argparse
import csv
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
from train_v2_b0 import load_initial_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init")
    parser.add_argument("--device")
    return parser.parse_args()


def freeze_except_fgtp(model: torch.nn.Module) -> list[str]:
    names = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith("fgtp.")
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    return names


def area_bucket(ratio: float) -> str:
    if ratio <= 0.0:
        return "background"
    if ratio < 0.005:
        return "lt_0p5"
    if ratio < 0.01:
        return "0p5_1"
    if ratio < 0.03:
        return "1_3"
    if ratio < 0.10:
        return "3_10"
    return "gte_10"


def add(values: dict[str, dict[str, float]], key: str, tensor: torch.Tensor, mask: torch.Tensor) -> None:
    selected = tensor[mask]
    if selected.numel() == 0:
        return
    values[key]["sum"] += float(selected.sum().cpu())
    values[key]["count"] += float(selected.numel())


@torch.inference_mode()
def fgtp_diagnostics(model: torch.nn.Module, loader, device: torch.device, max_batches: int) -> list[dict[str, float | int | str]]:
    buckets = ["background", "lt_0p5", "0p5_1", "1_3", "3_10", "gte_10"]
    fields = [
        "abs_t1_minus_n1", "abs_t2_minus_n2",
        "n1_fg_norm", "t1_fg_norm", "n2_fg_norm", "t2_fg_norm",
        "n1_bg_norm", "t1_bg_norm", "n2_bg_norm", "t2_bg_norm",
    ]
    stats = {bucket: {field: {"sum": 0.0, "count": 0.0} for field in fields} for bucket in buckets}
    counts = {bucket: 0 for bucket in buckets}
    model.eval()
    seen = 0
    for batch in loader:
        if seen >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        outputs = model(image)
        diag = outputs["fgtp"]
        seen += 1
        for sample in range(image.shape[0]):
            sample_mask = mask[sample:sample + 1]
            bucket = area_bucket(float(sample_mask.mean().cpu()))
            counts[bucket] += 1
            for scale, suffix in (("1", "1"), ("2", "2")):
                n = diag[f"n{suffix}"][sample:sample + 1].float()
                t = diag[f"t{suffix}"][sample:sample + 1].float()
                residual = (t - n).abs().mean(dim=1)
                n_norm = n.norm(dim=1)
                t_norm = t.norm(dim=1)
                resized = F.interpolate(sample_mask.float(), size=n.shape[-2:], mode="nearest")[:, 0] > 0.5
                bg = ~resized
                if bucket == "background":
                    add(stats[bucket], f"abs_t{scale}_minus_n{scale}", residual, bg)
                    add(stats[bucket], f"n{scale}_bg_norm", n_norm, bg)
                    add(stats[bucket], f"t{scale}_bg_norm", t_norm, bg)
                else:
                    add(stats[bucket], f"abs_t{scale}_minus_n{scale}", residual, resized)
                    add(stats[bucket], f"n{scale}_fg_norm", n_norm, resized)
                    add(stats[bucket], f"t{scale}_fg_norm", t_norm, resized)
                    add(stats[bucket], f"n{scale}_bg_norm", n_norm, bg)
                    add(stats[bucket], f"t{scale}_bg_norm", t_norm, bg)
    rows = []
    for bucket in buckets:
        row: dict[str, float | int | str] = {"bucket": bucket, "num_images": counts[bucket]}
        for field in fields:
            value = stats[bucket][field]
            row[field] = value["sum"] / max(1.0, value["count"])
        row["n1_fg_bg_contrast"] = float(row["n1_fg_norm"]) - float(row["n1_bg_norm"])
        row["t1_fg_bg_contrast"] = float(row["t1_fg_norm"]) - float(row["t1_bg_norm"])
        row["n2_fg_bg_contrast"] = float(row["n2_fg_norm"]) - float(row["n2_bg_norm"])
        row["t2_fg_bg_contrast"] = float(row["t2_fg_norm"]) - float(row["t2_bg_norm"])
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 3407)))
    device = choose_device(args.device)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output_dir / "config.yaml")

    train_loader = build_loader({**config["data"], "batch_size": config["training"]["batch_size"]}, "train")
    val_data = {**config["data"], "batch_size": config["evaluation"]["batch_size"]}
    val_loader = build_loader(val_data, "val", shuffle=False)
    diag_loader = build_loader({**val_data, "num_workers": 0}, "val", shuffle=False)

    model = build_model(config["model"]).to(device)
    init_path = args.init or config["training"].get("init_checkpoint")
    init_report = load_initial_weights(model, init_path, device, [])
    trainable_names = freeze_except_fgtp(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(yaml.safe_dump({
        "initial_weight_load": {"path": init_report["path"], "loaded": init_report["loaded"], "missing_count": len(init_report["missing"]), "missing_first12": init_report["missing"][:12], "unexpected_count": len(init_report["unexpected"]), "skipped_shape_first12": init_report["skipped_shape"][:12]},
        "trainable_parameter_count": trainable_params,
        "trainable_parameter_names": trainable_names,
    }, sort_keys=False).strip())
    if not trainable_names or any(not name.startswith("fgtp.") for name in trainable_names):
        raise RuntimeError("trainable parameters are not limited to FGTP")

    criterion = ReTraceLoss.from_config(config["loss"]).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    amp = bool(config["training"].get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_metric = -1.0
    selection_metric = str(config["training"].get("selection_metric", "pixel_f1_oracle"))

    history_path = output_dir / "history.csv"
    fieldnames = [
        "epoch", "lr", "train_total", "val_total", "val_f1", "val_f1_oracle",
        "val_precision_fixed", "val_recall_fixed", "val_iou_fixed",
        "val_precision_oracle", "val_recall_oracle", "val_iou_oracle",
        "val_auc", "oracle_threshold", "selection_metric", "selection_value",
        "fgtp_alpha1", "fgtp_alpha2", "tanh_alpha1", "tanh_alpha2",
        "mean_abs_t1_minus_n1", "mean_abs_t2_minus_n2", "fgtp_diag_path",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    for epoch in range(int(config["training"]["epochs"])):
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
        diag_rows = fgtp_diagnostics(model, diag_loader, device, int(config["training"].get("fgtp_diagnostic_batches", 256)))
        diag_path = output_dir / f"fgtp_diagnostics_epoch{epoch:03d}.yaml"
        diag_path.write_text(yaml.safe_dump(diag_rows, sort_keys=False), encoding="utf-8")
        alpha1 = model.fgtp.scale1.alpha.detach().float().cpu()
        alpha2 = model.fgtp.scale2.alpha.detach().float().cpu()
        current = float(metrics.get(selection_metric, metrics["pixel_f1_fixed"]))
        best_metric = max(best_metric, current)
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
            "trainable_parameter_names": trainable_names,
        }
        atomic_torch_save(state, output_dir / "last.pt")
        if current >= best_metric:
            atomic_torch_save(state, output_dir / "best.pt")
        if epoch + 1 in {1, 2, 3}:
            atomic_torch_save(state, output_dir / f"milestone_eplus{epoch + 1:02d}.pt")
        mean_t1 = next((float(r["abs_t1_minus_n1"]) for r in diag_rows if r["bucket"] == "lt_0p5"), 0.0)
        mean_t2 = next((float(r["abs_t2_minus_n2"]) for r in diag_rows if r["bucket"] == "lt_0p5"), 0.0)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
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
            "fgtp_alpha1": float(alpha1),
            "fgtp_alpha2": float(alpha2),
            "tanh_alpha1": float(torch.tanh(alpha1)),
            "tanh_alpha2": float(torch.tanh(alpha2)),
            "mean_abs_t1_minus_n1": mean_t1,
            "mean_abs_t2_minus_n2": mean_t2,
            "fgtp_diag_path": str(diag_path),
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        print(yaml.safe_dump(row, sort_keys=False).strip())
        scheduler.step()


if __name__ == "__main__":
    main()
