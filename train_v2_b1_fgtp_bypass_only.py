#!/usr/bin/env python
"""Train the B1 high-resolution FGTP bypass-only probe with the B0 body frozen."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrace.data import build_loader
from retrace.engine import evaluate_model, train_one_epoch
from retrace.losses import ReTraceLoss
from retrace.utils import atomic_torch_save, choose_device, load_config, set_seed
from retracenet.model import build_model
from train_v2_b0 import load_initial_weights


BUCKETS = ["background", "lt_0p25", "0p25_0p5", "0p5_1", "lt_0p5", "lt_1", "1_3", "3_10", "gte_10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init")
    parser.add_argument("--device")
    return parser.parse_args()


def freeze_except_bypass(model: torch.nn.Module) -> list[str]:
    names = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith("fgtp_bypass.")
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    return names


def strict_b0_load_report(report: dict[str, Any]) -> None:
    bad_missing = [name for name in report["missing"] if not name.startswith("fgtp_bypass.")]
    if bad_missing or report["unexpected"] or report["skipped_shape"]:
        raise RuntimeError(
            "B0 initialization is not clean for bypass: "
            f"bad_missing={bad_missing[:20]} unexpected={report['unexpected'][:20]} "
            f"skipped_shape={report['skipped_shape'][:20]}"
        )


def bucket_names(ratio: float) -> list[str]:
    if ratio <= 0.0:
        return []
    if ratio < 0.0025:
        names = ["lt_0p25"]
    elif ratio < 0.005:
        names = ["0p25_0p5"]
    elif ratio < 0.01:
        names = ["0p5_1"]
    elif ratio < 0.03:
        names = ["1_3"]
    elif ratio < 0.10:
        names = ["3_10"]
    else:
        names = ["gte_10"]
    if ratio < 0.005:
        names.append("lt_0p5")
    if ratio < 0.01:
        names.append("lt_1")
    return names


def add(stat: dict[str, float], values: torch.Tensor) -> None:
    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        return
    stat["sum"] += float(flat.sum().cpu())
    stat["sq"] += float(flat.square().sum().cpu())
    stat["count"] += float(flat.numel())


def mean(stat: dict[str, float]) -> float:
    return stat["sum"] / max(1.0, stat["count"])


def std(stat: dict[str, float]) -> float:
    m = mean(stat)
    return max(0.0, stat["sq"] / max(1.0, stat["count"]) - m * m) ** 0.5


@torch.inference_mode()
def bypass_diagnostics(model: torch.nn.Module, loader, device: torch.device, max_batches: int) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    stats = {bucket: {"q_hr": {"sum": 0.0, "sq": 0.0, "count": 0.0}} for bucket in BUCKETS}
    image_counts = {bucket: 0 for bucket in BUCKETS}
    delta_stats = {
        "delta_d1_abs": {"sum": 0.0, "sq": 0.0, "count": 0.0},
        "written_delta_d1_abs": {"sum": 0.0, "sq": 0.0, "count": 0.0},
        "p_hr_abs": {"sum": 0.0, "sq": 0.0, "count": 0.0},
    }
    model.eval()
    seen_batches = 0
    for batch in loader:
        if seen_batches >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        outputs = model(image)
        diag = outputs["fgtp_bypass"]
        p_hr = diag["p_hr"].float()
        q_hr = diag["p_hr_local_detail"].abs().mean(dim=1)
        add(delta_stats["delta_d1_abs"], diag["delta_d1"].abs())
        add(delta_stats["written_delta_d1_abs"], diag["written_delta_d1"].abs())
        add(delta_stats["p_hr_abs"], p_hr.abs())
        for sample in range(image.shape[0]):
            sample_mask = mask[sample:sample + 1].float()
            resized = F.interpolate(sample_mask, size=q_hr.shape[-2:], mode="nearest")[:, 0] > 0.5
            ratio = float(sample_mask.mean().cpu())
            bg_region = ~resized
            add(stats["background"]["q_hr"], q_hr[sample:sample + 1][bg_region])
            image_counts["background"] += 1
            for name in bucket_names(ratio):
                image_counts[name] += 1
                add(stats[name]["q_hr"], q_hr[sample:sample + 1][resized])
        seen_batches += 1

    bg = stats["background"]["q_hr"]
    bg_mean = mean(bg)
    bg_std = std(bg)
    eps = 1e-8
    rows = []
    for bucket in BUCKETS:
        q = stats[bucket]["q_hr"]
        rows.append({
            "bucket": bucket,
            "num_images": image_counts[bucket],
            "q_hr_mean": mean(q),
            "q_hr_std": std(q),
            "q_hr_pixels": int(q["count"]),
            "sep_vs_background": (mean(q) - bg_mean) / (bg_std + eps),
        })
    summary = {
        "mean_abs_delta_d1": mean(delta_stats["delta_d1_abs"]),
        "std_abs_delta_d1": std(delta_stats["delta_d1_abs"]),
        "mean_abs_written_delta_d1": mean(delta_stats["written_delta_d1_abs"]),
        "mean_abs_p_hr": mean(delta_stats["p_hr_abs"]),
        "sep_lt_0p5": next(row["sep_vs_background"] for row in rows if row["bucket"] == "lt_0p5"),
        "sep_lt_1": next(row["sep_vs_background"] for row in rows if row["bucket"] == "lt_1"),
    }
    return rows, summary


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
    strict_b0_load_report(init_report)
    trainable_names = freeze_except_bypass(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(yaml.safe_dump({
        "initial_weight_load": {
            "path": init_report["path"],
            "loaded": init_report["loaded"],
            "missing_count": len(init_report["missing"]),
            "missing_first20": init_report["missing"][:20],
            "unexpected_count": len(init_report["unexpected"]),
            "skipped_shape_count": len(init_report["skipped_shape"]),
        },
        "trainable_parameter_count": trainable_params,
        "trainable_parameter_names": trainable_names,
    }, sort_keys=False).strip())
    if not trainable_names or any(not name.startswith("fgtp_bypass.") for name in trainable_names):
        raise RuntimeError("trainable parameters are not limited to fgtp_bypass")

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
        "alpha_hr", "tanh_alpha_hr", "mean_abs_delta_d1", "std_abs_delta_d1",
        "mean_abs_written_delta_d1", "mean_abs_p_hr", "sep_lt_0p5", "sep_lt_1",
        "q_bg", "q_lt_0p25", "q_0p25_0p5", "q_0p5_1", "q_1_3", "q_3_10", "q_gte_10",
        "bypass_diag_path",
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
        diag_rows, diag_summary = bypass_diagnostics(model, diag_loader, device, int(config["training"].get("fgtp_diagnostic_batches", 256)))
        diag_path = output_dir / f"bypass_diagnostics_epoch{epoch:03d}.yaml"
        diag_path.write_text(yaml.safe_dump(diag_rows, sort_keys=False), encoding="utf-8")
        alpha = model.fgtp_bypass.alpha_hr.detach().float().cpu()
        current = float(metrics.get(selection_metric, metrics["pixel_f1_fixed"]))
        improved = current > best_metric
        if improved:
            best_metric = current
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
        if improved:
            atomic_torch_save(state, output_dir / "best.pt")
        atomic_torch_save(state, output_dir / f"milestone_eplus{epoch + 1:02d}.pt")
        by_bucket = {row["bucket"]: row for row in diag_rows}
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
            "alpha_hr": float(alpha),
            "tanh_alpha_hr": float(torch.tanh(alpha)),
            "mean_abs_delta_d1": diag_summary["mean_abs_delta_d1"],
            "std_abs_delta_d1": diag_summary["std_abs_delta_d1"],
            "mean_abs_written_delta_d1": diag_summary["mean_abs_written_delta_d1"],
            "mean_abs_p_hr": diag_summary["mean_abs_p_hr"],
            "sep_lt_0p5": diag_summary["sep_lt_0p5"],
            "sep_lt_1": diag_summary["sep_lt_1"],
            "q_bg": by_bucket["background"]["q_hr_mean"],
            "q_lt_0p25": by_bucket["lt_0p25"]["q_hr_mean"],
            "q_0p25_0p5": by_bucket["0p25_0p5"]["q_hr_mean"],
            "q_0p5_1": by_bucket["0p5_1"]["q_hr_mean"],
            "q_1_3": by_bucket["1_3"]["q_hr_mean"],
            "q_3_10": by_bucket["3_10"]["q_hr_mean"],
            "q_gte_10": by_bucket["gte_10"]["q_hr_mean"],
            "bypass_diag_path": str(diag_path),
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        print(yaml.safe_dump(row, sort_keys=False).strip())
        scheduler.step()


if __name__ == "__main__":
    main()
