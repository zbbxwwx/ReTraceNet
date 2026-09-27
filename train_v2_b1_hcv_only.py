#!/usr/bin/env python
"""Train the B1 HCV-only mechanism probe with the B0 body frozen."""

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


BUCKETS = ["background", "authentic", "lt_0p25", "0p25_0p5", "0p5_1", "lt_0p5", "lt_1", "1_3", "3_10", "gte_10"]
VALUES = ["proposal", "local_proposal", "energy", "delta", "abs_delta", "positive_delta"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--init")
    parser.add_argument("--device")
    return parser.parse_args()


def freeze_except_hcv(model: torch.nn.Module) -> list[str]:
    names = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith("hcv.")
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    return names


def strict_b0_load_report(report: dict[str, Any]) -> None:
    bad_missing = [name for name in report["missing"] if not name.startswith("hcv.")]
    if bad_missing or report["unexpected"] or report["skipped_shape"]:
        raise RuntimeError(
            "B0 initialization is not clean for HCV: "
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


def new_stat() -> dict[str, float]:
    return {"sum": 0.0, "sq": 0.0, "count": 0.0}


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
def hcv_diagnostics(model: torch.nn.Module, loader, device: torch.device, max_batches: int) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    stats = {
        scale: {bucket: {value: new_stat() for value in VALUES} for bucket in BUCKETS}
        for scale in range(3)
    }
    image_counts = {bucket: 0 for bucket in BUCKETS}
    model.eval()
    seen_batches = 0
    for batch in loader:
        if seen_batches >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        outputs = model(image)
        diag = outputs["hcv"]
        for sample in range(image.shape[0]):
            sample_mask = mask[sample:sample + 1].float()
            ratio = float(sample_mask.mean().cpu())
            is_authentic = bool(float(label[sample].detach().cpu()) <= 0.0)
            if is_authentic:
                image_counts["authentic"] += 1
            image_counts["background"] += 1
            for name in bucket_names(ratio):
                image_counts[name] += 1
            for scale in range(3):
                proposal = diag["proposals"][scale][sample:sample + 1].float()
                local_proposal = diag["local_proposals"][scale][sample:sample + 1].float()
                energy = diag["energies"][scale][sample:sample + 1].float()
                delta = diag["deltas"][scale][sample:sample + 1].float()
                resized = F.interpolate(sample_mask, size=proposal.shape[-2:], mode="nearest") > 0.5
                bg_region = ~resized
                values = {
                    "proposal": proposal,
                    "local_proposal": local_proposal,
                    "energy": energy,
                    "delta": delta,
                    "abs_delta": delta.abs(),
                    "positive_delta": (delta > 0).float(),
                }
                for value_name, value in values.items():
                    add(stats[scale]["background"][value_name], value[bg_region])
                    if is_authentic:
                        add(stats[scale]["authentic"][value_name], value.flatten())
                    for bucket in bucket_names(ratio):
                        add(stats[scale][bucket][value_name], value[resized])
        seen_batches += 1

    eps = 1e-8
    rows: list[dict[str, float | int | str]] = []
    for scale in range(3):
        bg_delta = stats[scale]["background"]["delta"]
        bg_abs = stats[scale]["background"]["abs_delta"]
        for bucket in BUCKETS:
            row: dict[str, float | int | str] = {"scale": f"s{scale + 1}", "bucket": bucket, "num_images": image_counts[bucket]}
            for value in VALUES:
                row[f"{value}_mean"] = mean(stats[scale][bucket][value])
                row[f"{value}_std"] = std(stats[scale][bucket][value])
                row[f"{value}_pixels"] = int(stats[scale][bucket][value]["count"])
            row["sep_delta"] = (mean(stats[scale][bucket]["delta"]) - mean(bg_delta)) / (std(bg_delta) + eps)
            row["sep_abs_delta"] = (mean(stats[scale][bucket]["abs_delta"]) - mean(bg_abs)) / (std(bg_abs) + eps)
            rows.append(row)
    summary: dict[str, float] = {}
    for scale in range(3):
        for bucket in ("lt_0p5", "lt_1", "authentic"):
            key = f"s{scale + 1}_{bucket}"
            row = next(r for r in rows if r["scale"] == f"s{scale + 1}" and r["bucket"] == bucket)
            summary[f"{key}_proposal"] = float(row["proposal_mean"])
            summary[f"{key}_energy"] = float(row["energy_mean"])
            summary[f"{key}_delta"] = float(row["delta_mean"])
            summary[f"{key}_abs_delta"] = float(row["abs_delta_mean"])
            summary[f"{key}_sep_delta"] = float(row["sep_delta"])
            summary[f"{key}_sep_abs_delta"] = float(row["sep_abs_delta"])
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
    trainable_names = freeze_except_hcv(model)
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
    if not trainable_names or any(not name.startswith("hcv.") for name in trainable_names):
        raise RuntimeError("trainable parameters are not limited to hcv")

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
        "alpha_s1", "alpha_s2", "alpha_s3", "tanh_alpha_s1", "tanh_alpha_s2", "tanh_alpha_s3",
        "lambda_1", "lambda_2", "sigmoid_lambda_1", "sigmoid_lambda_2",
        "s1_lt_0p5_proposal", "s1_lt_0p5_delta", "s1_lt_0p5_abs_delta", "s1_lt_0p5_sep_delta", "s1_lt_0p5_sep_abs_delta",
        "s2_lt_0p5_proposal", "s2_lt_0p5_delta", "s2_lt_0p5_abs_delta", "s2_lt_0p5_sep_delta", "s2_lt_0p5_sep_abs_delta",
        "s3_lt_0p5_proposal", "s3_lt_0p5_delta", "s3_lt_0p5_abs_delta", "s3_lt_0p5_sep_delta", "s3_lt_0p5_sep_abs_delta",
        "s1_lt_1_proposal", "s1_lt_1_delta", "s1_lt_1_abs_delta", "s1_lt_1_sep_delta", "s1_lt_1_sep_abs_delta",
        "s2_lt_1_proposal", "s2_lt_1_delta", "s2_lt_1_abs_delta", "s2_lt_1_sep_delta", "s2_lt_1_sep_abs_delta",
        "s3_lt_1_proposal", "s3_lt_1_delta", "s3_lt_1_abs_delta", "s3_lt_1_sep_delta", "s3_lt_1_sep_abs_delta",
        "s1_authentic_proposal", "s1_authentic_abs_delta", "s2_authentic_proposal", "s2_authentic_abs_delta", "s3_authentic_proposal", "s3_authentic_abs_delta",
        "hcv_diag_path",
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
        diag_rows, diag_summary = hcv_diagnostics(model, diag_loader, device, int(config["training"].get("hcv_diagnostic_batches", 512)))
        diag_path = output_dir / f"hcv_diagnostics_epoch{epoch:03d}.yaml"
        diag_path.write_text(yaml.safe_dump(diag_rows, sort_keys=False), encoding="utf-8")
        alpha = model.hcv.alpha.detach().float().cpu()
        lambdas = torch.stack([model.hcv.lambda_1.detach(), model.hcv.lambda_2.detach()]).float().cpu()
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
            "alpha_s1": float(alpha[0]),
            "alpha_s2": float(alpha[1]),
            "alpha_s3": float(alpha[2]),
            "tanh_alpha_s1": float(torch.tanh(alpha[0])),
            "tanh_alpha_s2": float(torch.tanh(alpha[1])),
            "tanh_alpha_s3": float(torch.tanh(alpha[2])),
            "lambda_1": float(lambdas[0]),
            "lambda_2": float(lambdas[1]),
            "sigmoid_lambda_1": float(torch.sigmoid(lambdas[0])),
            "sigmoid_lambda_2": float(torch.sigmoid(lambdas[1])),
            "hcv_diag_path": str(diag_path),
        }
        for scale in range(1, 4):
            for bucket in ("lt_0p5", "lt_1"):
                prefix = f"s{scale}_{bucket}"
                row[f"{prefix}_proposal"] = diag_summary[f"{prefix}_proposal"]
                row[f"{prefix}_delta"] = diag_summary[f"{prefix}_delta"]
                row[f"{prefix}_abs_delta"] = diag_summary[f"{prefix}_abs_delta"]
                row[f"{prefix}_sep_delta"] = diag_summary[f"{prefix}_sep_delta"]
                row[f"{prefix}_sep_abs_delta"] = diag_summary[f"{prefix}_sep_abs_delta"]
            row[f"s{scale}_authentic_proposal"] = diag_summary[f"s{scale}_authentic_proposal"]
            row[f"s{scale}_authentic_abs_delta"] = diag_summary[f"s{scale}_authentic_abs_delta"]
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        print(yaml.safe_dump(row, sort_keys=False).strip())
        scheduler.step()


if __name__ == "__main__":
    main()
