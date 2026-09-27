#!/usr/bin/env python
"""Train B1 FGTP-v2-only probe with the B0 body frozen."""

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


def local_detail_score(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    avg = F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)
    return (x - avg).abs().mean(dim=1)


@torch.inference_mode()
def fgtp_diagnostics(model: torch.nn.Module, loader, device: torch.device, max_batches: int) -> list[dict[str, float | int | str]]:
    buckets = ["background", "lt_0p5", "0p5_1", "1_3", "3_10", "gte_10"]
    names = [
        "abs_t1_minus_n1", "abs_t2_minus_n2", "q_n1", "q_t1", "q_n2", "q_t2",
        "n1_norm", "t1_norm", "n2_norm", "t2_norm", "gate1", "gate2",
    ]
    stats = {bucket: {name: {"sum": 0.0, "sq": 0.0, "count": 0.0} for name in names} for bucket in buckets}
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
        seen += image.shape[0]
        for sample in range(image.shape[0]):
            sample_mask = mask[sample:sample + 1]
            bucket = area_bucket(float(sample_mask.mean().cpu()))
            counts[bucket] += 1
            for scale, kernel in ((1, 5), (2, 3)):
                n = diag[f"n{scale}"][sample:sample + 1].float()
                t = diag[f"t{scale}"][sample:sample + 1].float()
                residual = (t - n).abs().mean(dim=1)
                qn = local_detail_score(n, kernel)
                qt = local_detail_score(t, kernel)
                nnorm = n.norm(dim=1)
                tnorm = t.norm(dim=1)
                gate = diag.get(f"spatial_gate{scale}")
                if gate is not None:
                    gate = gate[sample:sample + 1, 0].float()
                resized = F.interpolate(sample_mask.float(), size=n.shape[-2:], mode="nearest")[:, 0] > 0.5
                region = ~resized if bucket == "background" else resized
                add(stats[bucket][f"abs_t{scale}_minus_n{scale}"], residual[region])
                add(stats[bucket][f"q_n{scale}"], qn[region])
                add(stats[bucket][f"q_t{scale}"], qt[region])
                add(stats[bucket][f"n{scale}_norm"], nnorm[region])
                add(stats[bucket][f"t{scale}_norm"], tnorm[region])
                if gate is not None:
                    add(stats[bucket][f"gate{scale}"], gate[region])
    rows = []
    eps = 1e-8
    for bucket in buckets:
        row: dict[str, float | int | str] = {"bucket": bucket, "num_images": counts[bucket]}
        for name in names:
            row[name] = mean(stats[bucket][name])
            row[f"{name}_std"] = std(stats[bucket][name])
        for scale in (1, 2):
            bg_n = stats["background"][f"q_n{scale}"]
            bg_t = stats["background"][f"q_t{scale}"]
            row[f"sep_n{scale}"] = (float(row[f"q_n{scale}"]) - mean(bg_n)) / (std(bg_n) + eps)
            row[f"sep_t{scale}"] = (float(row[f"q_t{scale}"]) - mean(bg_t)) / (std(bg_t) + eps)
            row[f"sep_t_minus_n{scale}"] = float(row[f"sep_t{scale}"]) - float(row[f"sep_n{scale}"])
            row[f"norm_t_minus_n{scale}"] = float(row[f"t{scale}_norm"]) - float(row[f"n{scale}_norm"])
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
        "selection_scale1", "selection_scale2", "mean_abs_t1_minus_n1", "mean_abs_t2_minus_n2",
        "sep_n1_lt_0p5", "sep_t1_lt_0p5", "sep_delta1_lt_0p5",
        "sep_n2_lt_0p5", "sep_t2_lt_0p5", "sep_delta2_lt_0p5", "fgtp_diag_path",
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
        scale1 = model.fgtp.scale1.selection_scale.detach().float().cpu()
        scale2 = model.fgtp.scale2.selection_scale.detach().float().cpu()
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
        if epoch + 1 in {1, 2, 3}:
            atomic_torch_save(state, output_dir / f"milestone_eplus{epoch + 1:02d}.pt")
        tiny = next((r for r in diag_rows if r["bucket"] == "lt_0p5"), {})
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
            "selection_scale1": float(scale1),
            "selection_scale2": float(scale2),
            "mean_abs_t1_minus_n1": float(tiny.get("abs_t1_minus_n1", 0.0)),
            "mean_abs_t2_minus_n2": float(tiny.get("abs_t2_minus_n2", 0.0)),
            "sep_n1_lt_0p5": float(tiny.get("sep_n1", 0.0)),
            "sep_t1_lt_0p5": float(tiny.get("sep_t1", 0.0)),
            "sep_delta1_lt_0p5": float(tiny.get("sep_t_minus_n1", 0.0)),
            "sep_n2_lt_0p5": float(tiny.get("sep_n2", 0.0)),
            "sep_t2_lt_0p5": float(tiny.get("sep_t2", 0.0)),
            "sep_delta2_lt_0p5": float(tiny.get("sep_t_minus_n2", 0.0)),
            "fgtp_diag_path": str(diag_path),
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
        print(yaml.safe_dump(row, sort_keys=False).strip())
        scheduler.step()


if __name__ == "__main__":
    main()
