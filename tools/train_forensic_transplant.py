#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.utils import atomic_torch_save, choose_device, load_checkpoint, load_config, set_seed
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import (
    clean_loss_config,
    evaluate_tampered_pooled,
    forward_from_batch,
    overlay_checkpoint,
    train_loader_for,
)


def load_model_state(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)


def norm_feature(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def forensic_feature_alignment(student_detail: list[torch.Tensor], teacher_detail: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    total = student_detail[0].new_zeros(())
    for i, weight in enumerate(weights):
        if i >= len(student_detail) or i >= len(teacher_detail) or float(weight) <= 0:
            continue
        s = norm_feature(student_detail[i])
        t = norm_feature(teacher_detail[i].detach())
        total = total + float(weight) * F.smooth_l1_loss(s, t)
    return total


def set_only_forensic_trainable(model: torch.nn.Module) -> list[str]:
    names = []
    for name, param in model.named_parameters():
        train = name.startswith("forensic_stream.")
        param.requires_grad_(train)
        if train:
            names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 3407)))
    device = choose_device(args.device)
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    teacher = build_final_model(cfg["model"]).to(device).eval()
    student = build_final_model(cfg["model"]).to(device)
    init_checkpoint = cfg["training"]["init_checkpoint"]
    load_model_state(teacher, init_checkpoint)
    load_model_state(student, init_checkpoint)
    secondary = overlay_checkpoint(
        student,
        cfg["training"]["secondary_init_checkpoint"],
        cfg["training"].get("secondary_init_load_patterns") or ["forensic_stream."],
    )
    if secondary["unexpected"] or secondary["skipped_shape"]:
        raise RuntimeError(f"bad secondary init: {secondary}")
    for param in teacher.parameters():
        param.requires_grad_(False)
    trainable = set_only_forensic_trainable(student)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=float(cfg["training"].get("learning_rate", 1e-6)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )
    train_loader = train_loader_for(cfg)
    val_loader = build_loader({**cfg["data"], "batch_size": cfg["evaluation"]["batch_size"]}, "val", shuffle=False)
    seg_loss = ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).to(device)
    amp = bool(cfg["training"].get("amp", False) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    epochs = int(cfg["training"].get("epochs", 4))
    acc = max(1, int(cfg["training"].get("gradient_accumulation_steps", 1)))
    align_weights = [float(x) for x in cfg["training"].get("feature_alignment_weights", [1.0, 0.7, 0.5, 0.3])]
    seg_weight = float(cfg["training"].get("segmentation_weight", 0.25))
    align_weight = float(cfg["training"].get("alignment_weight", 1.0))
    fixed_th = float(cfg["evaluation"].get("threshold", 0.5))
    threshold_count = int(cfg["evaluation"].get("thresholds", 101))
    hist_path = outdir / "history.csv"
    fields = ["epoch", "lr", "train_total", "train_align", "train_seg", "selection_value", "val_fixed_f1", "val_fixed_precision", "val_fixed_recall", "val_auc", "extra"]
    with hist_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    best = -1e9
    print(yaml.safe_dump({"secondary_init": secondary, "trainable_count": sum(p.numel() for p in params), "trainable_examples": trainable[:30]}, sort_keys=False), flush=True)
    for epoch in range(epochs):
        student.eval()
        student.forensic_stream.train()
        sums = {"total": 0.0, "align": 0.0, "seg": 0.0}
        count = 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)
            with torch.no_grad():
                teacher_detail = teacher.forensic_stream(image)
            with torch.autocast(device_type=device.type, enabled=amp):
                student_detail = student.forensic_stream(image)
                align = forensic_feature_alignment(student_detail, teacher_detail, align_weights)
                out = forward_from_batch(student, batch, device, image)
                seg = seg_loss(out, mask, label)["total"]
                loss = align_weight * align + seg_weight * seg
            scaler.scale(loss / acc).backward()
            if (step + 1) % acc == 0 or step + 1 == len(train_loader):
                if float(cfg["training"].get("grad_clip", 0.0)) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, float(cfg["training"].get("grad_clip", 1.0)))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            bs = image.shape[0]
            sums["total"] += float(loss.detach().cpu()) * bs
            sums["align"] += float(align.detach().cpu()) * bs
            sums["seg"] += float(seg.detach().cpu()) * bs
            count += bs
        val = evaluate_tampered_pooled(student, val_loader, device, amp, fixed_th, threshold_count)
        cur = float(val["f1_best"])
        state = {
            "epoch": epoch,
            "model": {k: v.detach().cpu().clone() for k, v in student.state_dict().items()},
            "optimizer": opt.state_dict(),
            "best_metric": max(best, cur),
            "config": cfg,
            "phase": "forensic_transplant",
            "secondary_init_report": secondary,
            "eval_variant": "raw",
        }
        atomic_torch_save(state, outdir / "last.pt")
        if cur >= best:
            best = cur
            atomic_torch_save(state, outdir / "best.pt")
        row = {
            "epoch": epoch,
            "lr": opt.param_groups[0]["lr"],
            "train_total": sums["total"] / max(1, count),
            "train_align": sums["align"] / max(1, count),
            "train_seg": sums["seg"] / max(1, count),
            "selection_value": cur,
            "val_fixed_f1": val["fixed_f1"],
            "val_fixed_precision": val["fixed_precision"],
            "val_fixed_recall": val["fixed_recall"],
            "val_auc": val["auc"],
            "extra": json.dumps({"val_tampered_pooled": val}, sort_keys=True),
        }
        with hist_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        (outdir / "last_metrics.yaml").write_text(yaml.safe_dump(row, sort_keys=False), encoding="utf-8")
        print(yaml.safe_dump(row, sort_keys=False).strip(), flush=True)


if __name__ == "__main__":
    main()
