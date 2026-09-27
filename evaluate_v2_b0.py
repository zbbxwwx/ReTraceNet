#!/usr/bin/env python
"""Evaluate the clean v2 B0 base and write size-bucket diagnostics."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrace.data import build_loader
from retrace.engine import evaluate_model
from retrace.losses import ReTraceLoss
from retrace.metrics import Confusion
from retrace.utils import choose_device, load_checkpoint, load_config
from retracenet.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--profile-output", required=True)
    parser.add_argument("--device")
    return parser.parse_args()


def load_model(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(config["model"]).to(device)
    state = checkpoint["model"]
    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
    result = model.load_state_dict(compatible, strict=False)
    missing = list(result.missing_keys)
    skipped = len(state) - len(compatible)
    if missing or skipped:
        print(f"non-strict v2_b0 load: loaded={len(compatible)} missing={missing[:12]} skipped={skipped}")
    model.eval()
    return model


def scores(confusion: Confusion) -> dict[str, float]:
    f1, iou, _ = confusion.scores()
    precision = confusion.tp / max(1.0, confusion.tp + confusion.fp)
    recall = confusion.tp / max(1.0, confusion.tp + confusion.fn)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def bucket_name(ratio: float) -> str:
    if ratio < 0.0025:
        return "lt_0p25"
    if ratio < 0.005:
        return "0p25_0p5"
    if ratio < 0.01:
        return "0p5_1"
    if ratio < 0.03:
        return "1_3"
    if ratio < 0.10:
        return "3_10"
    return "gte_10"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    criterion = ReTraceLoss.from_config(config["loss"]).to(device)
    model = load_model(config, args.checkpoint, device)
    loader_config = {**config["data"], "batch_size": config["evaluation"]["batch_size"]}
    loader = build_loader(loader_config, args.split, shuffle=False)

    losses, metrics = evaluate_model(
        model,
        criterion,
        loader,
        device,
        bool(config["training"].get("amp", True) and device.type == "cuda"),
        float(config["evaluation"].get("threshold", 0.5)),
        int(config["evaluation"].get("thresholds", 101)),
    )

    buckets = OrderedDict((name, {"confusion": Confusion(), "count": 0, "ratio_sum": 0.0}) for name in [
        "lt_0p25", "0p25_0p5", "0p5_1", "1_3", "3_10", "gte_10", "lt_0p5", "lt_1"
    ])
    fixed_threshold = float(config["evaluation"].get("threshold", 0.5))
    with torch.inference_mode():
        for batch in build_loader(loader_config, args.split, shuffle=False):
            image = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True).bool()
            paths = batch.get("path", [""] * image.shape[0])
            outputs = model(image)
            pred = outputs["mask_logits"].sigmoid() >= fixed_threshold
            for index in range(image.shape[0]):
                target = mask[index:index + 1]
                ratio = float(target.float().mean().cpu())
                if ratio <= 0.0:
                    continue
                names = [bucket_name(ratio)]
                if ratio < 0.005:
                    names.append("lt_0p5")
                if ratio < 0.01:
                    names.append("lt_1")
                for name in names:
                    entry = buckets[name]
                    entry["count"] += 1
                    entry["ratio_sum"] += ratio
                    p = pred[index:index + 1]
                    entry["confusion"].tp += float((p & target).sum().cpu())
                    entry["confusion"].fp += float((p & ~target).sum().cpu())
                    entry["confusion"].fn += float((~p & target).sum().cpu())
                    entry["confusion"].tn += float((~p & ~target).sum().cpu())

    rows = []
    for name, entry in buckets.items():
        row = {"bucket": name, "num_images": entry["count"], "mean_area_ratio": entry["ratio_sum"] / max(1, entry["count"])}
        row.update(scores(entry["confusion"]))
        rows.append(row)
    Path(args.profile_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.profile_output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bucket", "num_images", "mean_area_ratio", "precision", "recall", "f1", "iou"])
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "samples": len(loader.dataset),
        "losses": losses,
        "metrics": metrics,
        "size_profile_fixed_threshold": rows,
        "warning": "oracle metrics are diagnostic; model selection remains validation-only",
    }
    Path(args.metrics_output).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump({"metrics": metrics, "size_profile_fixed_threshold": rows}, sort_keys=False))


if __name__ == "__main__":
    main()
