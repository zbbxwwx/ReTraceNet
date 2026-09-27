#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrace.data import pil_to_tensor, read_manifest
from retrace.utils import choose_device, load_checkpoint, load_config
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import forward_from_batch


def kind_from_path(path: Path) -> str:
    name = path.name
    if name.startswith("Tp_S") or "_S_" in name:
        return "Tp_S"
    if name.startswith("Tp_D") or "_D_" in name:
        return "Tp_D"
    return "unknown"


def bucket_name(ratio: float) -> str:
    if ratio <= 0:
        return "authentic"
    if ratio < 0.005:
        return "lt_0p5"
    if ratio < 0.01:
        return "0p5_1"
    if ratio < 0.05:
        return "1_5"
    if ratio < 0.20:
        return "5_20"
    return "gte_20"


def safe_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        return float("nan")
    return float(roc_auc_score(labels.astype(np.uint8), scores.astype(np.float32)))


def confusion_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= float(threshold)
    lab = labels.astype(bool)
    tp = float(np.logical_and(pred, lab).sum())
    fp = float(np.logical_and(pred, ~lab).sum())
    fn = float(np.logical_and(~pred, lab).sum())
    tn = float(np.logical_and(~pred, ~lab).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": tp / max(1.0, tp + fp + fn),
        "fpr": fp / max(1.0, fp + tn),
    }


def best_f1(scores: np.ndarray, labels: np.ndarray, threshold_count: int) -> dict[str, float]:
    if labels.size == 0 or int(labels.sum()) == 0:
        return {
            "threshold": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "iou": float("nan"),
        }
    best: dict[str, float] | None = None
    for threshold in np.linspace(0.0, 1.0, int(threshold_count)):
        metrics = confusion_at(scores, labels, float(threshold))
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics
    assert best is not None
    return best


def summarize(items: list[tuple[np.ndarray, np.ndarray]], threshold: float, threshold_count: int, select: bool) -> dict[str, Any]:
    if not items:
        empty = {
            "count": 0,
            "pixels": 0,
            "auc": float("nan"),
            "fixed": confusion_at(np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint8), threshold),
        }
        empty["oracle"] = best_f1(np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.uint8), threshold_count)
        return empty
    scores = np.concatenate([s for s, _ in items])
    labels = np.concatenate([y for _, y in items])
    out = {
        "count": len(items),
        "pixels": int(labels.size),
        "auc": safe_auc(scores, labels),
        "fixed": confusion_at(scores, labels, threshold),
    }
    if select:
        out["oracle"] = best_f1(scores, labels, threshold_count)
    return out


def preprocess_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    image = image.resize((int(image_size), int(image_size)), Image.Resampling.BILINEAR)
    return pil_to_tensor(image).unsqueeze(0)


def read_original_mask(record: Any, image_size: tuple[int, int]) -> np.ndarray:
    if record.mask is None:
        return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    with Image.open(record.mask) as handle:
        return (np.asarray(handle.convert("L")) > 0).astype(np.uint8)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    image_size = int(cfg["data"].get("image_size", 640))
    threshold_count = int(args.threshold_count or cfg.get("evaluation", {}).get("thresholds", 101))
    threshold = float(args.threshold if args.threshold is not None else cfg.get("evaluation", {}).get("threshold", 0.5))
    device = choose_device(args.device)
    model = build_final_model(cfg["model"]).to(device).eval()
    checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    if args.state not in checkpoint:
        raise KeyError(f"{args.state} not in checkpoint. keys={list(checkpoint)[:20]}")
    model.load_state_dict(checkpoint[args.state], strict=True)
    records = read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], args.split)

    full_items: list[tuple[np.ndarray, np.ndarray]] = []
    fake_items: list[tuple[np.ndarray, np.ndarray]] = []
    by_kind: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    by_bucket: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    by_kind_bucket: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}

    for record in records:
        with Image.open(record.image) as handle:
            original_size = handle.size
        image_tensor = preprocess_image(record.image, image_size).to(device)
        batch = {"image": image_tensor, "label": torch.tensor([[float(record.label)]], device=device)}
        out = forward_from_batch(model, batch, device, image_tensor)
        prob_640 = out["mask_logits"].sigmoid().detach().float()
        prob = F.interpolate(prob_640, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False)
        scores = prob.cpu().numpy()[0, 0].astype(np.float32).reshape(-1)
        mask = read_original_mask(record, original_size).reshape(-1)
        full_items.append((scores, mask))
        if record.label == 1 and mask.any():
            fake_items.append((scores, mask))
            kind = kind_from_path(record.image)
            bucket = bucket_name(float(mask.mean()))
            by_kind.setdefault(kind, []).append((scores, mask))
            by_bucket.setdefault(bucket, []).append((scores, mask))
            by_kind_bucket.setdefault(kind, {}).setdefault(bucket, []).append((scores, mask))

    select = bool(args.select_threshold)
    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "state": args.state,
        "split": args.split,
        "input_image_size": image_size,
        "probability_grid": "original_gt_size",
        "valid_pixels": "all_original_mask_pixels",
        "threshold_source": "selected_on_this_split" if select else "locked_or_config",
        "locked_threshold": threshold,
        "num_images": len(records),
        "full_image": summarize(full_items, threshold, threshold_count, select),
        "fake_only": summarize(fake_items, threshold, threshold_count, select),
        "by_kind": {k: summarize(v, threshold, threshold_count, select) for k, v in sorted(by_kind.items())},
        "by_bucket": {k: summarize(by_bucket.get(k, []), threshold, threshold_count, select) for k in ["lt_0p5", "0p5_1", "1_5", "5_20", "gte_20"]},
        "by_kind_bucket": {
            k: {b: summarize(v.get(b, []), threshold, threshold_count, select) for b in ["lt_0p5", "0p5_1", "1_5", "5_20", "gte_20"]}
            for k, v in sorted(by_kind_bucket.items())
        },
    }
    if select:
        result["selected_threshold_fake_only"] = result["fake_only"]["oracle"]["threshold"]
        result["selected_threshold_full_image"] = result["full_image"]["oracle"]["threshold"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state", required=True, choices=["model", "ema_model"])
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-count", type=int)
    parser.add_argument("--select-threshold", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = evaluate(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
