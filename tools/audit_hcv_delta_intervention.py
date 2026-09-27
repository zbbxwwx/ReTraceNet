"""Diagnostic-only HCV delta interventions for E21-style checkpoints.

This script never trains and never reports GT-intervention results as model
accuracy. It asks one causal question: if HCV signed deltas were better, would
the unchanged DHER/decoder path convert that evidence into a better mask?
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrace.data import ManipulationDataset, read_manifest
from retracenet.final_modules import build_final_model


def dump_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def forward_from_batch(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    kwargs = {}
    if "dct" in batch:
        kwargs = {
            "dct": batch["dct"],
            "qtable": batch["qtable"],
            "dct_available": batch["dct_available"],
        }
    return model(batch["image"], **kwargs)


def mask_for_delta(target: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return F.interpolate(target.float(), size=delta.shape[-2:], mode="nearest")


def install_delta_hook(model: torch.nn.Module, mode: str, target: torch.Tensor, amplitude: float):
    if mode == "baseline":
        return None

    def hook(_module, _inputs, output):
        refined = dict(output)
        deltas = []
        for delta in output["deltas"]:
            if mode == "zero":
                new_delta = torch.zeros_like(delta)
            elif mode.startswith("scale:"):
                factor = float(mode.split(":", 1)[1])
                new_delta = delta * factor
            elif mode == "ideal_signed":
                gt = mask_for_delta(target, delta)
                new_delta = (gt * 2.0 - 1.0) * float(amplitude)
            elif mode == "ideal_fg_only":
                gt = mask_for_delta(target, delta)
                new_delta = gt * float(amplitude)
            else:
                raise ValueError(f"unknown intervention mode: {mode}")
            deltas.append(new_delta)
        refined["deltas"] = deltas
        return refined

    return model.hcv2.register_forward_hook(hook)


def confusion(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= float(threshold)
    lab = labels.astype(bool)
    tp = int(np.logical_and(pred, lab).sum())
    fp = int(np.logical_and(pred, ~lab).sum())
    fn = int(np.logical_and(~pred, lab).sum())
    tn = int(np.logical_and(~pred, ~lab).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    iou = tp / max(1, tp + fp + fn)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def best_f1(scores: np.ndarray, labels: np.ndarray, threshold_count: int) -> dict[str, float]:
    if scores.size == 0:
        return confusion(scores, labels, 0.5)
    best = None
    for threshold in np.linspace(0.0, 1.0, int(threshold_count)):
        row = confusion(scores, labels, float(threshold))
        if best is None or row["f1"] > best["f1"]:
            best = row
    assert best is not None
    return best


def tensor_rms(value: torch.Tensor | None) -> float:
    if value is None:
        return 0.0
    return float(value.detach().float().square().mean().sqrt().cpu())


def tensor_mean(value: torch.Tensor | None) -> float:
    if value is None:
        return 0.0
    return float(value.detach().float().mean().cpu())


def low_map_stats(value: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    gt = mask_for_delta(target, value) > 0.5
    v = value.detach().float()
    fg = v[gt.expand_as(v)]
    bg = v[(~gt).expand_as(v)]
    return {
        "rms": tensor_rms(v),
        "mean": tensor_mean(v),
        "fg_mean": float(fg.mean().cpu()) if fg.numel() else float("nan"),
        "bg_mean": float(bg.mean().cpu()) if bg.numel() else float("nan"),
        "fg_bg_gap": float((fg.mean() - bg.mean()).cpu()) if fg.numel() and bg.numel() else float("nan"),
    }


def save_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return
    arr = np.nan_to_num(arr)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), mode="L").save(path)


def save_snapshot(
    sample_dir: Path,
    target_640: torch.Tensor,
    baseline: dict[str, Any],
    mode_outputs: dict[str, dict[str, Any]],
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_gray(sample_dir / "gt_640.png", target_640[0, 0].detach().cpu().numpy())
    arrays: dict[str, np.ndarray] = {"gt_640": target_640[0, 0].detach().cpu().numpy().astype("float32")}
    for mode, output in mode_outputs.items():
        prob = output["mask_logits"].detach().float().sigmoid()[0, 0].cpu().numpy().astype("float32")
        arrays[f"{mode}_prob_640"] = prob
        save_gray(sample_dir / f"{mode}_prob_640.png", prob)
    hcv = baseline.get("hcv2", {})
    for i, delta in enumerate(hcv.get("deltas", [])):
        arr = delta.detach().float()[0, 0].cpu().numpy().astype("float32")
        arrays[f"hcv_delta_s{i+1}"] = arr
        save_gray(sample_dir / f"hcv_delta_s{i+1}.png", arr)
    dher = baseline.get("dher", {})
    for key in ("support_write", "conflict_write", "write"):
        value = dher.get(key)
        if torch.is_tensor(value):
            arr = value.detach().float().square().mean(1)[0].sqrt().cpu().numpy().astype("float32")
            arrays[f"dher_{key}_rms_map"] = arr
            save_gray(sample_dir / f"dher_{key}_rms_map.png", arr)
    np.savez_compressed(sample_dir / "diagnostic_arrays.npz", **arrays)


def original_target(record) -> np.ndarray:
    if record.mask is None:
        with Image.open(record.image) as image:
            return np.zeros((image.height, image.width), dtype=np.uint8)
    with Image.open(record.mask) as mask:
        return (np.asarray(mask.convert("L")) > 127).astype(np.uint8)


def select_records(records, cfg: dict[str, Any], count: int) -> tuple[list[Any], list[dict[str, Any]]]:
    selected = []
    rows = []
    dataset_kwargs = {
        "image_size": int(cfg["data"].get("image_size", 640)),
        "training": False,
        "dct_forensic": bool(cfg["data"].get("dct_forensic", False)),
        "dct_source_mode": cfg["data"].get("dct_source_mode", "reencoded"),
        "dct_qtable_from_source": cfg["data"].get("dct_qtable_from_source", True),
        "dct_jpeg_quality": int(cfg["data"].get("dct_jpeg_quality", 100)),
        "dct_cat_root": cfg["data"].get("dct_cat_root"),
    }
    for record in records:
        path_text = str(record.image).lower()
        if "tp_s" not in path_text or record.mask is None:
            continue
        target = original_target(record)
        area = float(target.mean())
        if area < 0.20:
            continue
        sample = ManipulationDataset([record], **dataset_kwargs)[0]
        dct_available = float(sample.get("dct_available", torch.tensor([1.0])).mean())
        if bool(cfg["data"].get("dct_forensic", False)) and dct_available <= 0:
            continue
        selected.append(record)
        rows.append({
            "path": str(record.image),
            "mask": str(record.mask),
            "area": area,
            "dct_available": dct_available,
        })
        if len(selected) >= count:
            break
    return selected, rows


def summarize_modes(mode_scores: dict[str, list[np.ndarray]], labels: list[np.ndarray], threshold_count: int, locked_threshold: float) -> dict[str, Any]:
    lab = np.concatenate([x.reshape(-1) for x in labels]).astype(np.uint8)
    out = {}
    for mode, chunks in mode_scores.items():
        scores = np.concatenate([x.reshape(-1) for x in chunks]).astype(np.float32)
        out[mode] = {
            "oracle": best_f1(scores, lab, threshold_count),
            "locked": confusion(scores, lab, locked_threshold),
        }
    base = out["baseline"]["oracle"]["f1"]
    base_locked = out["baseline"]["locked"]["f1"]
    for mode, row in out.items():
        row["delta_oracle_f1_vs_baseline"] = float(row["oracle"]["f1"] - base)
        row["delta_locked_f1_vs_baseline"] = float(row["locked"]["f1"] - base_locked)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold-count", type=int, default=101)
    parser.add_argument("--amplitude", type=float, default=1.0)
    args = parser.parse_args()

    started = time.time()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint.get("eval_variant", "raw") == "raw"
    cfg = checkpoint["config"]
    model = build_final_model(copy.deepcopy(cfg["model"])).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    if model.hcv2 is None or model.dher is None:
        raise RuntimeError("This audit requires HCV and DHER to be present.")

    records = read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], args.split)
    selected, selected_rows = select_records(records, cfg, args.samples)
    if not selected:
        raise RuntimeError(f"No JPEG-available Tp_S >=20% records found in split={args.split}")

    dataset = ManipulationDataset(
        selected,
        image_size=int(cfg["data"].get("image_size", 640)),
        training=False,
        dct_forensic=bool(cfg["data"].get("dct_forensic", False)),
        dct_source_mode=cfg["data"].get("dct_source_mode", "reencoded"),
        dct_qtable_from_source=cfg["data"].get("dct_qtable_from_source", True),
        dct_jpeg_quality=int(cfg["data"].get("dct_jpeg_quality", 100)),
        dct_cat_root=cfg["data"].get("dct_cat_root"),
    )
    modes = [
        "baseline",
        "zero",
        "scale:0.5",
        "scale:2.0",
        "scale:4.0",
        "ideal_fg_only",
        "ideal_signed",
    ]
    mode_scores = {mode: [] for mode in modes}
    labels = []
    per_sample = []
    locked_threshold = float(checkpoint.get("selection_threshold", cfg.get("evaluation", {}).get("threshold", 0.5)))

    with torch.no_grad():
        for index, (record, selected_row) in enumerate(zip(selected, selected_rows)):
            raw_target = original_target(record)
            labels.append(raw_target.astype(np.uint8))
            batch = dataset[index]
            batch = {k: v[None] if torch.is_tensor(v) else v for k, v in batch.items()}
            batch = to_device(batch, device)
            target = batch["mask"].float()
            outputs = {}
            base_output = None
            sample_row: dict[str, Any] = {
                **selected_row,
                "index": index,
                "original_shape": list(raw_target.shape),
                "modes": {},
            }
            baseline_logits = None
            baseline_prob = None
            for mode in modes:
                handle = install_delta_hook(model, mode, target, args.amplitude)
                try:
                    output = forward_from_batch(model, batch)
                finally:
                    if handle is not None:
                        handle.remove()
                if mode == "baseline":
                    base_output = output
                    baseline_logits = output["mask_logits"].detach().float()
                    baseline_prob = baseline_logits.sigmoid()
                prob = F.interpolate(
                    output["mask_logits"].detach().float().sigmoid(),
                    size=raw_target.shape,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].cpu().numpy().astype(np.float32)
                mode_scores[mode].append(prob)
                outputs[mode] = output
                logits = output["mask_logits"].detach().float()
                prob640 = logits.sigmoid()
                assert baseline_logits is not None and baseline_prob is not None
                sample_metrics = {
                    "oracle": best_f1(prob.reshape(-1), raw_target.reshape(-1), args.threshold_count),
                    "locked": confusion(prob.reshape(-1), raw_target.reshape(-1), locked_threshold),
                    "logit_mean_abs_diff_vs_baseline_640": float((logits - baseline_logits).abs().mean().cpu()),
                    "logit_max_abs_diff_vs_baseline_640": float((logits - baseline_logits).abs().max().cpu()),
                    "prob_mean_abs_diff_vs_baseline_640": float((prob640 - baseline_prob).abs().mean().cpu()),
                    "fg_prob_mean_640": float(prob640[target > 0.5].mean().cpu()),
                    "bg_prob_mean_640": float(prob640[target <= 0.5].mean().cpu()),
                }
                dher = output.get("dher", {})
                if dher:
                    sample_metrics["dher"] = {
                        "support_write_rms": tensor_rms(dher.get("support_write")),
                        "conflict_write_rms": tensor_rms(dher.get("conflict_write")),
                        "write_rms": tensor_rms(dher.get("write")),
                        "support_mean": tensor_mean(dher.get("support")[0] if isinstance(dher.get("support"), list) else dher.get("support")),
                        "conflict_mean": tensor_mean(dher.get("conflict")[0] if isinstance(dher.get("conflict"), list) else dher.get("conflict")),
                    }
                hcv = output.get("hcv2", {})
                if hcv:
                    sample_metrics["hcv_delta"] = [
                        low_map_stats(delta, target)
                        for delta in hcv.get("deltas", [])
                    ]
                sample_row["modes"][mode] = sample_metrics
            assert base_output is not None
            save_snapshot(outdir / f"sample_{index:02d}", target.detach().cpu(), base_output, outputs)
            per_sample.append(sample_row)
            print(f"[audit] {index + 1}/{len(selected)} {record.image.name}", flush=True)

    pooled = summarize_modes(mode_scores, labels, args.threshold_count, locked_threshold)
    baseline_oracle = pooled["baseline"]["oracle"]["f1"]
    best_ideal = max(
        pooled["ideal_fg_only"]["oracle"]["f1"],
        pooled["ideal_signed"]["oracle"]["f1"],
    )
    best_scaled = max(
        pooled["scale:0.5"]["oracle"]["f1"],
        pooled["scale:2.0"]["oracle"]["f1"],
        pooled["scale:4.0"]["oracle"]["f1"],
    )
    decision = {
        "ideal_delta_oracle_gain": float(best_ideal - baseline_oracle),
        "scale_delta_oracle_gain": float(best_scaled - baseline_oracle),
        "downstream_can_use_ideal_evidence": bool(best_ideal - baseline_oracle >= 0.02),
        "current_delta_amplitude_sensitive": bool(abs(best_scaled - baseline_oracle) >= 0.005),
        "suggestion": "",
    }
    if not decision["downstream_can_use_ideal_evidence"]:
        decision["suggestion"] = "Do not train HCV refinement yet; diagnose DHER/decoder evidence consumption."
    elif decision["current_delta_amplitude_sensitive"]:
        decision["suggestion"] = "Delta path can influence masks; HCV refinement is worth one matched-budget run."
    else:
        decision["suggestion"] = "Ideal deltas help but simple scaling does not; train HCV only if the refinement can change spatial structure, not just magnitude."

    report = {
        "status": "PASS",
        "checkpoint": args.checkpoint,
        "split": args.split,
        "device": str(device),
        "protocol_note": "diagnostic only; GT-derived interventions are not model scores",
        "selected_records": selected_rows,
        "modes": modes,
        "locked_threshold": locked_threshold,
        "pooled": pooled,
        "decision": decision,
        "per_sample": per_sample,
        "elapsed_seconds": time.time() - started,
    }
    dump_yaml(outdir / "delta_intervention_report.yaml", report)
    (outdir / "delta_intervention_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "samples": len(selected),
        "baseline_oracle_f1": baseline_oracle,
        "best_ideal_oracle_f1": best_ideal,
        "best_scaled_oracle_f1": best_scaled,
        "decision": decision,
        "out": str(outdir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
