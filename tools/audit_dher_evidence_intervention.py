"""Diagnostic-only DHER/D1 evidence visibility interventions.

This does not train and does not treat GT-derived interventions as model
accuracy. It checks whether evidence can affect the final mask once it is made
visible at DHER/D1.
"""
from __future__ import annotations

import argparse
import copy
import json
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


def original_target(record) -> np.ndarray:
    if record.mask is None:
        with Image.open(record.image) as image:
            return np.zeros((image.height, image.width), dtype=np.uint8)
    with Image.open(record.mask) as mask:
        return (np.asarray(mask.convert("L")) > 127).astype(np.uint8)


def safe_name(name: str) -> str:
    return name.replace(":", "_").replace(".", "p").replace("-", "m")


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward_from_batch(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    kwargs = {}
    if "dct" in batch:
        kwargs = {
            "dct": batch["dct"],
            "qtable": batch["qtable"],
            "dct_available": batch["dct_available"],
        }
    return model(batch["image"], **kwargs)


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
    best = None
    for threshold in np.linspace(0.0, 1.0, int(threshold_count)):
        row = confusion(scores, labels, float(threshold))
        if best is None or row["f1"] > best["f1"]:
            best = row
    assert best is not None
    return best


def rms(value: torch.Tensor | None) -> float:
    if value is None:
        return 0.0
    return float(value.detach().float().square().mean().sqrt().cpu())


def save_gray(path: Path, array: np.ndarray) -> None:
    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255), mode="L").save(path)


def select_records(records, cfg: dict[str, Any], count: int) -> tuple[list[Any], list[dict[str, Any]]]:
    selected = []
    rows = []
    dataset_kwargs = dict(
        image_size=int(cfg["data"].get("image_size", 640)),
        training=False,
        dct_forensic=bool(cfg["data"].get("dct_forensic", False)),
        dct_source_mode=cfg["data"].get("dct_source_mode", "reencoded"),
        dct_qtable_from_source=cfg["data"].get("dct_qtable_from_source", True),
        dct_jpeg_quality=int(cfg["data"].get("dct_jpeg_quality", 100)),
        dct_cat_root=cfg["data"].get("dct_cat_root"),
    )
    for record in records:
        if "tp_s" not in str(record.image).lower() or record.mask is None:
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
        rows.append({"path": str(record.image), "mask": str(record.mask), "area": area, "dct_available": dct_available})
        if len(selected) >= count:
            break
    return selected, rows


def install_dher_scale_hook(model: torch.nn.Module, scale: float):
    if float(scale) == 1.0:
        return None

    def hook(_module, inputs, output):
        d1_pre = inputs[0]
        d1_out, diag = output
        write = d1_out - d1_pre
        scaled_write = write * float(scale)
        new_diag = dict(diag)
        if "support_write" in new_diag and torch.is_tensor(new_diag["support_write"]):
            new_diag["support_write"] = new_diag["support_write"] * float(scale)
        if "conflict_write" in new_diag and torch.is_tensor(new_diag["conflict_write"]):
            new_diag["conflict_write"] = new_diag["conflict_write"] * float(scale)
        if "write" in new_diag and torch.is_tensor(new_diag["write"]):
            new_diag["write"] = new_diag["write"] * float(scale)
        new_diag["diagnostic_dher_write_scale"] = torch.tensor(float(scale), device=d1_pre.device)
        return d1_pre + scaled_write, new_diag

    return model.dher.register_forward_hook(hook)


def install_d1_ideal_hook(model: torch.nn.Module, target: torch.Tensor, alpha: float, signed: bool):
    weight = model.mask_head.weight.detach()
    denom = weight.float().square().sum().clamp_min(1e-8)
    direction = weight / denom.to(dtype=weight.dtype)

    def pre_hook(_module, inputs):
        x = inputs[0]
        gt = F.interpolate(target.float(), size=x.shape[-2:], mode="nearest")
        evidence = gt * 2.0 - 1.0 if signed else gt
        residual = float(alpha) * evidence.to(dtype=x.dtype, device=x.device) * direction.to(dtype=x.dtype, device=x.device)
        return (x + residual,)

    return model.mask_head.register_forward_pre_hook(pre_hook)


def run_mode(model: torch.nn.Module, batch: dict[str, Any], target: torch.Tensor, mode: str) -> dict[str, Any]:
    handles = []
    try:
        if mode.startswith("dher_scale:"):
            handles.append(install_dher_scale_hook(model, float(mode.split(":", 1)[1])))
        elif mode.startswith("ideal_d1_signed:"):
            handles.append(install_d1_ideal_hook(model, target, float(mode.split(":", 1)[1]), signed=True))
        elif mode.startswith("ideal_d1_fg:"):
            handles.append(install_d1_ideal_hook(model, target, float(mode.split(":", 1)[1]), signed=False))
        elif mode != "baseline":
            raise ValueError(f"unknown mode: {mode}")
        return forward_from_batch(model, batch)
    finally:
        for handle in handles:
            if handle is not None:
                handle.remove()


def summarize_modes(mode_scores: dict[str, list[np.ndarray]], labels: list[np.ndarray], threshold_count: int, locked_threshold: float) -> dict[str, Any]:
    lab = np.concatenate([x.reshape(-1) for x in labels]).astype(np.uint8)
    out = {}
    for mode, chunks in mode_scores.items():
        scores = np.concatenate([x.reshape(-1) for x in chunks]).astype(np.float32)
        out[mode] = {
            "oracle": best_f1(scores, lab, threshold_count),
            "locked": confusion(scores, lab, locked_threshold),
        }
    base_oracle = out["baseline"]["oracle"]["f1"]
    base_locked = out["baseline"]["locked"]["f1"]
    for row in out.values():
        row["delta_oracle_f1_vs_baseline"] = float(row["oracle"]["f1"] - base_oracle)
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
    if model.dher is None:
        raise RuntimeError("This audit requires DHER.")

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

    modes = ["baseline"]
    modes += [f"dher_scale:{value}" for value in (0.0, 10.0, 100.0, 1000.0)]
    modes += [f"ideal_d1_fg:{value}" for value in (0.1, 0.5, 1.0, 2.0, 5.0)]
    modes += [f"ideal_d1_signed:{value}" for value in (0.1, 0.5, 1.0, 2.0, 5.0)]
    mode_scores = {mode: [] for mode in modes}
    labels = []
    per_sample = []
    locked_threshold = float(checkpoint.get("selection_threshold", cfg.get("evaluation", {}).get("threshold", 0.5)))

    with torch.no_grad():
        for index, (record, row) in enumerate(zip(selected, selected_rows)):
            raw_target = original_target(record)
            labels.append(raw_target.astype(np.uint8))
            batch = dataset[index]
            batch = {k: v[None] if torch.is_tensor(v) else v for k, v in batch.items()}
            batch = to_device(batch, device)
            target = batch["mask"].float()
            baseline = run_mode(model, batch, target, "baseline")
            baseline_logits = baseline["mask_logits"].detach().float()
            baseline_prob_640 = baseline_logits.sigmoid()
            sample_dir = outdir / f"sample_{index:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            save_gray(sample_dir / "gt_640.png", target[0, 0].detach().cpu().numpy())
            sample_row: dict[str, Any] = {**row, "index": index, "original_shape": list(raw_target.shape), "modes": {}}
            arrays: dict[str, np.ndarray] = {"gt_640": target[0, 0].detach().cpu().numpy().astype("float32")}
            dher = baseline.get("dher", {})
            dher_write = None
            if "support_write" in dher and "conflict_write" in dher:
                dher_write = dher["support_write"] + dher["conflict_write"]
            elif "write" in dher:
                dher_write = dher["write"]
            if dher_write is not None:
                write_map = dher_write.detach().float().square().mean(1)[0].sqrt().cpu().numpy().astype("float32")
                arrays["baseline_dher_write_rms_map"] = write_map
                save_gray(sample_dir / "baseline_dher_write_rms_map.png", write_map)

            for mode in modes:
                output = baseline if mode == "baseline" else run_mode(model, batch, target, mode)
                logits = output["mask_logits"].detach().float()
                prob_640 = logits.sigmoid()
                prob = F.interpolate(prob_640, size=raw_target.shape, mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float32)
                mode_scores[mode].append(prob)
                arrays[f"{safe_name(mode)}_prob_640"] = prob_640[0, 0].cpu().numpy().astype("float32")
                if mode in {"baseline", "dher_scale:100.0", "dher_scale:1000.0", "ideal_d1_signed:1.0", "ideal_d1_signed:2.0"}:
                    save_gray(sample_dir / f"{safe_name(mode)}_prob_640.png", prob_640[0, 0].cpu().numpy())
                d = output.get("dher", {})
                sample_row["modes"][mode] = {
                    "oracle": best_f1(prob.reshape(-1), raw_target.reshape(-1), args.threshold_count),
                    "locked": confusion(prob.reshape(-1), raw_target.reshape(-1), locked_threshold),
                    "logit_mean_abs_diff_vs_baseline_640": float((logits - baseline_logits).abs().mean().cpu()),
                    "logit_max_abs_diff_vs_baseline_640": float((logits - baseline_logits).abs().max().cpu()),
                    "prob_mean_abs_diff_vs_baseline_640": float((prob_640 - baseline_prob_640).abs().mean().cpu()),
                    "fg_prob_mean_640": float(prob_640[target > 0.5].mean().cpu()),
                    "bg_prob_mean_640": float(prob_640[target <= 0.5].mean().cpu()),
                    "dher_support_write_rms": rms(d.get("support_write")),
                    "dher_conflict_write_rms": rms(d.get("conflict_write")),
                    "dher_write_rms": rms(d.get("write")) if "write" in d else rms(dher_write),
                }
            np.savez_compressed(sample_dir / "dher_evidence_intervention_arrays.npz", **arrays)
            per_sample.append(sample_row)
            print(f"[audit] {index + 1}/{len(selected)} {record.image.name}", flush=True)

    pooled = summarize_modes(mode_scores, labels, args.threshold_count, locked_threshold)
    baseline_oracle = pooled["baseline"]["oracle"]["f1"]
    best_dher = max((pooled[m]["oracle"]["f1"], m) for m in modes if m.startswith("dher_scale:"))
    best_ideal = max((pooled[m]["oracle"]["f1"], m) for m in modes if m.startswith("ideal_d1_"))
    decision = {
        "best_dher_scale_mode": best_dher[1],
        "best_dher_scale_oracle_gain": float(best_dher[0] - baseline_oracle),
        "best_ideal_d1_mode": best_ideal[1],
        "best_ideal_d1_oracle_gain": float(best_ideal[0] - baseline_oracle),
        "existing_dher_write_amplitude_is_enough": bool(best_dher[0] - baseline_oracle >= 0.02),
        "mask_head_path_can_use_strong_evidence": bool(best_ideal[0] - baseline_oracle >= 0.02),
        "suggestion": "",
    }
    if decision["existing_dher_write_amplitude_is_enough"]:
        decision["suggestion"] = "Existing DHER write direction can move masks if scaled; train a larger/learnable visibility gate with a matched continuation control."
    elif decision["mask_head_path_can_use_strong_evidence"]:
        decision["suggestion"] = "Current DHER write direction is too weak or poorly aligned; implement a zero-init evidence-to-logit/D1 visibility path before training."
    else:
        decision["suggestion"] = "Even ideal D1-visible evidence does not help these samples; investigate decoder calibration or data labels before adding DHER capacity."

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
    dump_yaml(outdir / "dher_evidence_intervention_report.yaml", report)
    (outdir / "dher_evidence_intervention_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "samples": len(selected),
        "baseline_oracle_f1": baseline_oracle,
        "best_dher_scale": {"mode": best_dher[1], "oracle_f1": best_dher[0], "gain": best_dher[0] - baseline_oracle},
        "best_ideal_d1": {"mode": best_ideal[1], "oracle_f1": best_ideal[0], "gain": best_ideal[0] - baseline_oracle},
        "decision": decision,
        "out": str(outdir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
