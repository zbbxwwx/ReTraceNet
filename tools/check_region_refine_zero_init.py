#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrace.data import build_loader
from retrace.utils import choose_device, load_checkpoint, load_config
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import forward_from_batch


def load_matching(model: torch.nn.Module, path: str) -> dict:
    checkpoint = load_checkpoint(path, torch.device("cpu"))
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    filtered = {k: v for k, v in source.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
    result = model.load_state_dict(filtered, strict=False)
    return {"loaded": len(filtered), "missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out")
    args = parser.parse_args()
    base_cfg = load_config(args.base_config)
    region_cfg = load_config(args.region_config)
    device = choose_device(args.device)
    base = build_final_model(base_cfg["model"]).to(device).eval()
    region = build_final_model(region_cfg["model"]).to(device).eval()
    base_report = load_matching(base, args.checkpoint)
    region_report = load_matching(region, args.checkpoint)
    loader = build_loader({**region_cfg["data"], "batch_size": 1, "num_workers": 0}, "val", shuffle=False)
    batch = next(iter(loader))
    image = batch["image"].to(device)
    base_out = forward_from_batch(base, batch, device, image)
    region_out = forward_from_batch(region, batch, device, image)
    diff = (base_out["mask_logits"] - region_out["mask_logits"]).abs()
    residual = region_out.get("region_refine_residual")
    result = {
        "base_report": base_report,
        "region_report": region_report,
        "max_abs_diff": float(diff.max().detach().cpu()),
        "mean_abs_diff": float(diff.mean().detach().cpu()),
        "residual_abs_max": float(residual.abs().max().detach().cpu()) if residual is not None else None,
    }
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
