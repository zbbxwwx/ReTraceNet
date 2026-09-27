#!/usr/bin/env python
"""Evaluate a checkpoint under deterministic post-processing conditions."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from retrace.data import build_loader, pil_to_tensor
from retrace.engine import evaluate_model
from retrace.losses import ReTraceLoss
from retrace.model import build_model
from retrace.utils import choose_device, load_checkpoint, load_config, write_json


DEFAULT_CONDITIONS = ["clean", "jpeg90", "jpeg70", "jpeg50", "scale075", "scale050", "blur1", "blur2", "combo"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    return parser.parse_args()


def jpeg_batch(image: torch.Tensor, quality: int) -> torch.Tensor:
    device, dtype = image.device, image.dtype
    converted = []
    for sample in image.detach().cpu():
        array = np.uint8(np.clip(sample.permute(1, 2, 0).numpy(), 0, 1) * 255)
        buffer = io.BytesIO()
        Image.fromarray(array, "RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as handle:
            converted.append(pil_to_tensor(handle.convert("RGB")))
    return torch.stack(converted).to(device=device, dtype=dtype)


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    coordinate = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-0.5 * (coordinate / sigma).square())
    kernel = kernel / kernel.sum()
    kernel_x = kernel[None, None, None, :].repeat(3, 1, 1, 1)
    kernel_y = kernel[None, None, :, None].repeat(3, 1, 1, 1)
    image = F.conv2d(image, kernel_x, padding=(0, radius), groups=3)
    return F.conv2d(image, kernel_y, padding=(radius, 0), groups=3)


def transform_for(condition: str):
    def transform(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        if condition == "clean":
            return image
        if condition.startswith("jpeg"):
            return jpeg_batch(image, int(condition[4:]))
        if condition.startswith("scale"):
            scale = int(condition[5:]) / 100.0
            small = F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False, antialias=True)
            return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
        if condition.startswith("blur"):
            return gaussian_blur(image, float(condition[4:]))
        if condition == "combo":
            small = F.interpolate(image, scale_factor=0.5, mode="bilinear", align_corners=False, antialias=True)
            restored = F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
            return jpeg_batch(restored, 50)
        raise ValueError(f"unknown condition: {condition}")
    return transform


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(config["model"]).to(device)
    state = checkpoint["model"]
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    load_result = model.load_state_dict(compatible_state, strict=False)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if missing or unexpected or len(compatible_state) != len(state):
        print(
            f"non-strict checkpoint load: loaded={len(compatible_state)} "
            f"missing={missing[:12]} unexpected={unexpected[:12]}"
        )
    criterion = ReTraceLoss.from_config(config["loss"]).to(device)
    loader_config = {**config["data"], "batch_size": config["evaluation"]["batch_size"]}
    loader = build_loader(loader_config, args.split, shuffle=False)
    results = {}
    for condition in args.conditions:
        losses, metrics = evaluate_model(
            model,
            criterion,
            loader,
            device,
            bool(config["training"].get("amp", True) and device.type == "cuda"),
            float(config["evaluation"].get("threshold", 0.5)),
            int(config["evaluation"].get("thresholds", 101)),
            image_transform=transform_for(condition),
        )
        results[condition] = {"losses": losses, "metrics": metrics}
        print(condition, metrics["pixel_f1_fixed"])
    write_json({"checkpoint": str(Path(args.checkpoint).resolve()), "conditions": results}, args.output)


if __name__ == "__main__":
    main()
