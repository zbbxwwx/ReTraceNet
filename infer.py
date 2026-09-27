#!/usr/bin/env python
"""Full-resolution or overlap-tiled ReTraceNet inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from retrace.data import pil_to_tensor
from retrace.model import build_model
from retrace.utils import choose_device, load_checkpoint, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="needed only when checkpoint lacks its resolved config")
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--device")
    return parser.parse_args()


def starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    values = list(range(0, max(1, length - tile + 1), stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


@torch.inference_mode()
def predict_tiled(
    model: torch.nn.Module, image: torch.Tensor, tile: int, overlap: int
) -> tuple[dict[str, torch.Tensor], float]:
    _, _, height, width = image.shape
    map_sums = {
        key: image.new_zeros(1, 1, height, width)
        for key in ("probability", "variance", "causal_effect", "conflict", "proposal")
    }
    weight_sum = image.new_zeros(1, 1, height, width)
    scores: list[float] = []
    for top in starts(height, tile, overlap):
        for left in starts(width, tile, overlap):
            crop = image[..., top : min(top + tile, height), left : min(left + tile, width)]
            crop_height, crop_width = crop.shape[-2:]
            pad_h = (16 - crop_height % 16) % 16
            pad_w = (16 - crop_width % 16) % 16
            padded = F.pad(crop, (0, pad_w, 0, pad_h), mode="reflect")
            output = model(padded)
            conflict = F.interpolate(
                output["ledger_conflicts"][0],
                size=padded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            tile_maps = {
                "probability": output["mask_logits"].sigmoid(),
                "variance": output["log_var"].exp(),
                "causal_effect": F.relu(output["causal_effect"]),
                "conflict": conflict,
                "proposal": output["proposal_logits"].sigmoid(),
            }
            wy = torch.hann_window(crop_height, periodic=False, device=image.device).clamp_min(0.05)
            wx = torch.hann_window(crop_width, periodic=False, device=image.device).clamp_min(0.05)
            weight = (wy[:, None] * wx[None, :])[None, None]
            region = (..., slice(top, top + crop_height), slice(left, left + crop_width))
            for key, value in tile_maps.items():
                map_sums[key][region] += value[..., :crop_height, :crop_width] * weight
            weight_sum[region] += weight
            scores.append(float(output["image_logit"].sigmoid().item()))
    return {key: value / weight_sum for key, value in map_sums.items()}, max(scores)


def save_outputs(
    source: Image.Image, maps: dict[str, np.ndarray], score: float, output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    probability = maps["probability"]
    variance = maps["variance"]
    mask = Image.fromarray(np.uint8(np.clip(probability, 0, 1) * 255), mode="L")
    uncertainty = variance / max(1e-6, float(np.percentile(variance, 99)))
    uncertainty_image = Image.fromarray(np.uint8(np.clip(uncertainty, 0, 1) * 255), mode="L")
    base = np.asarray(source.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    alpha = np.clip(probability[..., None] * 0.65, 0, 0.65)
    overlay = Image.fromarray(np.uint8(base * (1 - alpha) + red * alpha), mode="RGB")
    mask.save(output.with_name(output.stem + "_mask.png"))
    uncertainty_image.save(output.with_name(output.stem + "_uncertainty.png"))
    for key in ("causal_effect", "conflict", "proposal"):
        value = maps[key]
        if key == "causal_effect":
            value = value / max(1e-6, float(np.percentile(value, 99)))
        diagnostic = Image.fromarray(
            np.uint8(np.clip(value, 0, 1) * 255), mode="L"
        )
        diagnostic.save(output.with_name(output.stem + f"_{key}.png"))
    overlay.save(output.with_name(output.stem + f"_overlay_s{score:.3f}.png"))


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint.get("config")
    if config is None:
        if not args.config:
            raise ValueError("checkpoint has no config; pass --config")
        config = load_config(args.config)
    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    input_path = Path(args.input)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = [input_path] if input_path.is_file() else [p for p in input_path.rglob("*") if p.suffix.lower() in extensions]
    output_dir = Path(args.output)
    for path in paths:
        with Image.open(path) as handle:
            source = handle.convert("RGB")
        tensor = pil_to_tensor(source).unsqueeze(0).to(device)
        maps, score = predict_tiled(model, tensor, args.tile_size, args.overlap)
        save_outputs(
            source,
            {key: value[0, 0].cpu().numpy() for key, value in maps.items()},
            score,
            output_dir / path.name,
        )
        print(f"{path}: integrity manipulation score={score:.4f}")


if __name__ == "__main__":
    main()
