#!/usr/bin/env python
"""Generate a tiny procedural dataset for smoke testing, not for benchmarking."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/toy")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def base_image(size: int, rng: np.random.Generator) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    phase = rng.uniform(0, np.pi * 2, 3)
    channels = []
    for c in range(3):
        pattern = 112 + 45 * np.sin(xx / rng.uniform(12, 35) + phase[c])
        pattern += 35 * np.cos(yy / rng.uniform(15, 42) - phase[c])
        pattern += rng.normal(0, 7, (size, size))
        channels.append(pattern)
    return Image.fromarray(np.uint8(np.clip(np.stack(channels, -1), 0, 255)), "RGB")


def manipulate(image: Image.Image, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    size = image.width
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    width = rng.randint(size // 7, size // 3)
    height = rng.randint(size // 7, size // 3)
    left = rng.randint(6, size - width - 6)
    top = rng.randint(6, size - height - 6)
    if rng.random() < 0.5:
        draw.ellipse((left, top, left + width, top + height), fill=255)
    else:
        draw.rounded_rectangle((left, top, left + width, top + height), radius=5, fill=255)
    soft_mask = mask.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.0)))
    source_left = rng.randint(0, max(0, size - width - 1))
    source_top = rng.randint(0, max(0, size - height - 1))
    patch = image.crop((source_left, source_top, source_left + width, source_top + height))
    patch = patch.filter(ImageFilter.GaussianBlur(rng.uniform(0.0, 0.8)))
    altered = image.copy()
    altered.paste(patch, (left, top), soft_mask.crop((left, top, left + width, top + height)))
    return altered, mask


def main() -> None:
    args = parse_args()
    if args.count < 12:
        raise ValueError("--count must be at least 12 so every split is meaningful")
    output = Path(args.output)
    image_dir, mask_dir = output / "images", output / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    rows = []
    splits = ["train"] * max(4, round(args.count * 0.6))
    splits += ["val"] * max(2, round(args.count * 0.2))
    splits += ["test"] * max(2, args.count - len(splits))
    splits = splits[: args.count]
    for index in range(args.count):
        image = base_image(args.size, np_rng)
        authentic = index % 4 == 0
        if authentic:
            result, mask = image, Image.new("L", image.size, 0)
        else:
            result, mask = manipulate(image, rng)
        image_name = f"sample_{index:04d}.png"
        mask_name = f"sample_{index:04d}.png"
        result.save(image_dir / image_name)
        mask.save(mask_dir / mask_name)
        rows.append({
            "image": f"images/{image_name}",
            "mask": f"masks/{mask_name}",
            "split": splits[index],
            "dataset": "toy",
            "label": int(not authentic),
        })
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "mask", "split", "dataset", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"created {len(rows)} samples under {output}")


if __name__ == "__main__":
    main()
