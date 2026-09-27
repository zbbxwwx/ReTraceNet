#!/usr/bin/env python
"""Create a ReTraceNet CSV manifest from image and mask directories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--mask-suffix", default="")
    parser.add_argument("--relative-to", help="write paths relative to this dataset root")
    parser.add_argument("--allow-missing-masks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.images).resolve()
    mask_dir = Path(args.masks).resolve() if args.masks else None
    relative_to = Path(args.relative_to).resolve() if args.relative_to else None
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in extensions)
    mask_index = {}
    if mask_dir:
        for path in mask_dir.rglob("*"):
            if path.suffix.lower() in extensions:
                key = path.stem
                if args.mask_suffix and key.endswith(args.mask_suffix):
                    key = key[: -len(args.mask_suffix)]
                mask_index[key] = path
    rows = []
    missing = []
    for image in images:
        mask = mask_index.get(image.stem)
        if mask_dir and mask is None and not args.allow_missing_masks:
            missing.append(image.name)
            continue
        def display(path: Path | None) -> str:
            if path is None:
                return ""
            try:
                return path.relative_to(relative_to).as_posix() if relative_to else str(path)
            except ValueError:
                return str(path)
        label = 0
        if mask is not None:
            with Image.open(mask) as handle:
                extrema = handle.convert("L").getextrema()
            label = int(extrema[1] > 0)
        rows.append({
            "image": display(image),
            "mask": display(mask),
            "split": args.split,
            "dataset": args.dataset,
            "label": label,
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "mask", "split", "dataset", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} records to {output}; skipped {len(missing)} missing masks")


if __name__ == "__main__":
    main()
