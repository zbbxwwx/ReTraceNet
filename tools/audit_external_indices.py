from pathlib import Path
from PIL import Image
import numpy as np
import sys

sys.path.insert(0, "tools")
import run_cross_dataset_after_chain as cross

root = Path("/group4/zouwenxuan/iml_generalization/paper_compare/mvss_five_indices")
for name in ["CASIA1", "Columbia", "NIST16", "IMD2020"]:
    records = cross.parse_index(root / f"{name}.txt", name)
    bad_size = []
    empty = []
    full = []
    ratios = []
    max_pixels = (0, None, None)
    for record in records:
        with Image.open(record.image) as image:
            size = image.size
        with Image.open(record.mask) as mask_image:
            mask = np.asarray(mask_image.convert("L")) > 127
        if mask.shape != (size[1], size[0]):
            bad_size.append((str(record.image), size, mask.shape))
            continue
        ratio = float(mask.mean())
        ratios.append(ratio)
        if not mask.any():
            empty.append(str(record.mask))
        if mask.all():
            full.append(str(record.mask))
        pixels = size[0] * size[1]
        if pixels > max_pixels[0]:
            max_pixels = (pixels, str(record.image), size)
    print({"dataset": name, "count": len(records), "bad_size": len(bad_size),
           "empty_masks": len(empty), "full_masks": len(full),
           "min_ratio": min(ratios) if ratios else None,
           "median_ratio": float(np.median(ratios)) if ratios else None,
           "max_ratio": max(ratios) if ratios else None,
           "max_image": max_pixels, "examples_bad_size": bad_size[:3]})
