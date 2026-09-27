"""FairDataset variant exposing native quantized Y-DCT blocks.

RGB, mask, coefficient crop, and qtable all follow the same block-aligned crop
chosen by the established CASIA/IMD training pipeline.
"""

from __future__ import annotations

import numpy as np
import torch

from fair_training_data import FairDataset


class NativeJPEGFairDataset(FairDataset):
    def __getitem__(self, key):
        sample = super().__getitem__(key)
        import jpegio

        jpeg = jpegio.read(sample["jpeg_path"])
        coefficients = jpeg.coef_arrays[0]
        top, left = map(int, sample["crop_origin"].tolist())
        height, width = map(int, sample["image"].shape[-2:])
        if height % 8 or width % 8:
            raise ValueError(f"native JPEG crop must be block aligned: {(height, width)}")

        crop = np.zeros((height, width), dtype=np.float32)
        source = coefficients[
            top : min(top + height, coefficients.shape[0]),
            left : min(left + width, coefficients.shape[1]),
        ]
        crop[: source.shape[0], : source.shape[1]] = source

        # [H, W] coefficient layout -> [64, H/8, W/8] block-grid tensor.
        blocks = crop.reshape(height // 8, 8, width // 8, 8)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(height // 8, width // 8, 64)
        sample["dct"] = torch.from_numpy(blocks.transpose(2, 0, 1).copy())
        sample["native_jpeg_layout"] = "quantized_y_dct_64_block_grid"
        return sample

