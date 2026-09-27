"""Training-time degradations used by the consistency objective."""

from __future__ import annotations

import random

import torch
from torch import Tensor
import torch.nn.functional as F


def straight_through_round(x: Tensor) -> Tensor:
    return x + (x.round() - x).detach()


def degrade_batch(image: Tensor) -> Tensor:
    """Apply a geometry-preserving differentiable social-media degradation."""

    height, width = image.shape[-2:]
    scale = random.uniform(0.5, 0.9)
    small = F.interpolate(
        image,
        size=(max(8, int(height * scale)), max(8, int(width * scale))),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    restored = F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
    if random.random() < 0.5:
        restored = F.avg_pool2d(restored, 3, stride=1, padding=1)
    levels = random.choice((64.0, 96.0, 128.0, 192.0))
    restored = straight_through_round(restored * levels) / levels
    return restored.clamp(0.0, 1.0)
