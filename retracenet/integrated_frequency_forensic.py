"""
Integrated residual-frequency forensic stream.

Design:
    Image
      ├─ constrained residual bank
      └─ fixed local DCT frequency prior
             ↓
       gated early fusion
             ↓
       original forensic encoder
             ↓
          N1-N4

Important:
- This is ONE forensic stream, not a parallel frequency backbone.
- No CAT-Net module.
- No CAT-Net pretrained weight.
- No HRNet DCT branch.
- Frequency evidence is fused once before the forensic encoder.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _make_dct_filters() -> Tensor:
    """64 orthonormal 8x8 DCT-II filters."""
    filters = []

    for u in range(8):
        for v in range(8):
            basis = torch.zeros(8, 8, dtype=torch.float32)

            au = math.sqrt(1.0 / 8.0) if u == 0 else math.sqrt(2.0 / 8.0)
            av = math.sqrt(1.0 / 8.0) if v == 0 else math.sqrt(2.0 / 8.0)

            for x in range(8):
                for y in range(8):
                    basis[x, y] = (
                        au
                        * av
                        * math.cos((2 * x + 1) * u * math.pi / 16.0)
                        * math.cos((2 * y + 1) * v * math.pi / 16.0)
                    )

            filters.append(basis)

    return torch.stack(filters, dim=0).unsqueeze(1)


class LocalFrequencyPrior(nn.Module):
    """
    Fixed local spectral analysis.

    Produces six compact maps:
        1 low-frequency AC
        2 mid-frequency AC
        3 high-frequency AC
        4 horizontal-oriented AC
        5 vertical-oriented AC
        6 local block/high-frequency contrast

    DCT filters are fixed; only the following fusion is learned.
    """

    out_channels = 6

    def __init__(self) -> None:
        super().__init__()

        self.register_buffer(
            "dct_filters",
            _make_dct_filters(),
            persistent=False,
        )

        low = []
        mid = []
        high = []
        horizontal = []
        vertical = []

        for idx, (u, v) in enumerate(
            (u, v) for u in range(8) for v in range(8)
        ):
            if u == 0 and v == 0:
                continue

            radial = u + v

            if radial <= 2:
                low.append(idx)
            elif radial <= 5:
                mid.append(idx)
            else:
                high.append(idx)

            # Variation mainly along x/y directions.
            if u >= 2 and v <= 1:
                horizontal.append(idx)

            if v >= 2 and u <= 1:
                vertical.append(idx)

        self.groups = (
            low,
            mid,
            high,
            horizontal,
            vertical,
        )

    def forward(self, image: Tensor) -> Tensor:
        # RGB -> luminance. Input is already [0, 1].
        r = image[:, 0:1]
        g = image[:, 1:2]
        b = image[:, 2:3]

        y = 0.299 * r + 0.587 * g + 0.114 * b

        # Overlapping local DCT rather than JPEG-grid-only DCT.
        # stride=4 reduces dependence on an exact JPEG 8x8 phase.
        coeff = F.conv2d(
            y,
            self.dct_filters.to(dtype=y.dtype),
            stride=4,
            padding=2,
        )

        coeff = torch.log1p(coeff.abs())

        maps = [
            coeff[:, ids].mean(dim=1, keepdim=True)
            for ids in self.groups
        ]

        # A simple local high-frequency/block contrast cue.
        smooth = F.avg_pool2d(
            y,
            kernel_size=5,
            stride=4,
            padding=2,
        )

        local = F.interpolate(
            smooth,
            size=y.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        block = (y - local).abs()

        block = F.avg_pool2d(
            block,
            kernel_size=5,
            stride=4,
            padding=2,
        )

        # Align all six maps.
        target = maps[0].shape[-2:]

        if block.shape[-2:] != target:
            block = F.interpolate(
                block,
                size=target,
                mode="bilinear",
                align_corners=False,
            )

        return torch.cat([*maps, block], dim=1)


class IntegratedFrequencyForensicStream(nn.Module):
    """
    Wraps the existing LightweightForensicStream.

    The original residual bank and forensic encoder are preserved.
    Frequency evidence is injected ONCE before stage1.
    """

    def __init__(
        self,
        base_stream: nn.Module,
        residual_kernels: int = 8,
    ) -> None:
        super().__init__()

        if not hasattr(base_stream, "residual_bank"):
            raise TypeError(
                "base forensic stream must expose residual_bank"
            )

        if not hasattr(base_stream, "encode_from_residual"):
            raise TypeError(
                "base forensic stream must expose encode_from_residual"
            )

        self.base = base_stream
        self.frequency_prior = LocalFrequencyPrior()

        freq_channels = LocalFrequencyPrior.out_channels

        # Compact learnable projection.
        self.freq_project = nn.Sequential(
            nn.Conv2d(
                freq_channels,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(4, 16),
            nn.GELU(),

            nn.Conv2d(
                16,
                residual_kernels,
                kernel_size=1,
                bias=False,
            ),
        )

        # Reliability gate is ONLY an internal forensic operation.
        # It is not another network branch.
        self.freq_gate = nn.Sequential(
            nn.Conv2d(
                residual_kernels + freq_channels,
                residual_kernels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        # Do not suppress frequency completely at initialization.
        nn.init.zeros_(self.freq_gate[0].weight)
        nn.init.constant_(self.freq_gate[0].bias, -1.0)

    @property
    def residual_bank(self):
        return self.base.residual_bank

    def encode_from_residual(self, residual: Tensor):
        return self.base.encode_from_residual(residual)

    def forward(self, image: Tensor):
        residual = self.base.residual_bank(image)

        freq = self.frequency_prior(image)

        freq = F.interpolate(
            freq,
            size=residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        freq_delta = self.freq_project(freq)

        gate = self.freq_gate(
            torch.cat([residual, freq], dim=1)
        )

        # Residual-preserving fusion.
        fused = residual + gate * freq_delta

        return self.base.encode_from_residual(fused)
