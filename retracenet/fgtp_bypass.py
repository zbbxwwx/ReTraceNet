"""High-resolution forensic trace bypass for FGTP."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from retrace.blocks import LayerNorm2d


def local_average(x: Tensor, kernel_size: int = 3) -> Tensor:
    return F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)


class HighResolutionTraceBypass(nn.Module):
    """Preserve residual-bank traces before the forensic hierarchy downsamples them."""

    def __init__(self, residual_channels: int = 8, hr_channels: int = 64, decoder_channels: int = 128) -> None:
        super().__init__()
        self.full_res_projection = nn.Sequential(
            nn.Conv2d(residual_channels, hr_channels, 3, padding=1, bias=False),
            LayerNorm2d(hr_channels),
            nn.GELU(),
            nn.Conv2d(hr_channels, hr_channels, 3, padding=1, groups=hr_channels, bias=False),
            LayerNorm2d(hr_channels),
            nn.GELU(),
        )
        self.preserve = nn.Sequential(
            nn.Conv2d(hr_channels * 2, hr_channels, 1, bias=False),
            LayerNorm2d(hr_channels),
            nn.GELU(),
            nn.Conv2d(hr_channels, hr_channels, 3, padding=1, groups=hr_channels, bias=False),
            LayerNorm2d(hr_channels),
            nn.GELU(),
            nn.Conv2d(hr_channels, hr_channels, 1, bias=False),
            LayerNorm2d(hr_channels),
            nn.GELU(),
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(2)
        self.d1_projection = nn.Sequential(
            nn.Conv2d(hr_channels * 8, decoder_channels, 1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, groups=decoder_channels, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, 1),
        )
        self.alpha_hr = nn.Parameter(torch.tensor(0.0))

    def forward(self, residual: Tensor, d1: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        full = self.full_res_projection(residual)
        f_hr = F.avg_pool2d(full, 2, stride=2)
        local_detail = f_hr - local_average(f_hr, 3)
        p_hr = self.preserve(torch.cat([f_hr, local_detail.abs()], dim=1))
        p_detail = p_hr - local_average(p_hr, 3)
        packed = torch.cat([self.pixel_unshuffle(p_hr), self.pixel_unshuffle(p_detail.abs())], dim=1)
        delta_d1 = self.d1_projection(packed)
        if delta_d1.shape[-2:] != d1.shape[-2:]:
            delta_d1 = F.interpolate(delta_d1, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        written_delta = torch.tanh(self.alpha_hr) * delta_d1
        d1_updated = d1 + written_delta
        diagnostics = {
            "residual0": residual.detach(),
            "f_hr": f_hr.detach(),
            "p_hr": p_hr.detach(),
            "p_hr_local_detail": p_detail.detach(),
            "delta_d1": delta_d1.detach(),
            "written_delta_d1": written_delta.detach(),
            "alpha_hr": self.alpha_hr.detach(),
        }
        return d1_updated, diagnostics
