"""Lightweight residual feature stream for the v2 B0 base."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from retrace.blocks import ConstrainedResidualBank, ConvNeXtBlock, LayerNorm2d


class FixedHighFrequencyFFTPrior(nn.Module):
    """Fixed high-frequency spatial prior computed from FFT magnitude."""

    def __init__(self, cutoff: float = 0.18) -> None:
        super().__init__()
        self.cutoff = float(cutoff)

    def forward(self, image: Tensor) -> Tensor:
        x = image.float()
        if x.shape[1] >= 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x.mean(dim=1, keepdim=True)
        gray = gray - gray.mean(dim=(-2, -1), keepdim=True)
        h, w = gray.shape[-2:]
        spectrum = torch.fft.rfft2(gray, dim=(-2, -1), norm="ortho")
        fy = torch.fft.fftfreq(h, device=gray.device, dtype=gray.dtype).view(h, 1)
        fx = torch.fft.rfftfreq(w, device=gray.device, dtype=gray.dtype).view(1, -1)
        radius = torch.sqrt(fy.square() + fx.square()) / (0.5 * 2.0 ** 0.5)
        highpass = (radius >= self.cutoff).to(spectrum.dtype).view(1, 1, h, w // 2 + 1)
        high = torch.fft.irfft2(spectrum * highpass, s=(h, w), dim=(-2, -1), norm="ortho")
        prior = high.abs()
        mean = prior.mean(dim=(-2, -1), keepdim=True)
        std = prior.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        prior = (prior - mean) / std
        return prior.to(dtype=image.dtype)


class FFTPriorResidualFusion(nn.Module):
    """Zero-effect fusion from a fixed FFT prior into residual-bank feature space."""

    def __init__(self, residual_kernels: int, hidden_channels: int | None = None, cutoff: float = 0.18) -> None:
        super().__init__()
        hidden = int(hidden_channels or max(8, residual_kernels * 2))
        self.prior = FixedHighFrequencyFFTPrior(cutoff=cutoff)
        self.adapter = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1, bias=False),
            LayerNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, residual_kernels, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, image: Tensor, residual: Tensor) -> Tensor:
        prior = self.prior(image)
        if prior.shape[-2:] != residual.shape[-2:]:
            prior = F.interpolate(prior, size=residual.shape[-2:], mode="bilinear", align_corners=False)
        return residual + self.adapter(prior)


class FixedSRMPrior(nn.Module):
    """Fixed SRM/Bayar-style high-pass residual priors from grayscale input."""

    def __init__(self) -> None:
        super().__init__()
        kernels = torch.tensor([
            [[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, -1, 4, -1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
            [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -2, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
        ], dtype=torch.float32)
        denom = kernels.abs().sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        kernels = kernels / denom
        self.register_buffer("weight", kernels[:, None, :, :])

    def forward(self, image: Tensor) -> Tensor:
        x = image.float()
        if x.shape[1] >= 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x.mean(dim=1, keepdim=True)
        prior = F.conv2d(gray, self.weight.to(dtype=gray.dtype), padding=2)
        mean = prior.mean(dim=(-2, -1), keepdim=True)
        std = prior.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        prior = (prior - mean) / std
        return prior.to(dtype=image.dtype)


class SRMPriorResidualFusion(nn.Module):
    """Zero-effect fusion from fixed SRM priors into residual-bank feature space."""

    def __init__(self, residual_kernels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_channels or max(16, residual_kernels * 2))
        self.prior = FixedSRMPrior()
        self.adapter = nn.Sequential(
            nn.Conv2d(5, hidden, 3, padding=1, bias=False),
            LayerNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, residual_kernels, 1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, image: Tensor, residual: Tensor) -> Tensor:
        prior = self.prior(image)
        if prior.shape[-2:] != residual.shape[-2:]:
            prior = F.interpolate(prior, size=residual.shape[-2:], mode="bilinear", align_corners=False)
        return residual + self.adapter(prior)


class MultiResolutionForensicExchange(nn.Module):
    """Bidirectional residual exchange inside the forensic pyramid."""

    def __init__(self, channels: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.channels = tuple(int(c) for c in channels)
        self.from_fine = nn.ModuleList(
            nn.Sequential(nn.Conv2d(self.channels[i - 1], self.channels[i], 1, bias=False), LayerNorm2d(self.channels[i]), nn.GELU())
            for i in range(1, len(self.channels))
        )
        self.from_coarse = nn.ModuleList(
            nn.Sequential(nn.Conv2d(self.channels[i + 1], self.channels[i], 1, bias=False), LayerNorm2d(self.channels[i]), nn.GELU())
            for i in range(len(self.channels) - 1)
        )
        self.exchange = nn.ModuleList()
        for i, ch in enumerate(self.channels):
            sources = 1 + int(i > 0) + int(i < len(self.channels) - 1)
            block = nn.Sequential(
                nn.Conv2d(ch * sources, ch, 1, bias=False),
                LayerNorm2d(ch),
                nn.GELU(),
                ConvNeXtBlock(ch),
                nn.Conv2d(ch, ch, 1),
            )
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)
            self.exchange.append(block)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        exchanged: list[Tensor] = []
        for i, feat in enumerate(features):
            pieces = [feat]
            if i > 0:
                fine = F.interpolate(features[i - 1], size=feat.shape[-2:], mode="bilinear", align_corners=False)
                pieces.append(self.from_fine[i - 1](fine))
            if i < len(features) - 1:
                coarse = F.interpolate(features[i + 1], size=feat.shape[-2:], mode="bilinear", align_corners=False)
                pieces.append(self.from_coarse[i](coarse))
            exchanged.append(feat + self.exchange[i](torch.cat(pieces, dim=1)))
        return exchanged




class SRMGuidedForensicStream(nn.Module):
    """SRM/Bayar-guided independent forensic encoder returning N1-N4."""

    def __init__(self, residual_kernels: int = 8, channels: tuple[int, int, int, int] = (64, 128, 256, 512), enable_mrfe: bool = False) -> None:
        super().__init__()
        self.residual_bank = ConstrainedResidualBank(residual_kernels)
        self.srm_prior = FixedSRMPrior()
        c1, c2, c3, c4 = channels
        in_ch = residual_kernels + 5
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(c1),
            nn.GELU(),
            ConvNeXtBlock(c1),
            ConvNeXtBlock(c1),
        )
        self.stage2 = self._down_stage(c1, c2)
        self.stage3 = self._down_stage(c2, c3)
        self.stage4 = self._down_stage(c3, c4)
        self.mrfe = MultiResolutionForensicExchange(tuple(channels)) if enable_mrfe else None

    @staticmethod
    def _down_stage(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
            ConvNeXtBlock(out_channels),
            ConvNeXtBlock(out_channels),
        )

    def forward(self, image: Tensor) -> list[Tensor]:
        residual = self.residual_bank(image)
        srm = self.srm_prior(image)
        if srm.shape[-2:] != residual.shape[-2:]:
            srm = F.interpolate(srm, size=residual.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([residual, srm.to(dtype=residual.dtype)], dim=1)
        n1 = self.stage1(x)
        n2 = self.stage2(n1)
        n3 = self.stage3(n2)
        n4 = self.stage4(n3)
        features = [n1, n2, n3, n4]
        return self.mrfe(features) if self.mrfe is not None else features


class LightweightForensicStream(nn.Module):
    """Residual-bank stream returning N1-N4 at 1/4 to 1/32 resolution."""

    def __init__(self, residual_kernels: int = 8, channels: tuple[int, int, int, int] = (64, 128, 256, 512), enable_mrfe: bool = False, enable_fft_prior: bool = False, fft_prior_cutoff: float = 0.18, enable_srm_prior: bool = False) -> None:
        super().__init__()
        self.residual_bank = ConstrainedResidualBank(residual_kernels)
        self.fft_prior_fusion = FFTPriorResidualFusion(residual_kernels, cutoff=fft_prior_cutoff) if enable_fft_prior else None
        self.srm_prior_fusion = SRMPriorResidualFusion(residual_kernels) if enable_srm_prior else None
        c1, c2, c3, c4 = channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(residual_kernels, c1, 7, stride=2, padding=3, bias=False),
            LayerNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c1, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(c1),
            nn.GELU(),
            ConvNeXtBlock(c1),
        )
        self.stage2 = self._down_stage(c1, c2)
        self.stage3 = self._down_stage(c2, c3)
        self.stage4 = self._down_stage(c3, c4)
        self.mrfe = MultiResolutionForensicExchange(tuple(channels)) if enable_mrfe else None

    @staticmethod
    def _down_stage(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
            ConvNeXtBlock(out_channels),
        )

    def encode_from_residual(self, residual: Tensor) -> list[Tensor]:
        n1 = self.stage1(residual)
        n2 = self.stage2(n1)
        n3 = self.stage3(n2)
        n4 = self.stage4(n3)
        features = [n1, n2, n3, n4]
        return self.mrfe(features) if self.mrfe is not None else features

    def forward(self, image: Tensor) -> list[Tensor]:
        residual = self.residual_bank(image)
        if self.fft_prior_fusion is not None:
            residual = self.fft_prior_fusion(image, residual)
        if self.srm_prior_fusion is not None:
            residual = self.srm_prior_fusion(image, residual)
        return self.encode_from_residual(residual)
