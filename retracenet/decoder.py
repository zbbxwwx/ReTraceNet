"""Progressive feature decoder for the v2 B0 base."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from retrace.blocks import ConvNeXtBlock, LayerNorm2d


class DecoderStage(nn.Module):
    def __init__(self, content_channels: int, detail_channels: int, out_channels: int, has_parent: bool) -> None:
        super().__init__()
        self.content = nn.Sequential(nn.Conv2d(content_channels, out_channels, 1, bias=False), LayerNorm2d(out_channels))
        self.detail = nn.Sequential(nn.Conv2d(detail_channels, out_channels, 1, bias=False), LayerNorm2d(out_channels))
        in_channels = out_channels * (3 if has_parent else 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
            ConvNeXtBlock(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )
        self.has_parent = has_parent

    def forward(self, content: Tensor, detail: Tensor, parent: Tensor | None = None) -> Tensor:
        c = self.content(content)
        n = self.detail(detail)
        parts = [c, n]
        if self.has_parent:
            if parent is None:
                raise ValueError("parent feature is required for this decoder stage")
            parts.append(F.interpolate(parent, size=c.shape[-2:], mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(parts, dim=1))


class ForensicEvidenceReconstruction(nn.Module):
    """Use the forensic pyramid as a zero-effect residual reconstruction cue for D1."""

    def __init__(
        self,
        detail_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        decoder_channels: int = 128,
        proj_channels: int = 32,
    ) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(ch, proj_channels, 1, bias=False),
                LayerNorm2d(proj_channels),
                nn.GELU(),
            )
            for ch in detail_channels
        )
        in_channels = decoder_channels + proj_channels * len(detail_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, decoder_channels, 3, padding=1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            ConvNeXtBlock(decoder_channels),
            nn.Conv2d(decoder_channels, decoder_channels, 1),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, d1: Tensor, detail: list[Tensor]) -> Tensor:
        target_size = d1.shape[-2:]
        cues = []
        for feat, project in zip(detail, self.projections):
            cue = project(feat)
            if cue.shape[-2:] != target_size:
                cue = F.interpolate(cue, size=target_size, mode="bilinear", align_corners=False)
            cues.append(cue)
        residual = self.fuse(torch.cat([d1, *cues], dim=1))
        return d1 + residual


class ProgressiveDecoder(nn.Module):
    """Fuse C1-C4 and N1-N4 from coarse to fine."""

    def __init__(
        self,
        content_channels: tuple[int, int, int, int] = (256, 512, 1024, 2048),
        detail_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        decoder_channels: int = 128,
        enable_forensic_reconstruction: bool = False,
        forensic_reconstruction_proj_channels: int = 32,
    ) -> None:
        super().__init__()
        self.stage4 = DecoderStage(content_channels[3], detail_channels[3], decoder_channels, has_parent=False)
        self.stage3 = DecoderStage(content_channels[2], detail_channels[2], decoder_channels, has_parent=True)
        self.stage2 = DecoderStage(content_channels[1], detail_channels[1], decoder_channels, has_parent=True)
        self.stage1 = DecoderStage(content_channels[0], detail_channels[0], decoder_channels, has_parent=True)
        self.forensic_reconstruction = (
            ForensicEvidenceReconstruction(
                detail_channels=detail_channels,
                decoder_channels=decoder_channels,
                proj_channels=forensic_reconstruction_proj_channels,
            )
            if enable_forensic_reconstruction
            else None
        )

    def forward(self, content: list[Tensor], detail: list[Tensor]) -> list[Tensor]:
        d4 = self.stage4(content[3], detail[3])
        d3 = self.stage3(content[2], detail[2], d4)
        d2 = self.stage2(content[1], detail[1], d3)
        d1 = self.stage1(content[0], detail[0], d2)
        if self.forensic_reconstruction is not None:
            d1 = self.forensic_reconstruction(d1, detail)
        return [d1, d2, d3, d4]
