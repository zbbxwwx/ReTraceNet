"""Fine-grained trace preservation for high-resolution residual features."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from retrace.blocks import LayerNorm2d


class FineGrainedTracePreservationBlock(nn.Module):
    """Residual detail enhancer driven by content/detail discrepancy."""

    def __init__(self, content_channels: int, detail_channels: int) -> None:
        super().__init__()
        self.content_align = nn.Sequential(
            nn.Conv2d(content_channels, detail_channels, 1, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
        )
        self.detail_align = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels, 1, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
        )
        self.local3 = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels, 3, padding=1, groups=detail_channels, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
        )
        self.local5 = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels, 5, padding=2, groups=detail_channels, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(detail_channels * 5, detail_channels, 1, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, detail_channels, 3, padding=1, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, detail_channels, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, content: Tensor, detail: Tensor) -> tuple[Tensor, Tensor]:
        c = self.content_align(content)
        n = self.detail_align(detail)
        discrepancy = n - c
        product = n * c
        local = self.local3(discrepancy) + self.local5(discrepancy)
        delta = self.fuse(torch.cat([detail, n, c, discrepancy, product + local], dim=1))
        preserved = detail + torch.tanh(self.alpha) * delta
        return preserved, delta


class FineGrainedTracePreservation(nn.Module):
    """Apply trace preservation at 1/4 and 1/8 feature scales."""

    def __init__(self) -> None:
        super().__init__()
        self.scale1 = FineGrainedTracePreservationBlock(256, 64)
        self.scale2 = FineGrainedTracePreservationBlock(512, 128)

    def forward(self, content: list[Tensor], detail: list[Tensor]) -> tuple[list[Tensor], dict[str, Tensor]]:
        t1, delta1 = self.scale1(content[0], detail[0])
        t2, delta2 = self.scale2(content[1], detail[1])
        updated = [t1, t2, detail[2], detail[3]]
        diagnostics = {
            "n1": detail[0].detach(),
            "n2": detail[1].detach(),
            "t1": t1.detach(),
            "t2": t2.detach(),
            "t1_minus_n1": (t1 - detail[0]).detach(),
            "t2_minus_n2": (t2 - detail[1]).detach(),
            "delta1": delta1.detach(),
            "delta2": delta2.detach(),
            "alpha1": self.scale1.alpha.detach(),
            "alpha2": self.scale2.alpha.detach(),
        }
        return updated, diagnostics



def _local_average(x: Tensor, kernel_size: int) -> Tensor:
    return F.avg_pool2d(x, kernel_size, stride=1, padding=kernel_size // 2, count_include_pad=False)


class SelectiveFineGrainedTracePreservationBlock(nn.Module):
    """Center-surround detail preservation from content/detail discrepancy."""

    def __init__(self, content_channels: int, detail_channels: int, detail_kernel: int, context_kernel: int) -> None:
        super().__init__()
        self.detail_kernel = int(detail_kernel)
        self.context_kernel = int(context_kernel)
        self.residual_kernel = int(detail_kernel)
        self.content_projection = nn.Conv2d(content_channels, detail_channels, 1, bias=False)
        self.detail_projection = nn.Conv2d(detail_channels, detail_channels, 1, bias=False)
        self.content_norm = LayerNorm2d(detail_channels)
        self.detail_norm = LayerNorm2d(detail_channels)
        self.evidence = nn.Sequential(
            nn.Conv2d(detail_channels * 3, detail_channels, 1, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, detail_channels, 3, padding=1, groups=detail_channels, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, detail_channels, 1, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(detail_channels * 3, detail_channels, 1, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, detail_channels, 3, padding=1, groups=detail_channels, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, detail_channels, 1),
        )
        self.selection_scale = nn.Parameter(torch.tensor(2.0))
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, content: Tensor, detail: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        cbar = self.content_projection(content)
        nbar = self.detail_projection(detail)
        discrepancy = (self.detail_norm(nbar) - self.content_norm(cbar)).abs()
        local_detail = (nbar - _local_average(nbar, self.detail_kernel)).abs()
        interaction = discrepancy * local_detail
        x = torch.cat([discrepancy, local_detail, interaction], dim=1)
        evidence = self.evidence(x)
        evidence_map = evidence.abs().mean(dim=1, keepdim=True)
        center_surround = evidence_map - _local_average(evidence_map, self.context_kernel)
        spatial_gate = torch.sigmoid(self.selection_scale * center_surround)
        candidate = self.residual(x)
        local_candidate = candidate - _local_average(candidate, self.residual_kernel)
        update = spatial_gate * local_candidate
        preserved = detail + torch.tanh(self.alpha) * update
        return preserved, {
            "update": update.detach(),
            "spatial_gate": spatial_gate.detach(),
            "evidence_map": evidence_map.detach(),
            "center_surround": center_surround.detach(),
            "alpha": self.alpha.detach(),
            "selection_scale": self.selection_scale.detach(),
        }


class SelectiveFineGrainedTracePreservation(nn.Module):
    """FGTP-v2 at 1/4 and 1/8 feature scales."""

    def __init__(self) -> None:
        super().__init__()
        self.scale1 = SelectiveFineGrainedTracePreservationBlock(256, 64, detail_kernel=5, context_kernel=7)
        self.scale2 = SelectiveFineGrainedTracePreservationBlock(512, 128, detail_kernel=3, context_kernel=5)

    def forward(self, content: list[Tensor], detail: list[Tensor]) -> tuple[list[Tensor], dict[str, Tensor]]:
        t1, diag1 = self.scale1(content[0], detail[0])
        t2, diag2 = self.scale2(content[1], detail[1])
        updated = [t1, t2, detail[2], detail[3]]
        diagnostics = {
            "n1": detail[0].detach(),
            "n2": detail[1].detach(),
            "t1": t1.detach(),
            "t2": t2.detach(),
            "t1_minus_n1": (t1 - detail[0]).detach(),
            "t2_minus_n2": (t2 - detail[1]).detach(),
            "delta1": diag1["update"],
            "delta2": diag2["update"],
            "spatial_gate1": diag1["spatial_gate"],
            "spatial_gate2": diag2["spatial_gate"],
            "evidence_map1": diag1["evidence_map"],
            "evidence_map2": diag2["evidence_map"],
            "center_surround1": diag1["center_surround"],
            "center_surround2": diag2["center_surround"],
            "alpha1": diag1["alpha"],
            "alpha2": diag2["alpha"],
            "selection_scale1": diag1["selection_scale"],
            "selection_scale2": diag2["selection_scale"],
        }
        return updated, diagnostics
