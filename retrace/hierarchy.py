"""Optional hierarchical evidence modules for ReTraceNet.

All modules in this file are residual adapters with zero-initialized scalar
gates. Enabling a module should therefore be a structural ablation switch, not
an accidental change to the Stage3 Feature Fusion baseline at step 0.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .blocks import ConstrainedResidualBank, ConvNeXtBlock, LayerNorm2d


def reliability_to_map(reliability: Tensor, size: tuple[int, int]) -> Tensor:
    """Convert scalar or spatial reliability to a single-channel map."""

    if reliability.ndim == 2:
        return reliability[:, :, None, None].expand(-1, -1, size[0], size[1])
    if reliability.ndim == 4:
        if reliability.shape[-2:] != size:
            return F.interpolate(reliability, size=size, mode="bilinear", align_corners=False)
        return reliability
    raise ValueError(f"unsupported reliability shape: {tuple(reliability.shape)}")


class SpatialReliabilityPerception(nn.Module):
    """HDRP-v2: spatial reliability maps with exact scalar fallback at alpha=0."""

    def __init__(self, channels: Sequence[int], hidden_channels: int = 64) -> None:
        super().__init__()
        self.channels = tuple(int(width) for width in channels)
        self.alpha = nn.Parameter(torch.zeros(len(self.channels)))
        self.predictors = nn.ModuleList()
        for width in self.channels:
            hidden = min(max(16, int(hidden_channels)), int(width))
            self.predictors.append(
                nn.Sequential(
                    nn.Conv2d(3 * int(width) + 1, hidden, 1, bias=False),
                    LayerNorm2d(hidden),
                    nn.GELU(),
                    nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
                    LayerNorm2d(hidden),
                    nn.GELU(),
                    nn.Conv2d(hidden, 1, 1),
                )
            )

    def forward(
        self,
        trace_features: Sequence[Tensor],
        content_features: Sequence[Tensor],
        base_reliability: Sequence[Tensor],
    ) -> tuple[list[Tensor], list[Tensor]]:
        if not (len(trace_features) == len(content_features) == len(base_reliability) == len(self.channels)):
            raise ValueError("spatial reliability inputs must have one item per scale")
        reliability: list[Tensor] = []
        residuals: list[Tensor] = []
        for index, (trace, content, base) in enumerate(zip(trace_features, content_features, base_reliability)):
            base_map = reliability_to_map(base, trace.shape[-2:])
            local = torch.sigmoid(
                self.predictors[index](torch.cat([trace, content, (trace - content).abs(), base_map], dim=1))
            )
            residual = local - base_map
            reliability.append((base_map + self.alpha[index].tanh() * residual).clamp(0.0, 1.0))
            residuals.append(residual)
        return reliability, residuals


class LightweightResidualStream(nn.Module):
    """Lightweight forensic residual pyramid, avoiding a second ResNet branch."""

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128, 256),
        encoder_type: str = "lightweight_srm",
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("residual stream requires four channel widths")
        self.channels = tuple(int(width) for width in channels)
        self.encoder_type = str(encoder_type)
        if self.encoder_type not in {"lightweight_srm", "constrained"}:
            raise ValueError("residual_encoder_type must be lightweight_srm or constrained")
        self.residual_bank = ConstrainedResidualBank(out_channels=8, kernel_size=5)
        self.stem = nn.Sequential(
            nn.Conv2d(8, self.channels[0], 5, stride=4, padding=2, bias=False),
            LayerNorm2d(self.channels[0]),
            nn.GELU(),
            ConvNeXtBlock(self.channels[0]),
        )
        self.down = nn.ModuleList()
        self.stages = nn.ModuleList([nn.Identity()])
        for index in range(1, 4):
            self.down.append(
                nn.Sequential(
                    LayerNorm2d(self.channels[index - 1]),
                    nn.Conv2d(self.channels[index - 1], self.channels[index], 3, stride=2, padding=1, bias=False),
                    LayerNorm2d(self.channels[index]),
                    nn.GELU(),
                )
            )
            self.stages.append(ConvNeXtBlock(self.channels[index]))

    def forward(self, image: Tensor) -> list[Tensor]:
        residual = self.residual_bank(image)
        state = self.stem(residual)
        outputs = [state]
        for down, stage in zip(self.down, self.stages[1:]):
            state = stage(down(state))
            outputs.append(state)
        return outputs


class HierarchicalReliabilityPropagation(nn.Module):
    """Fine-to-coarse residual reliability propagation for HDRP/H-DRPM."""

    def __init__(self, scales: int = 4) -> None:
        super().__init__()
        self.scales = int(scales)
        self.alpha = nn.Parameter(torch.zeros(self.scales))
        self.transforms = nn.ModuleList([nn.Linear(1, 1) for _ in range(self.scales)])
        self.spatial_transforms = nn.ModuleList([nn.Conv2d(1, 1, 1) for _ in range(self.scales)])

    def forward(self, reliability: Sequence[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        if len(reliability) != self.scales:
            raise ValueError(f"expected {self.scales} reliability tensors")
        updated = [item for item in reliability]
        propagated = [item.new_zeros(item.shape) for item in reliability]
        for index in range(1, self.scales):
            previous = updated[index - 1]
            current = updated[index]
            if previous.ndim == 4 or current.ndim == 4:
                previous_map = reliability_to_map(previous, current.shape[-2:] if current.ndim == 4 else previous.shape[-2:])
                if current.ndim == 2:
                    current = reliability_to_map(current, previous_map.shape[-2:])
                    updated[index] = current
                    propagated[index] = current.new_zeros(current.shape)
                message = torch.tanh(self.spatial_transforms[index](previous_map))
            else:
                message = torch.tanh(self.transforms[index](previous))
            propagated[index] = message
            updated[index] = (updated[index] + self.alpha[index].tanh() * message).clamp(0.0, 1.0)
        return updated, propagated


class CoarseToFineICVMPrior(nn.Module):
    """Coarse-to-fine prior propagation for ICVM proposal maps."""

    def __init__(self, scales: int = 4) -> None:
        super().__init__()
        self.scales = int(scales)
        self.alpha = nn.Parameter(torch.zeros(self.scales))
        self.transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 1, 3, padding=1, bias=True),
                )
                for _ in range(self.scales)
            ]
        )

    def forward(self, local_priors: Sequence[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        if len(local_priors) != self.scales:
            raise ValueError(f"expected {self.scales} ICVM priors")
        guided = [item for item in local_priors]
        messages = [item.new_zeros(item.shape) for item in local_priors]
        for index in range(self.scales - 2, -1, -1):
            coarse = F.interpolate(guided[index + 1], size=guided[index].shape[-2:], mode="bilinear", align_corners=False)
            message = self.transforms[index](coarse)
            messages[index] = message
            guided[index] = guided[index] + self.alpha[index].tanh() * message
        return guided, messages


class ConflictConditionedEvidenceReasoning(nn.Module):
    """CDERF: conflict-conditioned cross-scale evidence reasoning."""

    def __init__(self, channels: int, scales: int = 4) -> None:
        super().__init__()
        self.scales = int(scales)
        self.alpha = nn.Parameter(torch.zeros(self.scales))
        self.transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 1, bias=False),
                    LayerNorm2d(channels),
                    nn.GELU(),
                    ConvNeXtBlock(channels),
                )
                for _ in range(self.scales)
            ]
        )
        self.conflict_gate = nn.ModuleList([nn.Conv2d(1, 1, 1) for _ in range(self.scales)])
        for gate in self.conflict_gate:
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(
        self,
        evidence: Sequence[Tensor],
        ledger_logits: Sequence[Tensor],
        ledger_conflicts: Sequence[Tensor],
        reliability: Sequence[Tensor],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        if not (len(evidence) == len(ledger_logits) == len(ledger_conflicts) == len(reliability) == self.scales):
            raise ValueError("CDERF inputs must have one item per scale")
        support: list[Tensor] = []
        refutation: list[Tensor] = []
        conflict: list[Tensor] = []
        for logits, conflict_i, reliability_i in zip(ledger_logits, ledger_conflicts, reliability):
            rho = reliability_to_map(reliability_i, logits.shape[-2:])
            support_i = logits.sigmoid() * rho
            refutation_i = (-logits).sigmoid() * (1.0 - rho)
            support.append(support_i)
            refutation.append(refutation_i)
            conflict.append(conflict_i.clamp(0.0, 1.0))

        updated = [item for item in evidence]
        for index, evidence_i in enumerate(evidence):
            neighbor = torch.zeros_like(evidence_i)
            if index > 0:
                neighbor = neighbor + F.interpolate(evidence[index - 1], size=evidence_i.shape[-2:], mode="bilinear", align_corners=False)
            if index + 1 < self.scales:
                neighbor = neighbor + F.interpolate(evidence[index + 1], size=evidence_i.shape[-2:], mode="bilinear", align_corners=False)
            gate = torch.sigmoid(self.conflict_gate[index](conflict[index]))
            updated[index] = evidence_i + self.alpha[index].tanh() * gate * self.transforms[index](neighbor)
        return updated, support, refutation, conflict


class MultiScaleEvidenceFeedback(nn.Module):
    """Feature-level evidence feedback with optional reliability/conflict gate."""

    def __init__(self, channels: int, scales: int = 4) -> None:
        super().__init__()
        self.scales = int(scales)
        self.beta = nn.Parameter(torch.zeros(self.scales))
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(2 * channels + 2, channels, 1, bias=False),
                    LayerNorm2d(channels),
                    nn.GELU(),
                    ConvNeXtBlock(channels),
                    nn.Conv2d(channels, channels, 1),
                )
                for _ in range(self.scales)
            ]
        )
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(2, 1, 1),
                    nn.Sigmoid(),
                )
                for _ in range(self.scales)
            ]
        )

    def forward(
        self,
        decoder_features: Sequence[Tensor],
        evidence: Sequence[Tensor],
        reliability: Sequence[Tensor],
        conflict: Sequence[Tensor],
        use_reliability_conflict_gate: bool = False,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        if not (len(decoder_features) == len(evidence) == len(reliability) == len(conflict) == self.scales):
            raise ValueError("feedback inputs must have one item per scale")
        updated: list[Tensor] = []
        deltas: list[Tensor] = []
        gates: list[Tensor] = []
        for index, (decoder_i, evidence_i, reliability_i, conflict_i) in enumerate(
            zip(decoder_features, evidence, reliability, conflict)
        ):
            if evidence_i.shape[-2:] != decoder_i.shape[-2:]:
                evidence_i = F.interpolate(evidence_i, size=decoder_i.shape[-2:], mode="bilinear", align_corners=False)
            if conflict_i.shape[-2:] != decoder_i.shape[-2:]:
                conflict_i = F.interpolate(conflict_i, size=decoder_i.shape[-2:], mode="bilinear", align_corners=False)
            rho = reliability_to_map(reliability_i, decoder_i.shape[-2:])
            gate_input = torch.cat([rho, conflict_i], dim=1)
            gate = self.gates[index](gate_input) if use_reliability_conflict_gate else torch.ones_like(conflict_i)
            delta = self.adapters[index](torch.cat([decoder_i, evidence_i, rho, conflict_i], dim=1))
            updated.append(decoder_i + self.beta[index].tanh() * gate * delta)
            deltas.append(delta)
            gates.append(gate)
        return updated, deltas, gates


class ConflictGuidedBoundaryRefinement(nn.Module):
    """CBR: residual correction for high-conflict boundary regions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(()))
        self.delta_head = nn.Sequential(
            nn.Conv2d(channels + 2, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            ConvNeXtBlock(channels),
            nn.Conv2d(channels, 1, 1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.Sigmoid(),
        )
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def boundary_cue(self, logits: Tensor) -> Tensor:
        probability = logits.sigmoid()
        gx = F.conv2d(probability, self.sobel_x, padding=1)
        gy = F.conv2d(probability, self.sobel_y, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6).clamp(0.0, 1.0)

    def forward(
        self,
        decoder_feature: Tensor,
        logits: Tensor,
        conflict: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if conflict.shape[-2:] != logits.shape[-2:]:
            conflict = F.interpolate(conflict, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        boundary = self.boundary_cue(logits)
        delta = self.delta_head(torch.cat([decoder_feature, conflict, boundary], dim=1))
        gate = self.gate_head(torch.cat([conflict.clamp(0.0, 1.0), boundary], dim=1))
        refined = logits + self.alpha.tanh() * gate * delta
        return refined, delta, gate, boundary
