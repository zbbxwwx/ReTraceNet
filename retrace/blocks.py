"""Building blocks used by ReTraceNet.

The implementation intentionally avoids third-party model libraries so that all
architectural claims are inspectable and ablations are straightforward.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class ConvNeXtBlock(nn.Module):
    """A compact local-context block well suited to high-resolution traces."""

    def __init__(self, channels: int, expansion: int = 4, drop_path: float = 0.0) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.contract = nn.Conv2d(hidden, channels, 1)
        self.scale = nn.Parameter(1e-6 * torch.ones(channels))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.expand(x)
        x = F.gelu(x)
        x = self.contract(x) * self.scale[:, None, None]
        return residual + self.drop_path(x)


class ConstrainedResidualBank(nn.Module):
    """Learnable zero-sum residual kernels.

    Every output kernel assigns a total weight of -1 to its RGB center samples
    and a softmax-normalized total weight of +1 to all non-center samples. The
    constraint is imposed on every forward pass and therefore cannot drift.
    """

    def __init__(self, out_channels: int = 8, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.raw = nn.Parameter(torch.empty(out_channels, 3, kernel_size, kernel_size))
        nn.init.normal_(self.raw, std=0.2)
        center = torch.zeros(1, 3, kernel_size, kernel_size, dtype=torch.bool)
        center[..., kernel_size // 2, kernel_size // 2] = True
        self.register_buffer("center_mask", center)

    def constrained_weight(self) -> Tensor:
        center = self.center_mask.expand_as(self.raw)
        logits = self.raw.masked_fill(center, -1e4).flatten(1)
        neighbors = F.softmax(logits, dim=1).view_as(self.raw)
        center_weight = -center.to(self.raw.dtype) / 3.0
        return neighbors + center_weight

    def forward(self, x: Tensor) -> Tensor:
        return F.conv2d(x, self.constrained_weight(), padding=self.kernel_size // 2)


class QualityDescriptor(nn.Module):
    """Differentiable no-reference degradation descriptor.

    The descriptor measures intensity statistics, first-order residual energy,
    and the excess gradient across nominal 8-pixel block boundaries. It does not
    assume that the input is JPEG, so PNG and rescaled images are valid inputs.
    """

    output_dim = 8

    @staticmethod
    def _safe_mean(x: Tensor, dims: Iterable[int]) -> Tensor:
        if x.numel() == 0:
            return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return x.mean(dim=tuple(dims))[:, None]

    def forward(self, x: Tensor) -> Tensor:
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        dx = (gray[..., 1:] - gray[..., :-1]).abs()
        dy = (gray[..., 1:, :] - gray[..., :-1, :]).abs()
        lap = F.avg_pool2d(gray, 3, stride=1, padding=1) - gray
        block_x = dx[..., 7::8]
        block_y = dy[..., 7::8, :]
        mean = gray.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        std = gray.std(dim=(1, 2, 3), keepdim=False, unbiased=False)[:, None]
        sat = ((x < 0.02) | (x > 0.98)).to(x.dtype).mean(dim=(1, 2, 3))[:, None]
        stats = [
            mean,
            std,
            sat,
            dx.mean(dim=(1, 2, 3))[:, None],
            dy.mean(dim=(1, 2, 3))[:, None],
            lap.abs().mean(dim=(1, 2, 3))[:, None],
            self._safe_mean(block_x, (1, 2, 3)),
            self._safe_mean(block_y, (1, 2, 3)),
        ]
        return torch.cat(stats, dim=1)


class QualityConditioner(nn.Module):
    """Generates FiLM parameters and forensic reliability for every scale."""

    def __init__(self, channels: Sequence[int], hidden: int = 64) -> None:
        super().__init__()
        self.channels = tuple(channels)
        total = 2 * sum(channels) + len(channels)
        self.mlp = nn.Sequential(
            nn.Linear(QualityDescriptor.output_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, total),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, descriptor: Tensor) -> list[tuple[Tensor, Tensor, Tensor]]:
        params = self.mlp(descriptor)
        feature_params = params[:, : 2 * sum(self.channels)]
        reliability_logits = params[:, 2 * sum(self.channels) :]
        output: list[tuple[Tensor, Tensor, Tensor]] = []
        offset = 0
        for index, channels in enumerate(self.channels):
            gamma, beta = feature_params[:, offset : offset + 2 * channels].chunk(2, dim=1)
            offset += 2 * channels
            reliability = torch.sigmoid(reliability_logits[:, index : index + 1])
            output.append((0.5 * torch.tanh(gamma), 0.5 * torch.tanh(beta), reliability))
        return output


class ContentOrthogonalization(nn.Module):
    """Removes the local trace component aligned with RGB content."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.trace_align = nn.Conv2d(channels, channels, 1, bias=False)
        self.content_align = nn.Conv2d(channels, channels, 1, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )
        self.projection_strength = nn.Parameter(torch.tensor(0.0))

    def forward(self, trace: Tensor, content: Tensor, enabled: bool = True) -> tuple[Tensor, Tensor]:
        t = self.trace_align(trace)
        c = self.content_align(content)
        coefficient = (t * c).sum(1, keepdim=True) / (c.square().sum(1, keepdim=True) + 1e-6)
        strength = torch.sigmoid(self.projection_strength) if enabled else 0.0
        orthogonal = t - strength * coefficient * c
        return self.mix(torch.cat([orthogonal, trace], dim=1)), c


class RecyclingCell(nn.Module):
    """A gated, scale-shared evidence update cell."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        merged = 3 * channels
        self.gate = nn.Conv2d(merged, channels, 1)
        self.proposal = nn.Sequential(
            nn.Conv2d(merged, channels, 3, padding=1, groups=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            ConvNeXtBlock(channels),
        )

    def forward(self, state: Tensor, neighbor: Tensor, anchor: Tensor) -> Tensor:
        neighbor = F.interpolate(neighbor, size=state.shape[-2:], mode="bilinear", align_corners=False)
        merged = torch.cat([state, neighbor, anchor], dim=1)
        return state + torch.sigmoid(self.gate(merged)) * self.proposal(merged)


class EvidenceRecycler(nn.Module):
    """Bidirectionally recycles immutable evidence across a feature pyramid."""

    def __init__(self, channels: int, scales: int = 4, steps: int = 2) -> None:
        super().__init__()
        self.steps = steps
        self.scales = scales
        self.top_down = RecyclingCell(channels)
        self.bottom_up = RecyclingCell(channels)
        self.iteration_scale = nn.Parameter(torch.ones(steps))

    def forward(self, anchors: Sequence[Tensor]) -> list[Tensor]:
        if len(anchors) != self.scales:
            raise ValueError(f"expected {self.scales} scales, got {len(anchors)}")
        states = [x for x in anchors]
        for step in range(self.steps):
            previous = [x for x in states]
            for index in range(self.scales - 2, -1, -1):
                states[index] = self.top_down(states[index], states[index + 1], anchors[index])
            for index in range(1, self.scales):
                states[index] = self.bottom_up(states[index], states[index - 1], anchors[index])
            scale = self.iteration_scale[step].tanh()
            states = [old + scale * (new - old) for old, new in zip(previous, states)]
        return states


class IntraImageCounterfactualIntervention(nn.Module):
    """Erases candidate forensic traces while preserving the content pathway.

    The replacement trace is estimated exclusively from low-suspicion pixels in
    the same image. A local estimator preserves spatially varying acquisition
    statistics, while a global estimator prevents an empty neighborhood from
    producing an unstable counterfactual.
    """

    def __init__(
        self,
        channels: int,
        neighborhood: int = 9,
        gate_mode: str = "probability",
        gate_threshold: float = 0.2,
        gate_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if neighborhood % 2 != 1:
            raise ValueError("neighborhood must be odd")
        if gate_mode not in {"probability", "centered"}:
            raise ValueError("gate_mode must be one of: probability, centered")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        self.neighborhood = neighborhood
        self.gate_mode = gate_mode
        self.gate_threshold = float(gate_threshold)
        self.gate_temperature = float(gate_temperature)
        self.proposal_head = nn.Sequential(
            ConvNeXtBlock(channels),
            nn.Conv2d(channels, 1, 1),
        )
        self.replacement = nn.Sequential(
            nn.Conv2d(3 * channels + 1, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            ConvNeXtBlock(channels),
        )
        self.context_gate = nn.Sequential(
            nn.Conv2d(3 * channels + 1, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        self.strength = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1, groups=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )

    def propose(self, finest_anchor: Tensor) -> Tensor:
        return self.proposal_head(finest_anchor)

    def forward(
        self,
        anchor: Tensor,
        proposal_logits: Tensor,
        intervention_mask: Tensor | None = None,
        teacher_mix: float = 0.5,
        strength_scale: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        proposal = F.interpolate(
            proposal_logits, size=anchor.shape[-2:], mode="bilinear", align_corners=False
        ).sigmoid()
        if intervention_mask is not None:
            teacher = F.interpolate(
                intervention_mask, size=anchor.shape[-2:], mode="nearest"
            ).to(anchor.dtype)
            proposal = (1.0 - teacher_mix) * proposal + teacher_mix * teacher

        # The proposal is detached only for prototype estimation. This blocks the
        # trivial solution in which the proposer changes solely to ease filling,
        # while gradients still reach it through the intervention strength.
        background = (1.0 - proposal.detach()).clamp_min(1e-3)
        padding = self.neighborhood // 2
        numerator = F.avg_pool2d(
            anchor * background, self.neighborhood, stride=1, padding=padding
        )
        denominator = F.avg_pool2d(
            background, self.neighborhood, stride=1, padding=padding
        )
        local_pristine = numerator / denominator.clamp_min(1e-4)
        global_pristine = (anchor * background).sum((2, 3), keepdim=True) / (
            background.sum((2, 3), keepdim=True) + 1e-4
        )
        global_pristine = global_pristine.expand_as(anchor)
        replacement_input = torch.cat(
            [anchor, local_pristine, global_pristine, proposal], dim=1
        )
        context_gate = torch.sigmoid(self.context_gate(replacement_input))
        pristine_context = (
            context_gate * local_pristine + (1.0 - context_gate) * global_pristine
        )
        # A bounded adapter corrects estimator bias but cannot synthesize an
        # unconstrained replacement unrelated to the pristine context.
        replacement = pristine_context + 0.1 * torch.tanh(
            self.replacement(replacement_input)
        )
        if self.gate_mode == "centered":
            gate = torch.sigmoid((proposal - self.gate_threshold) / self.gate_temperature)
        else:
            gate = proposal
        strength = gate * torch.sigmoid(
            self.strength(torch.cat([anchor - replacement, proposal], dim=1))
        )
        strength = strength * float(strength_scale)
        counterfactual = anchor + strength * (replacement - anchor)
        intervention_energy = (anchor - counterfactual).abs().mean(1, keepdim=True)
        return counterfactual, intervention_energy, strength


class DualHypothesisEvidenceLedger(nn.Module):
    """Accumulates manipulation support and pristine counter-evidence.

    Unlike feature fusion, the ledger keeps both hypotheses explicit. Agreement
    raises confidence; simultaneous support and refutation define conflict and
    become a mechanistic uncertainty cue for the boundary head.
    """

    def __init__(self, channels: int, scales: int = 4) -> None:
        super().__init__()
        self.scales = scales
        self.local_evidence = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            ConvNeXtBlock(channels),
            nn.Conv2d(channels, 2, 1),
        )
        self.top_down_discount = nn.Parameter(torch.zeros(scales - 1))
        self.bottom_up_discount = nn.Parameter(torch.zeros(scales - 1))
        self.write_back = nn.Sequential(
            nn.Conv2d(5, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        )

    def forward(
        self,
        factual: Sequence[Tensor],
        counterfactual: Sequence[Tensor],
        reliability: Sequence[Tensor],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        if not (len(factual) == len(counterfactual) == len(reliability) == self.scales):
            raise ValueError("ledger inputs must have one item per scale")
        local: list[Tensor] = []
        effects: list[Tensor] = []
        for factual_i, counterfactual_i, reliability_i in zip(
            factual, counterfactual, reliability
        ):
            difference = factual_i - counterfactual_i
            raw = self.local_evidence(
                torch.cat([factual_i, counterfactual_i, difference], dim=1)
            )
            evidence = F.softplus(raw) + 1e-4
            if reliability_i.ndim == 2:
                rho = reliability_i[:, :, None, None]
            elif reliability_i.ndim == 4:
                rho = reliability_i
                if rho.shape[-2:] != evidence.shape[-2:]:
                    rho = F.interpolate(rho, size=evidence.shape[-2:], mode="bilinear", align_corners=False)
            else:
                raise ValueError(f"unsupported reliability shape: {tuple(reliability_i.shape)}")
            evidence = evidence * rho
            local.append(evidence)
            effects.append(difference.abs().mean(1, keepdim=True))

        top_down = [item for item in local]
        for index in range(self.scales - 2, -1, -1):
            neighbor = F.interpolate(
                top_down[index + 1], size=top_down[index].shape[-2:], mode="bilinear", align_corners=False
            )
            top_down[index] = top_down[index] + torch.sigmoid(
                self.top_down_discount[index]
            ) * neighbor

        ledger = [item for item in top_down]
        for index in range(1, self.scales):
            neighbor = F.interpolate(
                ledger[index - 1], size=ledger[index].shape[-2:], mode="bilinear", align_corners=False
            )
            ledger[index] = ledger[index] + torch.sigmoid(
                self.bottom_up_discount[index - 1]
            ) * neighbor

        enriched: list[Tensor] = []
        log_odds: list[Tensor] = []
        conflicts: list[Tensor] = []
        for factual_i, ledger_i, effect_i in zip(factual, ledger, effects):
            support, refutation = ledger_i.chunk(2, dim=1)
            total = support + refutation + 1e-6
            normalized_support = support / total
            normalized_refutation = refutation / total
            evidence_mass = 1.0 - torch.exp(-total)
            conflict = (
                2.0 * torch.minimum(support, refutation) / total
            ) * evidence_mass
            normalized_effect = effect_i / (
                effect_i.mean((2, 3), keepdim=True).detach() + 1e-4
            )
            normalized_mass = torch.log1p(total) / (
                torch.log1p(total).mean((2, 3), keepdim=True).detach() + 1e-4
            )
            guidance = torch.cat(
                [
                    normalized_support,
                    normalized_refutation,
                    conflict,
                    normalized_mass,
                    normalized_effect,
                ],
                dim=1,
            )
            enriched.append(factual_i + self.write_back(guidance))
            log_odds.append(torch.log(support) - torch.log(refutation))
            conflicts.append(conflict.clamp(0.0, 1.0))
        return enriched, log_odds, conflicts, effects


class UncertaintyBoundaryRefiner(nn.Module):
    """Refines boundaries with ledger conflict and intervention effect."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.coarse_head = nn.Conv2d(channels, 1, 1)
        self.variance_head = nn.Conv2d(channels, 1, 1)
        self.conflict_weight = nn.Parameter(torch.tensor(0.0))
        self.refine = nn.Sequential(
            nn.Conv2d(channels + 5, channels, 3, padding=1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
            ConvNeXtBlock(channels),
            nn.Conv2d(channels, 2, 1),
        )
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(
        self,
        feature: Tensor,
        conflict: Tensor | None = None,
        intervention_effect: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        coarse = self.coarse_head(feature)
        probability = coarse.sigmoid()
        gx = F.conv2d(probability, self.sobel_x, padding=1)
        gy = F.conv2d(probability, self.sobel_y, padding=1)
        boundary = torch.sqrt(gx.square() + gy.square() + 1e-6)
        learned_log_var = self.variance_head(feature).clamp(-5.0, 5.0)
        learned_uncertainty = torch.sigmoid(learned_log_var)
        if conflict is None:
            conflict = torch.zeros_like(learned_uncertainty)
        if intervention_effect is None:
            intervention_effect = torch.zeros_like(learned_uncertainty)
        conflict = F.interpolate(conflict, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        intervention_effect = F.interpolate(
            intervention_effect, size=feature.shape[-2:], mode="bilinear", align_corners=False
        )
        conflict_mix = torch.sigmoid(self.conflict_weight)
        uncertainty = (
            (1.0 - conflict_mix) * learned_uncertainty + conflict_mix * conflict.clamp(0.0, 1.0)
        ).clamp(1e-4, 1.0 - 1e-4)
        initial_log_var = torch.logit(uncertainty)
        correction, delta_var = self.refine(
            torch.cat(
                [feature, probability, boundary, uncertainty, conflict, intervention_effect], dim=1
            )
        ).chunk(2, dim=1)
        refined = coarse + correction * (0.5 + uncertainty)
        log_var = (initial_log_var + delta_var).clamp(-5.0, 5.0)
        return refined, log_var, coarse


def make_stage(channels: int, depth: int, drop_paths: Sequence[float]) -> nn.Sequential:
    return nn.Sequential(*[ConvNeXtBlock(channels, drop_path=p) for p in drop_paths[:depth]])


def linear_drop_path_schedule(depths: Sequence[int], maximum: float) -> list[list[float]]:
    total = sum(depths)
    values = torch.linspace(0.0, maximum, total).tolist() if total > 1 else [maximum]
    schedule: list[list[float]] = []
    offset = 0
    for depth in depths:
        schedule.append(values[offset : offset + depth])
        offset += depth
    return schedule
