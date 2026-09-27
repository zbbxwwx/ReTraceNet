"""Hierarchical Counterfactual Verification for ReTraceNet."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from retrace.blocks import LayerNorm2d


def stat_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    mean = x.mean(dim=1, keepdim=True)
    var = (x - mean).square().mean(dim=1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps)


class LocalProposal(nn.Module):
    def __init__(self, content_channels: int, detail_channels: int) -> None:
        super().__init__()
        self.content_align = nn.Sequential(
            nn.Conv2d(content_channels, detail_channels, 1, bias=False),
            LayerNorm2d(detail_channels),
        )
        self.proposal = nn.Sequential(
            nn.Conv2d(detail_channels * 3, detail_channels, 3, padding=1, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, detail_channels, 3, padding=1, groups=detail_channels, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, 1, 1),
        )

    def align_content(self, content: Tensor, detail: Tensor) -> Tensor:
        aligned = self.content_align(content)
        if aligned.shape[-2:] != detail.shape[-2:]:
            aligned = F.interpolate(aligned, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        return aligned

    def features(self, content: Tensor, detail: Tensor) -> tuple[Tensor, Tensor]:
        aligned = self.align_content(content, detail)
        diff = (stat_norm(detail) - stat_norm(aligned)).abs()
        return torch.cat([detail, aligned, diff], dim=1), aligned

    def forward(self, content: Tensor, detail: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, aligned = self.features(content, detail)
        return torch.sigmoid(self.proposal(x)), aligned, x


class ReferenceEstimator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(16, channels // 2)
        self.mix_gate = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, 1, bias=False),
            LayerNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.intervention_gate = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, 1, bias=False),
            LayerNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, detail: Tensor, proposal: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        background = (1.0 - proposal).detach()
        denom = F.avg_pool2d(background, 5, stride=1, padding=2, count_include_pad=False).clamp_min(1e-4)
        local_ref = F.avg_pool2d(detail * background, 5, stride=1, padding=2, count_include_pad=False) / denom
        global_denom = background.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        global_ref = (detail * background).sum(dim=(2, 3), keepdim=True) / global_denom
        global_ref = global_ref.expand_as(detail)
        gate_input = torch.cat([detail, proposal], dim=1)
        mix = torch.sigmoid(self.mix_gate(gate_input))
        strength = torch.sigmoid(self.intervention_gate(gate_input))
        reference = mix * local_ref + (1.0 - mix) * global_ref
        counterfactual = detail + proposal * strength * (reference - detail)
        energy = (proposal * strength * (reference - detail)).abs().mean(dim=1, keepdim=True)
        return counterfactual, reference, energy


class SharedVerifier(nn.Module):
    def __init__(self, content_channels: int, detail_channels: int) -> None:
        super().__init__()
        self.content_align = nn.Sequential(
            nn.Conv2d(content_channels, detail_channels, 1, bias=False),
            LayerNorm2d(detail_channels),
        )
        self.net = nn.Sequential(
            nn.Conv2d(detail_channels * 3, detail_channels, 3, padding=1, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, detail_channels, 3, padding=1, groups=detail_channels, bias=False),
            LayerNorm2d(detail_channels),
            nn.GELU(),
            nn.Conv2d(detail_channels, 1, 1),
        )

    def aligned_content(self, content: Tensor, detail: Tensor) -> Tensor:
        aligned = self.content_align(content)
        if aligned.shape[-2:] != detail.shape[-2:]:
            aligned = F.interpolate(aligned, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        return aligned

    def forward(self, content: Tensor, detail: Tensor) -> Tensor:
        aligned = self.aligned_content(content, detail)
        x = torch.cat([detail, aligned, (stat_norm(detail) - stat_norm(aligned)).abs()], dim=1)
        return self.net(x)


class EvidenceAdapter(nn.Module):
    def __init__(self, detail_channels: int, decoder_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(detail_channels + 3, decoder_channels, 1, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1, groups=decoder_channels, bias=False),
            LayerNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, 1),
        )

    def forward(self, detail: Tensor, proposal: Tensor, delta: Tensor) -> Tensor:
        return self.net(torch.cat([detail, proposal, delta, delta.abs()], dim=1))


@dataclass
class HCVScaleOutput:
    proposal: Tensor
    counterfactual: Tensor
    reference: Tensor
    v_factual: Tensor
    v_counterfactual: Tensor
    delta: Tensor
    energy: Tensor
    evidence: Tensor


class HCVScale(nn.Module):
    def __init__(self, content_channels: int, detail_channels: int, decoder_channels: int) -> None:
        super().__init__()
        self.local_proposal = LocalProposal(content_channels, detail_channels)
        self.reference = ReferenceEstimator(detail_channels)
        self.verifier = SharedVerifier(content_channels, detail_channels)
        self.adapter = EvidenceAdapter(detail_channels, decoder_channels)

    def forward(self, content: Tensor, detail: Tensor, proposal: Tensor) -> HCVScaleOutput:
        counterfactual, reference, energy = self.reference(detail, proposal)
        v_factual = self.verifier(content, detail)
        v_counterfactual = self.verifier(content, counterfactual)
        delta = v_factual - v_counterfactual
        evidence = self.adapter(detail, proposal, delta)
        return HCVScaleOutput(proposal, counterfactual, reference, v_factual, v_counterfactual, delta, energy, evidence)


class HierarchicalCounterfactualVerification(nn.Module):
    """Verify suspicious traces with same-image counterfactual intervention."""

    def __init__(
        self,
        content_channels: tuple[int, int, int] = (256, 512, 1024),
        detail_channels: tuple[int, int, int] = (64, 128, 256),
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        self.scales = nn.ModuleList(
            HCVScale(c, n, decoder_channels) for c, n in zip(content_channels, detail_channels)
        )
        self.lambda_1 = nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.zeros(3))

    def _hierarchical_proposals(self, local: list[Tensor]) -> list[Tensor]:
        m3 = local[2]
        prior2 = F.interpolate(m3, size=local[1].shape[-2:], mode="bilinear", align_corners=False)
        m2 = 1.0 - (1.0 - local[1]) * (1.0 - torch.sigmoid(self.lambda_2) * prior2)
        prior1 = F.interpolate(m2, size=local[0].shape[-2:], mode="bilinear", align_corners=False)
        m1 = 1.0 - (1.0 - local[0]) * (1.0 - torch.sigmoid(self.lambda_1) * prior1)
        return [m1.clamp(0.0, 1.0), m2.clamp(0.0, 1.0), m3.clamp(0.0, 1.0)]

    def forward(self, content: list[Tensor], detail: list[Tensor], decoded: list[Tensor]) -> tuple[list[Tensor], dict[str, list[Tensor] | Tensor]]:
        local = [scale.local_proposal(content[i], detail[i])[0] for i, scale in enumerate(self.scales)]
        proposals = self._hierarchical_proposals(local)
        scale_outputs = [self.scales[i](content[i], detail[i], proposals[i]) for i in range(3)]
        updated = list(decoded)
        written = []
        for i, scale_output in enumerate(scale_outputs):
            evidence = scale_output.evidence
            if evidence.shape[-2:] != updated[i].shape[-2:]:
                evidence = F.interpolate(evidence, size=updated[i].shape[-2:], mode="bilinear", align_corners=False)
            delta_feature = torch.tanh(self.alpha[i]) * evidence
            updated[i] = updated[i] + delta_feature
            written.append(delta_feature)
        diagnostics: dict[str, list[Tensor] | Tensor] = {
            "local_proposals": [x.detach() for x in local],
            "proposals": [x.proposal.detach() for x in scale_outputs],
            "counterfactuals": [x.counterfactual.detach() for x in scale_outputs],
            "references": [x.reference.detach() for x in scale_outputs],
            "v_factual": [x.v_factual.detach() for x in scale_outputs],
            "v_counterfactual": [x.v_counterfactual.detach() for x in scale_outputs],
            "deltas": [x.delta.detach() for x in scale_outputs],
            "energies": [x.energy.detach() for x in scale_outputs],
            "evidence_features": [x.evidence.detach() for x in scale_outputs],
            "written_features": [x.detach() for x in written],
            "alpha": self.alpha.detach(),
            "lambda": torch.stack([self.lambda_1.detach(), self.lambda_2.detach()]),
        }
        return updated, diagnostics
