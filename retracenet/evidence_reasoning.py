"""Dual-hypothesis evidence reasoning skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .counterfactual_verification import CounterfactualEvidence


@dataclass
class ReasonedEvidence:
    support: list[Tensor]
    refutation: list[Tensor]
    conflict: list[Tensor]
    evidence_features: list[Tensor]


class DualHypothesisEvidenceReasoning(nn.Module):
    """Maintain support, refutation, and conflict evidence at each scale."""

    def forward(self, traces: list[Tensor], verification: CounterfactualEvidence) -> ReasonedEvidence:
        raise NotImplementedError
