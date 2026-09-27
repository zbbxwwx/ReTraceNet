"""Conflict-guided feature-space re-examination skeleton."""

from __future__ import annotations

from torch import Tensor, nn

from .counterfactual_verification import CounterfactualEvidence
from .evidence_reasoning import ReasonedEvidence


class ConflictGuidedReexamination(nn.Module):
    """Revisit unresolved regions in feature space, not as direct logit correction."""

    def forward(
        self,
        decoder_features: list[Tensor],
        traces: list[Tensor],
        verification: CounterfactualEvidence,
        evidence: ReasonedEvidence,
    ) -> list[Tensor]:
        raise NotImplementedError
