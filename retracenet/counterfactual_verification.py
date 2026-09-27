"""Hierarchical Counterfactual Verification skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class CounterfactualEvidence:
    factual: list[Tensor]
    counterfactual: list[Tensor]
    effect: list[Tensor]
    local_proposal: list[Tensor]
    hierarchical_prior: list[Tensor]
    verified_feature: list[Tensor]


class HierarchicalCounterfactualVerification(nn.Module):
    """Verify whether preserved traces causally support manipulation evidence."""

    def forward(self, traces: list[Tensor], content: list[Tensor]) -> CounterfactualEvidence:
        raise NotImplementedError
