"""Composite objective for ReTraceNet."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def soft_dice_loss(logits: Tensor, target: Tensor, eps: float = 1.0) -> Tensor:
    probability = logits.sigmoid().flatten(1)
    target = target.flatten(1)
    intersection = (probability * target).sum(1)
    denominator = probability.sum(1) + target.sum(1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def morphological_boundary(x: Tensor, kernel_size: int) -> Tensor:
    padding = kernel_size // 2
    dilated = F.max_pool2d(x, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-x, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


def safe_probability(x: Tensor, eps: float = 1e-5) -> Tensor:
    return torch.nan_to_num(x.float(), nan=0.5, posinf=1.0, neginf=0.0).clamp(
        eps, 1.0 - eps
    )


def boundary_loss(logits: Tensor, target: Tensor, kernel_size: int) -> Tensor:
    # BCE on probabilities is unsafe under CUDA autocast; compute this small
    # boundary term in float32 while leaving the rest of training AMP-enabled.
    if logits.is_cuda:
        with torch.amp.autocast(device_type="cuda", enabled=False):
            predicted_boundary = morphological_boundary(logits.float().sigmoid(), kernel_size)
            target_boundary = morphological_boundary(target.float(), kernel_size)
            return F.binary_cross_entropy(
                safe_probability(predicted_boundary),
                target_boundary.float().clamp(0.0, 1.0),
            )
    predicted_boundary = morphological_boundary(logits.sigmoid(), kernel_size)
    target_boundary = morphological_boundary(target, kernel_size)
    return F.binary_cross_entropy(
        safe_probability(predicted_boundary), target_boundary.float().clamp(0.0, 1.0)
    )


def area_balanced_bce_with_logits(
    logits: Tensor,
    target: Tensor,
    clip: tuple[float, float],
) -> Tensor:
    positive = target.sum(dim=(2, 3), keepdim=True)
    negative = target.shape[-2] * target.shape[-1] - positive
    positive_weight = torch.sqrt((negative + 1.0) / (positive + 1.0)).clamp(*clip)
    spatial_weight = torch.where(target > 0.5, positive_weight, torch.ones_like(target))
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (spatial_weight * bce).mean()


def area_balanced_heteroscedastic_bce(
    logits: Tensor,
    log_var: Tensor,
    target: Tensor,
    clip: tuple[float, float],
) -> Tensor:
    """Area-balanced BCE attenuated by learned aleatoric uncertainty."""

    positive = target.sum(dim=(2, 3), keepdim=True)
    negative = target.shape[-2] * target.shape[-1] - positive
    positive_weight = torch.sqrt((negative + 1.0) / (positive + 1.0)).clamp(*clip)
    spatial_weight = torch.where(target > 0.5, positive_weight, torch.ones_like(target))
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    log_var = log_var.float().clamp(-5.0, 5.0)
    precision = torch.exp(-log_var)
    uncertainty_penalty = 0.5 * F.softplus(log_var)
    return (spatial_weight * precision * bce + uncertainty_penalty).mean()


def orthogonality_loss(outputs: Mapping[str, Any]) -> Tensor:
    losses = []
    for trace, content in zip(outputs["trace_features"], outputs["content_features"]):
        losses.append(F.cosine_similarity(trace, content, dim=1, eps=1e-6).abs().mean())
    return torch.stack(losses).mean()


def degradation_consistency_loss(clean_logits: Tensor, degraded_logits: Tensor) -> Tensor:
    """Teacher-detached symmetric Bernoulli consistency."""

    clean_target = safe_probability(clean_logits.detach().sigmoid())
    degraded_target = safe_probability(degraded_logits.detach().sigmoid())
    forward = F.binary_cross_entropy_with_logits(degraded_logits, clean_target)
    backward = F.binary_cross_entropy_with_logits(clean_logits, degraded_target)
    return 0.5 * (forward + backward)


def counterfactual_ranking_loss(
    factual_logits: Tensor,
    counterfactual_logits: Tensor,
    target: Tensor,
    margin: float,
) -> Tensor:
    """Require trace erasure to lower only genuinely manipulated evidence."""

    causal_drop = factual_logits - counterfactual_logits
    positive = (
        target * F.softplus(margin - causal_drop)
    ).sum(dim=(1, 2, 3)) / target.sum(dim=(1, 2, 3)).clamp_min(1.0)
    background = 1.0 - target
    stability_map = (
        factual_logits.sigmoid() - counterfactual_logits.sigmoid()
    ).abs()
    stability = (background * stability_map).sum(dim=(1, 2, 3)) / background.sum(
        dim=(1, 2, 3)
    ).clamp_min(1.0)
    return (positive + stability).mean()


def ledger_supervision_loss(ledger_logits: list[Tensor], target: Tensor) -> Tensor:
    return torch.stack(
        [
            F.binary_cross_entropy_with_logits(logits, target)
            + 0.5 * soft_dice_loss(logits, target)
            for logits in ledger_logits
        ]
    ).mean()


def dhel_directional_residual_loss(
    outputs: Mapping[str, Any],
    target: Tensor,
    threshold: float,
    clip: tuple[float, float],
    stable_weight: float,
    margin: float,
) -> Tensor:
    if "dhel_residual_logits" not in outputs:
        return target.new_zeros(())
    residual = outputs["dhel_residual_logits"]
    base_logits = outputs.get("dhel_base_logits", outputs.get("base_logits", outputs["mask_logits"]))
    base_probability = base_logits.detach().sigmoid()
    foreground = target > 0.5
    need_up = (foreground & (base_probability < threshold)).to(target.dtype)
    need_down = ((~foreground) & (base_probability >= threshold)).to(target.dtype)
    stable = ((need_up + need_down) < 0.5).to(target.dtype)

    # Directional correction target: false negatives should push logits up,
    # false positives should push logits down, already-correct pixels should stay quiet.
    fn_loss = (need_up * F.softplus(float(margin) - residual)).sum() / need_up.sum().clamp_min(1.0)
    fp_loss = (need_down * F.softplus(float(margin) + residual)).sum() / need_down.sum().clamp_min(1.0)
    stable_loss = (stable * F.smooth_l1_loss(residual, residual.detach().new_zeros(residual.shape), reduction="none", beta=0.5)).sum() / stable.sum().clamp_min(1.0)
    return 0.5 * (fn_loss + fp_loss) + float(stable_weight) * stable_loss


def intervention_locality_loss(outputs: Mapping[str, Any], target: Tensor) -> Tensor:
    penalties = []
    for strength in outputs["intervention_strengths"]:
        resized_target = F.interpolate(target, size=strength.shape[-2:], mode="nearest")
        outside = 1.0 - resized_target
        penalties.append(
            (outside * strength).sum() / outside.sum().clamp_min(1.0)
        )
    return torch.stack(penalties).mean()


def causal_degradation_consistency_loss(
    clean_outputs: Mapping[str, Any], degraded_outputs: Mapping[str, Any]
) -> Tensor:
    mask_agreement = degradation_consistency_loss(
        clean_outputs["mask_logits"], degraded_outputs["mask_logits"]
    )
    clean_effect = torch.tanh(clean_outputs["causal_effect"]).detach()
    degraded_effect = torch.tanh(degraded_outputs["causal_effect"])
    effect_agreement = F.smooth_l1_loss(degraded_effect, clean_effect)
    return mask_agreement + effect_agreement


class ReTraceLoss(nn.Module):
    def __init__(
        self,
        weights: Mapping[str, float],
        positive_weight_clip: tuple[float, float] | list[float] = (1.0, 20.0),
        proposal_positive_weight_clip: tuple[float, float] | list[float] | None = None,
        boundary_kernel: int = 5,
        counterfactual_margin: float = 0.5,
        causal_correction_mode: str = "logits",
        causal_correction_threshold: float = 0.5,
        causal_correction_positive_weight_clip: tuple[float, float] | list[float] | None = None,
        dhel_directional_threshold: float | None = None,
        dhel_directional_positive_weight_clip: tuple[float, float] | list[float] | None = None,
        dhel_directional_stable_weight: float = 0.05,
        dhel_directional_margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights = dict(weights)
        self.causal_correction_mode = str(causal_correction_mode)
        self.causal_correction_threshold = float(causal_correction_threshold)
        self.positive_weight_clip = tuple(float(x) for x in positive_weight_clip)
        if proposal_positive_weight_clip is None:
            proposal_positive_weight_clip = positive_weight_clip
        if causal_correction_positive_weight_clip is None:
            causal_correction_positive_weight_clip = proposal_positive_weight_clip
        self.proposal_positive_weight_clip = tuple(float(x) for x in proposal_positive_weight_clip)
        self.causal_correction_positive_weight_clip = tuple(float(x) for x in causal_correction_positive_weight_clip)
        if dhel_directional_threshold is None:
            dhel_directional_threshold = self.causal_correction_threshold
        if dhel_directional_positive_weight_clip is None:
            dhel_directional_positive_weight_clip = self.causal_correction_positive_weight_clip
        self.dhel_directional_threshold = float(dhel_directional_threshold)
        self.dhel_directional_positive_weight_clip = tuple(float(x) for x in dhel_directional_positive_weight_clip)
        self.dhel_directional_stable_weight = float(dhel_directional_stable_weight)
        self.dhel_directional_margin = float(dhel_directional_margin)
        self.boundary_kernel = int(boundary_kernel)
        self.counterfactual_margin = float(counterfactual_margin)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ReTraceLoss":
        return cls(**dict(config))

    def forward(
        self,
        outputs: Mapping[str, Any],
        target: Tensor,
        image_target: Tensor | None = None,
        degraded_outputs: Mapping[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        if image_target is None:
            image_target = (target.flatten(1).amax(1) > 0.5).to(target.dtype)[:, None]

        components: dict[str, Tensor] = {}
        components["localization"] = area_balanced_heteroscedastic_bce(
            outputs["mask_logits"], outputs["log_var"], target, self.positive_weight_clip
        )
        components["dice"] = soft_dice_loss(outputs["mask_logits"], target)
        components["boundary"] = boundary_loss(
            outputs["mask_logits"], target, self.boundary_kernel
        )

        auxiliary = []
        for logits in outputs["aux_logits"]:
            auxiliary.append(
                F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)
            )
        components["deep_supervision"] = torch.stack(auxiliary).mean()
        components["proposal"] = (
            area_balanced_bce_with_logits(
                outputs["proposal_logits"], target, self.proposal_positive_weight_clip
            )
            + soft_dice_loss(outputs["proposal_logits"], target)
        )
        if self.causal_correction_mode == "directional" and "correction_pos_logits" in outputs:
            base_probability = outputs["base_logits"].detach().sigmoid()
            foreground = target > 0.5
            need_up = (foreground & (base_probability < self.causal_correction_threshold)).to(target.dtype)
            need_down = ((~foreground) & (base_probability >= self.causal_correction_threshold)).to(target.dtype)
            components["causal_correction"] = 0.5 * (
                area_balanced_bce_with_logits(
                    outputs["correction_pos_logits"], need_up, self.causal_correction_positive_weight_clip
                )
                + area_balanced_bce_with_logits(
                    outputs["correction_neg_logits"], need_down, self.causal_correction_positive_weight_clip
                )
            )
            if "correction_pos_gate_logits" in outputs:
                components["causal_gate"] = 0.5 * (
                    area_balanced_bce_with_logits(
                        outputs["correction_pos_gate_logits"], need_up, self.causal_correction_positive_weight_clip
                    )
                    + area_balanced_bce_with_logits(
                        outputs["correction_neg_gate_logits"], need_down, self.causal_correction_positive_weight_clip
                    )
                )
            else:
                components["causal_gate"] = target.new_zeros(())
        elif self.causal_correction_mode == "residual" and "correction_delta" in outputs:
            components["causal_gate"] = target.new_zeros(())
            residual_target = target - outputs["base_logits"].detach().sigmoid()
            components["causal_correction"] = F.smooth_l1_loss(
                outputs["correction_delta"], residual_target, beta=0.1
            )
        elif "correction_logits" in outputs:
            components["causal_gate"] = target.new_zeros(())
            components["causal_correction"] = (
                F.binary_cross_entropy_with_logits(outputs["correction_logits"], target)
                + soft_dice_loss(outputs["correction_logits"], target)
            )
        else:
            components["causal_correction"] = target.new_zeros(())
            components["causal_gate"] = target.new_zeros(())
        components["counterfactual"] = counterfactual_ranking_loss(
            outputs["factual_logits"],
            outputs["counterfactual_logits"],
            target,
            self.counterfactual_margin,
        )
        components["ledger"] = ledger_supervision_loss(outputs["ledger_logits"], target)
        components["dhel_directional"] = dhel_directional_residual_loss(
            outputs,
            target,
            self.dhel_directional_threshold,
            self.dhel_directional_positive_weight_clip,
            self.dhel_directional_stable_weight,
            self.dhel_directional_margin,
        )
        components["intervention_locality"] = intervention_locality_loss(outputs, target)
        components["orthogonality"] = orthogonality_loss(outputs)
        components["image"] = F.binary_cross_entropy_with_logits(
            outputs["image_logit"], image_target
        )
        components["variance_regularization"] = outputs["log_var"].square().mean()
        if degraded_outputs is None:
            components["degradation_consistency"] = outputs["mask_logits"].new_zeros(())
        else:
            components["degradation_consistency"] = causal_degradation_consistency_loss(
                outputs, degraded_outputs
            )

        total = target.new_zeros(())
        active_names: list[str] = []
        for name, value in components.items():
            weight = float(self.weights.get(name, 0.0))
            if weight == 0.0:
                if not torch.isfinite(value).all():
                    components[name] = value.detach().new_zeros(())
                continue
            active_names.append(name)
            total = total + weight * value
        if not torch.isfinite(total):
            bad = []
            for name in active_names:
                value = components[name]
                if torch.isfinite(value).all():
                    continue
                raw = value.detach().float()
                safe = raw.nan_to_num(nan=0.0, posinf=1e6, neginf=-1e6)
                bad.append(
                    f"{name}=nan{bool(torch.isnan(raw).any())}/inf{bool(torch.isinf(raw).any())}"
                    f"/min{safe.min().item():.6g}/max{safe.max().item():.6g}"
                )
            detail = ", ".join(bad) if bad else "active component values are finite but total is not"
            raise FloatingPointError(f"non-finite ReTraceLoss total: {detail}")
        return {"total": total, **components}
