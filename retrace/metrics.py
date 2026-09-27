"""Dependency-free pixel- and image-level forensic metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


def binary_auc(scores: list[float], targets: list[int]) -> float:
    """AUROC with tie-aware trapezoidal integration."""

    if not scores or len(set(targets)) < 2:
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    y = np.asarray(targets, dtype=np.int64)[order]
    s = np.asarray(scores)[order]
    distinct = np.r_[np.where(np.diff(s))[0], y.size - 1]
    tp = np.cumsum(y)[distinct]
    fp = 1 + distinct - tp
    tp = np.r_[0, tp] / max(1, y.sum())
    fp = np.r_[0, fp] / max(1, (1 - y).sum())
    return float(np.trapz(tp, fp))


@dataclass
class Confusion:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    def scores(self) -> tuple[float, float, float]:
        f1 = 2 * self.tp / max(1.0, 2 * self.tp + self.fp + self.fn)
        iou = self.tp / max(1.0, self.tp + self.fp + self.fn)
        fpr = self.fp / max(1.0, self.fp + self.tn)
        return f1, iou, fpr


class ForensicMetrics:
    def __init__(
        self,
        fixed_threshold: float = 0.5,
        bins: int = 2048,
        calibration_bins: int = 15,
        gate_threshold: float | None = None,
    ) -> None:
        self.fixed_threshold = fixed_threshold
        self.gate_threshold = fixed_threshold if gate_threshold is None else float(gate_threshold)
        self.bins = bins
        self.calibration_bins = calibration_bins
        self.positive_hist = torch.zeros(bins, dtype=torch.float64)
        self.negative_hist = torch.zeros(bins, dtype=torch.float64)
        self.calibration_count = torch.zeros(calibration_bins, dtype=torch.float64)
        self.calibration_probability = torch.zeros(calibration_bins, dtype=torch.float64)
        self.calibration_target = torch.zeros(calibration_bins, dtype=torch.float64)
        self.uncertainty_error_hist = torch.zeros(bins, dtype=torch.float64)
        self.uncertainty_correct_hist = torch.zeros(bins, dtype=torch.float64)
        self.fixed = Confusion()
        self.authentic_fp = 0.0
        self.authentic_pixels = 0.0
        self.absolute_error = 0.0
        self.pixels = 0
        self.manipulated_causal_drop = 0.0
        self.manipulated_pixels = 0.0
        self.background_counterfactual_change = 0.0
        self.background_pixels = 0.0
        self.intervention_energy_inside = 0.0
        self.intervention_energy_total = 0.0
        self.ledger_conflict_pixels = 0.0
        self.ledger_pixels = 0.0
        self.pos_gate_positive_hist = torch.zeros(bins, dtype=torch.float64)
        self.pos_gate_negative_hist = torch.zeros(bins, dtype=torch.float64)
        self.neg_gate_positive_hist = torch.zeros(bins, dtype=torch.float64)
        self.neg_gate_negative_hist = torch.zeros(bins, dtype=torch.float64)
        self.fn_to_tp = 0.0
        self.fp_to_tn = 0.0
        self.tp_to_fn = 0.0
        self.tn_to_fp = 0.0
        self.image_scores: list[float] = []
        self.image_targets: list[int] = []

    @torch.no_grad()
    def update(self, outputs: dict, target: Tensor, image_target: Tensor) -> None:
        probability = outputs["mask_logits"].sigmoid().detach().cpu()
        target = target.detach().cpu().bool()
        prediction = probability >= self.fixed_threshold
        self.fixed.tp += float((prediction & target).sum())
        self.fixed.fp += float((prediction & ~target).sum())
        self.fixed.fn += float((~prediction & target).sum())
        self.fixed.tn += float((~prediction & ~target).sum())
        self.absolute_error += float((probability - target.float()).abs().sum())
        self.pixels += target.numel()

        causal_effect = outputs["causal_effect"].detach().cpu()
        factual_probability = outputs["factual_logits"].sigmoid().detach().cpu()
        counterfactual_probability = (
            outputs["counterfactual_logits"].sigmoid().detach().cpu()
        )
        self.manipulated_causal_drop += float((causal_effect * target).sum())
        self.manipulated_pixels += float(target.sum())
        background = ~target
        self.background_counterfactual_change += float(
            ((factual_probability - counterfactual_probability).abs() * background).sum()
        )
        self.background_pixels += float(background.sum())

        intervention_energy = outputs["intervention_effects"][0].detach().cpu()
        intervention_target = torch.nn.functional.interpolate(
            target.float(), size=intervention_energy.shape[-2:], mode="nearest"
        ).bool()
        self.intervention_energy_inside += float(
            (intervention_energy * intervention_target).sum()
        )
        self.intervention_energy_total += float(intervention_energy.sum())
        ledger_conflict = outputs["ledger_conflicts"][0].detach().cpu()
        self.ledger_conflict_pixels += float((ledger_conflict > 0.5).sum())
        self.ledger_pixels += ledger_conflict.numel()

        indices = (probability.flatten() * (self.bins - 1)).long().clamp(0, self.bins - 1)
        labels = target.flatten()
        self.positive_hist += torch.bincount(indices[labels], minlength=self.bins).double()
        self.negative_hist += torch.bincount(indices[~labels], minlength=self.bins).double()
        calibration_indices = (
            probability.flatten() * (self.calibration_bins - 1)
        ).long().clamp(0, self.calibration_bins - 1)
        self.calibration_count += torch.bincount(
            calibration_indices, minlength=self.calibration_bins
        ).double()
        self.calibration_probability += torch.bincount(
            calibration_indices,
            weights=probability.flatten(),
            minlength=self.calibration_bins,
        ).double()
        self.calibration_target += torch.bincount(
            calibration_indices,
            weights=labels.float(),
            minlength=self.calibration_bins,
        ).double()
        uncertainty = outputs["log_var"].sigmoid().detach().cpu().flatten()
        uncertainty_indices = (uncertainty * (self.bins - 1)).long().clamp(0, self.bins - 1)
        errors = prediction.flatten() != labels
        self.uncertainty_error_hist += torch.bincount(
            uncertainty_indices[errors], minlength=self.bins
        ).double()
        self.uncertainty_correct_hist += torch.bincount(
            uncertainty_indices[~errors], minlength=self.bins
        ).double()

        authentic = ~target.flatten(1).any(1)
        if authentic.any():
            authentic_prediction = prediction.flatten(1)[authentic]
            self.authentic_fp += float(authentic_prediction.sum())
            self.authentic_pixels += authentic_prediction.numel()

        if "correction_pos_gate_logits" in outputs and "base_logits" in outputs:
            base_probability = outputs["base_logits"].sigmoid().detach().cpu()
            need_up = target & (base_probability < self.gate_threshold)
            need_down = (~target) & (base_probability >= self.gate_threshold)
            pos_gate = outputs["correction_pos_gate_logits"].sigmoid().detach().cpu()
            neg_gate = outputs["correction_neg_gate_logits"].sigmoid().detach().cpu()
            pos_indices = (pos_gate.flatten() * (self.bins - 1)).long().clamp(0, self.bins - 1)
            neg_indices = (neg_gate.flatten() * (self.bins - 1)).long().clamp(0, self.bins - 1)
            need_up_flat = need_up.flatten()
            need_down_flat = need_down.flatten()
            self.pos_gate_positive_hist += torch.bincount(pos_indices[need_up_flat], minlength=self.bins).double()
            self.pos_gate_negative_hist += torch.bincount(pos_indices[~need_up_flat], minlength=self.bins).double()
            self.neg_gate_positive_hist += torch.bincount(neg_indices[need_down_flat], minlength=self.bins).double()
            self.neg_gate_negative_hist += torch.bincount(neg_indices[~need_down_flat], minlength=self.bins).double()
            base_prediction = base_probability >= self.gate_threshold
            final_prediction = probability >= self.gate_threshold
            self.fn_to_tp += float((~base_prediction & target & final_prediction).sum())
            self.fp_to_tn += float((base_prediction & ~target & ~final_prediction).sum())
            self.tp_to_fn += float((base_prediction & target & ~final_prediction).sum())
            self.tn_to_fp += float((~base_prediction & ~target & final_prediction).sum())

        self.image_scores.extend(outputs["image_logit"].sigmoid().flatten().detach().cpu().tolist())
        self.image_targets.extend(image_target.flatten().detach().cpu().int().tolist())

    def _histogram_curves(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tp = torch.flip(torch.cumsum(torch.flip(self.positive_hist, dims=(0,)), 0), dims=(0,))
        fp = torch.flip(torch.cumsum(torch.flip(self.negative_hist, dims=(0,)), 0), dims=(0,))
        fn = self.positive_hist.sum() - tp
        tn = self.negative_hist.sum() - fp
        return tp, fp, fn, tn

    def _histogram_auc(self, positive_hist: Tensor, negative_hist: Tensor) -> float:
        if positive_hist.sum() == 0 or negative_hist.sum() == 0:
            return float("nan")
        tp = torch.flip(torch.cumsum(torch.flip(positive_hist, (0,)), 0), (0,))
        fp = torch.flip(torch.cumsum(torch.flip(negative_hist, (0,)), 0), (0,))
        tpr = tp / positive_hist.sum().clamp_min(1.0)
        fpr = fp / negative_hist.sum().clamp_min(1.0)
        tpr = torch.cat([tpr.new_zeros(1), torch.flip(tpr, (0,))])
        fpr = torch.cat([fpr.new_zeros(1), torch.flip(fpr, (0,))])
        return float(torch.trapz(tpr, fpr))

    def compute(self, threshold_count: int = 101) -> dict[str, float]:
        fixed_f1, fixed_iou, fixed_fpr = self.fixed.scores()
        fixed_precision = self.fixed.tp / max(1.0, self.fixed.tp + self.fixed.fp)
        fixed_recall = self.fixed.tp / max(1.0, self.fixed.tp + self.fixed.fn)
        tp, fp, fn, _ = self._histogram_curves()
        f1_curve = 2 * tp / (2 * tp + fp + fn).clamp_min(1.0)
        sample_bins = torch.linspace(0, self.bins - 1, threshold_count).long()
        sampled_f1 = f1_curve[sample_bins]
        best_index = int(sample_bins[int(sampled_f1.argmax())])
        best_threshold = best_index / (self.bins - 1)
        best_f1 = float(f1_curve[best_index])
        best_iou = best_f1 / max(1e-12, 2.0 - best_f1)
        best_precision = float(tp[best_index] / (tp[best_index] + fp[best_index]).clamp_min(1.0))
        best_recall = float(tp[best_index] / (tp[best_index] + fn[best_index]).clamp_min(1.0))

        tpr = tp / self.positive_hist.sum().clamp_min(1.0)
        fpr = fp / self.negative_hist.sum().clamp_min(1.0)
        if self.positive_hist.sum() == 0 or self.negative_hist.sum() == 0:
            pixel_auc = float("nan")
        else:
            pixel_auc = float(torch.trapz(
                torch.cat([tpr.new_zeros(1), torch.flip(tpr, (0,))]),
                torch.cat([fpr.new_zeros(1), torch.flip(fpr, (0,))]),
            ))
        nonempty = self.calibration_count > 0
        bin_probability = self.calibration_probability[nonempty] / self.calibration_count[nonempty]
        bin_target = self.calibration_target[nonempty] / self.calibration_count[nonempty]
        pixel_ece = float(
            ((bin_probability - bin_target).abs() * self.calibration_count[nonempty]).sum()
            / self.calibration_count.sum().clamp_min(1.0)
        )
        uncertainty_positive = self.uncertainty_error_hist
        uncertainty_negative = self.uncertainty_correct_hist
        if uncertainty_positive.sum() == 0 or uncertainty_negative.sum() == 0:
            uncertainty_error_auc = float("nan")
        else:
            utp = torch.flip(torch.cumsum(torch.flip(uncertainty_positive, (0,)), 0), (0,))
            ufp = torch.flip(torch.cumsum(torch.flip(uncertainty_negative, (0,)), 0), (0,))
            utpr = utp / uncertainty_positive.sum()
            ufpr = ufp / uncertainty_negative.sum()
            uncertainty_error_auc = float(
                torch.trapz(
                    torch.cat([utpr.new_zeros(1), torch.flip(utpr, (0,))]),
                    torch.cat([ufpr.new_zeros(1), torch.flip(ufpr, (0,))]),
                )
            )
        return {
            "pixel_f1_fixed": fixed_f1,
            "pixel_iou_fixed": fixed_iou,
            "pixel_precision_fixed": fixed_precision,
            "pixel_recall_fixed": fixed_recall,
            "pixel_fpr_fixed": fixed_fpr,
            "pixel_mae": self.absolute_error / max(1, self.pixels),
            "pixel_auc": pixel_auc,
            "pixel_ece": pixel_ece,
            "pixel_f1_oracle": best_f1,
            "pixel_iou_oracle": best_iou,
            "pixel_precision_oracle": best_precision,
            "pixel_recall_oracle": best_recall,
            "oracle_threshold": best_threshold,
            "pristine_pixel_fpr": self.authentic_fp / max(1.0, self.authentic_pixels),
            "image_auc": binary_auc(self.image_scores, self.image_targets),
            "uncertainty_error_auc": uncertainty_error_auc,
            "manipulated_causal_drop": self.manipulated_causal_drop
            / max(1.0, self.manipulated_pixels),
            "background_counterfactual_change": self.background_counterfactual_change
            / max(1.0, self.background_pixels),
            "intervention_precision": self.intervention_energy_inside
            / max(1e-8, self.intervention_energy_total),
            "ledger_contradiction_rate": self.ledger_conflict_pixels
            / max(1.0, self.ledger_pixels),
            "pos_gate_auc_for_need_up": self._histogram_auc(
                self.pos_gate_positive_hist, self.pos_gate_negative_hist
            ),
            "neg_gate_auc_for_need_down": self._histogram_auc(
                self.neg_gate_positive_hist, self.neg_gate_negative_hist
            ),
            "gate_fn_to_tp": self.fn_to_tp,
            "gate_fp_to_tn": self.fp_to_tn,
            "gate_tp_to_fn": self.tp_to_fn,
            "gate_tn_to_fp": self.tn_to_fp,
            "gate_good_flips": self.fn_to_tp + self.fp_to_tn,
            "gate_bad_flips": self.tp_to_fn + self.tn_to_fp,
        }
