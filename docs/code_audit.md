# Current ReTraceNet Code Audit

This audit describes the current legacy implementation before the ReTraceNet cleanup. The legacy implementation is preserved, but it should no longer be used as the place for new architectural patching.

## Main Files

| File | Role |
| --- | --- |
| `retrace/model.py` | Main accumulated model class, ResNet/FPN sanity class, `build_model`. |
| `retrace/hierarchy.py` | Optional HMSI/HDRP/ICVM/CDERF/feedback/CBR modules. |
| `retrace/blocks.py` | Shared ConvNeXt blocks, residual bank, quality conditioner, content orthogonalization, ICVM intervention, DHEL, refiner. |
| `retrace/data.py` | Unified manifest dataset, 512 resize, positive/boundary crop support. |
| `retrace/engine.py` | Train/evaluate epoch loops. |
| `retrace/metrics.py` | Official fixed/oracle/AUC metrics via histogram. |
| `train.py` | Config-driven training, trainable patterns, parameter groups, best/last checkpointing. |
| `evaluate.py` | Config/checkpoint evaluation with non-strict compatible load. |

## Current Forward Graph

```text
Input RGB image [B,3,H,W]
 |
 |-- QualityDescriptor -> QualityConditioner -> per-scale FiLM gamma/beta and scalar reliability
 |
 |-- ConstrainedResidualBank(image)
 |
 |-- Content Encoder
 |     - if ResNet-50: ResNetFeaturePyramid -> C1/C2/C3/C4 = C2/C3/C4/C5 at strides 4/8/16/32
 |     - if custom: content stem/stages
 |
 |-- Trace Encoder
 |     - if ResNet-50: ResNetFeaturePyramid(residual bank output) -> N1/N2/N3/N4 at strides 4/8/16/32
 |     - if custom: trace stem/stages
 |
 |-- Quality calibration and ContentOrthogonalization
 |     C_i, N_i -> aligned content features and trace features
 |
 |-- Optional HDRP
 |     scalar reliability -> HierarchicalReliabilityPropagation
 |
 |-- Optional HDRP-v2
 |     trace/content/base reliability -> SpatialReliabilityPerception -> reliability maps
 |
 |-- Lateral projection
 |     trace/content features -> decoder anchors
 |
 |-- Optional LightweightResidualStream
 |     image -> residual stream pyramid -> residual_stream_lateral -> anchors += tanh(gain_i) * residual_anchor_i
 |
 |-- ICVM proposal
 |     anchors[0] -> IntraImageCounterfactualIntervention.propose -> proposal logits
 |     optional CoarseToFineICVMPrior: local proposals at all scales, coarse-to-fine residual prior
 |
 |-- Counterfactual intervention
 |     each anchor + proposal -> counterfactual anchor, intervention effect, intervention strength
 |
 |-- EvidenceRecycler
 |     anchors -> recycled features
 |     counterfactual anchors -> counterfactual recycled features
 |
 |-- DHEL
 |     DualHypothesisEvidenceLedger(recycled, counterfactual_recycled, reliability)
 |     -> dhel evidence features, ledger logits, ledger conflicts, ledger effects
 |
 |-- Optional CDERF
 |     ConflictConditionedEvidenceReasoning -> support/refutation/conflict and cross-scale evidence update
 |
 |-- Optional feature fusion
 |     dhel_feature_adapter([ledger feature, evidence feature, logits, conflict, effect, strength])
 |     ledger_features[0] += sigmoid(dhel_feature_fusion_gain) * delta
 |
 |-- Optional MultiScaleEvidenceFeedback
 |     feature-level evidence feedback with reliability/conflict gate
 |
 |-- Aux heads
 |     each ledger feature -> aux logits
 |
 |-- Shared refiner/verifier
 |     recycled[0] -> factual logits
 |     counterfactual_recycled[0] -> counterfactual logits
 |     ledger_features[0] + conflict/effect -> ledger refined logits and log_var
 |
 |-- Historical direct/logit correction paths
 |     optional DHEL residual correction
 |     optional causal correction head
 |     optional high-res decoder refinement
 |
 |-- Optional CBR-v1
 |     ConflictGuidedBoundaryRefinement(decoder feature, logits, conflict)
 |     -> logit residual correction
 |
 |-- Upsample to input size
 |     mask_logits, factual/counterfactual logits, causal effect, proposal, evidence maps
```

## Scale Answers

1. ResNet-50 content encoder outputs four scales: C1/C2/C3/C4 in code are ResNet C2/C3/C4/C5, strides 4/8/16/32, channels 256/512/1024/2048.
2. LightweightResidualStream outputs four residual scales from the RGB image through SRM-like residual bank: strides 4/8/16/32 with configured channels 32/64/128/256, then lateral projected to decoder channels.
3. HDRP enters after FiLM quality reliability and content orthogonalization, before lateral projection and ICVM.
4. ICVM enters after anchors are built. It proposes manipulation logits and then performs same-image counterfactual intervention on each anchor.
5. DHEL enters after the EvidenceRecycler; CDERF optionally updates DHEL evidence features and produces support/refutation/conflict.
6. MultiScaleEvidenceFeedback enters after CDERF/DHEL feature fusion and before aux heads/refiner. It modifies decoder/evidence features in feature space.
7. CBR-v1 acts near the end on the low-resolution final logit feature, using decoder feature, conflict, and boundary cue to produce a residual logit correction.
8. Old direct-logit paths still exist: `use_causal_correction_head`, `use_dhel_residual_correction`, `use_highres_decoder_refinement`, and the default `causal_gain * relu(causal_effect_low)` fallback when neither DHEL feature fusion nor DHEL residual correction is enabled.
9. Forward-changing config flags include at least: `encoder_backbone`, `use_quality_calibration`, `use_content_orthogonalization`, `use_counterfactual_intervention`, `use_evidence_ledger`, `use_uncertainty_refinement`, `simple_rgb_trace_baseline`, `use_content_decoder_fusion`, `decoder_source`, `detach_counterfactual_anchor`, `residual_intervention_output`, `use_causal_correction_head`, `causal_correction_injection`, `use_highres_decoder_refinement`, `use_dhel_residual_correction`, `use_dhel_feature_fusion`, `use_residual_stream`, `use_cross_scale_reliability`, `use_spatial_reliability`, `use_coarse_to_fine_icvm`, `use_conflict_cross_scale_reasoning`, `use_multiscale_evidence_feedback`, `use_reliability_conflict_feedback_gate`, `use_cbr`.
10. Historical modules still mixed into the formal forward: HDRP, HDRP-v2 spatial reliability, CBR-v1 boundary refinement, direct causal correction heads, DHEL residual logit correction, and high-res logit refinement.

## Source Mapping

| Module | Source Class | File | Status |
| --- | --- | --- | --- |
| Content encoder | `ResNetFeaturePyramid` | `retrace/model.py` | Keep concept; may remain ResNet-50 initially. |
| Residual bank | `ConstrainedResidualBank` | `retrace/blocks.py` | Keep as low-level forensic preprocessing. |
| Residual stream | `LightweightResidualStream` | `retrace/hierarchy.py` | Keep as auxiliary forensic stream, not a standalone innovation. |
| Quality FiLM / scalar reliability | `QualityDescriptor`, `QualityConditioner` | `retrace/blocks.py` | Downgrade to auxiliary conditioning. |
| Content orthogonalization | `ContentOrthogonalization` | `retrace/blocks.py` | Keep only if it supports v2 trace features. |
| HDRP | `HierarchicalReliabilityPropagation` | `retrace/hierarchy.py` | Historical/rejected as core innovation. |
| HDRP-v2 | `SpatialReliabilityPerception` | `retrace/hierarchy.py` | Historical/rejected as core innovation. |
| ICVM | `IntraImageCounterfactualIntervention`, `CoarseToFineICVMPrior` | `retrace/blocks.py`, `retrace/hierarchy.py` | Keep idea; redesign proposal fusion. |
| DHEL | `DualHypothesisEvidenceLedger` | `retrace/blocks.py` | Keep idea as evidence accounting. |
| CDERF | `ConflictConditionedEvidenceReasoning` | `retrace/hierarchy.py` | Keep and fold into v2 Innovation 3. |
| Evidence feedback | `MultiScaleEvidenceFeedback` | `retrace/hierarchy.py` | Keep concept as feature-space feedback. |
| CBR-v1 | `ConflictGuidedBoundaryRefinement` | `retrace/hierarchy.py` | Downgrade; replace boundary refinement with re-examination. |

## Metric And Split Check

- `train.py` and `evaluate.py` both use `build_loader` and `evaluate_model`.
- `ForensicMetrics` computes fixed threshold F1, oracle F1, IoU, AUC from the same mask logits and target tensors.
- Validation and test both use manifest split entries with `shuffle=False` and resize to config `image_size`.
- No test-only postprocessing was found in `evaluate.py`.
- The 0.754 vs 0.706 difference is a validation/test gap, not a training collapse.

## Main Code Problem

The current `ReTraceNet` class is no longer a clean final-method implementation. It is an experiment harness with many historical toggles. Future v2 work should not continue adding flags here. Preserve it as legacy and implement a clean v2 mainline.
