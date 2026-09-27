# ReTraceNet Experiment Plan

The new plan separates method innovation from performance engineering. The target is to improve the unified CASIA2 test F1 from about 0.706 toward the long-term target around 0.84, without changing the official metric protocol.

## Final Innovation Story

```text
Preserve -> Verify -> Reason & Re-examine
```

1. Fine-Grained Trace Preservation (FGTP): preserve faint/tiny forensic traces before deep downsampling destroys them.
2. Hierarchical Counterfactual Verification (HCV): verify whether a trace truly supports manipulation through same-image counterfactual replacement.
3. Conflict-Driven Evidence Reasoning and Re-examination (CDER): reason with support/refutation/conflict, then re-examine hard regions in feature space.

## Recommended Dependency Order

The order below is more useful than the requested phase numbering because several performance items must become part of B0 before testing innovations.

| Stage | Purpose | Depends On |
| --- | --- | --- |
| P0 | Freeze legacy and implement clean v2 skeleton. | None |
| B0 | Build strong non-innovation baseline. | P0 |
| B1 | Add FGTP as a single variable. | B0 |
| B2 | Add HCV/ICVM-v2 as a single variable. | B1 |
| B3 | Add CDER re-examination as a single variable. | B2 |
| P4 | Resolution 512 -> 640. | Stable B0 or B3, run as controlled performance test |
| P5 | Small-aware sampling and recall-oriented objective. | Stable B0 first, then reused for B1/B2/B3 ablations |
| P6 | ConvNeXt-T backbone. | After v2 structure and training recipe are stable |

## B0 Strong Base

- Change: clean ResNet-50 content encoder, lightweight forensic stream, high-resolution decoder with 1/4 and optional 1/2 detail path, 512 input minimum.
- Do not change: no counterfactual verification, no support/refutation/conflict, no re-examination, no metric protocol.
- Initialization: ImageNet ResNet-50 and normal decoder initialization; optionally initialize compatible encoder weights from legacy Feature Fusion only after a scratch sanity run.
- Trainable parameters: encoder with conservative LR, forensic stream, decoder, mask/aux heads.
- Epochs: 40-80 with early stopping 8, depending on convergence.
- Primary metric: unified test fixed F1 and val oracle F1 selected by validation only.
- Mechanism metric: small bucket F1/Recall for `<0.25%`, `0.25%-0.5%`, `<1%`; global precision/recall; FP/FN ratio.
- Stop criterion: early stopping or no val oracle F1 improvement for 8 epochs.
- Pass criterion: test oracle/fixed F1 must clearly exceed legacy Run C by at least +0.01, or small `<1%` F1 must improve by at least +0.03 without global F1 loss. Otherwise the baseline is still too weak.

## B1: B0 + Fine-Grained Trace Preservation

- Change: add FGTP producing `T1/T2/T3/T4`, with emphasis on 1/2 and 1/4 forensic detail preservation.
- Do not change: no HCV, no CDER, no new loss beyond B0 recipe.
- Initialization: B0 best checkpoint, non-strict load only for new FGTP parameters.
- Trainable parameters: FGTP, high-res decoder fusion, mask/aux heads; optionally freeze encoder for first short run.
- Epochs: 20-40 with early stopping 6.
- Primary metric: small `<1%` fixed F1 and overall fixed/oracle F1.
- Mechanism metric: T_i foreground-background contrast for tiny masks, especially T1/T2; small probability distribution near threshold.
- Stop criterion: no small-val improvement for 6 epochs or global val oracle F1 degrades by more than 0.005 for 3 consecutive epochs.
- Pass criterion: `<0.25%` or `0.25%-0.5%` F1 improves by at least +0.02, or `<1%` recall improves by at least +0.04 with no overall F1 drop greater than 0.003.

## B2: B1 + Hierarchical Counterfactual Verification

- Change: add HCV using local fine forensic proposal plus hierarchical context prior. Coarse prior cannot override fine proposal.
- Do not change: no CDER re-examination, no backbone swap, no metric changes.
- Initialization: B1 best checkpoint.
- Trainable parameters: HCV proposal/intervention/verifier, feature fusion from HCV into decoder, optionally last decoder block; keep encoder LR very low if unfrozen.
- Epochs: 20-40 with early stopping 6.
- Primary metric: overall fixed F1, small fixed F1/Recall, val oracle F1.
- Mechanism metric: manipulated causal drop, background counterfactual change, local proposal vs hierarchical prior fg-bg gap by size bucket.
- Stop criterion: causal drop fails to exceed background change by a clear margin after 5 epochs, or val F1 stagnates for 6 epochs.
- Pass criterion: manipulated causal drop increases, background change remains low, small recall improves by at least +0.03, and no mid-scale prior reversal for tiny regions.

## B3: B2 + Conflict-Driven Evidence Reasoning And Re-examination

- Change: add support/refutation/conflict reasoning and feature-space re-examination. Replace CBR-v1 boundary-only logic.
- Do not change: no direct logit correction, no CBR-v1 boundary residual, no new backbone.
- Initialization: B2 best checkpoint.
- Trainable parameters: evidence reasoning, re-examination adapter/gate, high-resolution decoder head; optionally freeze earlier components first.
- Epochs: 20-40 with early stopping 6.
- Primary metric: overall fixed F1 and small F1; oracle F1/AUC are secondary.
- Mechanism metric: support/refutation separation, conflict vs error/hard-region correlation, re-examination delta inside FN-prone pixels vs background.
- Stop criterion: AUC improves but fixed/oracle F1 does not improve after 6 epochs, or re-examination mostly increases FP.
- Pass criterion: F1 improves, not only AUC. Target at least +0.005 over B2 overall and +0.02 over B2 small `<1%` F1.

## P4: Resolution 512 -> 640

- Change: raise input/eval resolution from 512 to 640 for the best stable architecture.
- Do not change: architecture and loss.
- Initialization: best stable checkpoint at 512 with compatible interpolation where needed, or train from same initialization for controlled comparison.
- Trainable parameters: same as selected stage.
- Epochs: 15-30 continuation, or full schedule if memory allows.
- Primary metric: unified test fixed/oracle F1.
- Mechanism metric: tiny bucket F1 and recall; inference memory/time.
- Stop criterion: validation F1 does not improve after 5 epochs.
- Pass criterion: test F1 improves by at least +0.01 and small `<0.5%` improves without unacceptable memory cost.

## P5: Small-Aware Sampling / Recall-Oriented Optimization

- Change: training-only mask-area-aware sampling and one recall-oriented loss variant at a time.
- Do not change: validation/test protocol; no GT guidance outside training.
- Initialization: B0 best first, then apply the winning recipe consistently to B1/B2/B3 ablations.
- Trainable parameters: unchanged from the selected stage.
- Epochs: 20-40.
- Primary metric: recall and fixed F1.
- Mechanism metric: FP/FN ratio, precision loss, tiny bucket recall.
- Stop criterion: precision collapses or global F1 degrades for 3 consecutive validations.
- Pass criterion: recall increases enough to improve F1, not merely more positive pixels. Require `<1%` recall +0.05 or global recall +0.02 with fixed F1 non-negative.

## P6: ConvNeXt-T Backbone

- Change: replace ResNet-50 content encoder only after v2 structure is stable.
- Do not change: v2 modules, training recipe, metric protocol.
- Initialization: ImageNet ConvNeXt-T; compare against ResNet-50 v2 under same recipe.
- Trainable parameters: backbone with low LR, decoder/modules normal LR.
- Epochs: 40-80 with early stopping.
- Primary metric: unified test fixed/oracle F1.
- Mechanism metric: small bucket F1, calibration, AUC.
- Stop criterion: no val oracle F1 improvement for 8 epochs.
- Pass criterion: improves test F1 by at least +0.01 over ResNet-50 v2 or materially improves tiny buckets.

## The Next Single Experiment

The next experiment to run should be B0 Strong Base, not another HDRP/CBR patch. Reason: the legacy best is only about 0.706 test F1, and the gap to 0.84 cannot plausibly be closed by small gated adapters. A stronger base with high-resolution decoding, small-aware training, and recall-aware evaluation is the foundation needed before testing the three innovations fairly.

## Audit: Legacy High-Res C2 Refinement Failure

This audit is included before starting B0 so the new baseline does not repeat the old negative result.

- Legacy run: `resnet50_stage1_poscrop70_highres_c2_refine_lr1e4`.
- Config path: `configs/casia2_retrace_resnet50_stage1_poscrop70_highres_c2_refine_lr1e4.yaml`.
- Initialization: resumed from the Stage1 positive-crop checkpoint around epoch 57; optimizer was reset.
- Trainable scope: only `highres_decoder_refinement` and `highres_refinement_gain`.
- Structure: a late low-resolution logit residual was added after the existing prediction. It concatenated `ledger_features[0]`, `anchors[0]`, and upsampled `ledger_features[1]`, then predicted one residual logit map.
- Zero-effect mechanism: `sigmoid(highres_refinement_gain)` controlled the residual, and the final residual conv was zero-initialized.
- Result: early stopped after epochs 57-61. Best validation oracle F1 was `0.746903` at epoch 59, below the Stage1 positive-crop best validation oracle F1 of about `0.747288`.
- Diagnosis: this was not a true high-resolution feature decoder. It was a narrow late residual on logits with a highly restricted trainable scope, so failure here does not rule out a clean B0 decoder that fuses content and forensic features progressively from 1/32 to 1/4 resolution.

B0 therefore must be evaluated as a fresh strong base: no legacy gated patch, no old innovation modules, no logit-only C2 residual, and no resume of Run C innovation weights.
