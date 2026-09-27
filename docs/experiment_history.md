# ReTraceNet Experiment History

This document records completed exploratory results before the formal Verify -> Reason -> Re-examine chain.

## B0 Strong Base

- Purpose: establish a strong ReTraceNet baseline under the unified CASIA2 7/2/1 split and current metric protocol.
- Structure: ResNet-50 content encoder, constrained residual bank, lightweight forensic hierarchy, progressive decoder, mask/log-var/aux heads.
- Best validation metric: oracle F1 = 0.7452157, fixed F1 = 0.7451277, AUC = 0.9185182 from `outputs/retracenet_v2/b0_fulltrain_v2_from_e5/best.pt`.
- Mechanism diagnostic: baseline localization is stable, but tiny manipulation recall/F1 remains limited.
- Final judgment: keep as the official initialization and parent model for B1 probes.
- Why stop: B0 is now a reference point; further LR/scheduler/epoch tuning was paused to avoid turning the innovation study into baseline tuning.

## FGTP-v1

- Purpose: test whether feature-space trace preservation on already-downsampled N1/N2 can improve fine-grained and tiny manipulation localization.
- Structure: N1/N2 trace-preservation residual path with zero-gated feature replacement.
- Best metric: joint E0 validation oracle F1 = 0.7324607; FGTP-only probe did not exceed the B0 reference.
- Mechanism diagnostic: alpha opened, but T-N showed global residual drift rather than localized tiny manipulation evidence.
- Final judgment: negative exploratory result.
- Why stop: the mechanism changed feature statistics without producing meaningful tiny improvement.

## FGTP-v2

- Purpose: stabilize FGTP-v1 by making the preservation path more selective.
- Structure: selective N1/N2 trace-preservation path with guarded residual feature replacement.
- Best metric: FGTP-v2-only validation oracle F1 remained essentially tied with B0, around 0.745215.
- Mechanism diagnostic: drift was reduced, but <0.5% and <1% tiny F1/recall improvements were numerical rather than meaningful.
- Final judgment: negative exploratory result.
- Why stop: technical stabilization did not translate into a convincing tiny-manipulation mechanism.

## FGTP-Bypass

- Purpose: test the revised hypothesis that traces must be preserved before downsampling rather than repaired after downsampling.
- Structure: high-resolution residual-bank bypass from R0, producing P_hr at 1/2 resolution, packed by PixelUnshuffle into a zero-gated D1 feature residual.
- Best metric: validation best at epoch 1, oracle F1 = 0.7452045, fixed F1 = 0.7451455, AUC = 0.9173588.
- Mechanism diagnostic: tiny F1 had a small precision-driven increase, but recall decreased and q_hr tiny/background separation was not established; Sep_<0.5 and Sep_<1 became negative by the final epoch.
- Final judgment: negative exploratory result.
- Why stop: static trace enhancement/preservation could not reliably distinguish tiny manipulation traces from background forensic or high-frequency responses.

## FGTP Line Termination

FGTP line was terminated after three structurally distinct probes, rather than repeatedly tuned. The formal ReTraceNet innovation chain now removes Preserve as a core innovation and switches to Verify -> Reason -> Re-examine:

1. HCV: Hierarchical Counterfactual Verification.
2. DHE: Dual-Hypothesis Evidence Reasoning.
3. CFR: Conflict-Guided Feature Re-examination.
