# Experiment History Before ReTraceNet

This document freezes the exploratory ReTraceNet line before the v2 redesign. Do not delete these branches or checkpoints; they are historical evidence for the next design.

## Unified Protocol

- Dataset split: `/group4/zouwenxuan/iml_generalization/paper_compare/mvss_casia2_721`
- Manifest: `/group4/zouwenxuan/iml_generalization/repos/ReTraceNet/data/casia2_721_manifest.csv`
- Split size: train 8731, val 2494, test 1247
- Official evaluation space: config `image_size=512`, no post-hoc metric relaxation
- Selection metric in these runs: validation `pixel_f1_oracle`

## Frozen Lines

| Line | Purpose | Main Change | Best Checkpoint | Best Known Result | Decision |
| --- | --- | --- | --- | --- | --- |
| Feature Fusion Parent | Test whether evidence should enter feature space instead of directly correcting logits. | `use_dhel_feature_fusion=true`, train feature adapter and fusion gain. | `/group4/zouwenxuan/iml_generalization/outputs/retracenet/resnet50_poscrop70_stage3_featurefusion_adapter_lr5e5/best.pt` | test fixed F1 0.702694, oracle F1 0.703397, AUC 0.907485 | Keep as historical parent. |
| Run C Full Hierarchy | Test full HMSI exploratory hierarchy. | Residual stream + HDRP + coarse-to-fine ICVM + CDERF + multi-scale evidence feedback. | `/group4/zouwenxuan/iml_generalization/outputs/retracenet/hmsi_overnight_20260828_094605/run_c_full_hierarchy/best.pt` | val fixed F1 0.752307, oracle F1 0.754068, AUC 0.926877; test fixed F1 0.706121, oracle F1 0.706176, AUC 0.910760 | Keep as best legacy hierarchy baseline. |
| Run C + CBR-v1 | Test conflict-guided boundary residual after Run C. | Add CBR head and train CBR/coarse head. | `/group4/zouwenxuan/iml_generalization/outputs/retracenet/hmsi_overnight_20260828_094605/run_c_plus_cbr/best.pt` | test fixed F1 0.705611, oracle F1 0.706018, AUC 0.920430 | Keep idea, reject boundary-only formulation. |
| HDRP-v2 Spatial Reliability | Test whether spatial reliability fixes small-trace weakness. | Add `SpatialReliabilityPerception`, train only spatial reliability from Run C best. | `/group4/zouwenxuan/iml_generalization/outputs/retracenet/resnet50_poscrop70_hmsi_run_c_hdrp_v2_spatial/best.pt` | E92 val oracle F1 0.754136; E89 test oracle about 0.706196 | Reject as core innovation route. |

## Important Diagnostics

### Parent vs Run C

- Overall test oracle F1: 0.703397 -> 0.706176, delta +0.002779.
- Small `<1%` F1: 0.3890 -> 0.3903, only +0.0013.
- Large `>=10%` F1: about 0.7360 -> 0.7431, larger than small gain.

Conclusion: Run C has a real but very small overall gain; it is not a strong small-manipulation solution.

### Tiny Manipulation Failure

Run C test fixed-threshold profile:

| Bucket | F1 |
| --- | ---: |
| `<0.25%` | about 0.247 |
| `0.25%-0.5%` | about 0.271 |
| `0.5%-1%` | about 0.452 |
| `<1% total` | about 0.390 |
| `1%-10%` | about 0.734 |
| `>=10%` | about 0.743 |

The main weakness is tiny manipulation, especially below 0.5% area.

### Recall Bottleneck

Run C test precision is about 0.751 and recall is about 0.667. FN pixels are substantially higher than FP pixels. Future performance work must explicitly target recall without changing the official metric.

### HDRP-v2 Sensitivity Probe

E89 raw spatial reliability had statistical discrimination, especially s1 raw small-bg gap around +0.0286. However, even with temporary sensitivity modulation `lambda=1.0`:

- all-scale small F1: 0.390347 -> 0.389815
- s1-only small F1: 0.390347 -> 0.390378
- overall fixed F1 decreased slightly in both modes

Conclusion: current HDRP reliability has statistical signal but no meaningful downstream mask utility. Do not continue HDRP-v2/v3 as a core route.

### CBR-v1

CBR-v1 improved AUC from 0.910760 to 0.920430 but did not improve F1. This suggests conflict information is useful for ranking, but boundary-only logit refinement is not the right conversion mechanism. Recast this as feature-space conflict-guided re-examination.

## Rejected Or Downgraded Branches

- `direct logit correction`: rejected. It did not produce stable F1 gains; future evidence must flow through feature space before prediction.
- `HDRP` and `HDRP-v2`: rejected as final core innovation. Keep code only as exploratory record.
- `CBR-v1`: downgraded. Keep the conflict-guided idea, but replace boundary-only refinement with feature-space re-examination.
- `Stage4-v1/v2-style head/decoder patching`: paused. It belongs to historical exploration, not the new mainline.

## Verified Design Constraints

1. Evidence should not directly correct logits. Use `Evidence -> Feature Space -> Prediction`.
2. Feature fusion is the strongest verified improvement direction in the legacy line.
3. Current hierarchy is not small-specialized.
4. Tiny manipulation and recall are the real bottlenecks.
5. Small manipulation still has weak internal evidence; the issue is conversion to mask, not complete absence of signal.
6. CBR should become re-examination, not boundary-only refinement.
