#!/usr/bin/env bash
set -euo pipefail

ROOT=/group4/zouwenxuan/iml_generalization/repos/ReTraceNet
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
OUT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e14_protocol
CKPT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e13_mechanism_full_adapt_after_transplant/milestone_eplus02.pt

mkdir -p "${OUT}"
cd "${ROOT}"

"${PY}" tools/check_region_refine_zero_init.py \
  --base-config configs/casia2_retracenet_e13_mechanism_full_adapt_after_transplant.yaml \
  --region-config configs/casia2_retracenet_e14_region_refine_from_e13b_ep1.yaml \
  --checkpoint "${CKPT}" \
  --out "${OUT}/e14_zero_init_check.json"

"${PY}" train_v2_final_pipeline.py \
  --config configs/casia2_retracenet_e14_region_refine_from_e13b_ep1.yaml \
  --phase region_refine_only
