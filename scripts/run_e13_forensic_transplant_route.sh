#!/usr/bin/env bash
set -euo pipefail

ROOT=/group4/zouwenxuan/iml_generalization/repos/ReTraceNet
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
OUT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e13_protocol

mkdir -p "${OUT}"
cd "${ROOT}"

"${PY}" tools/train_forensic_transplant.py \
  --config configs/casia2_retracenet_e13_forensic_transplant.yaml

"${PY}" train_v2_final_pipeline.py \
  --config configs/casia2_retracenet_e13_mechanism_full_adapt_after_transplant.yaml \
  --phase mechanism_full_adapt

"${PY}" train_v2_final_pipeline.py \
  --config configs/casia2_retracenet_e13_mechanism_full_joint_after_transplant.yaml \
  --phase mechanism_full_joint

"${PY}" tools/eval_retrace_unified_original_grid.py \
  --config configs/casia2_retracenet_e13_mechanism_full_joint_after_transplant.yaml \
  --checkpoint /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e13_mechanism_full_joint_after_transplant/best.pt \
  --state model \
  --split val \
  --select-threshold \
  --out "${OUT}/e13_joint_val_original_grid.json"

TH=$("${PY}" -c 'import json; print(json.load(open("'"${OUT}"'/e13_joint_val_original_grid.json"))["selected_threshold_fake_only"])')

"${PY}" tools/eval_retrace_unified_original_grid.py \
  --config configs/casia2_retracenet_e13_mechanism_full_joint_after_transplant.yaml \
  --checkpoint /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e13_mechanism_full_joint_after_transplant/best.pt \
  --state model \
  --split test \
  --threshold "${TH}" \
  --out "${OUT}/e13_joint_test_original_grid_locked_fake_threshold.json"
