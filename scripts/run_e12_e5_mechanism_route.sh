#!/usr/bin/env bash
set -euo pipefail

ROOT=/group4/zouwenxuan/iml_generalization/repos/ReTraceNet
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
E5_OUT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/score_sprint_c_e5_mrfe_20260901_075256
E12_OUT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e12_protocol

mkdir -p "${E12_OUT}"
cd "${ROOT}"

"${PY}" tools/eval_retrace_unified_original_grid.py \
  --config configs/casia2_retracenet_score_sprint_c_e5_mrfe.yaml \
  --checkpoint "${E5_OUT}/best.pt" \
  --state model \
  --split val \
  --select-threshold \
  --out "${E12_OUT}/e5_raw_val_original_grid.json"

E5_RAW_TH=$("${PY}" -c 'import json; print(json.load(open("'"${E12_OUT}"'/e5_raw_val_original_grid.json"))["selected_threshold_fake_only"])')

"${PY}" tools/eval_retrace_unified_original_grid.py \
  --config configs/casia2_retracenet_score_sprint_c_e5_mrfe.yaml \
  --checkpoint "${E5_OUT}/best.pt" \
  --state model \
  --split test \
  --threshold "${E5_RAW_TH}" \
  --out "${E12_OUT}/e5_raw_test_original_grid_locked_fake_threshold.json"

"${PY}" tools/eval_retrace_unified_original_grid.py \
  --config configs/casia2_retracenet_score_sprint_c_e5_mrfe.yaml \
  --checkpoint "${E5_OUT}/best.pt" \
  --state ema_model \
  --split val \
  --select-threshold \
  --out "${E12_OUT}/e5_ema_val_original_grid.json"

E5_EMA_TH=$("${PY}" -c 'import json; print(json.load(open("'"${E12_OUT}"'/e5_ema_val_original_grid.json"))["selected_threshold_fake_only"])')

"${PY}" tools/eval_retrace_unified_original_grid.py \
  --config configs/casia2_retracenet_score_sprint_c_e5_mrfe.yaml \
  --checkpoint "${E5_OUT}/best.pt" \
  --state ema_model \
  --split test \
  --threshold "${E5_EMA_TH}" \
  --out "${E12_OUT}/e5_ema_test_original_grid_locked_fake_threshold.json"

"${PY}" train_v2_final_pipeline.py \
  --config configs/casia2_retracenet_e12_e5_control_continue.yaml \
  --phase score_sprint_c

"${PY}" tools/eval_retrace_unified_original_grid.py \
  --config configs/casia2_retracenet_e12_e5_control_continue.yaml \
  --checkpoint /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e12_e5_control_continue/best.pt \
  --state model \
  --split val \
  --select-threshold \
  --out "${E12_OUT}/e12_control_val_original_grid.json"

CONTROL_TH=$("${PY}" -c 'import json; print(json.load(open("'"${E12_OUT}"'/e12_control_val_original_grid.json"))["selected_threshold_fake_only"])')

"${PY}" tools/eval_retrace_unified_original_grid.py \
  --config configs/casia2_retracenet_e12_e5_control_continue.yaml \
  --checkpoint /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e12_e5_control_continue/best.pt \
  --state model \
  --split test \
  --threshold "${CONTROL_TH}" \
  --out "${E12_OUT}/e12_control_test_original_grid_locked_fake_threshold.json"

"${PY}" train_v2_final_pipeline.py \
  --config configs/casia2_retracenet_e12_mechanism_full_adapt_from_e5.yaml \
  --phase mechanism_full_adapt

"${PY}" train_v2_final_pipeline.py \
  --config configs/casia2_retracenet_e12_mechanism_full_joint_from_e5.yaml \
  --phase mechanism_full_joint

"${PY}" tools/eval_retrace_unified_original_grid.py \
  --config configs/casia2_retracenet_e12_mechanism_full_joint_from_e5.yaml \
  --checkpoint /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e12_mechanism_full_joint_from_e5/best.pt \
  --state model \
  --split val \
  --select-threshold \
  --out "${E12_OUT}/e12_joint_val_original_grid.json"

JOINT_TH=$("${PY}" -c 'import json; print(json.load(open("'"${E12_OUT}"'/e12_joint_val_original_grid.json"))["selected_threshold_fake_only"])')

"${PY}" tools/eval_retrace_unified_original_grid.py \
  --config configs/casia2_retracenet_e12_mechanism_full_joint_from_e5.yaml \
  --checkpoint /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e12_mechanism_full_joint_from_e5/best.pt \
  --state model \
  --split test \
  --threshold "${JOINT_TH}" \
  --out "${E12_OUT}/e12_joint_test_original_grid_locked_fake_threshold.json"
