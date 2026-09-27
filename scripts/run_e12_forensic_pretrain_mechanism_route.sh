#!/usr/bin/env bash
set -euo pipefail

ROOT=/group4/zouwenxuan/iml_generalization/repos/ReTraceNet
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
E12_OUT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e12_protocol_pretrained_forensic

mkdir -p "${E12_OUT}"
cd "${ROOT}"

"${PY}" tools/pretrain_forensic_stream.py \
  --config configs/casia2_retracenet_e12_forensic_history_pretrain.yaml

"${PY}" train_v2_final_pipeline.py \
  --config configs/casia2_retracenet_e12_mechanism_full_adapt_from_e5_pretrained_forensic.yaml \
  --phase mechanism_full_adapt

"${PY}" train_v2_final_pipeline.py \
  --config configs/casia2_retracenet_e12_mechanism_full_joint_from_e5_pretrained_forensic.yaml \
  --phase mechanism_full_joint

"${PY}" tools/eval_retrace_unified_original_grid.py \
  --config configs/casia2_retracenet_e12_mechanism_full_joint_from_e5_pretrained_forensic.yaml \
  --checkpoint /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e12_mechanism_full_joint_from_e5_pretrained_forensic/best.pt \
  --state model \
  --split val \
  --select-threshold \
  --out "${E12_OUT}/e12_joint_pretrained_forensic_val_original_grid.json"

JOINT_TH=$("${PY}" -c 'import json; print(json.load(open("'"${E12_OUT}"'/e12_joint_pretrained_forensic_val_original_grid.json"))["selected_threshold_fake_only"])')

"${PY}" tools/eval_retrace_unified_original_grid.py \
  --config configs/casia2_retracenet_e12_mechanism_full_joint_from_e5_pretrained_forensic.yaml \
  --checkpoint /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e12_mechanism_full_joint_from_e5_pretrained_forensic/best.pt \
  --state model \
  --split test \
  --threshold "${JOINT_TH}" \
  --out "${E12_OUT}/e12_joint_pretrained_forensic_test_original_grid_locked_fake_threshold.json"
