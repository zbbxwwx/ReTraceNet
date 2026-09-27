#!/usr/bin/env bash
set -euo pipefail

ROOT=/group4/zouwenxuan/iml_generalization/repos/ReTraceNet
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
CONFIG=configs/casia2_retracenet_e15_hr_decoder_adapt.yaml

cd "${ROOT}"

echo "architecture=e15_hr parent=E5 phase=decoder_adapt"
"${PY}" train_v2_final_pipeline.py \
  --config "${CONFIG}" \
  --phase e15_hr_decoder_adapt
