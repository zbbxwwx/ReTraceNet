#!/usr/bin/env bash
set -euo pipefail

ROOT=/group4/zouwenxuan/iml_generalization/repos/ReTraceNet
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
OUT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e15_hr
CONFIG=configs/casia2_retracenet_e15_hr_warmup.yaml

mkdir -p "${OUT}"
cd "${ROOT}"

echo "architecture=e15_hr parent=E5 phase=warmup"
"${PY}" tools/preflight_e15_hr.py \
  --config "${CONFIG}" \
  --decoder-config configs/casia2_retracenet_e15_hr_decoder_adapt.yaml \
  --out "${OUT}/preflight_warmup.json"

"${PY}" train_v2_final_pipeline.py \
  --config "${CONFIG}" \
  --phase e15_hr_warmup
