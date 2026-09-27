#!/usr/bin/env bash
set -euo pipefail
cd /group4/zouwenxuan/iml_generalization/repos/ReTraceNet
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
CFG=configs/casia2_retracenet_b0_same_image_correspondence.yaml
$PY train_v2_final_pipeline.py --config "$CFG" --phase correspondence_probe --device cuda
