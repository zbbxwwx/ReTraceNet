#!/usr/bin/env bash
set -euo pipefail
cd /group4/zouwenxuan/iml_generalization/repos/ReTraceNet
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
CFG=/group4/zouwenxuan/iml_generalization/repos/ReTraceNet/configs/casia2_retracenet_b0_dct_compression_forensic.yaml
OUT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/b0_dct_compression_forensic
mkdir -p "$OUT"
echo "[START] $(date) DCT forensic replacement screening on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[CFG] $CFG"
$PY train_v2_final_pipeline.py --config "$CFG" --phase dct_forensic --device cuda:0 2>&1 | tee "$OUT/train.log"
echo "[TRAIN_DONE] $(date)"
if [ ! -f "$OUT/best.pt" ]; then
  echo "[ERROR] missing best checkpoint: $OUT/best.pt" >&2
  exit 2
fi
$PY tools/eval_retracenet_stratified.py --config "$OUT/config.yaml" --checkpoint "$OUT/best.pt" --split val --device cuda:0 --out "$OUT/stratified_val_best" 2>&1 | tee "$OUT/stratified_val_best.log"
$PY tools/eval_retracenet_stratified.py --config "$OUT/config.yaml" --checkpoint "$OUT/best.pt" --split test --device cuda:0 --out "$OUT/stratified_test_best" 2>&1 | tee "$OUT/stratified_test_best.log"
echo "[ALL_DONE] $(date)"
