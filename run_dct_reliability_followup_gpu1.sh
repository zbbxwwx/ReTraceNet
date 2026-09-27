#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/group4/zouwenxuan/iml_generalization
REPO=$R/repos/ReTraceNet
PY=$R/envs/mesorch/bin/python
PARENT=$R/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt
GATE=$R/outputs/retracenet_v2/casia2_dct_reliability_gate_pilot1_20260921/epoch01.pt
EXT=$R/outputs/retracenet_v2/casia2_dct_reliability_external4_20260921
CTRL=$R/outputs/retracenet_v2/casia2_dct_fixed_gate_control1_20260921
LOG=$R/outputs/retracenet_v2/casia2_dct_reliability_followup_20260921.log
mkdir -p "$EXT" "$CTRL"
cd "$REPO"
echo "[$(date -Is)] external probe start" >> "$LOG"
if "$PY" -u tools/eval_dct_reliability_external.py --checkpoint "$GATE" --parent "$PARENT" --out "$EXT" >> "$LOG" 2>&1; then
  echo "[$(date -Is)] external probe complete" >> "$LOG"
else
  echo "[$(date -Is)] external probe failed rc=$?" >> "$LOG"
fi
echo "[$(date -Is)] fixed-gate control start" >> "$LOG"
if "$PY" -u tools/run_casia2_dct_fixed_gate_control.py >> "$LOG" 2>&1; then
  echo "[$(date -Is)] fixed-gate control complete" >> "$LOG"
else
  echo "[$(date -Is)] fixed-gate control failed rc=$?" >> "$LOG"
fi
