#!/usr/bin/env bash
set -u
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/group4/zouwenxuan/iml_generalization; REPO=$R/repos/ReTraceNet; PY=$R/envs/mesorch/bin/python
LOG=$R/outputs/retracenet_v2/gpu1_after_fixed_abcdef_20260921.log
PID=823462
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
cd "$REPO"
echo "[$(date -Is)] A JPEG Full/DCT-off start" >> "$LOG"
PYTHONPATH="$REPO:$REPO/tools" "$PY" -u /group4/zouwenxuan/iml_generalization/tools/evaluate_casia2_epoch76_jpeg_cue_diagnostic.py >> "$LOG" 2>&1 || true
echo "[$(date -Is)] A external4 Full/DCT-off start" >> "$LOG"
"$PY" -u tools/eval_casia2_full_dctoff_external4.py >> "$LOG" 2>&1 || true
echo "[$(date -Is)] B existing result retained: imd coverage ep29 ensemble" >> "$LOG"
echo "[$(date -Is)] C ManipDiv posttrain retry start" >> "$LOG"
"$PY" -u tools/run_casia2_mit_manipdiv_posttrain.py >> "$LOG" 2>&1 || true
echo "[$(date -Is)] A-C complete; D-F require conditional decision from produced summaries" >> "$LOG"
