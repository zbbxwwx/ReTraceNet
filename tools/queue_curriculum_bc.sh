#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/group4/zouwenxuan/iml_generalization
TASK_REPO=$TASK_ROOT/repos/ReTraceNet
TASK_PY=$TASK_ROOT/envs/mesorch/bin/python
TASK_SPLIT=$TASK_ROOT/outputs/retracenet_v2/native_jpeg_hrnet_djpeg_stage2_global_div2k_20260925/split.json
TASK_A=$TASK_ROOT/outputs/retracenet_v2/native_jpeg_local_curriculum_a_20260926/best.pt
TASK_OUT=$TASK_ROOT/outputs/retracenet_v2/native_jpeg_local_curriculum_bc_20260926
mkdir -p "$TASK_OUT"
exec >>"$TASK_OUT/queue.log" 2>&1
export CUDA_VISIBLE_DEVICES=1
echo "$(date -Is) Stage B start"
"$TASK_PY" "$TASK_REPO/tools/pretrain_local_curriculum_bc.py" --mode B --init "$TASK_A" --global-split "$TASK_SPLIT" --out "$TASK_OUT/B" --epochs 15
if "$TASK_PY" -c "import json;raise SystemExit(0 if json.load(open('$TASK_OUT/B/decision.json'))['passed'] else 1)"; then
  echo "$(date -Is) Stage B pass, Stage C start"
  "$TASK_PY" "$TASK_REPO/tools/pretrain_local_curriculum_bc.py" --mode C --init "$TASK_OUT/B/best.pt" --global-split "$TASK_SPLIT" --out "$TASK_OUT/C" --epochs 15
else
  echo "$(date -Is) Stage B failed; Stage C not started"
fi
