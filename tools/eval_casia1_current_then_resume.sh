#!/usr/bin/env bash
set -euo pipefail
REPO=/group4/zouwenxuan/iml_generalization/repos/ReTraceNet
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
ROOT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2
MASTER=$ROOT/v3_offline_master_20260912
CAS1=$ROOT/casia1_mit_v3_indomain100_20260921
OUT=$CAS1/test_current_best_1024_768
cd "$REPO"
echo "[$(date -Iseconds)] evaluate current CASIA1 best" > "$CAS1/current_best_test_queue.log"
CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/eval_split_1024_locked_all.py --dataset CASIA1 --split "$MASTER/indomain_CASIA1/split_721.json" --checkpoint "$CAS1/best.pt" --out "$OUT" >> "$CAS1/current_best_test_queue.log" 2>&1
echo "[$(date -Iseconds)] test complete; resume CASIA1-Columbia queue" >> "$CAS1/current_best_test_queue.log"
nohup bash tools/run_casia1_columbia_mit_queue.sh >> "$ROOT/casia1_columbia_mit_queue_20260921/nohup.log" 2>&1 < /dev/null &
