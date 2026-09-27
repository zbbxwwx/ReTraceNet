#!/usr/bin/env bash
set -euo pipefail
REPO=/group4/zouwenxuan/iml_generalization/repos/ReTraceNet
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
ROOT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2
MASTER=$ROOT/v3_offline_master_20260912
TEMPLATE=$ROOT/imd2020_stable_bn_full100_20260916/best.pt
COL=$ROOT/columbia_mit_v3_indomain200_20260921
QUEUE=$ROOT/columbia_mit_only_queue_20260921
mkdir -p "$QUEUE";cd "$REPO"
echo "[$(date -Iseconds)] start Columbia MiT-V3 200 epochs" | tee "$QUEUE/queue.log"
CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/run_imd_mit_full_v3.py --dataset Columbia \
 --split "$MASTER/indomain_Columbia/split_721.json" --template "$TEMPLATE" --out "$COL" \
 --epochs 200 --batch-size 4 --learning-rate .001 --skip-test > "$QUEUE/columbia_train.log" 2>&1
echo "[$(date -Iseconds)] Columbia 1024/768 val-select and test" | tee -a "$QUEUE/queue.log"
CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/eval_split_1024_locked_all.py --dataset Columbia \
 --split "$MASTER/indomain_Columbia/split_721.json" --checkpoint "$COL/best.pt" \
 --out "$COL/test_1024_768" > "$QUEUE/columbia_test.log" 2>&1
echo "[$(date -Iseconds)] ALL COMPLETE" | tee -a "$QUEUE/queue.log"
