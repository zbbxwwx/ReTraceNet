#!/usr/bin/env bash
set -euo pipefail
REPO=/group4/zouwenxuan/iml_generalization/repos/ReTraceNet
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
ROOT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2
MASTER=$ROOT/v3_offline_master_20260912
TEMPLATE=$ROOT/imd2020_stable_bn_full100_20260916/best.pt
CAS1=$ROOT/casia1_mit_v3_indomain100_20260921
COL=$ROOT/columbia_mit_v3_indomain200_20260921
QUEUE=$ROOT/casia1_columbia_mit_queue_20260921
mkdir -p "$QUEUE"; cd "$REPO"
log(){ echo "[$(date -Iseconds)] $*" | tee -a "$QUEUE/queue.log"; }
log "resume CASIA1 MiT-V3 from saved epoch9 through epoch100"
CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/run_imd_mit_full_v3.py \
 --dataset CASIA1 --split "$MASTER/indomain_CASIA1/split_721.json" --template "$TEMPLATE" \
 --out "$CAS1" --epochs 100 --batch-size 4 --learning-rate .001 --resume --skip-test \
 > "$QUEUE/casia1_train.log" 2>&1
log "CASIA1 select threshold on val at 1024/768 and evaluate test once"
CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/eval_split_1024_locked.py \
 --dataset CASIA1 --split "$MASTER/indomain_CASIA1/split_721.json" --checkpoint "$CAS1/best.pt" \
 --out "$CAS1/test_1024_768" > "$QUEUE/casia1_test_1024.log" 2>&1
log "start Columbia MiT-V3 public-initialized 200 epochs"
CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/run_imd_mit_full_v3.py \
 --dataset Columbia --split "$MASTER/indomain_Columbia/split_721.json" --template "$TEMPLATE" \
 --out "$COL" --epochs 200 --batch-size 4 --learning-rate .001 --skip-test \
 > "$QUEUE/columbia_train.log" 2>&1
log "Columbia select threshold on val at 1024/768 and evaluate test once"
CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/eval_split_1024_locked.py \
 --dataset Columbia --split "$MASTER/indomain_Columbia/split_721.json" --checkpoint "$COL/best.pt" \
 --out "$COL/test_1024_768" > "$QUEUE/columbia_test_1024.log" 2>&1
log "ALL COMPLETE"
