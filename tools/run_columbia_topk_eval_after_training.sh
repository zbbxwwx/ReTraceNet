#!/usr/bin/env bash
set -euo pipefail
REPO="/group4/zouwenxuan/iml_generalization/repos/ReTraceNet"
OUT="/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/columbia_mit_dedup_topk_20260921"
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
SPLIT="/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json"
TRAIN_PID=800993
LOG="$OUT/eval_queue.log"
cd "$REPO"
echo "[$(date -Is)] waiting for training queue pid $TRAIN_PID" >> "$LOG"
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "[$(date -Is)] training queue ended; checking all seeds" >> "$LOG"
for SEED in 3407 2026 42; do
  RUN="$OUT/seed_$SEED"
  if [ ! -f "$RUN/last.pt" ]; then
    echo "[$(date -Is)] ERROR missing $RUN/last.pt" >> "$LOG"
    exit 2
  fi
  echo "[$(date -Is)] evaluating seed $SEED" >> "$LOG"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/eval_columbia_topk_ensemble.py \
    --run "$RUN" --split "$SPLIT" --out "$RUN/topk_ensemble_1024_768" \
    >> "$LOG" 2>&1
  echo "[$(date -Is)] completed seed $SEED" >> "$LOG"
done
"$PY" tools/summarize_columbia_seeds.py --root "$OUT" --seeds 3407,2026,42 >> "$LOG" 2>&1
echo "[$(date -Is)] ALL COMPLETE" >> "$LOG"
