#!/usr/bin/env bash
set -euo pipefail
cd /group4/zouwenxuan/iml_generalization/repos/ReTraceNet
OUT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/columbia_mit_dedup_topk_20260921
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
SPLIT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json
LOG="$OUT/val_selected_ensemble_eval.log"
: > "$LOG"
for SEED in 3407 2026 42; do
  echo "[$(date -Is)] evaluate seed $SEED" >> "$LOG"
  rm -rf "$OUT/seed_$SEED/topk_ensemble_1024_768"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/eval_columbia_topk_ensemble.py --run "$OUT/seed_$SEED" --split "$SPLIT" --out "$OUT/seed_$SEED/topk_ensemble_1024_768" >> "$LOG" 2>&1
  echo "[$(date -Is)] completed seed $SEED" >> "$LOG"
done
"$PY" tools/summarize_columbia_seeds.py --root "$OUT" --seeds 3407,2026,42 >> "$LOG" 2>&1
echo "[$(date -Is)] ALL COMPLETE" >> "$LOG"
