#!/usr/bin/env bash
set -euo pipefail
cd /group4/zouwenxuan/iml_generalization/repos/ReTraceNet
mkdir -p /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/columbia_mit_dedup_topk_20260921
for SEED in 3407 2026 42; do
 OUT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/columbia_mit_dedup_topk_20260921/seed_${SEED}
 mkdir -p "$OUT"
 echo "[$(date -Is)] start seed $SEED" >> /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/columbia_mit_dedup_topk_20260921/queue.log
 CUDA_VISIBLE_DEVICES=0 /group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python -u tools/run_columbia_mit_seed.py --dataset Columbia --split /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json --template /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/imd2020_stable_bn_full100_20260916/best.pt --out "$OUT" --epochs 200 --batch-size 4 --learning-rate .001 --seed "$SEED" --skip-test > "$OUT/train.log" 2>&1
 echo "[$(date -Is)] complete seed $SEED" >> /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/columbia_mit_dedup_topk_20260921/queue.log
done
echo "[$(date -Is)] TRAINING COMPLETE" >> /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/columbia_mit_dedup_topk_20260921/queue.log
