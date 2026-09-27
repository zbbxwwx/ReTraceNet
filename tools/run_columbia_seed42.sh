#!/usr/bin/env bash
set -euo pipefail
cd /group4/zouwenxuan/iml_generalization/repos/ReTraceNet
OUT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/columbia_mit_dedup_topk_20260921
SEED=42
mkdir -p "$OUT/seed_$SEED"
echo "[$(date -Is)] start seed $SEED; focus checkpoints epoch040-055" >> "$OUT/queue_seed42.log"
CUDA_VISIBLE_DEVICES=0 /group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python -u tools/run_columbia_mit_seed.py --dataset Columbia --split /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json --template /group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/imd2020_stable_bn_full100_20260916/best.pt --out "$OUT/seed_$SEED" --epochs 200 --batch-size 4 --learning-rate .001 --seed "$SEED" --skip-test >> "$OUT/seed_$SEED/train.log" 2>&1
echo "[$(date -Is)] complete seed $SEED" >> "$OUT/queue_seed42.log"
