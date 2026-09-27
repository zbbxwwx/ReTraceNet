#!/usr/bin/env bash
set -euo pipefail
cd /group4/zouwenxuan/iml_generalization/repos/ReTraceNet
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
SPLIT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_IMD2020/split_721.json
TPL=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/imd_mitb2_full_v3_100ep_20260918/best.pt
O0=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/imd_tune_e1_crop1024_4k_20260920
O1=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/imd_tune_e2_group_lr_4k_20260920
mkdir -p "$O0" "$O1"
nohup env CUDA_VISIBLE_DEVICES=0 "$PY" tools/run_imd_mit_tuning.py \
  --dataset IMD2020 --split "$SPLIT" --template "$TPL" --out "$O0" \
  --epochs 8 --batch-size 1 --learning-rate .001 --seed 3407 \
  --train-crop 1024 --variant baseline --skip-test > "$O0/train.log" 2>&1 &
echo $! > "$O0/pid"
nohup env CUDA_VISIBLE_DEVICES=1 "$PY" tools/run_imd_mit_tuning.py \
  --dataset IMD2020 --split "$SPLIT" --template "$TPL" --out "$O1" \
  --epochs 30 --batch-size 4 --learning-rate .001 --seed 3407 \
  --train-crop 512 --variant group_lr --skip-test > "$O1/train.log" 2>&1 &
echo $! > "$O1/pid"
echo "gpu0 pid $(cat "$O0/pid")"
echo "gpu1 pid $(cat "$O1/pid")"
