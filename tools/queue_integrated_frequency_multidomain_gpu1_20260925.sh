#!/usr/bin/env bash
set -u
R=/group4/zouwenxuan/iml_generalization
PY=$R/envs/mesorch/bin/python
REPO=$R/repos/ReTraceNet
RUN=$REPO/tools/run_casia2_integrated_frequency_scratch.py
TEMPLATE=$R/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt
ROOT=$R/outputs/retracenet_v2/integrated_frequency_multidomain_gpu1_20260925
mkdir -p "$ROOT"
exec >>"$ROOT/queue.log" 2>&1
echo "[$(date -Is)] queue_start gpu=1"
run_one() {
  name="$1"; epochs="$2"; split="$3"; out="$ROOT/$name"
  mkdir -p "$out"
  echo "[$(date -Is)] start $name epochs=$epochs split=$split"
  CUDA_VISIBLE_DEVICES=1 "$PY" -u "$RUN" \
    --dataset "$name" --split "$split" --template "$TEMPLATE" --out "$out" \
    --epochs "$epochs" --batch-size 4 --learning-rate 0.001 --seed 3407 \
    >"$out/train.log" 2>&1
  rc=$?
  echo "[$(date -Is)] finish $name rc=$rc"
  if [ "$rc" -ne 0 ]; then echo "$name rc=$rc" >>"$ROOT/FAILED"; fi
}
run_one CASIA1 100 "$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_CASIA1/split_721.json"
run_one Columbia 200 "$R/outputs/retracenet_v2/columbia_mit_dedup_topk_20260921/corrected_test_exclude_mask_polarity_anomaly_20260924/split_721_revised.json"
run_one IMD2020 100 "$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_IMD2020/split_721.json"
run_one FantasticReality 5 "$R/datasets/FantasticReality_v1_localization/split_721_random_stratified.json"
echo "[$(date -Is)] queue_complete"
touch "$ROOT/COMPLETE"
