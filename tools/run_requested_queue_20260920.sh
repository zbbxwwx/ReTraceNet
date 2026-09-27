#!/usr/bin/env bash
set -euo pipefail
ROOT=/group4/zouwenxuan/iml_generalization
REPO=$ROOT/repos/ReTraceNet
PY=$ROOT/envs/mesorch/bin/python
OUT=$ROOT/outputs/retracenet_v2
LOG=$OUT/requested_queue_20260920
mkdir -p "$LOG"
exec 9>"$LOG/queue.lock"
flock -n 9 || { echo "queue already running"; exit 1; }
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0

echo "[1/4] DG epoch1 cross-domain evaluation" | tee "$LOG/status.txt"
$PY tools/evaluate_casia2_mit_cross721.py \
  --checkpoint "$OUT/casia2_mit_jpeg_only_dg_probe1_20260920/epoch01.pt" \
  --out "$OUT/casia2_mit_jpeg_only_dg_epoch1_cross721_20260920" \
  --datasets CASIA1 Columbia IMD2020 NIST16 \
  > "$LOG/01_cross.log" 2>&1

echo "[2/4] IMD coverage-first resume cycles 6-30 and locked test" | tee "$LOG/status.txt"
$PY tools/run_imd_mit_tuning.py \
  --dataset IMD2020 \
  --split "$OUT/v3_offline_master_20260912/indomain_IMD2020/split_721.json" \
  --template "$OUT/balanced_residual_v3_full30_20260912/best.pt" \
  --out "$OUT/imd_tune_coverage_first_2k_20260920" \
  --epochs 30 --batch-size 4 --learning-rate 0.001 --seed 3407 \
  --variant coverage_first --train-crop 512 --schedule-total-steps 14000 --resume \
  > "$LOG/02_imd_coverage_resume.log" 2>&1

echo "[3/4] NIST16 coverage-first 7:2:1 train and locked test" | tee "$LOG/status.txt"
$PY tools/run_imd_mit_tuning.py \
  --dataset NIST16 \
  --split "$OUT/v3_offline_master_20260912/indomain_NIST16/split_721.json" \
  --template "$OUT/balanced_residual_v3_full30_20260912/best.pt" \
  --out "$OUT/nist16_tune_coverage_first_30_20260920" \
  --epochs 30 --batch-size 4 --learning-rate 0.001 --seed 3407 \
  --variant coverage_first --train-crop 512 --schedule-total-steps 14000 \
  > "$LOG/03_nist_coverage.log" 2>&1

echo "[4/4] CASIA2 two-level robustness" | tee "$LOG/status.txt"
$PY tools/evaluate_casia2_dg_robustness_two_levels.py \
  > "$LOG/04_robustness.log" 2>&1

echo "completed" | tee "$LOG/status.txt"
