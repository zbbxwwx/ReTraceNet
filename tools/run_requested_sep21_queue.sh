#!/usr/bin/env bash
set -uo pipefail
ROOT=/group4/zouwenxuan/iml_generalization
REPO=$ROOT/repos/ReTraceNet
PY=$ROOT/envs/mesorch/bin/python
OUT=$ROOT/outputs/retracenet_v2/sep21_requested_queue
mkdir -p "$OUT"
cd "$REPO"
run(){ echo "[$(date -Is)] START $1" | tee -a "$OUT/controller.log"; shift; CUDA_VISIBLE_DEVICES=0 "$@" >> "$OUT/controller.log" 2>&1; rc=$?; echo "[$(date -Is)] END rc=$rc" | tee -a "$OUT/controller.log"; return $rc; }

# 1) IMD coverage-first epoch29: select threshold on IMD val at 1024/768, then locked test.
run imd_coverage_ep29_1024 "$PY" -u tools/eval_split_1024_locked.py \
 --dataset IMD2020 \
 --checkpoint $ROOT/outputs/retracenet_v2/imd_tune_coverage_first_2k_20260920/best.pt \
 --split $ROOT/outputs/retracenet_v2/v3_offline_master_20260912/indomain_IMD2020/split_721.json \
 --out $ROOT/outputs/retracenet_v2/imd_coverage_ep29_1024_768_20260921 || exit 11

# 2) Hard-shift-balanced source-only one epoch probe.
run casia2_hard_shift_probe "$PY" -u tools/run_casia2_hard_shift_probe.py || exit 12

# 3/4) Four-domain cross evaluation only if source hard-shift gate passed.
PASS=$($PY - <<'PY'
import yaml
p='/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_mit_hard_shift_balanced_probe1_20260921/summary.yaml'
print(1 if yaml.safe_load(open(p)).get('robustness_gate_pass') else 0)
PY
)
if [[ "$PASS" == 1 ]]; then
 run hard_shift_four_domain "$PY" -u tools/evaluate_casia2_mit_cross721.py \
  --checkpoint $ROOT/outputs/retracenet_v2/casia2_mit_hard_shift_balanced_probe1_20260921/epoch01.pt \
  --out $ROOT/outputs/retracenet_v2/casia2_hard_shift_cross4_1024_768_20260921 \
  --datasets CASIA1 Columbia IMD2020 NIST16 || exit 13
else
 echo "[$(date -Is)] Hard-shift gate failed: JPEG/DCT tuning stopped; manipulation diversification is next route." | tee -a "$OUT/controller.log"
fi

# 6) Retrain the current MiT-V3 in-domain model (the historical bests were ResNet),
# then reselect threshold on each target val at 1024/768 and run locked test.
run casia1_mit_indomain100 "$PY" -u tools/run_imd_mit_full_v3.py \
 --dataset CASIA1 --split $ROOT/outputs/retracenet_v2/v3_offline_master_20260912/indomain_CASIA1/split_721.json \
 --template $ROOT/outputs/retracenet_v2/casia1_stable_bn_v1_full50_20260913/best.pt \
 --out $ROOT/outputs/retracenet_v2/casia1_mit_v3_indomain100_20260921 --epochs 100 --batch-size 4 || exit 14
run casia1_mit_indomain_1024 "$PY" -u tools/eval_split_1024_locked.py \
 --dataset CASIA1 --checkpoint $ROOT/outputs/retracenet_v2/casia1_mit_v3_indomain100_20260921/best.pt \
 --split $ROOT/outputs/retracenet_v2/v3_offline_master_20260912/indomain_CASIA1/split_721.json \
 --out $ROOT/outputs/retracenet_v2/casia1_mit_v3_indomain100_1024_768_20260921 || exit 15
run columbia_mit_indomain200 "$PY" -u tools/run_imd_mit_full_v3.py \
 --dataset Columbia --split $ROOT/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json \
 --template $ROOT/outputs/retracenet_v2/columbia_stable_bn_v1_full200_20260913/best.pt \
 --out $ROOT/outputs/retracenet_v2/columbia_mit_v3_indomain200_20260921 --epochs 200 --batch-size 4 || exit 16
run columbia_mit_indomain_1024 "$PY" -u tools/eval_split_1024_locked.py \
 --dataset Columbia --checkpoint $ROOT/outputs/retracenet_v2/columbia_mit_v3_indomain200_20260921/best.pt \
 --split $ROOT/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json \
 --out $ROOT/outputs/retracenet_v2/columbia_mit_v3_indomain200_1024_768_20260921 || exit 17
echo "[$(date -Is)] ALL REQUESTED TASKS COMPLETED" | tee -a "$OUT/controller.log"
