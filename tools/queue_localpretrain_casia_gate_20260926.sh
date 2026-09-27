#!/usr/bin/env bash
set -u
R=/group4/zouwenxuan/iml_generalization
P=$R/repos/ReTraceNet
PY=$R/envs/mesorch/bin/python
LOCAL=$R/outputs/retracenet_v2/native_jpeg_hrnet_local_stage_div2k_20260926
ROOT=$R/outputs/retracenet_v2/native_jpeg_hrnet_localpretrain_gate_20260926
TEMPLATE=$R/outputs/retracenet_v2/casia2_mit_ep76_manipdiv_posttrain1_20260921/best.pt
mkdir -p "$ROOT"
exec >>"$ROOT/queue.log" 2>&1
echo "$(date -Is) wait local pid=1401314"
while kill -0 1401314 2>/dev/null; do sleep 30; done
test -s "$LOCAL/best.pt" || { echo "$(date -Is) local pretrain failed"; exit 2; }
echo "$(date -Is) local complete"
CUDA_VISIBLE_DEVICES=1 "$PY" "$P/tools/run_casia2_native_jpeg_hrnet.py" --template "$TEMPLATE" --smoke-only --forensic-pretrain "$LOCAL/best.pt" --split "$R/outputs/retracenet_v2/v3_target_adapted_stable_20260913/CASIA1/split_721.json" --out "$ROOT/smoke" --dataset CASIA1
test $? -eq 0 || { echo "$(date -Is) smoke failed"; exit 3; }
echo "$(date -Is) CASIA1 start"
CUDA_VISIBLE_DEVICES=1 "$PY" "$P/tools/run_casia2_native_jpeg_hrnet.py" --dataset CASIA1 --split "$R/outputs/retracenet_v2/v3_target_adapted_stable_20260913/CASIA1/split_721.json" --template "$TEMPLATE" --out "$ROOT/CASIA1" --epochs 100 --batch-size 4 --learning-rate 1e-3 --seed 3407 --forensic-pretrain "$LOCAL/best.pt"
rc=$?; echo "$(date -Is) CASIA1 rc=$rc"; test $rc -eq 0 || exit $rc
BEST=$("$PY" - "$ROOT/CASIA1/summary.yaml" <<'PY'
import sys,yaml
x=yaml.safe_load(open(sys.argv[1]))
print(float(x.get('best_F1',x.get('best_f1',0))))
PY
)
echo "$(date -Is) CASIA1 best=$BEST"
"$PY" - "$BEST" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1])>=.82 else 1)
PY
if [ $? -ne 0 ]; then echo "$(date -Is) GATE_FAIL stop downstream"; exit 0; fi
echo "$(date -Is) GATE_PASS downstream start"
run(){
 ds=$1; split=$2; ep=$3
 echo "$(date -Is) $ds start epochs=$ep"
 CUDA_VISIBLE_DEVICES=1 "$PY" "$P/tools/run_casia2_native_jpeg_hrnet.py" --dataset "$ds" --split "$split" --template "$TEMPLATE" --out "$ROOT/$ds" --epochs "$ep" --batch-size 4 --learning-rate 1e-3 --seed 3407 --forensic-pretrain "$LOCAL/best.pt"
 echo "$(date -Is) $ds rc=$?"
}
run CASIA2 "$R/outputs/retracenet_v2/mit_full_offline_20260918/casia2_split.json" 100
run IMD2020 "$R/outputs/retracenet_v2/v3_target_adapted_stable_20260913/IMD2020/split_721.json" 100
run Columbia "$R/outputs/retracenet_v2/columbia_mit_dedup_topk_20260921/corrected_test_exclude_mask_polarity_anomaly_20260924/split_721_revised.json" 200
run FantasticReality "$R/datasets/FantasticReality_v1_localization/split_721_random_stratified.json" 5
echo "$(date -Is) ALL_DONE"
