#!/usr/bin/env bash
set -uo pipefail
# Wait for the current main queue and its FantasticReality retry.
for P in 1093493 1096746; do while kill -0 "$P" 2>/dev/null; do sleep 60; done; done
export CUDA_VISIBLE_DEVICES=0
R=/group4/zouwenxuan/iml_generalization
REPO=$R/repos/Mesorch
PY=$R/envs/mesorch/bin/python
DATA=$REPO/data/unified_721
OUT=$R/outputs/imd_other_models_70_20260922
LOG=$OUT/queue.log
mkdir -p "$OUT"; cd "$REPO"
note(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
run_model(){
 local NAME="$1"; shift; local O="$OUT/$NAME"; mkdir -p "$O"
 if [ -s "$O/best.pth" ]; then note "SKIP $NAME training: best exists"; return; fi
 note "START $NAME IMD2020 70ep"
 "$PY" -u train_imd721_locked.py --model "$NAME" --world_size 1 --batch_size "$1" --data_path "$DATA/IMD2020/train.json" --epochs 70 --lr "$2" --image_size "$3" --if_resizing --min_lr 5e-7 --weight_decay .05 --test_data_path "$DATA/IMD2020/val.json" --warmup_epochs 2 --output_dir "$O" --log_dir "$O/tensorboard" --accum_iter "$4" --seed 1337 --test_period 1 --num_workers 4 "${@:5}" >> "$O/train.log" 2>&1
 local RC=$?; note "END rc=$RC $NAME IMD2020"
}
# Effective batches are conservative for one 24GB GPU.
run_model MVSSNet 2 2e-5 512 4 --if_not_amp --find_unused_parameters --no_model_eval
run_model PSCC_Net 8 1e-4 256 1 --if_not_amp --if_predict_label --pretrain_path "$R/repos/PSCC-Net/models/hrnet_w18_small_v2.pth"
run_model Cat_Net 2 1e-4 512 4 --cfg_file "$R/envs/mesorch/lib/python3.10/site-packages/IMDLBenCo/statics/model_zoo/configs/CAT_full.yaml"
note "ALL OTHER IMD MODELS TRAINING FINISHED"
