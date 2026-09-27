#!/usr/bin/env bash
set -uo pipefail
R=/group4/zouwenxuan/iml_generalization
PY=$R/envs/mesorch/bin/python
REPO=$R/repos/Mesorch
DATA=$REPO/data/unified_721
OUT=$R/outputs/overnight_fr_imd_suite_20260923
LOG=$OUT/queue.log
mkdir -p "$OUT"
note(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
run(){ local name="$1"; shift; note "START $name"; "$@" >>"$OUT/$name.log" 2>&1; local rc=$?; note "END $name rc=$rc"; return 0; }
export CUDA_VISIBLE_DEVICES=0
note "WAIT current E6+IMD queue pid 1275773"
while kill -0 1275773 2>/dev/null; do sleep 30; done
note "CURRENT QUEUE COMPLETE"
cd "$REPO"
train_generic(){
 local MODEL="$1" BS="$2" LR="$3" SZ="$4" ACC="$5"; shift 5
 local O="$OUT/fr_indomain/$MODEL"; mkdir -p "$O"
 run "preflight_$MODEL" "$PY" -u train_imd721_locked.py --model "$MODEL" --world_size 1 --batch_size "$BS" --data_path "$DATA/FantasticReality/train_smoke.json" --epochs 1 --lr "$LR" --image_size "$SZ" --if_resizing --min_lr 5e-7 --weight_decay .05 --test_data_path "$DATA/FantasticReality/val_smoke.json" --warmup_epochs 0 --output_dir "$O/smoke" --log_dir "$O/smoke/tb" --accum_iter 1 --seed 1337 --test_period 1 --num_workers 0 "$@"
 if [ ! -s "$O/smoke/best.pth" ]; then note "SKIP FULL $MODEL: preflight checkpoint missing"; return; fi
 run "train_$MODEL" "$PY" -u train_imd721_locked.py --model "$MODEL" --world_size 1 --batch_size "$BS" --data_path "$DATA/FantasticReality/train.json" --epochs 5 --lr "$LR" --image_size "$SZ" --if_resizing --min_lr 5e-7 --weight_decay .05 --test_data_path "$DATA/FantasticReality/val.json" --warmup_epochs 1 --output_dir "$O" --log_dir "$O/tensorboard" --accum_iter "$ACC" --seed 1337 --test_period 1 --num_workers 4 "$@"
}
train_generic MesorchFull 2 1e-4 512 48 --conv_pretrain True --conv_pretrain_path "$R/weights/convnext_tiny-983f1562.pth" --seg_pretrain_path "$REPO/mit_b3.pth"
if [ -s "$OUT/fr_indomain/MesorchFull/best.pth" ]; then run eval_Mesorch "$PY" -u eval_mesorch_unified.py --checkpoint "$OUT/fr_indomain/MesorchFull/best.pth" --val "$DATA/FantasticReality/val.json" --test "$DATA/FantasticReality/test.json" --dataset FantasticReality --out "$OUT/fr_indomain/MesorchFull/test_unified.yaml"; fi
train_generic MVSSNet 2 2e-5 512 4 --if_not_amp --find_unused_parameters --no_model_eval --edge_mask_width 7
train_generic PSCC_Net 8 1e-4 256 1 --if_not_amp --if_predict_label --pretrain_path "$R/repos/PSCC-Net/models/hrnet_w18_small_v2.pth"
train_generic Cat_Net 2 1e-4 512 4 --cfg_file "$R/envs/mesorch/lib/python3.10/site-packages/IMDLBenCo/statics/model_zoo/configs/CAT_full.yaml"
note "START existing TruFor Columbia+IMD and SAFIRE IMD queue"
run post_trufor_safire bash "$R/repos/Mesorch/run_post_baselines_gpu0_70_fixed.sh"
note "CORE OVERNIGHT QUEUE FINISHED"
