#!/usr/bin/env bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
R=/group4/zouwenxuan/iml_generalization
PY=$R/envs/mesorch/bin/python
DATA=$R/repos/Mesorch/data/unified_721/IMD2020
OUT=$R/outputs/imd_failed4_fixed_70_20260923
LOG=$OUT/queue.log
mkdir -p "$OUT"
note(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
run(){ note "START $1"; shift; "$@"; rc=$?; note "END rc=$rc"; return $rc; }

# 1 SAFIRE: invoke torch distributed through the environment Python, not missing torchrun executable.
SOUT=$OUT/SAFIRE_IMD
mkdir -p "$SOUT"
cd $R/repos/SAFIRE
run "SAFIRE IMD 70ep" "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 train_imd721_locked.py \
  --task_name IMD2020_721 --model_type vit_b --checkpoint $R/weights/safire/sam_vit_b_01ec64.pth \
  --work_dir "$SOUT" --num_epochs 70 --batch_size 4 --num_workers 0 --lr 1e-4 \
  >> "$SOUT/train.log" 2>&1 || true

# 2-4 IMDLBenCo baselines using fixed validator and correct AMP flag semantics.
cd $R/repos/Mesorch
train_benco(){
  name="$1"; batch="$2"; lr="$3"; size="$4"; accum="$5"; shift 5
  o="$OUT/$name"; mkdir -p "$o"
  run "$name IMD 70ep" "$PY" -u train_imd721_locked_fixed.py --model "$name" --world_size 1 \
    --batch_size "$batch" --data_path "$DATA/train.json" --epochs 70 --lr "$lr" --image_size "$size" \
    --if_resizing --min_lr 5e-7 --weight_decay .05 --test_data_path "$DATA/val.json" \
    --warmup_epochs 2 --output_dir "$o" --log_dir "$o/tensorboard" --accum_iter "$accum" \
    --seed 1337 --test_period 1 --num_workers 4 "$@" >> "$o/train.log" 2>&1 || true
}
train_benco MVSSNet 2 2e-5 512 4 --if_not_amp --find_unused_parameters --no_model_eval
train_benco PSCC_Net 8 1e-4 256 1 --if_not_amp --if_predict_label \
  --pretrain_path $R/repos/PSCC-Net/models/hrnet_w18_small_v2.pth
train_benco Cat_Net 2 1e-4 512 4 \
  --cfg_file $R/envs/mesorch/lib/python3.10/site-packages/IMDLBenCo/statics/model_zoo/configs/CAT_full.yaml
note "QUEUE COMPLETE"
