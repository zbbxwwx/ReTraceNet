#!/usr/bin/env bash
set -uo pipefail
R=/group4/zouwenxuan/iml_generalization
MAINPID=$(cat $R/outputs/imd_failed4_fixed_70_20260923/queue.pid)
while kill -0 "$MAINPID" 2>/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=0
PY=$R/envs/mesorch/bin/python
OUT=$R/outputs/imd_failed4_fixed_70_20260923
LOG=$OUT/repair_queue.log
note(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
# SAFIRE corrected model type; launch through Python module.
cd $R/repos/SAFIRE
note 'START SAFIRE corrected vit_b_adaptor 70ep'
$PY -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 train_imd721_locked.py \
 --task_name IMD2020_721 --model_type vit_b_adaptor --checkpoint $R/weights/safire/sam_vit_b_01ec64.pth \
 --work_dir $OUT/SAFIRE_IMD --num_epochs 70 --batch_size 4 --num_workers 0 --lr 1e-4 \
 >> $OUT/SAFIRE_IMD/train_fixed.log 2>&1
note "END SAFIRE rc=$?"
# MVSS corrected edge-mask contract.
cd $R/repos/Mesorch
note 'START MVSS corrected edge mask 70ep'
$PY -u train_imd721_locked_fixed.py --model MVSSNet --world_size 1 --batch_size 2 \
 --data_path $R/repos/Mesorch/data/unified_721/IMD2020/train.json --epochs 70 --lr 2e-5 \
 --image_size 512 --if_resizing --edge_mask_width 7 --min_lr 5e-7 --weight_decay .05 \
 --test_data_path $R/repos/Mesorch/data/unified_721/IMD2020/val.json --warmup_epochs 2 \
 --output_dir $OUT/MVSSNet --log_dir $OUT/MVSSNet/tensorboard --accum_iter 4 --seed 1337 \
 --test_period 1 --num_workers 4 --if_not_amp --find_unused_parameters --no_model_eval \
 >> $OUT/MVSSNet/train_fixed.log 2>&1
note "END MVSS rc=$?"
