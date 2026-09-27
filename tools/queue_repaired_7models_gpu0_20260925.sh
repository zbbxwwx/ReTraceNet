#!/usr/bin/env bash
set -uo pipefail
R=/group4/zouwenxuan/iml_generalization
PY=$R/envs/mesorch/bin/python
TPY=$R/envs/trufor_from_docker/bin/python
OUT=$R/outputs/repaired_7models_gpu0_20260925
DATA=$R/repos/Mesorch/data/unified_721/FantasticReality_fixed
M=$R/repos/Mesorch
mkdir -p "$OUT"
exec >>"$OUT/queue.log" 2>&1
note(){ echo "[$(date -Is)] $*"; }
run(){ n="$1";shift;note "START $n";"$@" >"$OUT/$n.log" 2>&1;rc=$?;note "END $n rc=$rc";[ $rc -ne 0 ]&&echo "$n rc=$rc">>"$OUT/FAILED";return 0; }
export CUDA_VISIBLE_DEVICES=0
train_common(){ model="$1" bs="$2" lr="$3" size="$4" accum="$5";shift 5;o="$OUT/fr_$model";mkdir -p "$o";run "train_$model" "$PY" -u "$M/train_imd721_locked_fixed.py" --model "$model" --world_size 1 --batch_size "$bs" --data_path "$DATA/train.json" --epochs 5 --lr "$lr" --image_size "$size" --if_resizing --min_lr 5e-7 --weight_decay .05 --test_data_path "$DATA/val.json" --warmup_epochs 1 --output_dir "$o" --log_dir "$o/tb" --accum_iter "$accum" --seed 1337 --test_period 1 --num_workers 4 "$@"; }
note 'REPAIRED_7MODEL_QUEUE_START'
cd "$M"
train_common MesorchFull 2 1e-4 512 48 --conv_pretrain True --conv_pretrain_path "$R/weights/convnext_tiny-983f1562.pth" --seg_pretrain_path "$M/mit_b3.pth"
if [ -s "$OUT/fr_MesorchFull/best.pth" ];then run eval_Mesorch "$PY" -u "$M/eval_mesorch_unified.py" --checkpoint "$OUT/fr_MesorchFull/best.pth" --val "$DATA/val.json" --test "$DATA/test.json" --dataset FantasticReality --out "$OUT/fr_MesorchFull/test_unified.yaml";fi
train_common MVSSNet 2 2e-5 512 4 --if_not_amp --find_unused_parameters --no_model_eval --edge_mask_width 7
train_common PSCC_Net 8 1e-4 256 1 --if_not_amp --if_predict_label --pretrain_path "$R/repos/PSCC-Net/models/hrnet_w18_small_v2.pth"
train_common Cat_Net 2 1e-4 512 4 --cfg_file "$R/envs/mesorch/lib/python3.10/site-packages/IMDLBenCo/statics/model_zoo/configs/CAT_full.yaml"
# TruFor: repaired lists and absolute pretrained paths were smoke-tested.
T=$R/repos/TruFor/TruFor_train_test;cd "$T";unset PYTORCH_CUDA_ALLOC_CONF
run train_TruFor "$TPY" -u train.py -exp trufor_fr721_5 -g 0
CK=$T/weights/trufor_fr721_5/best.pth.tar;SPLIT=$R/datasets/FantasticReality_v1_localization/split_721_random_stratified.json
if [ -s "$CK" ];then mkdir -p "$OUT/fr_TruFor";run infer_TruFor_val "$TPY" -u "$R/repos/ReTraceNet/tools/infer_trufor_split.py" --repo "$T" --experiment trufor_fr721_5 --checkpoint "$CK" --split "$SPLIT" --part val --out "$OUT/fr_TruFor/val_pred" --gpu 0;run infer_TruFor_test "$TPY" -u "$R/repos/ReTraceNet/tools/infer_trufor_split.py" --repo "$T" --experiment trufor_fr721_5 --checkpoint "$CK" --split "$SPLIT" --part test --out "$OUT/fr_TruFor/test_pred" --gpu 0;run eval_TruFor "$TPY" -u "$R/repos/ReTraceNet/tools/evaluate_trufor_locked.py" --split "$SPLIT" --val-pred "$OUT/fr_TruFor/val_pred" --test-pred "$OUT/fr_TruFor/test_pred" --out "$OUT/fr_TruFor/test_locked.yaml";fi
# SAFIRE: corrected lists exist in ForensicsEval/data/img_lists and data/img_lists.
S=$R/repos/SAFIRE;cd "$S";export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True;mkdir -p "$OUT/fr_SAFIRE"
run train_SAFIRE "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 train_fr721_locked.py --task_name SAFIRE_FR721_5 --model_type vit_b_adaptor --checkpoint "$R/weights/safire/sam_vit_b_01ec64.pth" --work_dir "$OUT/fr_SAFIRE" --num_epochs 5 --batch_size 4 --num_workers 0 --lr 1e-4 --resume '' --encresume ''
# ManTra: pretrained inference only; unified locked/oracle evaluator.
cd "$R/repos/ManTra-net-pytorch";run eval_ManTra "$PY" -u eval_mantra_unified.py --weight "$R/repos/ManTra-net-pytorch/MantraNet/MantraNetv4.pt" --val "$DATA/val.json" --test "$DATA/test.json" --out "$OUT/fr_ManTra/test_unified.yaml" --max-side 1024
note 'REPAIRED_7MODEL_QUEUE_COMPLETE';touch "$OUT/COMPLETE"
