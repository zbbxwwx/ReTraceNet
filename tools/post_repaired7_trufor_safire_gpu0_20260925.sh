#!/usr/bin/env bash
set -uo pipefail
R=/group4/zouwenxuan/iml_generalization
MAIN_PID=1350465
OUT=$R/outputs/post_repaired7_trufor_safire_gpu0_20260925
TOOLS=$R/repos/ReTraceNet/tools
T=$R/repos/TruFor/TruFor_train_test
TPY=$R/envs/trufor_from_docker/bin/python
SPY=$R/envs/mesorch/bin/python
CSPLIT=$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json
ISPLIT=$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_IMD2020/split_721.json
mkdir -p "$OUT"
exec >>"$OUT/queue.log" 2>&1
note(){ echo "[$(date -Is)] $*"; }
run(){ name="$1"; shift; note "START $name"; "$@" >"$OUT/$name.log" 2>&1; rc=$?; note "END $name rc=$rc"; [ $rc -ne 0 ] && echo "$name rc=$rc" >>"$OUT/FAILED"; return 0; }
note "WAIT repaired7 pid=$MAIN_PID"
while kill -0 "$MAIN_PID" 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=0
unset PYTORCH_CUDA_ALLOC_CONF
cd "$T"
for spec in "Columbia trufor_columbia721_70 $CSPLIT" "IMD2020 trufor_imd721_70 $ISPLIT"; do
 set -- $spec; D=$1; E=$2; SPLIT=$3; CK="$T/weights/$E/best.pth.tar"; O="$OUT/trufor_$D"; mkdir -p "$O"
 if [ -s "$CK" ]; then
  run "trufor_D_val" "$TPY" -u "$TOOLS/infer_trufor_split.py" --repo "$T" --experiment "$E" --checkpoint "$CK" --split "$SPLIT" --part val --out "$O/val_pred" --gpu 0
  run "trufor_D_test" "$TPY" -u "$TOOLS/infer_trufor_split.py" --repo "$T" --experiment "$E" --checkpoint "$CK" --split "$SPLIT" --part test --out "$O/test_pred" --gpu 0
  run "trufor_D_eval" "$TPY" -u "$TOOLS/evaluate_trufor_locked.py" --split "$SPLIT" --val-pred "$O/val_pred" --test-pred "$O/test_pred" --out "$O/test_locked.yaml"
 else note "MISSING $CK"; fi
done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$R/repos/SAFIRE
SOUT=$OUT/safire_IMD2020
mkdir -p "$SOUT"
cd "$S"
run safire_IMD_train "$SPY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 train_imd721_locked.py --task_name SAFIRE_IMD2020_721_70 --model_type vit_b_adaptor --checkpoint "$R/weights/safire/sam_vit_b_01ec64.pth" --work_dir "$SOUT" --num_epochs 70 --batch_size 4 --num_workers 0 --lr 1e-4 --resume "" --encresume ""
CK=$(find "$SOUT" -name medsam_model_best.pth -type f 2>/dev/null | sort | tail -1)
if [ -n "$CK" ]; then
 run safire_IMD_val "$SPY" -u safire_infer_imd721.py --sam_checkpoint "$R/weights/safire/sam_vit_b_01ec64.pth" --resume "$CK" --out_root "$SOUT/pred" --dataset IMD721_val --points_per_batch 64 --points_per_side 16
 run safire_IMD_test "$SPY" -u safire_infer_imd721.py --sam_checkpoint "$R/weights/safire/sam_vit_b_01ec64.pth" --resume "$CK" --out_root "$SOUT/pred" --dataset IMD721_test --points_per_batch 64 --points_per_side 16
 run safire_IMD_eval "$SPY" -u "$TOOLS/evaluate_trufor_locked.py" --split "$ISPLIT" --val-pred "$SOUT/pred/IMD721_val" --test-pred "$SOUT/pred/IMD721_test" --out "$SOUT/test_locked.yaml"
else note "SAFIRE_NO_CHECKPOINT"; fi
note COMPLETE
touch "$OUT/COMPLETE"
