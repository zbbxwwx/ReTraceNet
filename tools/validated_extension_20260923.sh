#!/usr/bin/env bash
set -uo pipefail
R=/group4/zouwenxuan/iml_generalization
TOOLS=$R/repos/ReTraceNet/tools
OUT=$R/outputs/validated_extension_20260923
SPLIT=$R/datasets/FantasticReality_v1_localization/split_721_random_stratified.json
TPY=$R/envs/trufor_from_docker/bin/python
MPY=$R/envs/mesorch/bin/python
LOG=$OUT/queue.log
mkdir -p "$OUT"
note(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
run(){ N="$1";shift;note "START $N";"$@" >>"$OUT/$N.log" 2>&1;RC=$?;note "END $N rc=$RC";return $RC; }
note "WAIT existing verified TruFor/SAFIRE queue pid=1276539"
while kill -0 1276539 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=0
# ManTra smoke then complete val/test and unified evaluator
MR=$R/repos/ManTra-net-pytorch
MW=$MR/MantraNet/MantraNetv4.pt
run mantra_smoke "$MPY" -u "$TOOLS/infer_mantra_split.py" --repo "$MR" --checkpoint "$MW" --split "$SPLIT" --part val --out "$OUT/mantra/smoke" --gpu 0
if [ -s "$OUT/mantra/smoke/manifest.json" ]; then
 rm -rf "$OUT/mantra/val_pred"; run mantra_val "$MPY" -u "$TOOLS/infer_mantra_split.py" --repo "$MR" --checkpoint "$MW" --split "$SPLIT" --part val --out "$OUT/mantra/val_pred" --gpu 0
 run mantra_test "$MPY" -u "$TOOLS/infer_mantra_split.py" --repo "$MR" --checkpoint "$MW" --split "$SPLIT" --part test --out "$OUT/mantra/test_pred" --gpu 0
 run mantra_eval "$MPY" -u "$TOOLS/evaluate_trufor_locked.py" --split "$SPLIT" --val-pred "$OUT/mantra/val_pred" --test-pred "$OUT/mantra/test_pred" --out "$OUT/mantra/test_locked.yaml"
fi
# TruFor FR 5 epoch: CPU dataset smoke, then train, val/test, unified evaluator
TR=$R/repos/TruFor/TruFor_train_test
run trufor_fr_dataset_smoke "$TPY" -c "import os,sys;os.chdir('$TR');sys.path.insert(0,'.');from lib.config import config,update_config;from dataset.data_core import myDataset;A=type('A',(),{'experiment':'trufor_fr721_5','gpu':[0],'opts':[]})();update_config(config,A);d=myDataset(config,(512,512),False,'train',aug=None);print(len(d),d[0][0].shape)"
if grep -q "END trufor_fr_dataset_smoke rc=0" "$LOG"; then
 run trufor_fr_train "$TPY" -u "$TOOLS/train_trufor_imd_locked.py" --repo "$TR" --experiment trufor_fr721_5 --split "$SPLIT" --gpu 0 --seed 1337
 CK=$TR/weights/trufor_fr721_5/best.pth.tar
 if [ -s "$CK" ]; then
  run trufor_fr_val "$TPY" -u "$TOOLS/infer_trufor_split.py" --repo "$TR" --experiment trufor_fr721_5 --checkpoint "$CK" --split "$SPLIT" --part val --out "$OUT/trufor_fr/val_pred" --gpu 0
  run trufor_fr_test "$TPY" -u "$TOOLS/infer_trufor_split.py" --repo "$TR" --experiment trufor_fr721_5 --checkpoint "$CK" --split "$SPLIT" --part test --out "$OUT/trufor_fr/test_pred" --gpu 0
  run trufor_fr_eval "$MPY" -u "$TOOLS/evaluate_trufor_locked.py" --split "$SPLIT" --val-pred "$OUT/trufor_fr/val_pred" --test-pred "$OUT/trufor_fr/test_pred" --out "$OUT/trufor_fr/test_locked.yaml"
 fi
fi
# SAFIRE FR training smoke is dataset construction inside one-process run; full only if smoke exits clean.
SR=$R/repos/SAFIRE
run safire_fr_smoke "$MPY" -c "import os,sys;os.chdir('$SR');sys.path.insert(0,'.');import ForensicsEval as FE;d=FE.data.Dataset_FantasticReality('data/img_lists/FantasticReality_721_train_tamp.txt');x=d[0];print(len(d),type(x))"
if grep -q "END safire_fr_smoke rc=0" "$LOG"; then
 run safire_fr_train "$MPY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 "$SR/train_fr721_locked.py" --task_name SAFIRE_FR_721_5 --model_type vit_b_adaptor --checkpoint "$R/weights/safire/sam_vit_b_01ec64.pth" --work_dir "$OUT/safire_fr" --num_epochs 5 --batch_size 4 --num_workers 0 --lr 1e-4 --resume "" --encresume ""
fi
note "VALIDATED EXTENSION COMPLETE"
