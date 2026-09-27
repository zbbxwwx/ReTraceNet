#!/usr/bin/env bash
set -uo pipefail
R=/group4/zouwenxuan/iml_generalization; T=$R/repos/ReTraceNet/tools; OUT=$R/outputs/imd_cross_extension_20260923
PY=$R/envs/mesorch/bin/python; TPY=$R/envs/trufor_from_docker/bin/python; LOG=$OUT/queue.log
mkdir -p "$OUT"; note(){ echo "[$(date -Is)] $*"|tee -a "$LOG";}; run(){ N=$1;shift;note "START $N";"$@" >>"$OUT/$N.log" 2>&1;RC=$?;note "END $N rc=$RC";return 0;}
while kill -0 1276545 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=0
IMD=$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_IMD2020/split_721.json
MR=$R/repos/ManTra-net-pytorch; MW=$MR/MantraNet/MantraNetv4.pt
run mantra_imd_val "$PY" -u "$T/infer_mantra_split.py" --repo "$MR" --checkpoint "$MW" --split "$IMD" --part val --out "$OUT/mantra/imd_val" --gpu 0
for D in CASIA1 Columbia FantasticReality; do
 if [ "$D" = CASIA1 ];then S=$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_CASIA1/split_721.json;fi
 if [ "$D" = Columbia ];then S=$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json;fi
 if [ "$D" = FantasticReality ];then S=$R/datasets/FantasticReality_v1_localization/split_721_random_stratified.json;fi
 run mantra_$D "$PY" -u "$T/infer_mantra_split.py" --repo "$MR" --checkpoint "$MW" --split "$S" --part test --out "$OUT/mantra/$D" --gpu 0
 run mantra_eval_$D "$PY" -u "$T/evaluate_trufor_locked.py" --split "$S" --val-pred "$OUT/mantra/imd_val" --test-pred "$OUT/mantra/$D" --out "$OUT/mantra/$D.yaml"
done
TR=$R/repos/TruFor/TruFor_train_test; CK=$TR/weights/trufor_imd721_70/best.pth.tar
for D in CASIA1 Columbia FantasticReality; do
 if [ "$D" = CASIA1 ];then S=$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_CASIA1/split_721.json;fi
 if [ "$D" = Columbia ];then S=$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json;fi
 if [ "$D" = FantasticReality ];then S=$R/datasets/FantasticReality_v1_localization/split_721_random_stratified.json;fi
 run trufor_$D "$TPY" -u "$T/infer_trufor_split.py" --repo "$TR" --experiment trufor_imd721_70 --checkpoint "$CK" --split "$S" --part test --out "$OUT/trufor/$D" --gpu 0
done
note "MANTRA/TRUFOR IMD CROSS COMPLETE"
