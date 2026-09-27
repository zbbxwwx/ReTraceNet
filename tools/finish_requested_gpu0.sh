#!/usr/bin/env bash
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
R=/group4/zouwenxuan/iml_generalization
OUT=$R/outputs/baseline_long_gpu0_70_20260921
LOG=$OUT/finish_requested_20260922.log
MPY=$R/envs/mesorch/bin/python
MREPO=$R/repos/Mesorch
DATA=$MREPO/data/unified_721
TOOLS=$R/repos/ReTraceNet/tools
T=$R/repos/TruFor/TruFor_train_test
TPY=$R/envs/trufor_from_docker/bin/python
ISPLIT=$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_IMD2020/split_721.json
CSPLIT=$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json
FRSPLIT=$R/datasets/FantasticReality_v1_localization/split_721_random_stratified.json
mkdir -p "$OUT"
note(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
run(){ note "START $1"; shift; "$@" >>"$LOG" 2>&1; rc=$?; note "END rc=$rc $1"; return 0; }

# Materialize FantasticReality target test list for Mesorch and combined cross splits for TruFor.
"$MPY" - <<'PY'
import json
from pathlib import Path
R=Path('/group4/zouwenxuan/iml_generalization')
out=R/'outputs/baseline_long_gpu0_70_20260921/cross_manifests';out.mkdir(parents=True,exist_ok=True)
imd=json.loads((R/'outputs/retracenet_v2/v3_offline_master_20260912/indomain_IMD2020/split_721.json').read_text())
targets={
 'CASIA1':R/'outputs/retracenet_v2/v3_offline_master_20260912/indomain_CASIA1/split_721.json',
 'Columbia':R/'outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json',
 'CASIA2':R/'outputs/retracenet_v2/mit_full_offline_20260918/casia2_split.json',
 'FantasticReality':R/'datasets/FantasticReality_v1_localization/split_721_random_stratified.json'}
for name,p in targets.items():
 d=json.loads(p.read_text()); test=d['test']
 (out/f'{name}_test.json').write_text(json.dumps(test))
 (out/f'IMD2020_to_{name}.json').write_text(json.dumps({'train':imd.get('train',[]),'val':imd['val'],'test':test}))
PY

# 1) Mesorch CASIA2 best -> CASIA2 test. Existing complete result is preserved.
if [ -s "$OUT/mesorch_indomain/CASIA2/test_unified.yaml" ]; then note "SKIP Mesorch CASIA2 test: complete"; else
 run "Mesorch CASIA2 test" "$MPY" -u "$MREPO/eval_mesorch_unified.py" --checkpoint "$OUT/mesorch_indomain/CASIA2/best.pth" --val "$DATA/CASIA2/val.json" --test "$DATA/CASIA2/test.json" --dataset CASIA2 --out "$OUT/mesorch_indomain/CASIA2/test_unified.yaml"
fi

# 2) Mesorch IMD best -> all table targets, threshold selected only on IMD val.
mkdir -p "$OUT/mesorch_imd_cross"
for D in CASIA1 Columbia CASIA2 FantasticReality; do
 F="$OUT/mesorch_imd_cross/$D.yaml"
 if [ -s "$F" ]; then note "SKIP Mesorch IMD->$D: complete"; else
  run "Mesorch IMD->$D" "$MPY" -u "$MREPO/eval_mesorch_unified.py" --checkpoint "$OUT/mesorch_indomain/IMD2020/best.pth" --val "$DATA/IMD2020/val.json" --test "$OUT/cross_manifests/${D}_test.json" --dataset "$D" --threshold-source IMD2020_val --out "$F"
 fi
done

# 3) TruFor Columbia: training already reached 70; only clean/re-run inference and locked test if missing.
unset PYTORCH_CUDA_ALLOC_CONF
mkdir -p "$OUT/trufor/Columbia"; cd "$T"
if [ ! -s "$OUT/trufor/Columbia/test_locked.yaml" ]; then
 rm -rf "$OUT/trufor/Columbia/val_pred" "$OUT/trufor/Columbia/test_pred"
 run "TruFor Columbia val infer" "$TPY" -u "$TOOLS/infer_trufor_split.py" --repo "$T" --experiment trufor_columbia721_70 --checkpoint "$T/weights/trufor_columbia721_70/best.pth.tar" --split "$CSPLIT" --part val --out "$OUT/trufor/Columbia/val_pred" --gpu 0
 run "TruFor Columbia test infer" "$TPY" -u "$TOOLS/infer_trufor_split.py" --repo "$T" --experiment trufor_columbia721_70 --checkpoint "$T/weights/trufor_columbia721_70/best.pth.tar" --split "$CSPLIT" --part test --out "$OUT/trufor/Columbia/test_pred" --gpu 0
 run "TruFor Columbia locked eval" "$TPY" -u "$TOOLS/evaluate_trufor_locked.py" --split "$CSPLIT" --val-pred "$OUT/trufor/Columbia/val_pred" --test-pred "$OUT/trufor/Columbia/test_pred" --out "$OUT/trufor/Columbia/test_locked.yaml"
else note "SKIP TruFor Columbia: complete"; fi

# 4) TruFor IMD: resume exactly from epoch5 checkpoint to configured epoch70.
IMDCK=$T/weights/trufor_imd721_70/checkpoint.pth.tar
if ! grep -q '^status: completed' "$T/weights/trufor_imd721_70/summary.yaml" 2>/dev/null; then
 run "TruFor IMD resume to 70" "$TPY" -u "$TOOLS/train_trufor_locked_resume.py" --repo "$T" --experiment trufor_imd721_70 --split "$ISPLIT" --gpu 0 --seed 1337 --resume "$IMDCK"
fi
rm -rf "$OUT/trufor/IMD2020/val_pred" "$OUT/trufor/IMD2020/test_pred"; mkdir -p "$OUT/trufor/IMD2020"
run "TruFor IMD val infer" "$TPY" -u "$TOOLS/infer_trufor_split.py" --repo "$T" --experiment trufor_imd721_70 --checkpoint "$T/weights/trufor_imd721_70/best.pth.tar" --split "$ISPLIT" --part val --out "$OUT/trufor/IMD2020/val_pred" --gpu 0
run "TruFor IMD test infer" "$TPY" -u "$TOOLS/infer_trufor_split.py" --repo "$T" --experiment trufor_imd721_70 --checkpoint "$T/weights/trufor_imd721_70/best.pth.tar" --split "$ISPLIT" --part test --out "$OUT/trufor/IMD2020/test_pred" --gpu 0
run "TruFor IMD locked eval" "$TPY" -u "$TOOLS/evaluate_trufor_locked.py" --split "$ISPLIT" --val-pred "$OUT/trufor/IMD2020/val_pred" --test-pred "$OUT/trufor/IMD2020/test_pred" --out "$OUT/trufor/IMD2020/test_locked.yaml"

# Fill TruFor IMD-source table targets under the same IMD-val threshold protocol.
mkdir -p "$OUT/trufor_imd_cross"
for D in CASIA1 Columbia CASIA2 FantasticReality; do
 X="$OUT/cross_manifests/IMD2020_to_${D}.json"; O="$OUT/trufor_imd_cross/$D"; mkdir -p "$O"
 rm -rf "$O/val_pred" "$O/test_pred"
 run "TruFor IMD->$D val infer" "$TPY" -u "$TOOLS/infer_trufor_split.py" --repo "$T" --experiment trufor_imd721_70 --checkpoint "$T/weights/trufor_imd721_70/best.pth.tar" --split "$X" --part val --out "$O/val_pred" --gpu 0
 run "TruFor IMD->$D test infer" "$TPY" -u "$TOOLS/infer_trufor_split.py" --repo "$T" --experiment trufor_imd721_70 --checkpoint "$T/weights/trufor_imd721_70/best.pth.tar" --split "$X" --part test --out "$O/test_pred" --gpu 0
 run "TruFor IMD->$D locked eval" "$TPY" -u "$TOOLS/evaluate_trufor_locked.py" --split "$X" --val-pred "$O/val_pred" --test-pred "$O/test_pred" --out "$O/result.yaml"
done

# 5) SAFIRE IMD 70 epochs + locked test.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$R/repos/SAFIRE; SPY=$R/envs/mesorch/bin/python; SOUT=$OUT/safire_imd; mkdir -p "$SOUT"; cd "$S"
CK=$(find "$SOUT" -name medsam_model_best.pth -type f 2>/dev/null | sort | tail -1)
if [ -z "$CK" ]; then
 run "SAFIRE IMD train 70" "$SPY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 train_imd721_locked.py --task_name SAFIRE_IMD2020_721_70 --model_type vit_b_adaptor --checkpoint "$R/weights/safire/sam_vit_b_01ec64.pth" --work_dir "$SOUT" --num_epochs 70 --batch_size 4 --num_workers 0 --lr 1e-4 --resume "" --encresume ""
 CK=$(find "$SOUT" -name medsam_model_best.pth -type f 2>/dev/null | sort | tail -1)
fi
if [ -n "$CK" ]; then
 rm -rf "$SOUT/pred/IMD721_val" "$SOUT/pred/IMD721_test"
 run "SAFIRE IMD val infer" "$SPY" -u safire_infer_imd721.py --sam_checkpoint "$R/weights/safire/sam_vit_b_01ec64.pth" --resume "$CK" --out_root "$SOUT/pred" --dataset IMD721_val --points_per_batch 64 --points_per_side 16
 run "SAFIRE IMD test infer" "$SPY" -u safire_infer_imd721.py --sam_checkpoint "$R/weights/safire/sam_vit_b_01ec64.pth" --resume "$CK" --out_root "$SOUT/pred" --dataset IMD721_test --points_per_batch 64 --points_per_side 16
 run "SAFIRE IMD locked eval" "$SPY" -u "$TOOLS/evaluate_trufor_locked.py" --split "$ISPLIT" --val-pred "$SOUT/pred/IMD721_val" --test-pred "$SOUT/pred/IMD721_test" --out "$SOUT/test_locked.yaml"
else note "SAFIRE FAILED: no checkpoint"; fi
note "ALL REQUESTED GPU0 TASKS FINISHED"
