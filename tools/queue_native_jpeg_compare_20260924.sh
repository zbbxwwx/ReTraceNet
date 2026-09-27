#!/usr/bin/env bash
set -uo pipefail
ROOT=/group4/zouwenxuan/iml_generalization
REPO=$ROOT/repos/ReTraceNet
PY=$ROOT/envs/mesorch/bin/python
SPLIT=$ROOT/outputs/retracenet_v2/mit_full_offline_20260918/casia2_split.json
TEMPLATE=$ROOT/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt
LIGHT=$ROOT/outputs/retracenet_v2/casia2_integrated_frequency_scratch30_20260924
HEAVY=$ROOT/outputs/retracenet_v2/casia2_native_jpeg_hrnet_w32_80_20260924
QUEUE=$ROOT/outputs/retracenet_v2/native_jpeg_compare_queue_20260924
mkdir -p "$QUEUE"
exec >>"$QUEUE/queue.log" 2>&1
echo "[$(date -Is)] queue_start"
while kill -0 1307060 2>/dev/null; do sleep 30; done
echo "[$(date -Is)] light30_finished"
# Extend the light architecture to the same 80-epoch budget before comparison.
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$REPO/tools/run_casia2_integrated_frequency_scratch.py"   --dataset CASIA2 --split "$SPLIT" --template "$TEMPLATE" --out "$LIGHT"   --epochs 80 --batch-size 4 --learning-rate 0.001 --seed 3407 --resume   >"$LIGHT/resume_to80.log" 2>&1
rc=$?
echo "[$(date -Is)] light80_rc=$rc"
if [ $rc -ne 0 ]; then echo light80_failed >"$QUEUE/FAILED"; exit $rc; fi
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$REPO/tools/run_casia2_native_jpeg_hrnet.py"   --split "$SPLIT" --template "$TEMPLATE" --out "$HEAVY" --smoke-only   >"$HEAVY.smoke.log" 2>&1
rc=$?
echo "[$(date -Is)] heavy_smoke_rc=$rc"
if [ $rc -ne 0 ]; then echo heavy_smoke_failed >"$QUEUE/FAILED"; exit $rc; fi
mkdir -p "$HEAVY"
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$REPO/tools/run_casia2_native_jpeg_hrnet.py"   --dataset CASIA2 --split "$SPLIT" --template "$TEMPLATE" --out "$HEAVY"   --epochs 80 --batch-size 4 --learning-rate 0.001 --seed 3407   >"$HEAVY/train.log" 2>&1
rc=$?
echo "[$(date -Is)] heavy80_rc=$rc"
if [ $rc -ne 0 ]; then echo heavy80_failed >"$QUEUE/FAILED"; exit $rc; fi
"$PY" - "$LIGHT" "$HEAVY" "$QUEUE/winner.yaml" <<'PY'
import sys,yaml,pathlib
light,heavy,out=map(pathlib.Path,sys.argv[1:])
def load(p):
 d=yaml.safe_load((p/'summary.yaml').read_text())
 return float(d['best_F1']),int(d['best_epoch'])
lf,le=load(light); hf,he=load(heavy)
winner='light' if lf>=hf else 'heavy'
payload={'light':{'best_f1':lf,'best_epoch':le,'out':str(light)},'heavy':{'best_f1':hf,'best_epoch':he,'out':str(heavy)},'winner':winner,'winner_out':str(light if winner=='light' else heavy)}
out.write_text(yaml.safe_dump(payload,sort_keys=False))
print(payload)
PY
echo "[$(date -Is)] compare_complete"
