#!/usr/bin/env bash
set -uo pipefail
MAIN=1093493
while kill -0 "$MAIN" 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=0
R=/group4/zouwenxuan/iml_generalization
OUT=$R/outputs/baseline_long_gpu0_70_20260921
LOG=$OUT/finish_requested_20260922.log
PY=$R/envs/mesorch/bin/python
FR=$R/datasets/FantasticReality_v1_localization/split_721_random_stratified.json
TARGET=$OUT/cross_manifests/FantasticReality_test_mesorch.json
"$PY" - "$FR" "$TARGET" <<'PY'
import json,sys
src,dst=sys.argv[1:]
d=json.load(open(src));rows=[[r['image'],r['mask'] if r.get('mask') else 'Negative'] for r in d['test']]
json.dump(rows,open(dst,'w'))
print('converted',len(rows))
PY
F=$OUT/mesorch_imd_cross/FantasticReality.yaml
rm -f "$F"
echo "[$(date -Is)] RETRY Mesorch IMD->FantasticReality" >> "$LOG"
"$PY" -u $R/repos/Mesorch/eval_mesorch_unified.py --checkpoint "$OUT/mesorch_indomain/IMD2020/best.pth" --val "$R/repos/Mesorch/data/unified_721/IMD2020/val.json" --test "$TARGET" --dataset FantasticReality --threshold-source IMD2020_val --out "$F" >> "$LOG" 2>&1
RC=$?
echo "[$(date -Is)] RETRY END rc=$RC Mesorch IMD->FantasticReality" >> "$LOG"
