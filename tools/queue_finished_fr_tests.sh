#!/usr/bin/env bash
set -u
TASK_ROOT=/group4/zouwenxuan/iml_generalization
TASK_PY=$TASK_ROOT/envs/mesorch/bin/python
TASK_TPY=$TASK_ROOT/envs/trufor_from_docker/bin/python
TASK_REPO=$TASK_ROOT/repos/ReTraceNet
TASK_OUT=$TASK_ROOT/outputs/repaired_7models_gpu0_20260925
TASK_DATA=$TASK_ROOT/repos/Mesorch/data/unified_721/FantasticReality_fixed
mkdir -p "$TASK_OUT"
exec >>"$TASK_OUT/test_queue.log" 2>&1
echo "$(date -Is) wait ManTra/main queue"
while kill -0 1350465 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=0
echo "$(date -Is) TruFor manifest evaluation"
"$TASK_TPY" "$TASK_REPO/tools/evaluate_trufor_manifest.py" --val-pred "$TASK_OUT/fr_TruFor/val_pred" --test-pred "$TASK_OUT/fr_TruFor/test_pred" --out "$TASK_OUT/fr_TruFor/test_locked.yaml"
for spec in 'MVSSNet 512' 'PSCC_Net 256' 'Cat_Net 512'; do
  set -- $spec; model=$1; size=$2
  echo "$(date -Is) $model test start"
  "$TASK_PY" "$TASK_REPO/tools/eval_imdlbenco_unified.py" --model "$model" --size "$size" --checkpoint "$TASK_OUT/fr_$model/best.pth" --val "$TASK_DATA/val.json" --test "$TASK_DATA/test.json" --out "$TASK_OUT/fr_$model/test_unified.yaml"
  echo "$(date -Is) $model test rc=$?"
done
echo "$(date -Is) FINISHED_FR_TESTS_COMPLETE"
touch "$TASK_OUT/FINISHED_TESTS_COMPLETE"
