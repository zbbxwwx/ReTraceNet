#!/usr/bin/env bash
set -uo pipefail

R=/group4/zouwenxuan/iml_generalization
REPO=$R/repos/ReTraceNet
PY=$R/envs/mesorch/bin/python
SPLIT=$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_IMD2020/split_721.json
TEMPLATE=$R/outputs/retracenet_v2/imd_tune_coverage_first_2k_20260920/best.pt
ROOT=$R/outputs/retracenet_v2/imd_final_qtr_hcv_dher_ablation_20260926
SAFIRE_QUEUE_PID=1419085
mkdir -p "$ROOT"
exec >>"$ROOT/queue.log" 2>&1

echo "[$(date -Is)] waiting for SAFIRE IMD -> FR queue pid=$SAFIRE_QUEUE_PID"
while kill -0 "$SAFIRE_QUEUE_PID" 2>/dev/null; do sleep 30; done
echo "[$(date -Is)] SAFIRE FR queue complete; starting ReTraceNet ablations"
cd "$REPO"

run_one() {
  local name="$1"
  local out="$ROOT/without_${name}"
  mkdir -p "$out"
  echo "[$(date -Is)] START w/o $name"
  local resume=()
  if [ -f "$out/last.pt" ]; then resume=(--resume); fi
  CUDA_VISIBLE_DEVICES=1 "$PY" -u tools/run_imd_final_module_ablation.py \
    --dataset IMD2020 --split "$SPLIT" --template "$TEMPLATE" \
    --out "$out" --epochs 30 --batch-size 4 --learning-rate 0.001 \
    --seed 3407 --ablation "$name" --skip-test "${resume[@]}" \
    >"$out/train.log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$(date -Is)] first training attempt failed rc=$rc; retrying with --resume"
    CUDA_VISIBLE_DEVICES=1 "$PY" -u tools/run_imd_final_module_ablation.py \
      --dataset IMD2020 --split "$SPLIT" --template "$TEMPLATE" \
      --out "$out" --epochs 30 --batch-size 4 --learning-rate 0.001 \
      --seed 3407 --ablation "$name" --skip-test --resume \
      >>"$out/train.log" 2>&1
    rc=$?
  fi
  if [ "$rc" -ne 0 ] || [ ! -f "$out/best.pt" ]; then
    echo "[$(date -Is)] FAILED w/o $name training rc=$rc"
    return 0
  fi
  echo "[$(date -Is)] EVAL w/o $name at 1024/768"
  CUDA_VISIBLE_DEVICES=1 "$PY" -u tools/eval_final_module_ablation_1024.py \
    --dataset IMD2020 --checkpoint "$out/best.pt" --split "$SPLIT" \
    --out "$out/test_1024_768" --ablation "$name" \
    >"$out/test_1024_768.log" 2>&1
  echo "[$(date -Is)] DONE w/o $name eval_rc=$?"
}

run_one qtr
run_one hcv
run_one dher

"$PY" - <<'PY'
from pathlib import Path
import yaml
root=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/imd_final_qtr_hcv_dher_ablation_20260926')
rows=[]
for name in ('qtr','hcv','dher'):
    p=root/f'without_{name}'/'test_1024_768'/'result.yaml'
    if not p.exists():
        rows.append({'ablation':f'w/o {name.upper()}','status':'missing'})
        continue
    x=yaml.safe_load(p.read_text()); a=x['all_images_pooled']; f=x['fake_only_pooled']
    rows.append({'ablation':f'w/o {name.upper()}','status':'completed','checkpoint_epoch':x['checkpoint_epoch'],
                 'all_images_pixel_auc':a['auc'],'all_images_pixel_f1_best':a['f1_best']['f1'],
                 'all_images_pixel_f1_at_0p5':a['f1_at_0p5']['f1'],
                 'all_images_f1_best_precision':a['f1_best']['precision'],'all_images_f1_best_recall':a['f1_best']['recall'],
                 'fake_only_pixel_auc':f['auc'],'fake_only_pixel_f1_best':f['test_f1_oracle_diagnostic']['f1'],
                 'fake_only_pixel_f1_at_0p5':f['f1_at_0p5']['f1'],
                 'fake_only_f1_best_precision':f['test_f1_oracle_diagnostic']['precision'],
                 'fake_only_f1_best_recall':f['test_f1_oracle_diagnostic']['recall']})
(root/'final_results.yaml').write_text(yaml.safe_dump({'status':'completed','inference':'1024/768','rows':rows},sort_keys=False))
print(rows)
PY
echo "[$(date -Is)] ALL DONE"
