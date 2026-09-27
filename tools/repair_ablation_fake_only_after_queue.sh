#!/usr/bin/env bash
set -uo pipefail
R=/group4/zouwenxuan/iml_generalization
REPO=$R/repos/ReTraceNet
PY=$R/envs/mesorch/bin/python
SPLIT=$R/outputs/retracenet_v2/v3_offline_master_20260912/indomain_IMD2020/split_721.json
ROOT=$R/outputs/retracenet_v2/imd_final_qtr_hcv_dher_ablation_20260926
while kill -0 1421296 2>/dev/null; do sleep 30; done
cd "$REPO"
for name in qtr hcv; do
  out="$ROOT/without_$name"
  CUDA_VISIBLE_DEVICES=1 "$PY" -u tools/eval_final_module_ablation_1024.py \
    --dataset IMD2020 --checkpoint "$out/best.pt" --split "$SPLIT" \
    --out "$out/test_1024_768" --ablation "$name" >"$out/test_1024_768.log" 2>&1
done
"$PY" - <<'PY'
from pathlib import Path
import yaml
root=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/imd_final_qtr_hcv_dher_ablation_20260926')
rows=[]
for name in ('qtr','hcv','dher'):
    p=root/f'without_{name}'/'test_1024_768'/'result.yaml'
    if not p.exists(): rows.append({'ablation':f'w/o {name.upper()}','status':'missing'}); continue
    x=yaml.safe_load(p.read_text()); a=x['all_images_pooled']; f=x['fake_only_pooled']
    rows.append({'ablation':f'w/o {name.upper()}','status':'completed','checkpoint_epoch':x['checkpoint_epoch'],
      'fake_only_pixel_auc':f['auc'],'fake_only_pixel_f1_best':f['f1_best']['f1'],'fake_only_pixel_f1_at_0p5':f['f1_at_0p5']['f1'],
      'fake_only_f1_best_precision':f['f1_best']['precision'],'fake_only_f1_best_recall':f['f1_best']['recall'],
      'all_images_pixel_auc':a['auc'],'all_images_pixel_f1_best':a['f1_best']['f1'],'all_images_pixel_f1_at_0p5':a['f1_at_0p5']['f1'],
      'all_images_f1_best_precision':a['f1_best']['precision'],'all_images_f1_best_recall':a['f1_best']['recall']})
(root/'final_results.yaml').write_text(yaml.safe_dump({'status':'completed','inference':'1024/768','rows':rows},sort_keys=False))
PY
