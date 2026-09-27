#!/usr/bin/env python
"""Independent runner for DHER-v2 Clean and Performance Total experiments."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from overnight_final_total_runner import (
    B0_AUC, B0_CKPT, B0_FIXED, B0_IOU, B0_LT05_F1, B0_LT05_RECALL,
    B0_LT1_F1, B0_LT1_RECALL, B0_ORACLE, B0_PRECISION, B0_RECALL,
    PY, ROOT, base_cfg, best_row, gpu_env, md_table, model_row, read_diag,
    read_size, row_float, run, write_yaml,
)

PARENT_ROOT = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/final_overnight_20260829_143051')
QTR_DIR = PARENT_ROOT / 'qtr_hcv'
QTR_CKPT = QTR_DIR / 'best.pt'
QTR_CFG = PARENT_ROOT / 'configs' / 'qtr_hcv.yaml'
HCV_DIR = PARENT_ROOT / 'hcv2_calibrated'
HCV_CKPT = HCV_DIR / 'best.pt'
HCV_CFG = PARENT_ROOT / 'configs' / 'hcv2_calibrated.yaml'
OUT_BASE = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')


def now(): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def log(msg): print(f'[{now()}] {msg}', flush=True)

def launch(cmd, log_path: Path, env=None):
    log(f'LAUNCH {" ".join(str(x) for x in cmd)}')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open('w', encoding='utf-8')
    p = subprocess.Popen([str(x) for x in cmd], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    return p, f

def wait_proc(name, proc, f):
    rc = proc.wait(); f.close(); log(f'{name} EXIT rc={rc}')
    return rc

def best_size(outdir: Path, bucket: str, key: str) -> float:
    br = best_row(outdir)
    if not br: return 0.0
    size = read_size(outdir, int(br['epoch']))
    return float(size.get(bucket, {}).get(key, 0.0))

def hcv_sep(outdir: Path, scale: str) -> float:
    br=best_row(outdir)
    if not br: return 0.0
    diag=read_diag(outdir, int(br['epoch']))
    row=next((r for r in diag if r.get('scale')==scale and r.get('bucket')=='lt_1'), None)
    return float(row.get('sep_delta', 0.0)) if row else 0.0

def pass_clean(outdir: Path):
    br=best_row(outdir)
    if not br: return 'FAIL', {'reason':'no_history'}
    oracle=row_float(br,'val_f1_oracle'); recall=row_float(br,'val_recall_fixed')
    tiny05=best_size(outdir,'lt_0p5','f1')
    ok = oracle >= 0.748 or (oracle >= 0.746 and tiny05 >= 0.315 and recall >= 0.710)
    return ('PASS' if ok else 'FAIL'), {'oracle':oracle,'recall':recall,'lt_0p5_f1':tiny05,'rule':'oracle>=0.748 OR oracle>=0.746 and lt0.5F1>=0.315 and recall>=0.710'}

def pass_perf(outdir: Path):
    br=best_row(outdir)
    if not br: return 'FAIL', {'reason':'no_history'}
    oracle=row_float(br,'val_f1_oracle'); recall=row_float(br,'val_recall_fixed')
    tiny05_r=best_size(outdir,'lt_0p5','recall'); tiny1_r=best_size(outdir,'lt_1','recall')
    ok = oracle >= 0.755 and recall >= 0.73
    return ('PASS' if ok else 'FAIL'), {'oracle':oracle,'recall':recall,'lt_0p5_recall':tiny05_r,'lt_1_recall':tiny1_r,'rule':'oracle>=0.755 and recall>=0.73'}

def make_report(root: Path, clean_dir: Path, perf_dir: Path, perf_res: int, decisions: dict):
    rows=[{'Model':'B0','Resolution':512,'Innovation modules':'none','Fixed F1':B0_FIXED,'Oracle F1':B0_ORACLE,'Precision':B0_PRECISION,'Recall':B0_RECALL,'IoU':B0_IOU,'AUC':B0_AUC,'<0.5% F1':B0_LT05_F1,'<0.5% Recall':B0_LT05_RECALL,'<1% F1':B0_LT1_F1,'<1% Recall':B0_LT1_RECALL,'Epoch':2,'Checkpoint':str(B0_CKPT)}]
    rows.append(model_row('QTR+HCV parent',512,'QTR+HCV frozen evidence generator',QTR_DIR))
    rows.append(model_row('Clean DHER-v2',512,'QTR+HCV frozen + DHER-v2 + decoder stage1/head',clean_dir))
    rows.append(model_row('Performance Total DHER-v2',perf_res,'DHER-v2 + high-res/tiny-sampling/Tversky recipe; QTR/HCV frozen',perf_dir))
    numeric=[r for r in rows if isinstance(r.get('Oracle F1'),float)]
    best=max(numeric,key=lambda r:r['Oracle F1'])
    best_recall=max(numeric,key=lambda r:r['Recall'])
    best_tiny=max(numeric,key=lambda r:r.get('<1% F1',0.0))
    sep_rows=[]
    for name,out in [('QTR+HCV parent',QTR_DIR),('Clean DHER-v2',clean_dir),('Performance Total DHER-v2',perf_dir)]:
        sep_rows.append({'Model':name,'s1 <1% Sep_Delta':hcv_sep(out,'s1'),'s2 <1% Sep_Delta':hcv_sep(out,'s2')})
    report=[]
    report.append('# DHER-v2 A/B Validation Report\n')
    report.append(f'Run root: `{root}`')
    report.append('Selection/evaluation: validation only; no test run. QTR/HCV are frozen evidence generators for DHER-v2.\n')
    report.append(md_table(rows,['Model','Resolution','Innovation modules','Fixed F1','Oracle F1','Precision','Recall','IoU','AUC','<0.5% F1','<0.5% Recall','<1% F1','<1% Recall','Epoch','Checkpoint']))
    report.append('\n## HCV Evidence Preservation')
    report.append(md_table(sep_rows,['Model','s1 <1% Sep_Delta','s2 <1% Sep_Delta']))
    report.append('\n## Decisions')
    for k,v in decisions.items(): report.append(f'- {k}: **{v[0]}** `{json.dumps(v[1], sort_keys=True)}`')
    report.append('\n## Required Summary')
    report.append(f'- Best overall val F1: **{best["Oracle F1"]:.6f}** from `{best["Model"]}`.')
    report.append(f'- Best recall: **{best_recall["Recall"]:.6f}** from `{best_recall["Model"]}`.')
    report.append(f'- Best tiny F1 (<1%): **{best_tiny.get("<1% F1",0.0):.6f}** from `{best_tiny["Model"]}`.')
    report.append(f'- Clean DHER-v2 checkpoint: `{clean_dir / "best.pt"}`.')
    report.append(f'- Performance Total checkpoint: `{perf_dir / "best.pt"}`.')
    report.append('- Excluded from this run: old DHER joint training, HCV retraining, FGTP, HDRP, CBR, direct logits, new attention, new backbone.')
    md='\n'.join(report)+'\n'
    (root/'DHER_V2_AB_REPORT.md').write_text(md,encoding='utf-8')
    docs=ROOT/'docs'; docs.mkdir(exist_ok=True); (docs/'DHER_V2_AB_REPORT.md').write_text(md,encoding='utf-8')
    log(f'report written: {root/"DHER_V2_AB_REPORT.md"}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu-a',default='0'); ap.add_argument('--gpu-b',default='1'); args=ap.parse_args()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    root=OUT_BASE/f'dher_v2_ab_{stamp}'; root.mkdir(parents=True,exist_ok=True)
    (OUT_BASE/'CURRENT_DHER_V2_AB.txt').write_text(str(root),encoding='utf-8')
    logs=root/'logs'; checks=root/'checks'; cfgdir=root/'configs'; logs.mkdir(parents=True,exist_ok=True); checks.mkdir(exist_ok=True); cfgdir.mkdir(exist_ok=True)
    parent_ckpt = QTR_CKPT if QTR_CKPT.exists() else HCV_CKPT
    parent_cfg = QTR_CFG if QTR_CKPT.exists() else HCV_CFG
    if not parent_ckpt.exists(): raise FileNotFoundError(parent_ckpt)
    arch='retracenet_b3_qtr_hcv2_dher_v2' if QTR_CKPT.exists() else 'retracenet_b3_hcv2_dher_v2'
    log(f'run root={root} parent_ckpt={parent_ckpt} arch={arch}')
    run([PY,'-m','py_compile','retracenet/final_modules.py','train_v2_final_pipeline.py','tools/check_final_pipeline.py'], logs/'py_compile.log')

    clean_dir=root/'clean_dher_v2'
    clean_cfg=base_cfg(arch, clean_dir, 4, image_size=512, init=str(parent_ckpt), lr=2e-5)
    clean_cfg['training'].update({'constant_lr_epochs':1,'min_lr_ratio':0.1,'early_stopping':5,'diagnostic_batches':512})
    clean_cfg['training']['parameter_groups']=[{'patterns':['dher.'],'lr':2e-5},{'patterns':['decoder.stage1'],'lr':5e-6},{'patterns':['mask_head'],'lr':5e-6}]
    clean_cfg['dher_v2_loss']={'support':0.10,'refutation':0.05}
    clean_cfg_path=cfgdir/'clean_dher_v2.yaml'; write_yaml(clean_cfg_path,clean_cfg)

    perf_res=640
    perf_dir=root/'performance_total_640'
    perf_cfg=base_cfg(arch, perf_dir, 6, image_size=perf_res, init=str(parent_ckpt), lr=1e-5)
    perf_cfg['data']['tiny_stratified_sampling']=True
    perf_cfg['data']['tiny_sampling_bucket_masses']={'lt_1':0.35,'1_3':0.25,'3_10':0.25,'gte_10':0.15,'authentic':0.02}
    perf_cfg['loss']['weights']['tversky']=0.20
    perf_cfg['loss']['tversky_alpha_fp']=0.3; perf_cfg['loss']['tversky_beta_fn']=0.7
    perf_cfg['training'].update({'constant_lr_epochs':1,'min_lr_ratio':0.1,'early_stopping':3,'meaningful_delta':0.0005,'diagnostic_batches':512})
    perf_cfg['training']['parameter_groups']=[{'patterns':['content_encoder'],'lr':1e-6},{'patterns':['forensic_stream'],'lr':2e-6},{'patterns':['dher.'],'lr':1e-5},{'patterns':['decoder'],'lr':3e-6},{'patterns':['mask_head','log_var_head','aux_heads','image_head'],'lr':3e-6}]
    perf_cfg['dher_v2_loss']={'support':0.10,'refutation':0.05}
    perf_cfg_path=cfgdir/'performance_total.yaml'; write_yaml(perf_cfg_path,perf_cfg)

    run([PY,'tools/check_final_pipeline.py','--parent-config',parent_cfg,'--child-config',clean_cfg_path,'--checkpoint',parent_ckpt,'--phase','dher_v2','--device','cuda:0','--output',checks/'clean_preflight.json'], logs/'clean_preflight.log', env=gpu_env(args.gpu_a))
    rc=run([PY,'tools/check_final_pipeline.py','--parent-config',parent_cfg,'--child-config',perf_cfg_path,'--checkpoint',parent_ckpt,'--phase','total_v2','--device','cuda:0','--output',checks/'perf_640_preflight.json'], logs/'perf_640_preflight.log', env=gpu_env(args.gpu_b), allow_fail=True)
    if rc!=0:
        txt=(logs/'perf_640_preflight.log').read_text(errors='ignore') if (logs/'perf_640_preflight.log').exists() else ''
        if 'out of memory' not in txt.lower() and 'cuda' not in txt.lower(): raise RuntimeError('Performance 640 preflight failed for non-CUDA reason')
        perf_res=576; perf_dir=root/'performance_total_576'; perf_cfg['data']['image_size']=576; perf_cfg['output_dir']=str(perf_dir); write_yaml(perf_cfg_path,perf_cfg)
        run([PY,'tools/check_final_pipeline.py','--parent-config',parent_cfg,'--child-config',perf_cfg_path,'--checkpoint',parent_ckpt,'--phase','total_v2','--device','cuda:0','--output',checks/'perf_576_preflight.json'], logs/'perf_576_preflight.log', env=gpu_env(args.gpu_b))

    (root/'runner_status.json').write_text(json.dumps({'stage':'training','updated':now(),'gpu_a':args.gpu_a,'gpu_b':args.gpu_b,'perf_res':perf_res,'parent_ckpt':str(parent_ckpt)},indent=2),encoding='utf-8')
    pa,fa=launch([PY,'train_v2_final_pipeline.py','--config',clean_cfg_path,'--phase','dher_v2','--device','cuda:0'], logs/'clean_train.log', env=gpu_env(args.gpu_a))
    pb,fb=launch([PY,'train_v2_final_pipeline.py','--config',perf_cfg_path,'--phase','total_v2','--device','cuda:0'], logs/f'performance_{perf_res}_train.log', env=gpu_env(args.gpu_b))
    rb=wait_proc('Performance Total',pb,fb)
    if rb!=0 and perf_res==640:
        txt=(logs/'performance_640_train.log').read_text(errors='ignore') if (logs/'performance_640_train.log').exists() else ''
        if 'out of memory' in txt.lower() or 'cuda' in txt.lower():
            perf_res=576; perf_dir=root/'performance_total_576'; perf_cfg['data']['image_size']=576; perf_cfg['output_dir']=str(perf_dir); write_yaml(perf_cfg_path,perf_cfg)
            run([PY,'tools/check_final_pipeline.py','--parent-config',parent_cfg,'--child-config',perf_cfg_path,'--checkpoint',parent_ckpt,'--phase','total_v2','--device','cuda:0','--output',checks/'perf_576_preflight_after_oom.json'], logs/'perf_576_preflight_after_oom.log', env=gpu_env(args.gpu_b))
            pb,fb=launch([PY,'train_v2_final_pipeline.py','--config',perf_cfg_path,'--phase','total_v2','--device','cuda:0'], logs/'performance_576_train.log', env=gpu_env(args.gpu_b))
            rb=wait_proc('Performance Total 576',pb,fb)
        if rb!=0: raise RuntimeError('Performance Total failed')
    elif rb!=0:
        raise RuntimeError('Performance Total failed')
    ra=wait_proc('Clean DHER-v2',pa,fa)
    if ra!=0: raise RuntimeError('Clean DHER-v2 failed')
    decisions={'Clean DHER-v2':pass_clean(clean_dir),'Performance Total':pass_perf(perf_dir)}
    make_report(root,clean_dir,perf_dir,perf_res,decisions)
    (root/'runner_status.json').write_text(json.dumps({'stage':'complete','updated':now(),'gpu_a':args.gpu_a,'gpu_b':args.gpu_b,'perf_res':perf_res,'decisions':decisions,'report':str(root/'DHER_V2_AB_REPORT.md')},indent=2),encoding='utf-8')

if __name__=='__main__': main()
