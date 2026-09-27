#!/usr/bin/env python
"""Independent runner for DHER-v2.1 clean and performance experiments."""
from __future__ import annotations

import argparse, csv, json, subprocess
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
OUT_BASE = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')


def now(): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def log(msg): print(f'[{now()}] {msg}', flush=True)

def launch(cmd, log_path: Path, env=None):
    log(f'LAUNCH {" ".join(str(x) for x in cmd)}')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f=log_path.open('w',encoding='utf-8')
    p=subprocess.Popen([str(x) for x in cmd], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    return p,f

def wait_proc(name, proc, f):
    rc=proc.wait(); f.close(); log(f'{name} EXIT rc={rc}'); return rc

def best_size(outdir: Path, bucket: str, key: str) -> float:
    br=best_row(outdir)
    if not br: return 0.0
    size=read_size(outdir,int(br['epoch']))
    return float(size.get(bucket,{}).get(key,0.0))

def hcv_sep(outdir: Path, scale: str) -> float:
    br=best_row(outdir)
    if not br: return 0.0
    diag=read_diag(outdir,int(br['epoch']))
    row=next((r for r in diag if r.get('scale')==scale and r.get('bucket')=='lt_1'),None)
    return float(row.get('sep_delta',0.0)) if row else 0.0

def dher_gate(outdir: Path, key: str) -> float:
    br=best_row(outdir)
    if not br: return 0.0
    p=outdir/f'dher_v2_diag_epoch{int(br["epoch"]):03d}.yaml'
    if not p.exists(): return 0.0
    rows=yaml.safe_load(p.read_text(encoding='utf-8')) or []
    g=next((r for r in rows if r.get('scale')=='gates'),{})
    return float(g.get(key,0.0))

def pass_clean(outdir: Path):
    br=best_row(outdir)
    if not br: return 'FAIL', {'reason':'no_history'}
    oracle=row_float(br,'val_f1_oracle'); recall=row_float(br,'val_recall_fixed')
    tiny05=best_size(outdir,'lt_0p5','f1'); tiny1=best_size(outdir,'lt_1','f1')
    conflict=dher_gate(outdir,'conflict_gate_mean')
    precision=row_float(br,'val_precision_fixed')
    ok = (oracle >= 0.748 and recall >= 0.715) or (oracle >= 0.746 and tiny05 >= 0.315 and tiny1 >= 0.465 and recall >= 0.705)
    collapse = recall < 0.705 or conflict > 0.6
    return ('PASS' if ok and not collapse else 'FAIL'), {'oracle':oracle,'precision':precision,'recall':recall,'lt_0p5_f1':tiny05,'lt_1_f1':tiny1,'conflict_gate_mean':conflict,'conflict_gate_gt_0p6':conflict>0.6}

def pass_perf(outdir: Path):
    br=best_row(outdir)
    if not br: return 'FAIL', {'reason':'no_history'}
    oracle=row_float(br,'val_f1_oracle'); recall=row_float(br,'val_recall_fixed')
    conflict=dher_gate(outdir,'conflict_gate_mean')
    return ('PASS' if oracle>=0.755 and recall>=0.73 else 'FAIL'), {'oracle':oracle,'recall':recall,'lt_0p5_recall':best_size(outdir,'lt_0p5','recall'),'lt_1_recall':best_size(outdir,'lt_1','recall'),'conflict_gate_mean':conflict,'conflict_gate_gt_0p6':conflict>0.6}

def make_report(root: Path, clean_dir: Path, perf_dir: Path, perf_res: int, decisions: dict):
    rows=[{'Model':'B0','Resolution':512,'Innovation modules':'none','Fixed F1':B0_FIXED,'Oracle F1':B0_ORACLE,'Precision':B0_PRECISION,'Recall':B0_RECALL,'IoU':B0_IOU,'AUC':B0_AUC,'<0.5% F1':B0_LT05_F1,'<0.5% Recall':B0_LT05_RECALL,'<1% F1':B0_LT1_F1,'<1% Recall':B0_LT1_RECALL,'Epoch':2,'Checkpoint':str(B0_CKPT)}]
    rows.append(model_row('QTR+HCV parent',512,'frozen QTR + calibrated HCV evidence generator',QTR_DIR))
    rows.append(model_row('Clean DHER-v2.1',512,'sparse conflict + missed-support routing',clean_dir))
    rows.append(model_row('Performance Total DHER-v2.1',perf_res,'DHER-v2.1 + tiny sampling + Tversky; QTR/HCV/content/forensic frozen',perf_dir))
    numeric=[r for r in rows if isinstance(r.get('Oracle F1'),float)]
    best=max(numeric,key=lambda r:r['Oracle F1']); best_recall=max(numeric,key=lambda r:r['Recall']); best_tiny=max(numeric,key=lambda r:r.get('<1% F1',0.0))
    diag_rows=[]
    for name,out in [('QTR+HCV parent',QTR_DIR),('Clean DHER-v2.1',clean_dir),('Performance Total DHER-v2.1',perf_dir)]:
        diag_rows.append({'Model':name,'s1 <1% Sep_Delta':hcv_sep(out,'s1'),'s2 <1% Sep_Delta':hcv_sep(out,'s2'),'G_support mean':dher_gate(out,'support_gate_mean'),'G_support fg':dher_gate(out,'support_gate_fg_mean'),'G_support bg':dher_gate(out,'support_gate_bg_mean'),'G_conflict mean':dher_gate(out,'conflict_gate_mean'),'G_conflict fg':dher_gate(out,'conflict_gate_fg_mean'),'G_conflict bg':dher_gate(out,'conflict_gate_bg_mean')})
    report=['# DHER-v2.1 A/B Validation Report\n',f'Run root: `{root}`','Selection/evaluation: validation only; no test run. Parent is QTR + calibrated HCV checkpoint; DHER-v2 E0 is not used.\n']
    report.append(md_table(rows,['Model','Resolution','Innovation modules','Fixed F1','Oracle F1','Precision','Recall','IoU','AUC','<0.5% F1','<0.5% Recall','<1% F1','<1% Recall','Epoch','Checkpoint']))
    report.append('\n## Evidence and Routing Diagnostics')
    report.append(md_table(diag_rows,['Model','s1 <1% Sep_Delta','s2 <1% Sep_Delta','G_support mean','G_support fg','G_support bg','G_conflict mean','G_conflict fg','G_conflict bg']))
    report.append('\n## Decisions')
    for k,v in decisions.items(): report.append(f'- {k}: **{v[0]}** `{json.dumps(v[1], sort_keys=True)}`')
    report.append('\n## Required Summary')
    report.append(f'- Best overall val F1: **{best["Oracle F1"]:.6f}** from `{best["Model"]}`.')
    report.append(f'- Best recall: **{best_recall["Recall"]:.6f}** from `{best_recall["Model"]}`.')
    report.append(f'- Best tiny F1 (<1%): **{best_tiny.get("<1% F1",0.0):.6f}** from `{best_tiny["Model"]}`.')
    report.append(f'- Clean checkpoint: `{clean_dir/"best.pt"}`.')
    report.append(f'- Performance checkpoint: `{perf_dir/"best.pt"}`.')
    report.append('- Excluded: old DHER joint training, DHER-v2 E0 continuation, HCV tuning, FGTP, HDRP, CBR, direct logits, new attention, new backbone.')
    md='\n'.join(report)+'\n'
    (root/'DHER_V21_AB_REPORT.md').write_text(md,encoding='utf-8')
    docs=ROOT/'docs'; docs.mkdir(exist_ok=True); (docs/'DHER_V21_AB_REPORT.md').write_text(md,encoding='utf-8')
    log(f'report written: {root/"DHER_V21_AB_REPORT.md"}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu-a',default='0'); ap.add_argument('--gpu-b',default='1'); args=ap.parse_args()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    root=OUT_BASE/f'dher_v21_ab_{stamp}'; root.mkdir(parents=True,exist_ok=True)
    (OUT_BASE/'CURRENT_DHER_V21_AB.txt').write_text(str(root),encoding='utf-8')
    logs=root/'logs'; checks=root/'checks'; cfgdir=root/'configs'; logs.mkdir(parents=True,exist_ok=True); checks.mkdir(exist_ok=True); cfgdir.mkdir(exist_ok=True)
    if not QTR_CKPT.exists(): raise FileNotFoundError(QTR_CKPT)
    arch='retracenet_b3_qtr_hcv2_dher_v21'
    log(f'run root={root} parent_ckpt={QTR_CKPT} arch={arch}')
    run([PY,'-m','py_compile','retracenet/final_modules.py','train_v2_final_pipeline.py','tools/check_final_pipeline.py'], logs/'py_compile.log')
    clean_dir=root/'clean_dher_v21'
    clean_cfg=base_cfg(arch,clean_dir,3,image_size=512,init=str(QTR_CKPT),lr=2e-5)
    clean_cfg['training'].update({'constant_lr_epochs':1,'min_lr_ratio':0.1,'early_stopping':4,'diagnostic_batches':512})
    clean_cfg['training']['parameter_groups']=[{'patterns':['dher.'],'lr':2e-5},{'patterns':['decoder.stage1'],'lr':5e-6},{'patterns':['mask_head'],'lr':5e-6}]
    clean_cfg['dher_v2_loss']={'support':0.10,'refutation':0.05}
    clean_cfg_path=cfgdir/'clean_dher_v21.yaml'; write_yaml(clean_cfg_path,clean_cfg)
    perf_res=640; perf_dir=root/'performance_total_640'
    perf_cfg=base_cfg(arch,perf_dir,4,image_size=perf_res,init=str(QTR_CKPT),lr=1e-5)
    perf_cfg['data']['tiny_stratified_sampling']=True
    perf_cfg['data']['tiny_sampling_bucket_masses']={'lt_1':0.35,'1_3':0.25,'3_10':0.25,'gte_10':0.15,'authentic':0.02}
    perf_cfg['loss']['weights']['tversky']=0.20
    perf_cfg['loss']['tversky_alpha_fp']=0.3; perf_cfg['loss']['tversky_beta_fn']=0.7
    perf_cfg['training'].update({'constant_lr_epochs':1,'min_lr_ratio':0.1,'early_stopping':3,'meaningful_delta':0.0005,'diagnostic_batches':512})
    perf_cfg['training']['parameter_groups']=[{'patterns':['dher.'],'lr':1e-5},{'patterns':['decoder'],'lr':3e-6},{'patterns':['mask_head','log_var_head','aux_heads','image_head'],'lr':3e-6}]
    perf_cfg['dher_v2_loss']={'support':0.10,'refutation':0.05}
    perf_cfg_path=cfgdir/'performance_total.yaml'; write_yaml(perf_cfg_path,perf_cfg)
    run([PY,'tools/check_final_pipeline.py','--parent-config',QTR_CFG,'--child-config',clean_cfg_path,'--checkpoint',QTR_CKPT,'--phase','dher_v21','--device','cuda:0','--output',checks/'clean_preflight.json'], logs/'clean_preflight.log', env=gpu_env(args.gpu_a))
    rc=run([PY,'tools/check_final_pipeline.py','--parent-config',QTR_CFG,'--child-config',perf_cfg_path,'--checkpoint',QTR_CKPT,'--phase','total_v21','--device','cuda:0','--output',checks/'perf_640_preflight.json'], logs/'perf_640_preflight.log', env=gpu_env(args.gpu_b), allow_fail=True)
    if rc!=0:
        txt=(logs/'perf_640_preflight.log').read_text(errors='ignore') if (logs/'perf_640_preflight.log').exists() else ''
        if 'out of memory' not in txt.lower() and 'cuda' not in txt.lower(): raise RuntimeError('Performance 640 preflight failed for non-CUDA reason')
        perf_res=576; perf_dir=root/'performance_total_576'; perf_cfg['data']['image_size']=576; perf_cfg['output_dir']=str(perf_dir); write_yaml(perf_cfg_path,perf_cfg)
        run([PY,'tools/check_final_pipeline.py','--parent-config',QTR_CFG,'--child-config',perf_cfg_path,'--checkpoint',QTR_CKPT,'--phase','total_v21','--device','cuda:0','--output',checks/'perf_576_preflight.json'], logs/'perf_576_preflight.log', env=gpu_env(args.gpu_b))
    (root/'runner_status.json').write_text(json.dumps({'stage':'training','updated':now(),'gpu_a':args.gpu_a,'gpu_b':args.gpu_b,'perf_res':perf_res,'parent_ckpt':str(QTR_CKPT)},indent=2),encoding='utf-8')
    pa,fa=launch([PY,'train_v2_final_pipeline.py','--config',clean_cfg_path,'--phase','dher_v21','--device','cuda:0'],logs/'clean_train.log',env=gpu_env(args.gpu_a))
    pb,fb=launch([PY,'train_v2_final_pipeline.py','--config',perf_cfg_path,'--phase','total_v21','--device','cuda:0'],logs/f'performance_{perf_res}_train.log',env=gpu_env(args.gpu_b))
    rb=wait_proc('Performance Total v2.1',pb,fb)
    if rb!=0 and perf_res==640:
        txt=(logs/'performance_640_train.log').read_text(errors='ignore') if (logs/'performance_640_train.log').exists() else ''
        if 'out of memory' in txt.lower() or 'cuda' in txt.lower():
            perf_res=576; perf_dir=root/'performance_total_576'; perf_cfg['data']['image_size']=576; perf_cfg['output_dir']=str(perf_dir); write_yaml(perf_cfg_path,perf_cfg)
            run([PY,'tools/check_final_pipeline.py','--parent-config',QTR_CFG,'--child-config',perf_cfg_path,'--checkpoint',QTR_CKPT,'--phase','total_v21','--device','cuda:0','--output',checks/'perf_576_preflight_after_oom.json'], logs/'perf_576_preflight_after_oom.log', env=gpu_env(args.gpu_b))
            pb,fb=launch([PY,'train_v2_final_pipeline.py','--config',perf_cfg_path,'--phase','total_v21','--device','cuda:0'],logs/'performance_576_train.log',env=gpu_env(args.gpu_b)); rb=wait_proc('Performance Total 576 v2.1',pb,fb)
    if rb!=0: raise RuntimeError('Performance Total failed')
    ra=wait_proc('Clean DHER-v2.1',pa,fa)
    if ra!=0: raise RuntimeError('Clean DHER-v2.1 failed')
    decisions={'Clean DHER-v2.1':pass_clean(clean_dir),'Performance Total DHER-v2.1':pass_perf(perf_dir)}
    make_report(root,clean_dir,perf_dir,perf_res,decisions)
    (root/'runner_status.json').write_text(json.dumps({'stage':'complete','updated':now(),'gpu_a':args.gpu_a,'gpu_b':args.gpu_b,'perf_res':perf_res,'decisions':decisions,'report':str(root/'DHER_V21_AB_REPORT.md')},indent=2),encoding='utf-8')

if __name__=='__main__': main()
