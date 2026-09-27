#!/usr/bin/env python
"""Performance sprint from locked Performance-640 E2 best."""
from __future__ import annotations
import argparse, copy, csv, json, shutil, subprocess
from datetime import datetime
from pathlib import Path
import torch
import yaml
from overnight_final_total_runner import PY, ROOT, gpu_env, md_table, model_row, read_size, row_float, run, write_yaml

OUT_BASE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')
BASE_DIR=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/dher_v21_ab_20260830_071856/performance_total_640')
BASE_CKPT=BASE_DIR/'best.pt'
BASE_CFG=BASE_DIR/'config.yaml'
BASE_ORACLE=0.7537580531653538
BASE_FIXED=0.7537256303649689
BASE_PREC=0.8234284756997015
BASE_REC=0.6949024541972082
BASE_IOU=0.6047830628063833
BASE_AUC=0.922398002261387
BASE_LT05_F1=0.420621996796583
BASE_LT05_REC=0.5442894770931387
BASE_LT1_F1=0.5773127467338568
BASE_LT1_REC=0.6236170641557572
BASE_SHA='3259f97eb8fd7ad8e6129344e5697e67357fcd8d713c654ed34dc76c902d0318'

def now(): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def log(msg): print(f'[{now()}] {msg}', flush=True)

def read_history(outdir: Path):
    p=outdir/'history.csv'
    if not p.exists(): return []
    with p.open(encoding='utf-8') as f: return list(csv.DictReader(f))

def best_row(outdir: Path):
    rows=read_history(outdir)
    if not rows: return None
    return max(rows,key=lambda r: float(r.get('val_f1_oracle') or -1))

def make_cfg(outdir: Path, res: int, epochs: int, init: Path):
    cfg=yaml.safe_load(BASE_CFG.read_text(encoding='utf-8'))
    cfg=copy.deepcopy(cfg)
    cfg['experiment']=outdir.name; cfg['output_dir']=str(outdir)
    cfg['data']['image_size']=res
    cfg['data']['tiny_stratified_sampling']=True
    cfg['data']['tiny_sampling_bucket_masses']={'lt_1':0.25,'1_3':0.25,'3_10':0.30,'gte_10':0.20,'authentic':0.02}
    cfg['training']['init_checkpoint']=str(init)
    cfg['training']['epochs']=epochs
    cfg['training']['constant_lr_epochs']=1
    cfg['training']['min_lr_ratio']=0.1
    cfg['training']['early_stopping']=epochs+1
    cfg['training']['meaningful_delta']=0.0005
    cfg['training']['ema']=True
    cfg['training']['ema_decay']=0.999
    cfg['training']['deep_supervision_weights']=[0.30,0.20,0.10]
    cfg['training']['selection_metric']='pixel_f1_oracle'
    cfg['training']['parameter_groups']=[
        {'patterns':['content_encoder.layer4'],'lr':5e-7},
        {'patterns':['forensic_stream.stage3','forensic_stream.stage4'],'lr':1e-6},
        {'patterns':['dher.'],'lr':3e-6},
        {'patterns':['decoder'],'lr':3e-6},
        {'patterns':['mask_head','aux_heads'],'lr':5e-6},
        {'patterns':['log_var_head','image_head'],'lr':5e-6},
    ]
    cfg['loss']['weights']['deep_supervision']=0.0
    cfg['loss']['weights']['tversky']=0.20
    cfg['loss']['tversky_alpha_fp']=0.3; cfg['loss']['tversky_beta_fn']=0.7
    return cfg

def launch(cmd, log_path, env=None):
    log(f'LAUNCH {" ".join(str(x) for x in cmd)}')
    log_path.parent.mkdir(parents=True,exist_ok=True)
    f=log_path.open('w',encoding='utf-8')
    p=subprocess.Popen([str(x) for x in cmd],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    return p,f

def wait_proc(name,p,f):
    rc=p.wait(); f.close(); log(f'{name} EXIT rc={rc}'); return rc

def oomish(log_path):
    txt=log_path.read_text(errors='ignore') if log_path.exists() else ''
    lo=txt.lower()
    return 'out of memory' in lo or 'cuda error' in lo or 'cublas' in lo

def train_branch(root, name, res, gpu, epochs=3, init=BASE_CKPT):
    logs=root/'logs'; checks=root/'checks'; cfgdir=root/'configs'
    outdir=root/f'{name}_{res}_e0_2'
    cfg=make_cfg(outdir,res,epochs,init)
    cfg_path=cfgdir/f'{name}_{res}.yaml'; write_yaml(cfg_path,cfg)
    pre=checks/f'{name}_{res}_preflight.json'; prelog=logs/f'{name}_{res}_preflight.log'
    rc=run([PY,'tools/check_final_pipeline.py','--parent-config',BASE_CFG,'--child-config',cfg_path,'--checkpoint',init,'--phase','perf_sprint','--device','cuda:0','--output',pre],prelog,env=gpu_env(gpu),allow_fail=True)
    if rc!=0: return {'name':name,'res':res,'status':'PREFLIGHT_FAIL','oom':oomish(prelog),'outdir':str(outdir),'cfg':str(cfg_path),'log':str(prelog)}
    p,f=launch([PY,'train_v2_final_pipeline.py','--config',cfg_path,'--phase','perf_sprint','--device','cuda:0'],logs/f'{name}_{res}_train.log',env=gpu_env(gpu))
    return {'name':name,'res':res,'status':'RUNNING','proc':p,'file':f,'outdir':str(outdir),'cfg':str(cfg_path),'log':str(logs/f'{name}_{res}_train.log')}

def branch_summary(outdir: Path):
    br=best_row(outdir)
    if not br: return {'status':'NO_HISTORY'}
    ep=int(br['epoch']); size=read_size(outdir,ep)
    def g(b,k): return float(size.get(b,{}).get(k,0.0))
    extra={}
    try: extra=json.loads(br.get('extra') or '{}')
    except Exception: extra={}
    return {'epoch':ep,'oracle':row_float(br,'val_f1_oracle'),'fixed':row_float(br,'val_f1'),'precision':row_float(br,'val_precision_fixed'),'recall':row_float(br,'val_recall_fixed'),'iou':row_float(br,'val_iou_fixed'),'auc':row_float(br,'val_auc'),'lt_0p5_f1':g('lt_0p5','f1'),'lt_0p5_recall':g('lt_0p5','recall'),'lt_1_f1':g('lt_1','f1'),'lt_1_recall':g('lt_1','recall'),'1_3_recall':g('1_3','recall'),'3_10_recall':g('3_10','recall'),'gte_10_recall':g('gte_10','recall'),'eval_variant':extra.get('eval_variant','raw'),'ema_oracle_f1':extra.get('ema_oracle_f1')}

def maybe_continue(root, branch, gpu):
    outdir=Path(branch['outdir']); summ=branch_summary(outdir); oracle=summ.get('oracle',0.0)
    branch['summary']=summ
    branch['continued']=False
    if oracle >= 0.765:
        branch['decision']='LOCK_NEW_PERFORMANCE_BEST'; return branch
    if oracle >= 0.758:
        logs=root/'logs'; cfgdir=root/'configs'
        contdir=root/f"{branch['name']}_{branch['res']}_extra2"
        cfg=make_cfg(contdir,int(branch['res']),2,outdir/'best.pt')
        cfg_path=cfgdir/f"{branch['name']}_{branch['res']}_extra2.yaml"; write_yaml(cfg_path,cfg)
        p,f=launch([PY,'train_v2_final_pipeline.py','--config',cfg_path,'--phase','perf_sprint','--device','cuda:0'],logs/f"{branch['name']}_{branch['res']}_extra2_train.log",env=gpu_env(gpu))
        rc=wait_proc(f"{branch['name']} extra2",p,f)
        if rc!=0: raise RuntimeError(f"extra2 failed for {branch['name']}")
        branch['continued']=True; branch['extra_outdir']=str(contdir); branch['extra_summary']=branch_summary(contdir)
        if branch['extra_summary'].get('oracle',0.0) > oracle:
            branch['best_outdir']=str(contdir); branch['best_summary']=branch['extra_summary']
        else:
            branch['best_outdir']=str(outdir); branch['best_summary']=summ
        branch['decision']='CONTINUED_EXTRA2'
    elif oracle < 0.756:
        branch['decision']='STOP_BELOW_0P756'
    else:
        branch['decision']='STOP_NO_0P758_TRIGGER'
    branch.setdefault('best_outdir',str(outdir)); branch.setdefault('best_summary',summ)
    return branch

def run_tta_if_needed(root, branch, gpu):
    outdir=Path(branch.get('best_outdir',branch['outdir'])); summ=branch.get('best_summary') or branch_summary(outdir)
    if summ.get('oracle',0.0) <= BASE_ORACLE:
        branch['tta']='SKIP_NO_NEW_TRAINING_BEST'; return branch
    cfg=outdir/'config.yaml'; ckpt=outdir/'best.pt'; out=root/'tta'/f"{branch['name']}_{branch['res']}_tta.json"
    run([PY,'tools/eval_tta_v2.py','--config',cfg,'--checkpoint',ckpt,'--device','cuda:0','--output',out],root/'logs'/f"{branch['name']}_{branch['res']}_tta.log",env=gpu_env(gpu))
    branch['tta']=str(out)
    branch['tta_result']=json.loads(out.read_text())
    return branch

def row_for_branch(label, branch):
    s=branch.get('best_summary') or branch.get('summary') or {}
    return {'Branch':label,'Resolution':branch.get('res'),'Decision':branch.get('decision',branch.get('status')),'Epoch':s.get('epoch',''),'Fixed F1':s.get('fixed',''),'Oracle F1':s.get('oracle',''),'Precision':s.get('precision',''),'Recall':s.get('recall',''),'IoU':s.get('iou',''),'AUC':s.get('auc',''),'<0.5% F1':s.get('lt_0p5_f1',''),'<0.5% Recall':s.get('lt_0p5_recall',''),'<1% F1':s.get('lt_1_f1',''),'<1% Recall':s.get('lt_1_recall',''),'EMA/raw':s.get('eval_variant',''),'Checkpoint':str(Path(branch.get('best_outdir',branch.get('outdir','')))/'best.pt')}

def make_report(root, branches):
    rows=[{'Branch':'Locked baseline','Resolution':640,'Decision':'LOCKED_INPUT','Epoch':2,'Fixed F1':BASE_FIXED,'Oracle F1':BASE_ORACLE,'Precision':BASE_PREC,'Recall':BASE_REC,'IoU':BASE_IOU,'AUC':BASE_AUC,'<0.5% F1':BASE_LT05_F1,'<0.5% Recall':BASE_LT05_REC,'<1% F1':BASE_LT1_F1,'<1% Recall':BASE_LT1_REC,'EMA/raw':'raw','Checkpoint':str(BASE_CKPT)}]
    rows += [row_for_branch('GPU0 v2-A',branches['A']), row_for_branch('GPU1 v2-B',branches['B'])]
    numeric=[r for r in rows if isinstance(r.get('Oracle F1'),float)]
    best=max(numeric,key=lambda r:r['Oracle F1'])
    best_recall=max(numeric,key=lambda r:r['Recall'])
    report=['# Performance Sprint Validation Report\n',f'Run root: `{root}`',f'Locked baseline checkpoint: `{BASE_CKPT}`',f'Locked baseline sha256: `{BASE_SHA}`','Selection/evaluation: validation only; no test run.\n']
    report.append(md_table(rows,['Branch','Resolution','Decision','Epoch','Fixed F1','Oracle F1','Precision','Recall','IoU','AUC','<0.5% F1','<0.5% Recall','<1% F1','<1% Recall','EMA/raw','Checkpoint']))
    report.append('\n## TTA')
    for k,b in branches.items():
        report.append(f'- {k}: `{b.get("tta")}`')
        if isinstance(b.get('tta_result'),dict):
            d=b['tta_result'].get('delta',{})
            report.append(f'  - delta fixed `{d.get("pixel_f1_fixed")}`, delta oracle `{d.get("pixel_f1_oracle")}`')
    report.append('\n## Decision')
    report.append(f'- Best validation oracle F1: **{best["Oracle F1"]:.6f}** from `{best["Branch"]}`.')
    report.append(f'- Best validation recall: **{best_recall["Recall"]:.6f}** from `{best_recall["Branch"]}`.')
    if best['Oracle F1'] >= 0.765: report.append('- Status: reached >=0.765, lock as new Performance Best.')
    elif best['Oracle F1'] >= 0.758: report.append('- Status: exceeded 0.758 screening threshold; extra epochs were allowed for that branch.')
    else: report.append('- Status: did not exceed ~0.758; next recommended direction is stronger pretrained content backbone, not more DHER routing.')
    md='\n'.join(report)+'\n'
    (root/'PERFORMANCE_SPRINT_REPORT.md').write_text(md,encoding='utf-8')
    (ROOT/'docs').mkdir(exist_ok=True); (ROOT/'docs'/'PERFORMANCE_SPRINT_REPORT.md').write_text(md,encoding='utf-8')
    log(f'report written {root/"PERFORMANCE_SPRINT_REPORT.md"}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu-a',default='0'); ap.add_argument('--gpu-b',default='1'); args=ap.parse_args()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    root=OUT_BASE/f'performance_sprint_{stamp}'; (root/'logs').mkdir(parents=True,exist_ok=True); (root/'checks').mkdir(exist_ok=True); (root/'configs').mkdir(exist_ok=True)
    (OUT_BASE/'CURRENT_PERFORMANCE_SPRINT.txt').write_text(str(root),encoding='utf-8')
    log(f'root={root} baseline={BASE_CKPT}')
    if not BASE_CKPT.exists(): raise FileNotFoundError(BASE_CKPT)
    actual=subprocess.check_output(['sha256sum',str(BASE_CKPT)],text=True).split()[0]
    if actual != BASE_SHA: raise RuntimeError(f'baseline sha mismatch {actual}')
    run([PY,'-m','py_compile','retracenet/final_modules.py','train_v2_final_pipeline.py','tools/check_final_pipeline.py','tools/eval_tta_v2.py'],root/'logs'/'py_compile.log')
    a=train_branch(root,'A',704,args.gpu_a,3,BASE_CKPT)
    if a['status']=='PREFLIGHT_FAIL' and a.get('oom'):
        log('A 704 preflight OOM/CUDA; fallback once to 672')
        a=train_branch(root,'A',672,args.gpu_a,3,BASE_CKPT)
    if a['status']=='PREFLIGHT_FAIL': raise RuntimeError(f'A preflight failed: {a}')
    b=train_branch(root,'B',640,args.gpu_b,3,BASE_CKPT)
    if b['status']=='PREFLIGHT_FAIL': raise RuntimeError(f'B preflight failed: {b}')
    (root/'runner_status.json').write_text(json.dumps({'stage':'training_initial3','updated':now(),'A':{k:v for k,v in a.items() if k not in ['proc','file']},'B':{k:v for k,v in b.items() if k not in ['proc','file']}},indent=2),encoding='utf-8')
    ra=wait_proc('A initial3',a['proc'],a['file']); rb=wait_proc('B initial3',b['proc'],b['file'])
    if ra!=0 and int(a['res'])==704 and oomish(Path(a['log'])):
        log('A 704 train OOM/CUDA; fallback once to 672')
        a=train_branch(root,'A',672,args.gpu_a,3,BASE_CKPT)
        if a['status']=='PREFLIGHT_FAIL': raise RuntimeError(f'A 672 preflight failed: {a}')
        ra=wait_proc('A 672 initial3',a['proc'],a['file'])
    if ra!=0: raise RuntimeError('A failed')
    if rb!=0: raise RuntimeError('B failed')
    a=maybe_continue(root,a,args.gpu_a); b=maybe_continue(root,b,args.gpu_b)
    a=run_tta_if_needed(root,a,args.gpu_a); b=run_tta_if_needed(root,b,args.gpu_b)
    branches={'A':a,'B':b}
    (root/'runner_status.json').write_text(json.dumps({'stage':'complete','updated':now(),'branches':branches},indent=2,default=str),encoding='utf-8')
    make_report(root,branches)

if __name__=='__main__': main()
