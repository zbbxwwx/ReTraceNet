#!/usr/bin/env python
"""B-only monitor for the performance sprint after stopping A=704."""
from __future__ import annotations
import argparse, csv, json, shutil, subprocess, time
from datetime import datetime
from pathlib import Path
import yaml
from overnight_final_total_runner import PY, ROOT, md_table, read_size, row_float, write_yaml

SPRINT_ROOT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/performance_sprint_20260830_081014')
B_DIR=SPRINT_ROOT/'B_640_e0_2'
B_CFG=SPRINT_ROOT/'configs'/'B_640.yaml'
B_LOG=SPRINT_ROOT/'logs'/'B_640_train.log'
BASE_ORACLE=0.7537580531653538

def now(): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def log(msg): print(f'[{now()}] {msg}', flush=True)
def read_history(outdir=B_DIR):
    p=outdir/'history.csv'
    if not p.exists(): return []
    with p.open(encoding='utf-8') as f: return list(csv.DictReader(f))
def best_row(outdir=B_DIR):
    rows=read_history(outdir)
    if not rows: return None
    return max(rows,key=lambda r: float(r.get('val_f1_oracle') or -1))
def proc_alive():
    cmd="ps -eo pid=,cmd= | awk '/train_v2_final_pipeline.py/ && /B_640.yaml/ {print $1}'"
    out=subprocess.check_output(cmd,shell=True,text=True).strip().split()
    return out
def run(cmd, log_path):
    log(f'RUN {" ".join(str(x) for x in cmd)}')
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('w',encoding='utf-8') as f:
        rc=subprocess.run([str(x) for x in cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
    log(f'EXIT rc={rc} log={log_path}')
    if rc!=0: raise RuntimeError(f'command failed rc={rc}: {cmd}')
def branch_summary(outdir):
    br=best_row(outdir)
    if not br: return {'status':'NO_HISTORY'}
    ep=int(br['epoch']); size=read_size(outdir,ep)
    def g(b,k): return float(size.get(b,{}).get(k,0.0))
    extra={}
    try: extra=json.loads(br.get('extra') or '{}')
    except Exception: extra={}
    return {'outdir':str(outdir),'epoch':ep,'fixed':row_float(br,'val_f1'),'oracle':row_float(br,'val_f1_oracle'),'precision':row_float(br,'val_precision_fixed'),'recall':row_float(br,'val_recall_fixed'),'iou':row_float(br,'val_iou_fixed'),'auc':row_float(br,'val_auc'),'lt_0p5_f1':g('lt_0p5','f1'),'lt_0p5_recall':g('lt_0p5','recall'),'lt_1_f1':g('lt_1','f1'),'lt_1_recall':g('lt_1','recall'),'1_3_f1':g('1_3','f1'),'1_3_recall':g('1_3','recall'),'3_10_f1':g('3_10','f1'),'3_10_recall':g('3_10','recall'),'gte_10_f1':g('gte_10','f1'),'gte_10_recall':g('gte_10','recall'),'eval_variant':extra.get('eval_variant','raw'),'checkpoint':str(outdir/'best.pt')}
def make_extra_config(src_cfg, outdir, init):
    cfg=yaml.safe_load(src_cfg.read_text(encoding='utf-8'))
    cfg['experiment']=outdir.name
    cfg['output_dir']=str(outdir)
    cfg['training']['init_checkpoint']=str(init)
    cfg['training']['epochs']=2
    cfg['training']['constant_lr_epochs']=1
    cfg['training']['early_stopping']=3
    return cfg
def write_report(decision, summaries):
    rows=[]
    for name,s in summaries.items():
        rows.append({'Run':name,'Decision':decision if name=='final_best' else '', 'Epoch':s.get('epoch',''),'Fixed F1':s.get('fixed',''),'Oracle F1':s.get('oracle',''),'Precision':s.get('precision',''),'Recall':s.get('recall',''),'IoU':s.get('iou',''),'AUC':s.get('auc',''),'<0.5% F1':s.get('lt_0p5_f1',''),'<0.5% Recall':s.get('lt_0p5_recall',''),'<1% F1':s.get('lt_1_f1',''),'<1% Recall':s.get('lt_1_recall',''),'1-3% Recall':s.get('1_3_recall',''),'3-10% Recall':s.get('3_10_recall',''),'>=10% Recall':s.get('gte_10_recall',''),'EMA/raw':s.get('eval_variant',''),'Checkpoint':s.get('checkpoint','')})
    md=['# Performance Sprint B-Only Report\n',f'Run root: `{SPRINT_ROOT}`','A=704 stopped after E0 by user decision. B=640 continued with original config through E2.\n',md_table(rows,['Run','Decision','Epoch','Fixed F1','Oracle F1','Precision','Recall','IoU','AUC','<0.5% F1','<0.5% Recall','<1% F1','<1% Recall','1-3% Recall','3-10% Recall','>=10% Recall','EMA/raw','Checkpoint'])]
    md.append('\n## Decision Rule')
    md.append('- If best oracle >= 0.758: continue at most 2 more epochs.')
    md.append('- Else if best >= 0.755: save best and stop.')
    md.append('- Else: stop this performance recipe and move to stronger pretrained backbone.')
    md.append(f'\nFinal decision: **{decision}**')
    text='\n'.join(md)+'\n'
    (SPRINT_ROOT/'PERFORMANCE_SPRINT_B_ONLY_REPORT.md').write_text(text,encoding='utf-8')
    (ROOT/'docs').mkdir(exist_ok=True); (ROOT/'docs'/'PERFORMANCE_SPRINT_B_ONLY_REPORT.md').write_text(text,encoding='utf-8')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--poll',type=int,default=60); args=ap.parse_args()
    status=SPRINT_ROOT/'B_ONLY_MONITOR_STATUS.json'
    log(f'B-only monitor start root={SPRINT_ROOT}')
    while True:
        rows=read_history(B_DIR); alive=proc_alive()
        status.write_text(json.dumps({'updated':now(),'stage':'waiting_B_E2','rows':len(rows),'alive_pids':alive},indent=2),encoding='utf-8')
        if len(rows)>=3 or (not alive and len(rows)>0): break
        time.sleep(args.poll)
    initial=branch_summary(B_DIR)
    best=initial
    decision=''
    if initial.get('oracle',0.0) >= 0.765:
        decision='LOCK_NEW_PERFORMANCE_BEST'
    elif initial.get('oracle',0.0) >= 0.758:
        decision='CONTINUE_EXTRA2_MAX'
        extra_dir=SPRINT_ROOT/'B_640_extra2'
        cfg=make_extra_config(B_CFG,extra_dir,B_DIR/'best.pt')
        cfg_path=SPRINT_ROOT/'configs'/'B_640_extra2.yaml'; write_yaml(cfg_path,cfg)
        run([PY,'train_v2_final_pipeline.py','--config',cfg_path,'--phase','perf_sprint','--device','cuda:0'],SPRINT_ROOT/'logs'/'B_640_extra2_train.log')
        extra=branch_summary(extra_dir)
        best=extra if extra.get('oracle',0.0)>initial.get('oracle',0.0) else initial
        if best.get('oracle',0.0)>=0.765: decision='LOCK_NEW_PERFORMANCE_BEST_AFTER_EXTRA2'
    elif initial.get('oracle',0.0) >= 0.755:
        decision='SAVE_BEST_AND_STOP'
    else:
        decision='STOP_RECIPE_NEXT_STRONGER_BACKBONE'
    summaries={'B_initial3':initial,'final_best':best}
    if (SPRINT_ROOT/'B_640_extra2'/'history.csv').exists(): summaries['B_extra2']=branch_summary(SPRINT_ROOT/'B_640_extra2')
    write_report(decision,summaries)
    status.write_text(json.dumps({'updated':now(),'stage':'complete','decision':decision,'initial':initial,'best':best,'report':str(SPRINT_ROOT/'PERFORMANCE_SPRINT_B_ONLY_REPORT.md')},indent=2),encoding='utf-8')
    log(f'complete decision={decision} report={SPRINT_ROOT/"PERFORMANCE_SPRINT_B_ONLY_REPORT.md"}')
if __name__=='__main__': main()
