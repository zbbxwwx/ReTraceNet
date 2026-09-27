"""Fixed E6 checkpoint; CASIA2 test robustness; locked and pooled oracle metrics."""
import copy, json, os, traceback, fcntl
from pathlib import Path
import torch, yaml
import eval_e5_locked_casia2_test as ev
import evaluate_casia2_dg_robustness_two_levels as rob
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e6_casia2_test_robustness_20260924')
CK=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e6_q90_q80_joint_adaptation_20260923/epoch02.pt')
JPEGS={'clean':(None,None),'q90_420':(90,2),'q80_420':(80,2),'q70_420':(70,2),'q90_444':(90,0),'q80_444':(80,0),'q70_444':(70,0)}
OPS={k:v for k,v in rob.CONDS.items() if v[0] not in ('jpeg','subsampling')}
def save(s):
 p=OUT/'summary.yaml';t=p.with_suffix('.tmp');t.write_text(yaml.safe_dump(s,sort_keys=False));t.replace(p)
 lines=['|Condition|Fake oracle F1|P|R|AUC|All oracle F1|P|R|AUC|Fake locked F1 @.18|Auth FPR|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for n,r in s['conditions'].items():
  f=r['oracle_diagnostic']['fake_only_pooled'];a=r['oracle_diagnostic']['all_images_pooled']
  vals=[f['f1'],f['precision'],f['recall'],r['auc']['fake_only_pooled'],a['f1'],a['precision'],a['recall'],r['auc']['all_images_pooled'],r['locked']['fake']['f1'],r['locked']['authentic']['fpr']]
  lines.append('|'+n+'|'+'|'.join(f'{v:.6f}' for v in vals)+'|')
 (OUT/'table.md').write_text('\n'.join(lines)+'\n')
def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4);ev.TH=.18
 par=torch.load(ev.PARENT,map_location='cpu');cfg=copy.deepcopy(par['config']);cfg['training']['dct_sample_dropout']=0.
 ck=torch.load(CK,map_location='cpu');model=ev.fair.e.build_final_model(cfg['model']).cuda().eval();model.load_state_dict(ck['model'],strict=True)
 rows=ev.fair.e.read_manifest(ev.MANIFEST,cfg['data']['root'],'test');base=ev.FairDataset(rows,OUT/'cache_test',False)
 s={'status':'running','checkpoint':str(CK),'dataset':'CASIA2_test','manifest':str(ev.MANIFEST),'test_images':len(rows),'fake_images':sum(r.label==1 for r in rows),'locked_threshold':.18,'threshold_source':'E6_epoch02_CASIA2_clean_val','oracle_scope':'one pooled threshold per condition and population, diagnostic','nonjpeg_operations_encoding':'JPEG Q100 4:4:4, matching previous robustness script','inference':'same full-image forward and FairDataset preprocessing as E5 formal test','conditions':{}}
 if (OUT/'summary.yaml').exists():s['conditions']=yaml.safe_load((OUT/'summary.yaml').read_text()).get('conditions',{})
 print('START',s,flush=True)
 try:
  for n in list(JPEGS)+list(OPS):
   if n in s['conditions']:continue
   ds=ev.View(base,*JPEGS[n]) if n in JPEGS else rob.DS(base,n)
   r=ev.evaluate(model,ds,cfg,n)
   (OUT/(n+'.json')).write_text(json.dumps(r));s['conditions'][n]=r;save(s)
   print('DONE',n,json.dumps(r),flush=True)
  s['status']='completed';save(s)
 except BaseException:
  s['status']='failed';s['traceback']=traceback.format_exc();save(s);raise
if __name__=='__main__':main()
