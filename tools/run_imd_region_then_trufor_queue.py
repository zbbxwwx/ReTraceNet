"""Offline queue: feature audit -> matched region A/B -> winner test -> TruFor IMD 7:2:1."""
import json,os,subprocess,time,traceback
from pathlib import Path
import yaml

R=Path('/group4/zouwenxuan/iml_generalization/repos/ReTraceNet');T=Path('/group4/zouwenxuan/iml_generalization/repos/TruFor/TruFor_train_test')
PY='/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python';TPY='/group4/zouwenxuan/iml_generalization/envs/trufor_from_docker/bin/python'
B=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2');SPLIT=B/'v3_offline_master_20260912/indomain_IMD2020/split_721.json';TEMPLATE=B/'balanced_residual_v3_full30_20260912/best.pt'
ROOT=B/'imd_region_then_trufor_queue_20260916';ROOT.mkdir(parents=True,exist_ok=True);STATE=ROOT/'summary.yaml'
BASE=B/'imd2020_region_matched_baseline30_20260916';REG=B/'imd2020_region_affinity30_20260916';AUD=B/'imd2020_region_feature_audit_20260916'
state={'status':'initializing','policy':'val-only matched ReTrace pilots; test once on val winner; then independent TruFor IMD 7:2:1','existing_pids_waited':[461917,464336]}
def dump():STATE.write_text(yaml.safe_dump(state,sort_keys=False))
def alive(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
 except PermissionError:return True
def run(name,cmd,cwd,gpu=None):
 state.update(status='running_'+name,current_command=cmd,current_output=str(ROOT/(name+'.log')));dump();env=os.environ.copy()
 if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 with (ROOT/(name+'.log')).open('a') as log:return subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
def read_summary(p):
 try:return yaml.safe_load((p/'summary.yaml').read_text()) or {}
 except:return {}

dump()
try:
 while any(alive(p) for p in state['existing_pids_waited']):
  state.update(status='waiting_existing_jobs',alive_pids=[p for p in state['existing_pids_waited'] if alive(p)]);dump();time.sleep(60)
 checkpoint=B/'imd2020_stable_bn_full100_20260916/best.pt'
 if not checkpoint.exists():raise FileNotFoundError(checkpoint)
 rc=run('feature_audit',[PY,'tools/audit_imd_region_features.py','--checkpoint',str(checkpoint),'--split',str(SPLIT),'--out',str(AUD)],R,1);state['feature_audit_returncode']=rc
 if rc:raise RuntimeError('feature audit failed')
 common=[PY,'tools/run_v3_stable_bn_indomain.py','--dataset','IMD2020','--split',str(SPLIT),'--template',str(TEMPLATE),'--epochs','30','--batch-size','4','--seed','1337','--skip-test']
 BASE.mkdir(parents=True,exist_ok=True);REG.mkdir(parents=True,exist_ok=True);state.update(status='running_matched_pair',baseline=str(BASE),region=str(REG));dump()
 with (ROOT/'matched_baseline30.log').open('a') as l0,(ROOT/'region_affinity30.log').open('a') as l1:
  e0={**os.environ,'CUDA_VISIBLE_DEVICES':'0'};e1={**os.environ,'CUDA_VISIBLE_DEVICES':'1'}
  p0=subprocess.Popen(common+['--out',str(BASE)],cwd=R,env=e0,stdout=l0,stderr=subprocess.STDOUT)
  p1=subprocess.Popen(common+['--out',str(REG),'--splicing-region-affinity','--region-affinity-weight','0.10'],cwd=R,env=e1,stdout=l1,stderr=subprocess.STDOUT)
  state['pair_pids']=[p0.pid,p1.pid];dump()
  r0=p0.wait();r1=p1.wait()
 state['pair_returncodes']={'baseline':r0,'region':r1};state['pair_summaries']={'baseline':read_summary(BASE),'region':read_summary(REG)};dump()
 if r0 or r1:raise RuntimeError(f'matched pair failed baseline={r0} region={r1}')
 scores={k:float(v.get('best_F1',-1)) for k,v in state['pair_summaries'].items()};winner=max(scores,key=scores.get);win=BASE if winner=='baseline' else REG;state.update(pair_scores=scores,winner=winner,winner_checkpoint=str(win/'best.pt'));dump()
 rc=run('winner_test',[PY,'tools/eval_stable_domain_checkpoint.py','--dataset','IMD2020','--split',str(SPLIT),'--checkpoint',str(win/'best.pt'),'--out',str(ROOT/'retracenet_winner_test')],R,0);state['winner_test_returncode']=rc
 if rc:raise RuntimeError('winner test failed')
 rc=run('prepare_trufor',[TPY,'tools/prepare_trufor_imd721.py','--split',str(SPLIT),'--root','/group4/zouwenxuan/iml_generalization/data/IMD2020','--repo',str(T)],R,None);state['prepare_trufor_returncode']=rc
 if rc:raise RuntimeError('TruFor manifest preparation failed')
 rc=run('trufor_train',[TPY,str(R/'tools/train_trufor_imd_locked.py'),'--repo',str(T),'--experiment','trufor_imd721','--split',str(SPLIT),'--gpu','1','--seed','1337'],T,None);state['trufor_train_returncode']=rc
 if rc:raise RuntimeError('TruFor training failed')
 ck=T/'weights/trufor_imd721/best.pth.tar';pred=ROOT/'trufor_predictions'
 for part in ('val','test'):
  rc=run('trufor_infer_'+part,[TPY,str(R/'tools/infer_trufor_split.py'),'--repo',str(T),'--checkpoint',str(ck),'--split',str(SPLIT),'--part',part,'--out',str(pred/part),'--gpu','1'],T,None);state['trufor_infer_'+part+'_returncode']=rc
  if rc:raise RuntimeError('TruFor inference failed: '+part)
 rc=run('trufor_evaluate',[TPY,str(R/'tools/evaluate_trufor_locked.py'),'--split',str(SPLIT),'--val-pred',str(pred/'val'),'--test-pred',str(pred/'test'),'--out',str(ROOT/'trufor_locked_result.yaml')],T,None);state['trufor_evaluate_returncode']=rc
 if rc:raise RuntimeError('TruFor evaluation failed')
 state.update(status='completed',trufor_checkpoint=str(ck),trufor_result=str(ROOT/'trufor_locked_result.yaml'));dump()
except BaseException:
 state.update(status='failed',error=traceback.format_exc());dump();raise
