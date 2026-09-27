"""Resume the stopped offline chain after the matched baseline OOM."""
import os,subprocess,traceback
from pathlib import Path
import yaml
R=Path('/group4/zouwenxuan/iml_generalization/repos/ReTraceNet');T=Path('/group4/zouwenxuan/iml_generalization/repos/TruFor/TruFor_train_test');PY='/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python';TPY='/group4/zouwenxuan/iml_generalization/envs/trufor_from_docker/bin/python';B=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2');SPLIT=B/'v3_offline_master_20260912/indomain_IMD2020/split_721.json';TEMPLATE=B/'balanced_residual_v3_full30_20260912/best.pt';Q=B/'imd_region_then_trufor_queue_20260916';BASE=B/'imd2020_region_matched_baseline30_retry_20260917';REG=B/'imd2020_region_affinity30_20260916';S=Q/'continuation_summary.yaml';state={'status':'initializing'}
def dump():S.write_text(yaml.safe_dump(state,sort_keys=False))
def read(p):
 try:return yaml.safe_load((p/'summary.yaml').read_text()) or {}
 except:return {}
def run(name,cmd,cwd,gpu=None):
 state.update(status='running_'+name,current_output=str(Q/(name+'.log')));dump();env=os.environ.copy()
 if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 with (Q/(name+'.log')).open('a') as log:return subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
dump()
try:
 common=[PY,'tools/run_v3_stable_bn_indomain.py','--dataset','IMD2020','--split',str(SPLIT),'--template',str(TEMPLATE),'--out',str(BASE),'--epochs','30','--batch-size','4','--seed','1337','--skip-test']
 rc=run('matched_baseline30_retry',common,R,0);state['baseline_returncode']=rc;state['baseline']=read(BASE);state['region']=read(REG);dump()
 if rc:raise RuntimeError('matched baseline retry failed')
 scores={'baseline':float(state['baseline'].get('best_F1',-1)),'region':float(state['region'].get('best_F1',-1))};winner=max(scores,key=scores.get);win=BASE if winner=='baseline' else REG;state.update(scores=scores,winner=winner,winner_checkpoint=str(win/'best.pt'));dump()
 rc=run('winner_test',[PY,'tools/eval_stable_domain_checkpoint.py','--dataset','IMD2020','--split',str(SPLIT),'--checkpoint',str(win/'best.pt'),'--out',str(Q/'retracenet_winner_test')],R,0);state['winner_test_returncode']=rc;dump()
 if rc:raise RuntimeError('winner test failed')
 rc=run('prepare_trufor',[TPY,'tools/prepare_trufor_imd721.py','--split',str(SPLIT),'--root','/group4/zouwenxuan/iml_generalization/data/IMD2020','--repo',str(T)],R);state['prepare_returncode']=rc
 if rc:raise RuntimeError('prepare TruFor failed')
 rc=run('trufor_train',[TPY,str(R/'tools/train_trufor_imd_locked.py'),'--repo',str(T),'--experiment','trufor_imd721','--split',str(SPLIT),'--gpu','1','--seed','1337'],T);state['trufor_train_returncode']=rc;dump()
 if rc:raise RuntimeError('TruFor train failed')
 ck=T/'weights/trufor_imd721/best.pth.tar';pred=Q/'trufor_predictions'
 for part in ('val','test'):
  rc=run('trufor_infer_'+part,[TPY,str(R/'tools/infer_trufor_split.py'),'--repo',str(T),'--checkpoint',str(ck),'--split',str(SPLIT),'--part',part,'--out',str(pred/part),'--gpu','1'],T);state['trufor_infer_'+part]=rc;dump()
  if rc:raise RuntimeError('TruFor infer '+part+' failed')
 rc=run('trufor_evaluate',[TPY,str(R/'tools/evaluate_trufor_locked.py'),'--split',str(SPLIT),'--val-pred',str(pred/'val'),'--test-pred',str(pred/'test'),'--out',str(Q/'trufor_locked_result.yaml')],T);state['trufor_evaluate_returncode']=rc
 if rc:raise RuntimeError('TruFor evaluate failed')
 state.update(status='completed',trufor_result=str(Q/'trufor_locked_result.yaml'));dump()
except BaseException:state.update(status='failed',error=traceback.format_exc());dump();raise
