"""Offline IMD structural experiment queue; pilots use val only, test only final winner."""
import subprocess,time,yaml,traceback
from pathlib import Path
ROOT=Path('/group4/zouwenxuan/iml_generalization/repos/ReTraceNet');PY='/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python';BASE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')
SPLIT=BASE/'v3_offline_master_20260912/indomain_IMD2020/split_721.json';TEMPLATE=BASE/'balanced_residual_v3_full30_20260912/best.pt';Q=BASE/'imd2020_structural_queue_20260915';Q.mkdir(parents=True,exist_ok=True);S=Q/'summary.yaml';baseline=.4833455106808563
A=BASE/'imd2020_ctxcore_pilot15_v2_20260915';B=BASE/'imd2020_decoder_ds_pilot15_20260915';D=BASE/'imd2020_verified_residual_decoder_pilot15_20260915'
state={'status':'waiting_context_core','baseline_val_f1':baseline,'test_policy':'pilots never read test; one final winner test only','experiments':{'C':{'status':'skipped','reason':'baseline val precision 0.468 and recall 0.500 are balanced; adaptive P/R loss trigger not met'},'B':{'note':'existing D2-D4 supervision is replaced by the requested weaker D1pre-D4 schedule, not duplicated'}}}
def dump():S.write_text(yaml.safe_dump(state,sort_keys=False),encoding='utf-8')
def read(p):
 try:return yaml.safe_load((p/'summary.yaml').read_text()) or {}
 except:return {}
def run(name,args,out):
 state['status']='running_'+name;state['current_output']=str(out);dump();out.mkdir(parents=True,exist_ok=True)
 with (out/'queue_runner.log').open('a') as log:return subprocess.run([PY,*args],cwd=ROOT,env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':'1'},stdout=log,stderr=subprocess.STDOUT).returncode
dump()
try:
 while read(A).get('status') not in ('completed','failed'):time.sleep(60)
 state['experiments']['A']=read(A);dump()
 profile=BASE/'imd2020_error_profile_20260915'
 rc=run('error_profile',['tools/profile_imd2020_errors.py'],profile);state['error_profile_returncode']=rc;state['error_profile_file']=str(profile/'imd2020_error_profile.yaml');dump()
 common=['tools/run_v3_stable_bn_indomain.py','--dataset','IMD2020','--split',str(SPLIT),'--template',str(TEMPLATE),'--epochs','15','--batch-size','4','--seed','1337','--skip-test']
 rc=run('B_decoder_ds',common+['--out',str(B),'--decoder-ds-targeted'],B);state['experiments']['B'].update(read(B));state['experiments']['B']['returncode']=rc;dump()
 rc=run('D_verified_residual_decoder',common+['--out',str(D),'--verified-residual-decoder'],D);state['experiments']['D']=read(D);state['experiments']['D']['returncode']=rc;dump()
 scores={k:float(state['experiments'].get(k,{}).get('best_F1',-1)) for k in ('A','B','D')};state['pilot_scores']=scores
 passed={k:v for k,v in scores.items() if v>=baseline+.02};state['passed']=passed
 if not passed:state.update(status='completed_no_winner',stop_reason='no pilot improved overall val F1 by 2pp; no formal run and no test');dump();raise SystemExit(0)
 winner=max(passed,key=passed.get);state['winner']=winner;formal=BASE/f'imd2020_structural_winner_{winner}_formal50_20260915'
 if winner=='A':args=['tools/run_imd_context_core.py','--split',str(SPLIT),'--template',str(TEMPLATE),'--out',str(formal),'--epochs','50','--context','768','--batch-size','2','--accum','2','--seed','1337','--test']
 else:args=['tools/run_v3_stable_bn_indomain.py','--dataset','IMD2020','--split',str(SPLIT),'--template',str(TEMPLATE),'--out',str(formal),'--epochs','50','--batch-size','4','--seed','1337',('--decoder-ds-targeted' if winner=='B' else '--verified-residual-decoder')]
 rc=run('formal_'+winner,args,formal);state['formal']=read(formal);state['formal']['returncode']=rc;state.update(status='completed' if rc==0 else 'failed_formal',final_output=str(formal));dump()
except SystemExit:raise
except BaseException:state.update(status='failed',stop_reason=traceback.format_exc());dump();raise
