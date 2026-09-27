"""Matched 20-epoch Full/R/D cue experiment; val only, never test."""
import os,subprocess,traceback,yaml
from pathlib import Path
ROOT=Path('/group4/zouwenxuan/iml_generalization/repos/ReTraceNet');PY='/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python';BASE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')
SPLIT=BASE/'v3_offline_master_20260912/indomain_IMD2020/split_721.json';TEMPLATE=BASE/'balanced_residual_v3_full30_20260912/best.pt';Q=BASE/'imd2020_single_cue_queue_20260916';Q.mkdir(parents=True,exist_ok=True);S=Q/'summary.yaml'
state={'status':'preparing','test_read':False,'protocol':'same public init, seed, split, 20-epoch schedule, loss, sampler and evaluator','runs':{}}
def dump():S.write_text(yaml.safe_dump(state,sort_keys=False))
def read(p):
 try:return yaml.safe_load((p/'summary.yaml').read_text()) or {}
 except:return {}
try:
 for tag,mode in [('full','full'),('content_residual','residual_only'),('content_dct','dct_only')]:
  out=BASE/f'imd2020_cue_{tag}_pilot20_20260916';out.mkdir(parents=True,exist_ok=True);state.update(status='running_'+tag,current_output=str(out));dump()
  args=[PY,'-u','tools/run_v3_stable_bn_indomain.py','--dataset','IMD2020','--split',str(SPLIT),'--template',str(TEMPLATE),'--out',str(out),'--epochs','20','--batch-size','4','--seed','1337','--forensic-mode',mode,'--skip-test']
  with (out/'queue_runner.log').open('a') as log:rc=subprocess.run(args,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'},stdout=log,stderr=subprocess.STDOUT).returncode
  state['runs'][tag]=read(out);state['runs'][tag]['returncode']=rc;dump()
  if rc:raise RuntimeError(f'{tag} failed rc={rc}')
 state.update(status='completed',stop_reason='matched cue pilots complete; no test read');dump()
except BaseException:state.update(status='failed',stop_reason=traceback.format_exc());dump();raise
