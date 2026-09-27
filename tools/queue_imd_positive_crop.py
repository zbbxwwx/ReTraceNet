"""Wait for a quiet GPU, then run the single-variable IMD positive-crop comparison."""
import os,subprocess,time,yaml
from pathlib import Path
ROOT=Path('/group4/zouwenxuan/iml_generalization/repos/ReTraceNet')
PY=Path('/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python')
BASE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')
OUT=BASE/'imd2020_stable_bn_positive_crop75_full50_20260915'
OUT.mkdir(parents=True,exist_ok=True)
SUMMARY=OUT/'queue_summary.yaml'
def dump(x):SUMMARY.write_text(yaml.safe_dump(x,sort_keys=False),encoding='utf-8')
state={'status':'waiting_for_quiet_gpu','experiment':'IMD2020 Stable-BN positive-crop 0.75','only_change':'75% fake crops retry until foreground coverage; 25% remain random','baseline':'imd2020_stable_bn_v1_full50_20260913','epochs':50}
dump(state)
gpu=None
while gpu is None:
 out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 rows=[]
 for line in out.strip().splitlines():
  i,m,u=[int(x.strip()) for x in line.split(',')];rows.append((m,u,i))
 quiet=[x for x in rows if x[0]<3000 and x[1]<20]
 if quiet:gpu=min(quiet)[2];break
 state.update(last_gpu_status=out.strip(),last_checked=time.strftime('%Y-%m-%d %H:%M:%S'));dump(state);time.sleep(60)
state.update(status='running',gpu=gpu,started_at=time.strftime('%Y-%m-%d %H:%M:%S'));dump(state)
cmd=[str(PY),str(ROOT/'tools/run_v3_stable_bn_indomain.py'),'--dataset','IMD2020','--split',str(BASE/'v3_offline_master_20260912/indomain_IMD2020/split_721.json'),'--template',str(BASE/'balanced_residual_v3_full30_20260912/best.pt'),'--out',str(OUT),'--epochs','50','--batch-size','4','--seed','1337','--positive-crop-prob','0.75','--positive-crop-retries','20']
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
with (OUT/'queue_run.log').open('a',encoding='utf-8') as log:rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
state.update(status='completed' if rc==0 else 'failed',returncode=rc,finished_at=time.strftime('%Y-%m-%d %H:%M:%S'));dump(state)
