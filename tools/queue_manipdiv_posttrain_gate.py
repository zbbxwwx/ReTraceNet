import os,time,subprocess,yaml
from pathlib import Path
pid=791947
train=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_mit_ep76_manipdiv_posttrain1_20260921')
cross=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_mit_ep76_manipdiv_posttrain1_cross721_20260921')
repo=Path('/group4/zouwenxuan/iml_generalization/repos/ReTraceNet')
while Path(f'/proc/{pid}').exists(): time.sleep(30)
try: s=yaml.safe_load((train/'summary.yaml').read_text()) or {}
except Exception as e: s={'status':'unreadable','error':str(e)}
h=(s.get('history') or []); f1=float(h[-1]['val_fake_f1']) if h else -1
report={'training_status':s.get('status'),'clean_val_f1':f1,'required_clean_f1':0.936,'source_gate_pass':f1>=0.936,'cross_launched':False}
(train/'posttrain_gate.yaml').write_text(yaml.safe_dump(report,sort_keys=False))
if f1>=0.936:
 cross.mkdir(parents=True,exist_ok=True)
 cmd=['/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python','-u',str(repo/'tools/evaluate_casia2_mit_cross721.py'),'--checkpoint',str(train/'best.pt'),'--out',str(cross),'--datasets','CASIA1','Columbia','IMD2020','NIST16']
 with (cross/'run.log').open('w') as log:
  rc=subprocess.call(cmd,cwd=repo,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT)
 report.update(cross_launched=True,cross_returncode=rc,cross_output=str(cross));(train/'posttrain_gate.yaml').write_text(yaml.safe_dump(report,sort_keys=False))
