import time,yaml
from pathlib import Path
new=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_mit_manipdiv_ep1_cross721_1024_20260921/summary.yaml')
base=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_mit_ep76_cross721_ctx1024_complete_20260920/summary.yaml')
out=new.parent/'generalization_decision.yaml'
while True:
 try: n=yaml.safe_load(new.read_text()) or {}
 except Exception: n={}
 if n.get('status') in ('completed','failed'): break
 time.sleep(30)
report={'status':n.get('status'),'new_checkpoint':'ManipDiv epoch1','baseline_checkpoint':'CASIA2 MiT-V3 epoch76','threshold_source':'CASIA2_val (separate source-val locked threshold per checkpoint)','inference':'1024/768','datasets':{}}
if n.get('status')=='completed':
 b=yaml.safe_load(base.read_text())
 deltas=[];wins=0
 for name in ['CASIA1','Columbia','IMD2020','NIST16']:
  nd=n['datasets'][name];bd=b['datasets'][name]
  nm=nd['pixel_localization_fake_only'];bm=bd['pixel_localization_fake_only']
  nf=nm['test_f1_locked']['f1'];bf=bm['test_f1_locked']['f1'];delta=nf-bf
  if delta>0:wins+=1
  deltas.append(delta)
  report['datasets'][name]={'baseline_f1':bf,'new_f1':nf,'delta_f1':delta,'baseline_auc':bm['auc'],'new_auc':nm['auc'],'delta_auc':nm['auc']-bm['auc'],'new_precision':nm['test_f1_locked']['precision'],'new_recall':nm['test_f1_locked']['recall'],'new_oracle_f1_diagnostic':nm['test_f1_oracle_diagnostic']['f1']}
 report['wins']=wins;report['mean_delta_f1']=sum(deltas)/len(deltas);report['gate_pass']=wins>=3 and report['mean_delta_f1']>=0.02
 report['decision']='continue_manipulation_diversity_training' if report['gate_pass'] else 'do_not_long_train_under_predeclared_gate'
out.write_text(yaml.safe_dump(report,sort_keys=False),encoding='utf8')
