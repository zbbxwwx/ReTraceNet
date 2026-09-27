"""Corrected fake-only locked audit plus pure HCV neutral and pure DHER bypass."""
import copy,json,traceback
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
import run_fair_training_v1 as fair
from run_cross_dataset_after_chain import LockedCounts
BASE=Path('/group4/zouwenxuan/iml_generalization')
PARENT=BASE/'outputs/retracenet_v2/effective_chain_full_training_clean_v2_20260912'
CKPT=PARENT/'epoch02_audit_snapshot.pt'; OUT=BASE/'outputs/retracenet_v2/effective_chain_epoch02_ablation_audit_20260912'
VARS=['full','qtr_bypass','hcv_neutral_dher_active','dher_only_bypass','dct_off']; TH=0.28
class NeutralHCV(nn.Module):
 def __init__(self,inner): super().__init__(); self.inner=inner
 def forward(self,*a,**kw):
  x=self.inner(*a,**kw); z=lambda t:torch.zeros_like(t)
  # Neutral proposal is 0.5, signed/counterfactual effects and energy are zero.
  x['local_proposals']=[torch.full_like(t,.5) for t in x['local_proposals']]
  x['proposals']=[torch.full_like(t,.5) for t in x['proposals']]
  x['deltas']=[z(t) for t in x['deltas']]
  x['energies']=[z(t) for t in x['energies']]
  x['p_counterfactual']=[t.detach().clone() for t in x['p_factual']]
  return x
def build(v):
 ck=torch.load(CKPT,map_location='cpu');cfg=copy.deepcopy(ck['config'])
 m=fair.e.build_final_model(cfg['model']).cuda().eval();m.load_state_dict(ck['model'],strict=True)
 if v=='qtr_bypass':m.qtr=None
 elif v=='hcv_neutral_dher_active':m.hcv2=NeutralHCV(m.hcv2)
 elif v=='dher_only_bypass':m.dher=None
 if v=='dct_off':m.disable_dct_evidence=True
 return cfg,m
@torch.inference_mode()
def evaluate(v,ds,cfg,m):
 stats=fair.e.ValidationStats(); locked=LockedCounts(TH); count=0
 for i,r in enumerate(ds.records):
  b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)))
  o=fair.fwd(m,b);vh,vw=map(int,b['valid_hw'][0]);p=o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy();y=b['mask'][0,0,:vh,:vw].numpy().astype(np.int16)
  stats.add(p,y,r.label,r.image)
  if r.label:locked.add(p,y);count+=1
  if i==0 or (i+1)%200==0:print('[pure]',v,i+1,len(ds),flush=True)
 return {'variant':v,'fake_count':count,'fake_locked':locked.result(),'metrics':stats.result()}
def main():
 OUT.mkdir(parents=True,exist_ok=True);state={'status':'running','training_run':False,'variants':VARS}
 (OUT/'state.json').write_text(json.dumps(state,indent=2))
 try:
  ck=torch.load(CKPT,map_location='cpu');cfg=copy.deepcopy(ck['config'])
  rec=fair.e.read_manifest(PARENT/'clean_manifest.csv',cfg['data']['root'],'val');ds=fair.FairDataset(rec,PARENT/'jpeg_cache',False)
  # QTR structural/runtime audit on one real batch.
  b=next(iter(DataLoader(ds,batch_size=1,sampler=[0],num_workers=0)));_,m=build('full')
  inp={}
  def pre(mod,args): inp['detail']=[x.detach() for x in args[2]]
  h=m.qtr.register_forward_pre_hook(pre)
  with torch.inference_mode(): o=fair.fwd(m,b)
  h.remove();q=o['qtr']
  qa=m.hcv2.q_alpha.detach().cpu()
  qinfo={'use_reliability_gate':m.qtr.use_reliability_gate,'qualified_detail_present':'qualified_detail' in q,
         'reliability_maps_present':'reliability_maps' in q,'hcv_q_alpha':qa.tolist(),
         'hcv_tanh_q_alpha':torch.tanh(qa).tolist(),
         'per_scale_qfeature_over_input_rms':[float(a.float().square().mean().sqrt()/d.float().square().mean().sqrt().clamp_min(1e-12)) for a,d in zip(q['features'],inp['detail'])],
         'dher_class':type(m.dher).__name__,'dher_forward_source_fact':'DHERv21 assigns _ = qtr and does not read QTR tensors',
         'historical_preflight_qtr_grad_norm_max':0.00019451970654801252,
         'historical_preflight_qtr_parameter_delta_L1_4_steps':6.366833859294818e-07}
  fair.e.dump(OUT/'qtr_wiring.yaml',qinfo);del m,o,q,inp,b;torch.cuda.empty_cache()
  fake_ids=[i for i,r in enumerate(ds.records) if r.label][:16]
  _,base_m=build('full'); base_logits=[]
  for i in fake_ids:
   bb=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)))
   
   with torch.inference_mode(): base_logits.append(fair.fwd(base_m,bb)['mask_logits'].float().cpu())
  del base_m;torch.cuda.empty_cache(); sanity={}
  for vv in VARS[1:]:
   _,mm=build(vv); means=[]; mx=0.0
   for j,i in enumerate(fake_ids):
    bb=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)))
    
    with torch.inference_mode(): pred=fair.fwd(mm,bb)['mask_logits'].float().cpu()
    d=(pred-base_logits[j]).abs()
    means.append(float(d.mean())); mx=max(mx,float(d.max()))
   sanity[vv]={'samples':len(fake_ids),'mean_abs_logit_diff':float(np.mean(means)),'max_abs_logit_diff':mx}
   del mm;torch.cuda.empty_cache()
  fair.e.dump(OUT/'switch_sanity.yaml',sanity)
  results={}
  for v in VARS:
   p=OUT/(v+'.yaml')
   if p.exists():
    import yaml;results[v]=yaml.safe_load(p.read_text());continue
   cfg,m=build(v);results[v]=evaluate(v,ds,cfg,m);fair.e.dump(p,results[v]);del m;torch.cuda.empty_cache()
  base=results['full'];table=[]
  for v in VARS:
   r=results[v];table.append({'variant':v,'fake_locked_f1':r['fake_locked']['f1'],'locked_delta':r['fake_locked']['f1']-base['fake_locked']['f1'],
    'fake_oracle_f1':r['metrics']['fake']['oracle']['f1'],'oracle_delta':r['metrics']['fake']['oracle']['f1']-base['metrics']['fake']['oracle']['f1'],'fake_auc':r['metrics']['fake']['auc']})
  summary={'status':'completed','training_run':False,'threshold':TH,'qtr_wiring':qinfo,'switch_sanity':sanity,'table':table,
   'hcv_intervention':'proposals=0.5; deltas=0; energies=0; DHER remains active',
   'dher_intervention':'HCV executes normally; only DHER write-back is bypassed'}
  fair.e.dump(OUT/'summary.yaml',summary);state.update(status='completed',table=table);(OUT/'state.json').write_text(json.dumps(state,indent=2))
 except BaseException:
  state.update(status='failed',error=traceback.format_exc());(OUT/'state.json').write_text(json.dumps(state,indent=2));raise
if __name__=='__main__':main()
