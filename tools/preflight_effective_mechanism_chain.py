"""Four-step preflight for the single effective QTR->HCV->DHER repair."""
import copy,json,traceback
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
import run_fair_training_v1 as fair
BASE=Path('/group4/zouwenxuan/iml_generalization')
PARENT=BASE/'outputs/retracenet_v2/fair_training_v1_full_20260911';CK=PARENT/'best.pt'
OUT=BASE/'outputs/retracenet_v2/effective_mechanism_chain_preflight_20260912'
class NeutralHCV(nn.Module):
 def __init__(self,x):super().__init__();self.x=x
 def forward(self,*a,**k):
  d=self.x(*a,**k)
  d['proposals']=[torch.full_like(x,.5) for x in d['proposals']]
  d['local_proposals']=[torch.full_like(x,.5) for x in d['local_proposals']]
  d['deltas']=[torch.zeros_like(x) for x in d['deltas']]
  d['energies']=[torch.zeros_like(x) for x in d['energies']]
  return d
def rms(x):return float(x.detach().float().square().mean().sqrt())
def state_group(m,prefixes):return {n:p.detach().clone() for n,p in m.named_parameters() if any(n.startswith(x) for x in prefixes)}
def delta(m,before,prefix):
 return sum(float((p.detach()-before[n]).abs().sum()) for n,p in m.named_parameters() if n in before and n.startswith(prefix))
def grad(m,prefix):
 return sum(float(p.grad.detach().float().square().sum()) for n,p in m.named_parameters() if n.startswith(prefix) and p.grad is not None)**.5
def logits(m,b):return fair.fwd(m,b)['mask_logits'].float()
def main():
 OUT.mkdir(parents=True,exist_ok=True);st={'status':'running','training_run':False,'steps':4};(OUT/'state.json').write_text(json.dumps(st,indent=2))
 try:
  torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
  ck=torch.load(CK,map_location='cpu');cfg=copy.deepcopy(ck['config']);cfg['model']['enable_effective_mechanism_chain']=True
  model=fair.e.build_final_model(cfg['model']).cuda()
  load=model.load_state_dict(ck['model'],strict=False)
  expected=('qtr.reliability_gates.','hcv2.q_fusions.')
  unexpected_missing=[x for x in load.missing_keys if not x.startswith(expected)]
  assert not unexpected_missing,unexpected_missing
  assert not load.unexpected_keys,load.unexpected_keys
  rec=fair.e.read_manifest(PARENT/'clean_manifest.csv',cfg['data']['root'],'val');ds=fair.FairDataset(rec,PARENT/'jpeg_cache',False)
  sample_id=next(i for i,r in enumerate(rec) if r.label);b=next(iter(DataLoader(ds,batch_size=1,sampler=[sample_id],num_workers=0)))
  # Mechanisms only: no encoder/decoder movement during wiring preflight.
  prefixes=('qtr.','hcv2.','dher.','dher_visibility.','dher_calibrated_writeback.')
  for n,p in model.named_parameters():p.requires_grad_(any(n.startswith(x) for x in prefixes))
  opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0)
  seg=fair.e24.ReTraceLoss.from_config(fair.e24.clean_loss_config(cfg['loss'])).cuda()
  before=state_group(model,prefixes);losses=[];grad_max={x:0. for x in prefixes}
  model.train()
  for step in range(4):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    out=fair.fwd(model,b);loss=fair.objective(out,b,seg,cfg)['total']
   assert torch.isfinite(loss);loss.backward()
   for x in prefixes:grad_max[x]=max(grad_max[x],grad(model,x))
   torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
  updates={x:delta(model,before,x) for x in prefixes}
  core={x:grad_max[x] for x in ('qtr.','hcv2.','dher.')}
  core_u={x:updates[x] for x in ('qtr.','hcv2.','dher.')}
  assert all(v>0 for v in core.values()),core
  assert all(v>0 for v in core_u.values()),core_u
  model.eval()
  with torch.no_grad():
   full=logits(model,b)
   q=model.qtr;model.qtr=None;qoff=logits(model,b);model.qtr=q
   h=model.hcv2;model.hcv2=NeutralHCV(h);hoff=logits(model,b);model.hcv2=h
   d=model.dher;model.dher=None;doff=logits(model,b);model.dher=d
  diffs={'qtr_bypass':{'mean':float((full-qoff).abs().mean()),'max':float((full-qoff).abs().max())},
         'hcv_neutral':{'mean':float((full-hoff).abs().mean()),'max':float((full-hoff).abs().max())},
         'dher_bypass':{'mean':float((full-doff).abs().mean()),'max':float((full-doff).abs().max())}}
  assert all(v['max']>1e-5 for v in diffs.values()),diffs
  result={'status':'passed','training_run':False,'source_checkpoint':str(CK),'config':cfg['model'],
    'load':{'missing':load.missing_keys,'unexpected':load.unexpected_keys},'losses':losses,
    'grad_norm_max':grad_max,'parameter_delta_L1':updates,'post_step_bypass_logit_diff':diffs,
    'qtr_gate_enabled':model.qtr.use_reliability_gate,'hcv_mandatory_q_fusion':model.hcv2.mandatory_q_fusion,
    'dher_residual_form':'D1_refined = D1_pre + support_write + conflict_write'}
  fair.e.dump(OUT/'preflight.yaml',result);st.update(status='passed');(OUT/'state.json').write_text(json.dumps(st,indent=2));print(json.dumps(result,indent=2))
 except BaseException:
  st.update(status='failed',error=traceback.format_exc());(OUT/'state.json').write_text(json.dumps(st,indent=2));raise
if __name__=='__main__':main()
