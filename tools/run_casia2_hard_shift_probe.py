"""One epoch source-only hard-shift-balanced probe from CASIA2 epoch76."""
import copy, json, random, sys, time, traceback
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_imd_mit_full_v3
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
import takeover_views as views

PARENT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
MANIFEST=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_mit_hard_shift_balanced_probe1_20260921')
HARD=('jpeg50','resize075','blur','noise');SEED=3407

class TrainBalanced(torch.utils.data.Dataset):
 def __init__(self,base):self.base=base;self.records=base.records
 def __len__(self):return len(self.base)
 def __getitem__(self,key):
  s=self.base[key];seed=key[1] if isinstance(key,tuple) else SEED+int(key);rng=np.random.default_rng(seed)
  if int(rng.integers(2))==0:return s
  return views.make_view(s,HARD[int(rng.integers(len(HARD)))],seed+301)
class ShiftVal(torch.utils.data.Dataset):
 def __init__(self,base):self.base=base;self.records=base.records
 def __len__(self):return len(self.base)
 def __getitem__(self,key):
  s=self.base[key];seed=views.stable_seed(s['path']);return views.make_view(s,HARD[seed%len(HARD)],seed+911)
def metric(r):
 q=r['metrics']['fake']['oracle'];return {'f1':float(q['f1']),'precision':float(q['precision']),'recall':float(q['recall']),'auc':float(r['metrics']['fake']['auc']),'threshold':float(q['threshold'])}
def main():
 OUT.mkdir(parents=True,exist_ok=True);summary={'status':'preparing','parent':str(PARENT),'epoch':1,'lr':1e-5,'mixture':'50% clean + 50% hard shift, balanced hard operations','hard_operations':list(HARD),'dct_dropout':0.0,'consistency':0.0,'external_datasets_read':False};stable.dump(OUT/'summary.yaml',summary)
 try:
  torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);ck=torch.load(PARENT,map_location='cpu');assert int(ck['epoch'])==76
  cfg=copy.deepcopy(ck['config']);cfg['training']['dct_sample_dropout']=0.;cfg['training']['learning_rate']=1e-5;cfg['training']['epochs']=1
  tr=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'train');va=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'val');fake=[i for i,r in enumerate(tr) if r.label];auth=[i for i,r in enumerate(tr) if not r.label];nf=min(len(fake),len(auth))
  btr=FairDataset(tr,OUT/'jpeg_cache_train',True);td=TrainBalanced(btr);clean=FairDataset(va,OUT/'jpeg_cache_val',False);shift=ShiftVal(clean)
  model=fair.e.build_final_model(cfg['model']).cuda();model.load_state_dict(ck['model'],strict=True);stable.train_mode(model);seg=stable.v3.e24.ReTraceLoss.from_config(stable.v3.e24.clean_loss_config(cfg['loss'])).cuda();opt=torch.optim.SGD(model.parameters(),lr=1e-5,momentum=.9,weight_decay=.0005)
  # Exact same deterministic shifted source val before update.
  model.eval();base_clean=metric(fair.evaluate(model,clean,cfg,OUT,'parent_clean_val'));base_shift=metric(fair.evaluate(model,shift,cfg,OUT,'parent_hard_shift_val'));stable.train_mode(model)
  g=torch.Generator().manual_seed(SEED+1);ids=[fake[i] for i in torch.randperm(len(fake),generator=g)[:nf]]+[auth[i] for i in torch.randperm(len(auth),generator=g)[:nf]];ids=[ids[i] for i in torch.randperm(len(ids),generator=g)];plan=[(idx,SEED*1000000+100000+j) for j,idx in enumerate(ids)];loader=DataLoader(td,batch_size=4,sampler=plan,num_workers=2,pin_memory=True);total=0.;start=time.time()
  for st,b in enumerate(loader,1):
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):o=stable.v3.fwd(model,b,cfg,False);loss=stable.v3.objective(o,b,seg,cfg)['total']
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();total+=float(loss)
   if st==1 or st%25==0 or st==len(loader):stable.dump(OUT/'progress.yaml',{'batch':st,'batches':len(loader),'loss':total/st,'elapsed_minutes':(time.time()-start)/60});print('[hard-shift]',st,len(loader),flush=True)
  post_clean=metric(fair.evaluate(model,clean,cfg,OUT,'clean_val'));post_shift=metric(fair.evaluate(model,shift,cfg,OUT,'hard_shift_val'))
  delta_shift=post_shift['f1']-base_shift['f1'];delta_clean=post_clean['f1']-base_clean['f1'];passed=delta_shift>=.01 and delta_clean>=-.01
  state={'model':fair.e.copy_state(model),'optimizer':opt.state_dict(),'config':cfg,'epoch':1,'parent_epoch':76,'selection_threshold':post_clean['threshold'],'best_metric':post_clean['f1']};fair.e24.atomic_torch_save(state,OUT/'epoch01.pt')
  summary.update(status='completed',parent_clean_val=base_clean,parent_hard_shift_val=base_shift,clean_val=post_clean,hard_shift_val=post_shift,delta_clean_f1=delta_clean,delta_hard_shift_f1=delta_shift,robustness_gate_pass=passed,decision='run_four_domain_generalization' if passed else 'stop_jpeg_dct_tuning_and_move_to_manipulation_diversification');stable.dump(OUT/'summary.yaml',summary)
 except BaseException:summary.update(status='failed',stop_reason=traceback.format_exc());stable.dump(OUT/'summary.yaml',summary);raise
if __name__=='__main__':main()
