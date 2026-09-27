"""CASIA1 Stable-BN V1: full 50 epochs, with BN/evaluator as the only change."""
import copy, json, math, random, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from retrace.data import Record
import run_balanced_residual_v3_full30 as v3
import run_cross_dataset_after_chain as cross
from fair_training_data import FairDataset

BASE=Path('/group4/zouwenxuan/iml_generalization')
OLD=BASE/'outputs/retracenet_v2/v3_offline_master_20260912/indomain_CASIA1'
OUT=BASE/'outputs/retracenet_v2/casia1_stable_bn_v1_full50_20260913'
TEMPLATE=BASE/'outputs/retracenet_v2/balanced_residual_v3_full30_20260912/best.pt'

def dump(name,value):v3.e.dump(OUT/name,value)
def records(rows,split):return [Record(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') else None,split=split,dataset='CASIA1',label=int(r['label'])) for r in rows]

def freeze_pretrained_bn_stats(model):
 count=0
 for branch in (model.content_encoder,model.dct_branch):
  for module in branch.modules():
   if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval();count+=1
 return count

def mode_train_stable(model):
 v3.mode_train(model)
 count=freeze_pretrained_bn_stats(model)
 assert count>0
 return count

def evaluate_stable(model,*args,**kwargs):
 was_training=model.training
 model.eval()
 try:return cross.evaluate_dataset(model,*args,**kwargs)
 finally:
  if was_training:mode_train_stable(model)

def bn_state(model):
 return {root_name+'.'+n:(m.running_mean.detach().cpu().clone(),m.running_var.detach().cpu().clone(),int(m.num_batches_tracked))
  for root_name,root in [('content_encoder',model.content_encoder),('dct_branch',model.dct_branch)]
  for n,m in root.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}

def main():
 OUT.mkdir(parents=True,exist_ok=True);summary={'status':'preparing','experiment':'CASIA1-Stable-BN V1','only_change':'freeze Content/DCT BN running statistics and force evaluator model.eval()','test_run':False};dump('summary.yaml',summary)
 try:
  split=json.loads((OLD/'split_721.json').read_text());tr,va=(records(split[k],k) for k in ('train','val'))
  fake=[i for i,r in enumerate(tr) if r.label];auth=[i for i,r in enumerate(tr) if not r.label];assert fake and auth and va
  template=torch.load(TEMPLATE,map_location='cpu');cfg=copy.deepcopy(template['config']);cfg['training']['epochs']=50;cfg['training']['dct_sample_dropout']=.2
  torch.manual_seed(3407);np.random.seed(3407);random.seed(3407)
  # Public pretrained encoders are loaded by the constructor. CASIA2 localization weights are deliberately not loaded.
  model=v3.e.build_final_model(cfg['model']).cuda();bn_count=mode_train_stable(model)
  seg=v3.e24.ReTraceLoss.from_config(v3.e24.clean_loss_config(cfg['loss'])).cuda();opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=.0005)
  ds=FairDataset(tr,OUT/'jpeg_cache_train',True);nf=min(len(fake),len(auth));steps_per_epoch=math.ceil(2*nf/4);total_steps=50*steps_per_epoch;gs=0;history=[];best=-1
  initial_bn=bn_state(model);bn_weight_bias_trainable=all(p.requires_grad for branch in (model.content_encoder,model.dct_branch) for m in branch.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for p in (m.weight,m.bias) if p is not None)
  summary.update(status='training',initialization={'content':'ImageNet public pretrained','dct':'public pretrained','localization_chain':'random','CASIA2_localization_weights_loaded':False},bn_module_count=bn_count,bn_weight_bias_trainable=bn_weight_bias_trainable,epochs=50,counts={'train':len(tr),'val':len(va)},history=[]);dump('summary.yaml',summary)
  for ep in range(1,51):
   gen=torch.Generator().manual_seed(3407+ep);ids=[fake[i] for i in torch.randperm(len(fake),generator=gen)[:nf]]+[auth[i] for i in torch.randperm(len(auth),generator=gen)[:nf]];ids=[ids[i] for i in torch.randperm(len(ids),generator=gen)];plan=[(idx,3407*1000000+ep*100000+j) for j,idx in enumerate(ids)]
   loader=DataLoader(ds,batch_size=4,sampler=plan,num_workers=2,pin_memory=True);mode_train_stable(model);running=0
   for st,b in enumerate(loader,1):
    lr=.001*(1-gs/max(total_steps,1))**.9
    for group in opt.param_groups:group['lr']=lr
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):terms=v3.objective(v3.fwd(model,b,cfg,True),b,seg,cfg);loss=terms['total']
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();gs+=1;running+=float(loss)
    if st==1:
     bn_grads={root_name+'.'+n:float(sum((p.grad.float().square().sum() if p.grad is not None else 0) for p in (m.weight,m.bias) if p is not None).sqrt()) for root_name,root in [('content_encoder',model.content_encoder),('dct_branch',model.dct_branch)] for n,m in root.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)}
    if st==1 or st%25==0:dump('progress.yaml',{'epoch':ep,'epochs':50,'batch':st,'batches':len(loader),'loss':running/st,'lr':lr});print('[stable-bn]',ep,st,len(loader),running/st,flush=True)
   current_bn=bn_state(model);changed=[n for n in initial_bn if not torch.equal(initial_bn[n][0],current_bn[n][0]) or not torch.equal(initial_bn[n][1],current_bn[n][1]) or initial_bn[n][2]!=current_bn[n][2]]
   assert not changed,f'BN running statistics changed: {changed[:10]}'
   result=evaluate_stable(model,va,OUT/'jpeg_cache_val',None,None,'CASIA1',OUT/f'val_epoch{ep:02d}',evaluation_mode='validation_select_eval_forced',threshold_source='CASIA1_val_current')
   q=result['pixel_localization_fake_only']['test_f1_oracle_diagnostic'];score=float(q['f1']);row={'epoch':ep,'global_step':gs,'train_total':running/len(loader),'val_fake_f1':score,'val_fake_auc':result['pixel_localization_fake_only']['auc'],'threshold':q['threshold']};history.append(row)
   state={'model':v3.e.copy_state(model),'optimizer':opt.state_dict(),'config':cfg,'epoch':ep,'history':history,'best_metric':max(best,score),'selection_threshold':q['threshold']};v3.e24.atomic_torch_save(state,OUT/'last.pt')
   if score>best:best=score;best_ep=ep;v3.e24.atomic_torch_save(state,OUT/'best.pt')
   summary.update(history=history,best_F1=best,best_epoch=best_ep,best_checkpoint=str(OUT/'best.pt'),bn_running_stats_unchanged=True,bn_grad_norms_first_batch=bn_grads);dump('summary.yaml',summary)
  summary.update(status='completed',stop_reason='fixed_5_epoch_BN_only_pilot_complete',test_run=False);dump('summary.yaml',summary)
 except BaseException:
  summary.update(status='failed',stop_reason=traceback.format_exc());dump('summary.yaml',summary);raise
if __name__=='__main__':main()
