"""Locked CASIA1 epoch10 diagnostic: evaluator, mode, loss and final-only fit."""
import copy, json, math, random, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from retrace.data import Record
import run_balanced_residual_v3_full30 as v3
import run_cross_dataset_after_chain as cross
from fair_training_data import FairDataset, crop_outputs

BASE=Path('/group4/zouwenxuan/iml_generalization')
RUN=BASE/'outputs/retracenet_v2/v3_offline_master_20260912/indomain_CASIA1'
OUT=RUN/'epoch10_diagnostic_v2'; CKPT=RUN/'last.pt'

def dump(name,value):v3.e.dump(OUT/name,value)
def recs(rows,split):return [Record(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') else None,split=split,dataset='CASIA1',label=int(r['label'])) for r in rows]

def evaluate(model, records, name, mode):
    return cross.evaluate_dataset(model,records,OUT/'jpeg_cache',None,None,name,OUT/name,
        evaluation_mode=mode,threshold_source=name+'_oracle_diagnostic')

def main():
 OUT.mkdir(parents=True,exist_ok=True);state={'status':'starting'};dump('summary.yaml',state)
 try:
  split=json.loads((RUN/'split_721.json').read_text());train_fake=recs([r for r in split['train'] if int(r['label'])], 'train');val_fake=recs([r for r in split['val'] if int(r['label'])], 'val')
  ck=torch.load(CKPT,map_location='cpu');cfg=copy.deepcopy(ck['config'])
  def loaded():
   m=v3.e.build_final_model(cfg['model']).cuda();m.load_state_dict(ck['model'],strict=True);return m
  state.update(status='same_evaluator_train_val',checkpoint_epoch=int(ck.get('epoch',-1)));dump('summary.yaml',state)
  m=loaded().eval();tr=evaluate(m,train_fake,'train_fake_epoch10','diagnostic_same_evaluator');va=evaluate(m,val_fake,'val_fake_epoch10','diagnostic_same_evaluator')
  state['same_evaluator']={'train_fake':tr['pixel_localization_fake_only'],'val_fake':va['pixel_localization_fake_only']};dump('summary.yaml',state);del m;torch.cuda.empty_cache()

  eligible=[]
  for r in train_fake:
   with Image.open(r.image) as im:w,h=im.size
   if h<=512 and w<=512:eligible.append(r)
  random.Random(3407).shuffle(eligible);fixed=eligible[:32]
  (OUT/'fixed32.json').write_text(json.dumps([{'image':str(r.image),'mask':str(r.mask)} for r in fixed],indent=2))
  # Identical full-image inputs; only Module.training differs. Separate models avoid BN contamination.
  me=loaded().eval();mt=loaded().train()
  eval_mode=evaluate(me,fixed,'fixed32_eval_mode','diagnostic_mode_compare')
  train_mode=evaluate(mt,fixed,'fixed32_train_mode','diagnostic_mode_compare')
  ds_eval=FairDataset(fixed,OUT/'jpeg_cache',False)
  diffs=[];fg_e=[];bg_e=[];fg_t=[];bg_t=[]
  for i in range(len(fixed)):
   b=next(iter(DataLoader(ds_eval,batch_size=1,sampler=[i],num_workers=0)))
   with torch.no_grad():
    le=v3.fwd(me,b,cfg,False)['mask_logits'].float();lt=v3.fwd(mt,b,cfg,False)['mask_logits'].float()
   vh,vw=map(int,b['valid_hw'][0]);le=le[0,0,:vh,:vw];lt=lt[0,0,:vh,:vw];y=b['mask'][0,0,:vh,:vw].cuda()>0.5
   diffs.append(float((le-lt).abs().mean()));pe=le.sigmoid();pt=lt.sigmoid()
   fg_e.append(float(pe[y].mean()));bg_e.append(float(pe[~y].mean()));fg_t.append(float(pt[y].mean()));bg_t.append(float(pt[~y].mean()))
  state['mode_compare']={'eval':eval_mode['pixel_localization_fake_only'],'train':train_mode['pixel_localization_fake_only'],
    'mean_abs_logit_diff':float(np.mean(diffs)),'eval_fg_probability_mean':float(np.mean(fg_e)),'eval_bg_probability_mean':float(np.mean(bg_e)),
    'train_fg_probability_mean':float(np.mean(fg_t)),'train_bg_probability_mean':float(np.mean(bg_t))};dump('summary.yaml',state)
  del me,mt;torch.cuda.empty_cache()

  # Component audit on four fixed full-image fake samples, no augmentation/dropout.
  ds_train=FairDataset(fixed,OUT/'jpeg_cache',True);batch=next(iter(DataLoader(ds_train,batch_size=4,sampler=[(i,8000+i) for i in range(4)],num_workers=0)))
  m=loaded();v3.mode_train(m);seg=v3.e24.ReTraceLoss.from_config(v3.e24.clean_loss_config(cfg['loss'])).cuda()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):terms=v3.objective(v3.fwd(m,batch,cfg,False),batch,seg,cfg)
  state['loss_components']={k:float(x) for k,x in terms.items() if torch.is_tensor(x) and x.numel()==1};dump('summary.yaml',state);del m;torch.cuda.empty_cache()

  # Fresh identical initialization, final mask BCE+Dice only, no DCT dropout.
  torch.manual_seed(3407);np.random.seed(3407);random.seed(3407);m=v3.e.build_final_model(cfg['model']).cuda();v3.mode_train(m)
  opt=torch.optim.SGD(m.parameters(),lr=.001,momentum=.9,weight_decay=.0005);hist=[]
  def final_only(out,b):
   bs,_,fh,fw=b['image'].shape;loss=0
   for i in range(bs):
    vh,vw=map(int,b['valid_hw'][i]);z=out['mask_logits'][i:i+1,:,:vh,:vw].float();y=b['mask'][i:i+1,:,:vh,:vw].cuda().float()
    bce=F.binary_cross_entropy_with_logits(z,y);p=z.sigmoid();inter=(p*y).sum();dice=1-(2*inter+1)/(p.sum()+y.sum()+1);loss=loss+(bce+dice)/bs
   return loss
  for step in range(1,801):
   ids=[((step-1)*4+j)%32 for j in range(4)];b=next(iter(DataLoader(ds_train,batch_size=4,sampler=[(i,900000+step*10+j) for j,i in enumerate(ids)],num_workers=0)))
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=final_only(v3.fwd(m,b,cfg,False),b)
   loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);opt.step()
   if step==1 or step%25==0:dump('progress.yaml',{'stage':'final_only_overfit','step':step,'loss':float(loss)});print('[final-only]',step,float(loss),flush=True)
   if step%100==0:
    m.eval();r=evaluate(m,fixed,f'final_only_step{step:04d}','diagnostic_final_only_overfit');q=r['pixel_localization_fake_only']['test_f1_oracle_diagnostic'];hist.append({'step':step,'loss':float(loss),'f1':q['f1'],'precision':q['precision'],'recall':q['recall'],'threshold':q['threshold'],'auc':r['pixel_localization_fake_only']['auc']});state['final_only_history']=hist;dump('summary.yaml',state)
    if q['f1']>=.9:break
    v3.mode_train(m)
  state.update(status='completed',final_only_pass=bool(hist and hist[-1]['f1']>=.9),final_step=hist[-1]['step']);dump('summary.yaml',state)
 except BaseException:
  state.update(status='failed',stop_reason=traceback.format_exc());dump('summary.yaml',state);raise
if __name__=='__main__':main()
