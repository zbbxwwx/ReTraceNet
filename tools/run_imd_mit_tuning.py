"""Resumable public-initialized Stable-BN in-domain training for NIST16/IMD2020."""
import argparse,copy,json,math,random,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from retrace.data import Record
import run_balanced_residual_v3_full30 as v3
import run_cross_dataset_after_chain as cross
from fair_training_data_tuning import FairDataset,crop_outputs

TRU = Path('/group4/zouwenxuan/iml_generalization/repos/TruFor/TruFor_train_test')
sys.path.insert(0, str(TRU))
MIT_WEIGHTS = TRU / 'pretrained_models/segformers/mit_b2.pth'

class MiTB2ContentCompat(nn.Module):
 def __init__(self):
  super().__init__()
  from lib.models.cmx.encoders.dual_segformer import mit_b2
  self.net=mit_b2();self.net.init_weights_onepath(str(MIT_WEIGHTS))
  self.proj=nn.ModuleList([nn.Conv2d(64,256,1,bias=False),nn.Conv2d(128,512,1,bias=False),nn.Conv2d(320,1024,1,bias=False),nn.Conv2d(512,2048,1,bias=False)])
 def forward(self,x):
  n,b,feats=self.net,x.shape[0],[]
  stages=((n.patch_embed1,n.block1,n.norm1),(n.patch_embed2,n.block2,n.norm2),(n.patch_embed3,n.block3,n.norm3),(n.patch_embed4,n.block4,n.norm4))
  for i,(pe,blocks,norm) in enumerate(stages):
   x,h,w=pe(x)
   for blk in blocks:x=blk(x,h,w)
   x=norm(x).reshape(b,h,w,-1).permute(0,3,1,2).contiguous();feats.append(self.proj[i](x))
  return feats

_original_build=v3.e.build_final_model
def _build_mit(cfg):
 model=_original_build(cfg);model.content_encoder=MiTB2ContentCompat();return model
v3.e.build_final_model=_build_mit

def dump(p,x):v3.e.dump(Path(p),x)
def records(rows,split,name):return [Record(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') else None,split=split,dataset=name,label=int(r['label'])) for r in rows]
def freeze_bn(model):
 n=0
 for branch in (model.content_encoder,model.dct_branch):
  for m in branch.modules():
   if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval();n+=1
 assert n>0;return n
def train_mode(model):v3.mode_train(model);return freeze_bn(model)
def evaluate(model,*args,**kwargs):
 was=model.training;model.eval()
 try:return cross.evaluate_dataset(model,*args,**kwargs)
 finally:
  if was:train_mode(model)

class HCVSemanticBypass(nn.Module):
 """Keep HCV diagnostics/losses, but feed qualified Q features to the decoder.

 This removes only the learned verification correction.  It preserves feature
 shape/scale and avoids the distribution-breaking all-zero HCV ablation.
 """
 def __init__(self,inner):super().__init__();self.inner=inner
 def forward(self,content,detail,qtr=None):
  out=self.inner(content,detail,qtr)
  out['verified_features']=self.inner._apply_q(detail,qtr)
  return out

class VerifiedResidualDecoderStage(nn.Module):
 """Decoder stage that preserves verified evidence through a gated residual."""
 def __init__(self,inner):
  super().__init__();self.inner=inner
  # The existing detail projection produces decoder-width features.
  ch=inner.detail[0].out_channels
  self.evidence_write=nn.Conv2d(ch,ch,1)
  self.evidence_gate=nn.Conv2d(ch*2,ch,1)
  nn.init.zeros_(self.evidence_write.weight);nn.init.zeros_(self.evidence_write.bias)
  nn.init.zeros_(self.evidence_gate.weight);nn.init.zeros_(self.evidence_gate.bias)
 def forward(self,content,detail,parent=None):
  c=self.inner.content(content);v=self.inner.detail(detail);parts=[c,v]
  if self.inner.has_parent:
   if parent is None:raise ValueError('parent feature is required')
   parts.append(torch.nn.functional.interpolate(parent,size=c.shape[-2:],mode='bilinear',align_corners=False))
  base=self.inner.fuse(torch.cat(parts,dim=1))
  # These heads are injected after the base model has been moved to CUDA.
  # Run them explicitly in fp32 so BF16 autocast cannot mix BF16 activations
  # with FP32 bias parameters, then restore the decoder activation dtype.
  with torch.autocast(device_type='cuda',enabled=False):
   gate=torch.sigmoid(self.evidence_gate(torch.cat([base.float(),v.float()],dim=1)))
   write=self.evidence_write(v.float())
  return base+gate.to(base.dtype)*write.to(base.dtype)

def enable_verified_residual_decoder(model):
 for name in ('stage1','stage2','stage3','stage4'):
  old=getattr(model.decoder,name);device=next(old.parameters()).device
  setattr(model.decoder,name,VerifiedResidualDecoderStage(old).to(device))

def final_mask_only_objective(out,b):
 """Strict final-output BCE+Dice on valid, unpadded pixels only."""
 bs,_,fh,fw=b['image'].shape; total=out['mask_logits'].new_zeros(())
 for i in range(bs):
  vh,vw=map(int,b['valid_hw'][i].tolist())
  oo=crop_outputs(out,i,bs,vh,vw,fh,fw)
  logits=oo['mask_logits'].float();target=b['mask'][i:i+1,:,:vh,:vw].float().cuda()
  bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
  prob=logits.sigmoid();inter=(prob*target).sum((1,2,3));den=prob.sum((1,2,3))+target.sum((1,2,3))
  dice=(1-(2*inter+1)/(den+1)).mean()
  total=total+(bce+dice)/bs
 return {'total':total,'final_bce_dice':total}

def final_focal_loss(out,b,gamma=2.0,alpha=0.5):
 """Focal replacement for final localization BCE only; Dice/auxiliaries stay unchanged."""
 bs,_,fh,fw=b['image'].shape;total=out['mask_logits'].new_zeros(())
 for i in range(bs):
  vh,vw=map(int,b['valid_hw'][i].tolist())
  oo=crop_outputs(out,i,bs,vh,vw,fh,fw)
  logits=oo['mask_logits'].float();target=b['mask'][i:i+1,:,:vh,:vw].float().cuda()
  bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
  pt=torch.exp(-bce)
  alpha_t=alpha*target+(1.0-alpha)*(1.0-target)
  total=total+(alpha_t*(1.0-pt).pow(gamma)*bce).mean()/bs
 return total

def aux_seg_loss(logits,target):
 z=torch.nn.functional.interpolate(logits.float(),size=target.shape[-2:],mode='bilinear',align_corners=False)
 bce=torch.nn.functional.binary_cross_entropy_with_logits(z,target)
 p=z.sigmoid();inter=(p*target).sum((1,2,3));den=p.sum((1,2,3))+target.sum((1,2,3))
 return bce+(1-(2*inter+1)/(den+1)).mean()

def targeted_decoder_ds(out,b):
 """0.25*(.4 D1pre + .3 D2 + .2 D3 + .1 D4), all supervising GT mask."""
 bs,_,fh,fw=b['image'].shape;total=out['mask_logits'].new_zeros(())
 for i in range(bs):
  vh,vw=map(int,b['valid_hw'][i].tolist());oo=crop_outputs(out,i,bs,vh,vw,fh,fw);target=b['mask'][i:i+1,:,:vh,:vw].float().cuda()
  logits=[oo['decoder_d1_pre_logits'],*oo['aux_logits_low']]
  total=total+0.25*sum(w*aux_seg_loss(z,target) for w,z in zip((.4,.3,.2,.1),logits))/bs
 return total

def region_affinity_loss(out,b):
 """Balanced same-region / boundary supervision for the active propagation graph."""
 diag=out['splicing_region'];hl=diag['horizontal_logits'].float();vl=diag['vertical_logits'].float()
 raw=b['mask'].float().cuda();valid=torch.zeros_like(raw,dtype=torch.bool)
 for i,hw in enumerate(b['valid_hw']):
  vh,vw=map(int,hw.tolist());valid[i,:,:vh,:vw]=True
 target=F.interpolate(raw,size=(hl.shape[-2],vl.shape[-1]),mode='nearest')>0.5
 valid=F.interpolate(valid.float(),size=target.shape[-2:],mode='nearest')>0.5
 losses=[];used=0
 for i in range(target.shape[0]):
  if not bool((target[i]&valid[i]).any()):continue
  ht=(target[i:i+1,:,:,1:]==target[i:i+1,:,:,:-1]).float();vt=(target[i:i+1,:,1:,:]==target[i:i+1,:,:-1,:]).float()
  lh=F.binary_cross_entropy_with_logits(hl[i:i+1],ht,reduction='none');lv=F.binary_cross_entropy_with_logits(vl[i:i+1],vt,reduction='none')
  hv=valid[i:i+1,:,:,1:]&valid[i:i+1,:,:,:-1];vv=valid[i:i+1,:,1:,:]&valid[i:i+1,:,:-1,:]
  same=torch.cat([lh[(ht>0.5)&hv],lv[(vt>0.5)&vv]]);cross=torch.cat([lh[(ht<=0.5)&hv],lv[(vt<=0.5)&vv]])
  if cross.numel():losses.append(0.5*same.mean()+0.5*cross.mean());used+=1
 loss=torch.stack(losses).mean() if losses else hl.sum()*0.0+vl.sum()*0.0
 return loss,used

def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);summary={'status':'preparing','dataset':a.dataset,'experiment':'Stable-BN in-domain public initialization','test_run':False};dump(out/'summary.yaml',summary)
 try:
  split=json.loads(Path(a.split).read_text());tr,va,te=(records(split[k],k,a.dataset) for k in ('train','val','test'));fake=[i for i,r in enumerate(tr) if r.label];auth=[i for i,r in enumerate(tr) if not r.label];assert fake and auth and va and te
  template=torch.load(a.template,map_location='cpu');cfg=copy.deepcopy(template['config']);cfg['training']['epochs']=a.epochs;cfg['training']['dct_sample_dropout']=.2
  if a.splicing_region_affinity:
   cfg['model']['enable_splicing_region_affinity']=True;cfg['model']['splicing_affinity_channels']=32
  torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);model=v3.e.build_final_model(cfg['model']).cuda()
  cue_ablation_handle=None
  if a.forensic_mode=='residual_only':
   model.disable_dct_evidence=True
  elif a.forensic_mode=='dct_only':
   def _zero_residual(_module,_args,output):
    if isinstance(output,tuple):return tuple(torch.zeros_like(x) for x in output)
    if isinstance(output,list):return [torch.zeros_like(x) for x in output]
    raise TypeError(f'unsupported forensic output: {type(output)}')
   cue_ablation_handle=model.forensic_stream.register_forward_hook(_zero_residual)
  if a.hcv_semantic_bypass:model.hcv2=HCVSemanticBypass(model.hcv2)
  if a.verified_residual_decoder:enable_verified_residual_decoder(model)
  d1_capture={}
  capture_handle=None
  if a.decoder_ds_targeted:
   def _capture_d1(_module,args):d1_capture['feature']=args[0]
   capture_handle=model.dher.register_forward_pre_hook(_capture_d1)
  bn_count=train_mode(model);seg=v3.e24.ReTraceLoss.from_config(v3.e24.clean_loss_config(cfg['loss'])).cuda()
  if a.variant=='group_lr':
   content_ids={id(p) for p in model.content_encoder.parameters()}
   content=[p for p in model.parameters() if id(p) in content_ids]
   task=[p for p in model.parameters() if id(p) not in content_ids]
   opt=torch.optim.SGD([{'params':content,'lr':a.learning_rate*0.1,'lr_scale':0.1},{'params':task,'lr':a.learning_rate,'lr_scale':1.0}],lr=a.learning_rate,momentum=.9,weight_decay=.0005)
  else:opt=torch.optim.SGD(model.parameters(),lr=a.learning_rate,momentum=.9,weight_decay=.0005)
  history=[];best=-1.;best_ep=0;start=1;nf=min(len(fake),len(auth))
  coverage_auth=math.ceil(len(fake)/3) if a.variant=='coverage_first' else nf
  steps=math.ceil((len(fake)+coverage_auth)/a.batch_size) if a.variant=='coverage_first' else math.ceil(2*nf/a.batch_size)
  gs=0
  if a.resume and (out/'last.pt').exists():
   ck=torch.load(out/'last.pt',map_location='cpu');model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);history=ck.get('history',[]);best=float(ck.get('best_metric',-1));start=int(ck['epoch'])+1;gs=int(ck.get('global_step',int(ck['epoch'])*steps))
   try:best_ep=int((__import__('yaml').safe_load((out/'summary.yaml').read_text()) or {}).get('best_epoch',0))
   except Exception:best_ep=max(history,key=lambda r:r.get('val_fake_f1',r.get('val_fake_F1',-1)))['epoch'] if history else 0
  FairDataset.TRAIN_CROP=a.train_crop
  schedule_total=a.schedule_total_steps if a.schedule_total_steps>0 else a.epochs*steps
  total=schedule_total;ds=FairDataset(tr,out/'jpeg_cache_train',True,positive_crop_prob=a.positive_crop_prob,positive_crop_retries=a.positive_crop_retries);summary.update(status='training',epochs=a.epochs,start_epoch=start,counts={k:len(split[k]) for k in split},bn_stats_frozen=True,bn_module_count=bn_count,history=history,best_F1=best,best_epoch=best_ep,positive_crop_prob=a.positive_crop_prob,positive_crop_retries=a.positive_crop_retries,forensic_mode=a.forensic_mode,hcv_semantic_bypass=bool(a.hcv_semantic_bypass),verified_residual_decoder=bool(a.verified_residual_decoder),verified_residual_initialization='zero-effect 1x1 evidence write; spatial gate initialized 0 logits' if a.verified_residual_decoder else None,decoder_ds_targeted=bool(a.decoder_ds_targeted),decoder_ds_definition='0.25*(0.4*D1pre + 0.3*D2 + 0.2*D3 + 0.1*D4), BCE+Dice; replaces existing decoder DS' if a.decoder_ds_targeted else None,final_mask_only=bool(a.final_mask_only),training_objective=('final mask BCE+Dice only; all auxiliary/deep/boundary/image/uncertainty losses disabled' if a.final_mask_only else 'original composite objective'),decoder_forensic_input=('QTR-qualified features with HCV correction removed' if a.hcv_semantic_bypass else 'normal HCV verified features'),auxiliary_loss_scales={'hcv':a.hcv_aux_scale,'qtr':a.qtr_aux_scale,'dher':a.dher_aux_scale},splicing_region_affinity=bool(a.splicing_region_affinity),region_affinity_loss_weight=a.region_affinity_weight,variant=a.variant,train_crop=a.train_crop,schedule_total_steps=schedule_total,steps_per_cycle=steps,coverage_cycle={'fake':len(fake),'auth':coverage_auth,'ratio':len(fake)/coverage_auth} if a.variant=='coverage_first' else None);dump(out/'summary.yaml',summary)
  for ep in range(start,a.epochs+1):
   gen=torch.Generator().manual_seed(a.seed+ep)
   if a.variant=='coverage_first':
    fake_ids=[fake[i] for i in torch.randperm(len(fake),generator=gen)]
    auth_perm=torch.randperm(len(auth),generator=gen).tolist();auth_ids=[]
    while len(auth_ids)<coverage_auth:auth_ids.extend(auth[i] for i in auth_perm)
    ids=fake_ids+auth_ids[:coverage_auth]
   else:ids=[fake[i] for i in torch.randperm(len(fake),generator=gen)[:nf]]+[auth[i] for i in torch.randperm(len(auth),generator=gen)[:nf]]
   ids=[ids[i] for i in torch.randperm(len(ids),generator=gen)];plan=[(idx,a.seed*1000000+ep*100000+j) for j,idx in enumerate(ids)];loader=DataLoader(ds,batch_size=a.batch_size,sampler=plan,num_workers=2,pin_memory=True);train_mode(model);tot=0.
   for st,b in enumerate(loader,1):
    lr=a.learning_rate*(1-min(gs/max(total,1),1))**.9
    for g in opt.param_groups:g['lr']=lr*float(g.get('lr_scale',1.0))
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     raw_out=v3.fwd(model,b,cfg,True)
     if a.decoder_ds_targeted:
      if 'feature' not in d1_capture:raise RuntimeError('D1pre capture missing')
      raw_out['decoder_d1_pre_logits']=model.mask_head(d1_capture.pop('feature'))
     if a.final_mask_only:
      terms=final_mask_only_objective(raw_out,b);loss=terms['total']
     else:
      terms=v3.objective(raw_out,b,seg,cfg)
      loss=terms['total']+(a.hcv_aux_scale-1.0)*terms['hcv_existing_supervision']+(a.qtr_aux_scale-0.15)*terms['qtr_direct_supervision']+(a.dher_aux_scale-0.05)*terms['dher_direct_supervision']
      if a.variant=='focal_dice':
       focal=final_focal_loss(raw_out,b,gamma=2.0,alpha=0.5)
       # localization has weight 1.0 in the locked baseline config. Replace
       # only this final-mask term; Dice, boundary, Tversky and every module
       # auxiliary term remain bit-for-bit on the baseline path.
       loss=loss-terms['localization']+focal
       terms['final_focal']=focal
     if a.decoder_ds_targeted:
      old=terms.get('weighted_deep_supervision',loss.new_zeros(()));new=targeted_decoder_ds(raw_out,b);loss=loss-old+new;terms['decoder_ds_targeted']=new
     if a.splicing_region_affinity:
      affinity,used=region_affinity_loss(raw_out,b);loss=loss+a.region_affinity_weight*affinity;terms['region_affinity']=affinity;terms['region_affinity_samples']=loss.new_tensor(float(used))
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();gs+=1;tot+=float(loss)
    if st==1 or st%25==0:dump(out/'progress.yaml',{'dataset':a.dataset,'epoch':ep,'epochs':a.epochs,'batch':st,'batches':len(loader),'loss':tot/st,'lr':lr});print('[stable-domain]',a.dataset,ep,st,len(loader),tot/st,flush=True)
   vr=evaluate(model,va,out/'jpeg_cache_val',None,None,a.dataset,out/f'val_epoch{ep:03d}',evaluation_mode='validation_select_eval_forced',threshold_source=a.dataset+'_val_current');q=vr['pixel_localization_fake_only']['test_f1_oracle_diagnostic'];score=float(q['f1']);row={'epoch':ep,'global_step':gs,'train_total':tot/len(loader),'val_fake_f1':score,'val_fake_precision':q.get('precision'),'val_fake_recall':q.get('recall'),'val_fake_auc':vr['pixel_localization_fake_only']['auc'],'threshold':q['threshold'],'image_threshold':vr['image_level_all']['test_f1_oracle_diagnostic']['threshold']};history.append(row)
   if score>best:best=score;best_ep=ep
   state={'model':v3.e.copy_state(model),'optimizer':opt.state_dict(),'config':cfg,'epoch':ep,'global_step':gs,'history':history,'best_metric':best,'selection_threshold':q['threshold'],'image_selection_threshold':row['image_threshold']};v3.e24.atomic_torch_save(state,out/'last.pt')
   if best_ep==ep:v3.e24.atomic_torch_save(state,out/'best.pt')
   summary.update(history=history,best_F1=best,best_epoch=best_ep,best_checkpoint=str(out/'best.pt'));dump(out/'summary.yaml',summary)
   if a.early_gate_epoch and ep==a.early_gate_epoch:
    gate_rows=[r for r in history if max(1,a.early_gate_epoch-5)<=int(r['epoch'])<=a.early_gate_epoch]
    gate_best=max(float(r['val_fake_f1']) for r in gate_rows)
    summary['early_gate']={'epoch':a.early_gate_epoch,'window_start':max(1,a.early_gate_epoch-5),'window_best_f1':gate_best,'required_f1':a.early_gate_min_f1}
    if gate_best<=a.early_gate_min_f1:
     summary.update(status='completed',test_run=False,stop_reason='early_gate_failed_no_test');dump(out/'summary.yaml',summary);return
  if a.skip_test:
   if a.final_mask_only:
    bc=torch.load(out/'best.pt',map_location='cpu');model.load_state_dict(bc['model'],strict=True)
    train_eval=evaluate(model,tr,out/'jpeg_cache_train_eval',None,None,a.dataset,out/'train_best_eval',evaluation_mode='train_diagnostic_eval_forced',threshold_source=a.dataset+'_train_oracle_diagnostic')
    summary['train_best_diagnostic']=train_eval
   summary.update(status='completed',test_run=False,stop_reason='diagnostic_fixed_budget_complete_no_test');dump(out/'summary.yaml',summary)
  else:
   bc=torch.load(out/'best.pt',map_location='cpu');model.load_state_dict(bc['model'],strict=True);test=evaluate(model,te,out/'jpeg_cache_test',float(bc['selection_threshold']),float(bc['image_selection_threshold']),a.dataset,out/'test_once',evaluation_mode='in_domain_locked',threshold_source=a.dataset+'_val');summary.update(status='completed',test_run=True,test_checkpoint_epoch=best_ep,test=test,stop_reason='fixed_budget_complete_test_once');dump(out/'summary.yaml',summary)
 except BaseException:
  summary.update(status='failed',stop_reason=traceback.format_exc());dump(out/'summary.yaml',summary);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--split',required=True);p.add_argument('--template',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,required=True);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--seed',type=int,default=3407);p.add_argument('--resume',action='store_true');p.add_argument('--positive-crop-prob',type=float,default=0.0);p.add_argument('--positive-crop-retries',type=int,default=20);p.add_argument('--hcv-aux-scale',type=float,default=1.0);p.add_argument('--qtr-aux-scale',type=float,default=0.15);p.add_argument('--dher-aux-scale',type=float,default=0.05);p.add_argument('--forensic-mode',choices=('full','residual_only','dct_only'),default='full');p.add_argument('--hcv-semantic-bypass',action='store_true');p.add_argument('--verified-residual-decoder',action='store_true');p.add_argument('--decoder-ds-targeted',action='store_true');p.add_argument('--final-mask-only',action='store_true');p.add_argument('--splicing-region-affinity',action='store_true');p.add_argument('--region-affinity-weight',type=float,default=0.10);p.add_argument('--early-gate-epoch',type=int,default=0);p.add_argument('--early-gate-min-f1',type=float,default=0.5);p.add_argument('--skip-test',action='store_true');p.add_argument('--variant',choices=('baseline','group_lr','focal_dice','coverage_first'),default='baseline');p.add_argument('--train-crop',type=int,choices=(512,1024),default=512);p.add_argument('--schedule-total-steps',type=int,default=0);main(p.parse_args())
