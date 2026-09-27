#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.utils import choose_device, load_config
from retracenet.score_model_v2 import build_score_model_v2
from train_score_model_v2 import load_score_v2_parent, set_trainable_score, build_optimizer_score, positive_preserving_resize, aux_pp_loss
from train_v2_final_pipeline import clean_loss_config, tversky_loss

def rms(x): return float(x.detach().float().square().mean().sqrt())

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--device'); args=ap.parse_args()
 cfg=load_config(args.config); device=choose_device(args.device)
 model=build_score_model_v2(cfg['model']).to(device)
 rep=load_score_v2_parent(model,cfg['training']['init_checkpoint'],device)
 names=set_trainable_score(model,'stage_a'); opt=build_optimizer_score(model,cfg,'stage_a')
 loader=build_loader({**cfg['data'],'batch_size':1,'num_workers':0},'train',shuffle=False)
 b=next(iter(loader)); im=b['image'].to(device); mask=b['mask'].to(device); lab=b['label'].to(device)
 crit=ReTraceLoss.from_config(clean_loss_config(cfg['loss'])).to(device)
 out=model(im, return_teacher=True)
 losses=crit(out,mask,lab); loss=losses['total']
 tv=tversky_loss(out['mask_logits'],mask,float(cfg['loss'].get('tversky_alpha_fp',0.3)),float(cfg['loss'].get('tversky_beta_fn',0.7)))
 aux=aux_pp_loss(out,mask,cfg['training']['deep_supervision_weights'])
 kd=torch.nn.functional.binary_cross_entropy_with_logits(out['mask_logits'],out['teacher_logits'].detach().sigmoid())
 total=loss+float(cfg['loss']['weights'].get('tversky',0.0))*tv+aux+0.30*kd
 total.backward()
 grad={n:rms(p.grad) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
 report={
  'init_loaded_keys': rep['loaded_keys'],
  'missing_count': len(rep['missing_keys']),
  'missing_examples': rep['missing_key_examples'][:30],
  'trainable_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
  'trainable_examples': names[:40],
  'optimizer_groups':[{'name':g.get('name'),'lr':g['lr'],'parameter_count':sum(p.numel() for p in g['params'])} for g in opt.param_groups],
  'output_shapes': {'logits':list(out['mask_logits'].shape),'teacher_logits':list(out['teacher_logits'].shape),'aux_low':[list(x.shape) for x in out['aux_logits_low']], 'pyramid':[list(x.shape) for x in out['fpn_pyramid']]},
  'losses': {'main':float(loss.detach().cpu()),'tversky':float(tv.detach().cpu()),'pp_aux':float(aux.detach().cpu()),'distill_e0':float(kd.detach().cpu()),'total':float(total.detach().cpu())},
  'grad_rms_examples': {k:grad[k] for k in list(grad)[:30]},
  'fpn_grad_nonzero': any(k.startswith('fpn_decoder') and v>0 for k,v in grad.items()),
  'mask_grad_nonzero': any(k.startswith('mask_head') and v>0 for k,v in grad.items()),
 }
 print(yaml.safe_dump(report,sort_keys=False))
if __name__=='__main__': main()
