#!/usr/bin/env python
"""Preflight checks for final overnight pipeline models."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.utils import choose_device, load_config, load_checkpoint, set_seed
from retracenet.final_modules import build_final_model
from train_v2_b0 import load_initial_weights


def clean_loss_config(loss_cfg):
    cfg=dict(loss_cfg)
    cfg.pop('tversky_alpha_fp', None)
    cfg.pop('tversky_beta_fn', None)
    return cfg
from train_v2_final_pipeline import hcv_loss, set_trainable


def load_model(cfg, ckpt, device):
    m=build_final_model(cfg['model']).to(device).eval()
    rep=load_initial_weights(m, ckpt, device, [])
    return m, rep

def main():
    p=argparse.ArgumentParser(); p.add_argument('--parent-config',required=True); p.add_argument('--child-config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--phase',required=True); p.add_argument('--device'); p.add_argument('--output',required=True); a=p.parse_args()
    set_seed(3407); device=choose_device(a.device); pcfg=load_config(a.parent_config); ccfg=load_config(a.child_config)
    parent,_=load_model(pcfg,a.checkpoint,device); child,rep=load_model(ccfg,a.checkpoint,device)
    bad=[k for k in rep['missing'] if not (k.startswith('hcv2.') or k.startswith('qtr.') or k.startswith('dher.'))]
    if bad or rep['unexpected'] or rep['skipped_shape']: raise RuntimeError(f'bad child load bad_missing={bad[:20]} unexpected={rep["unexpected"][:20]} skipped={rep["skipped_shape"][:20]}')
    crit=ReTraceLoss.from_config(clean_loss_config(ccfg['loss'])).to(device); loader=build_loader({**ccfg['data'],'batch_size':1,'num_workers':0},'val',shuffle=False)
    max_logit=0.0; max_loss=0.0; checked=0; hcv_delta_diff=None
    with torch.inference_mode():
        for b in loader:
            if checked>=32: break
            im=b['image'].to(device); mask=b['mask'].to(device); lab=b['label'].to(device)
            o0=parent(im); o1=child(im)
            max_logit=max(max_logit,float((o0['mask_logits']-o1['mask_logits']).abs().max().cpu()))
            max_loss=max(max_loss,float((crit(o0,mask,lab)['total']-crit(o1,mask,lab)['total']).abs().cpu()))
            if 'hcv2' in o0 and 'hcv2' in o1:
                diffs=[float((x-y).abs().max().cpu()) for x,y in zip(o0['hcv2']['deltas'],o1['hcv2']['deltas'])]
                hcv_delta_diff=max(diffs) if hcv_delta_diff is None else max(hcv_delta_diff,max(diffs))
            checked+=1
    if max_logit>1e-6 or max_loss>1e-6: raise RuntimeError(f'step0 mismatch logits={max_logit} loss={max_loss}')
    child.train(); set_trainable(child,a.phase); opt=torch.optim.AdamW([p for p in child.parameters() if p.requires_grad],lr=float(ccfg['training'].get('learning_rate',2e-5)),weight_decay=float(ccfg['training'].get('weight_decay',0.01)))
    smoke=[]
    for step,b in zip(range(5),loader):
        im=b['image'].to(device); mask=b['mask'].to(device); lab=b['label'].to(device); out=child(im)
        losses=hcv_loss(out,mask,lab,ccfg) if a.phase in ['hcv2','qtr_hcv'] else crit(out,mask,lab); loss=losses['total']; loss.backward()
        g_alpha=0.0; g_internal=0.0
        for n,pv in child.named_parameters():
            if pv.grad is None: continue
            if 'alpha' in n or 'lambda' in n or 'q_alpha' in n: g_alpha += float(pv.grad.detach().abs().sum().cpu())
            if any(k in n for k in ['hcv2.','qtr.','dher.']) and 'alpha' not in n: g_internal += float(pv.grad.detach().abs().sum().cpu())
        opt.step(); opt.zero_grad(set_to_none=True)
        smoke.append({'step':step+1,'loss':float(loss.detach().cpu()),'gate_grad_abs_sum':g_alpha,'internal_grad_abs_sum':g_internal})
    if smoke[-1]['internal_grad_abs_sum']<=0: raise RuntimeError('internal gradients blocked')
    out={'status':'PASS','phase':a.phase,'load':{'loaded':rep['loaded'],'missing_count':len(rep['missing']),'unexpected_count':len(rep['unexpected']),'skipped_shape_count':len(rep['skipped_shape'])},'step0':{'batches':checked,'max_abs_logit_diff':max_logit,'max_loss_diff':max_loss,'max_hcv_delta_diff':hcv_delta_diff},'smoke':smoke,'trainable_count':sum(p.numel() for p in child.parameters() if p.requires_grad)}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
