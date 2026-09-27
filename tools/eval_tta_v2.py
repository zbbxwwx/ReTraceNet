#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.metrics import ForensicMetrics
from retrace.utils import choose_device, load_config, set_seed
from retracenet.final_modules import build_final_model
from train_v2_b0 import load_initial_weights
from train_v2_final_pipeline import clean_loss_config

@torch.inference_mode()
def tta_outputs(model, image, scales):
    probs=[]
    h,w=image.shape[-2:]
    for sc in scales:
        if abs(sc-1.0)<1e-8:
            x=image
        else:
            x=F.interpolate(image, size=(max(1,int(round(h*sc))), max(1,int(round(w*sc)))), mode='bilinear', align_corners=False)
        for flip in [False, True]:
            xi=torch.flip(x, dims=[-1]) if flip else x
            p=model(xi)['mask_logits'].sigmoid()
            if flip: p=torch.flip(p, dims=[-1])
            if p.shape[-2:] != (h,w): p=F.interpolate(p, size=(h,w), mode='bilinear', align_corners=False)
            probs.append(p)
    prob=torch.stack(probs).mean(0).clamp(1e-5,1-1e-5)
    logits=torch.logit(prob)
    zeros=torch.zeros_like(logits)
    return {'mask_logits':logits,'base_logits':logits,'proposal_logits':logits,'coarse_logits':logits,'factual_logits':logits,'counterfactual_logits':logits.detach(),'log_var':zeros,'image_logit':logits.mean((2,3)),'aux_logits':[logits, logits, logits],'content_features':[logits],'trace_features':[logits],'quality_descriptor':logits.new_zeros((logits.shape[0],8)),'scale_reliability':[logits.new_full((logits.shape[0],1,1,1),0.5)],'ledger_logits':[logits],'ledger_conflicts':[zeros],'causal_effect':zeros,'intervention_effects':[prob],'intervention_strengths':[prob]}

def evaluate(model, crit, loader, device, amp, scales=None):
    model.eval(); metrics=ForensicMetrics(fixed_threshold=0.5); loss_sum={}; count=0
    for b in loader:
        im=b['image'].to(device); mask=b['mask'].to(device); lab=b['label'].to(device)
        with torch.autocast(device_type=device.type,enabled=amp):
            out = tta_outputs(model, im, scales) if scales else model(im)
            losses=crit(out,mask,lab)
        for k,v in losses.items(): loss_sum[k]=loss_sum.get(k,0.0)+float(v.detach().cpu())*im.shape[0]
        count+=im.shape[0]; metrics.update(out,mask,lab)
    return {k:v/max(1,count) for k,v in loss_sum.items()}, metrics.compute(101)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--device'); ap.add_argument('--output',required=True); args=ap.parse_args()
    set_seed(3407); cfg=load_config(args.config); device=choose_device(args.device)
    model=build_final_model(cfg['model']).to(device)
    rep=load_initial_weights(model,args.checkpoint,device,[])
    if rep['unexpected'] or rep['skipped_shape']: raise RuntimeError(rep)
    crit=ReTraceLoss.from_config(clean_loss_config(cfg['loss'])).to(device)
    loader=build_loader({**cfg['data'],'batch_size':1,'num_workers':0},'val',shuffle=False)
    amp=bool(cfg['training'].get('amp',False) and device.type=='cuda')
    _,single=evaluate(model,crit,loader,device,amp,None)
    _,tta=evaluate(model,crit,loader,device,amp,[0.75,1.0,1.25])
    out={'single':single,'tta':tta,'delta':{k:tta.get(k,0.0)-single.get(k,0.0) for k in set(single)|set(tta)}}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
