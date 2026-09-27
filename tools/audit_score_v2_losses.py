#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch, yaml
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.utils import choose_device, load_checkpoint, load_config
from retracenet.score_model_v2 import build_score_model_v2
from train_v2_final_pipeline import clean_loss_config, tversky_loss
from train_score_model_v2 import aux_pp_loss

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--device'); args=ap.parse_args()
    cfg=load_config(args.config); device=choose_device(args.device)
    model=build_score_model_v2(cfg['model']).to(device)
    ck=load_checkpoint(args.checkpoint, torch.device('cpu'))
    model.load_state_dict(ck['model'], strict=False); model.freeze_teacher(); model.train()
    loader=build_loader({**cfg['data'],'batch_size':1,'num_workers':0},'train',shuffle=False)
    b=next(iter(loader)); im=b['image'].to(device); mask=b['mask'].to(device); lab=b['label'].to(device)
    crit=ReTraceLoss.from_config(clean_loss_config(cfg['loss'])).to(device)
    out=model(im, return_teacher=True)
    losses=crit(out,mask,lab)
    loss_weights=dict(cfg['loss'].get('weights',{}))
    rows=[]
    for k,v in losses.items():
        if k=='total': continue
        w=float(loss_weights.get(k, 1.0 if k in ['localization','dice'] else 0.0))
        rows.append({'term':k,'raw_value':float(v.detach().cpu()),'configured_weight':w,'weighted_contribution_to_total':float((v*w).detach().cpu()),'requires_grad':bool(v.requires_grad)})
    tv=tversky_loss(out['mask_logits'],mask,float(cfg['loss'].get('tversky_alpha_fp',0.3)),float(cfg['loss'].get('tversky_beta_fn',0.7)))
    tvw=float(loss_weights.get('tversky',0.0))
    rows.append({'term':'tversky_external','raw_value':float(tv.detach().cpu()),'configured_weight':tvw,'weighted_contribution_to_total':float((tv*tvw).detach().cpu()),'requires_grad':bool(tv.requires_grad)})
    pp=aux_pp_loss(out,mask,cfg['training'].get('deep_supervision_weights') or [0.3,0.2,0.1])
    rows.append({'term':'positive_preserving_deep_supervision_external','raw_value':float(pp.detach().cpu()),'configured_weight':1.0,'weighted_contribution_to_total':float(pp.detach().cpu()),'requires_grad':bool(pp.requires_grad)})
    kd=F.binary_cross_entropy_with_logits(out['mask_logits'],out['teacher_logits'].detach().sigmoid())
    rows.append({'term':'teacher_distill_external_E1','raw_value':float(kd.detach().cpu()),'configured_weight':0.15,'weighted_contribution_to_total':float((kd*0.15).detach().cpu()),'requires_grad':bool(kd.requires_grad)})
    intended_sum=sum(r['weighted_contribution_to_total'] for r in rows if r['term'] in ['localization','dice','tversky_external','positive_preserving_deep_supervision_external','teacher_distill_external_E1'])
    unintended=[r for r in rows if r['term'] in ['counterfactual','proposal','intervention_locality','orthogonality','variance_regularization','ledger','degradation_consistency'] and abs(r['weighted_contribution_to_total'])>1e-12]
    report={'checkpoint_epoch':ck.get('epoch'),'eval_variant':ck.get('eval_variant'),'loss_rows':rows,'intended_weighted_sum_E1':intended_sum,'unintended_nonzero_terms':unintended,'status':'PASS' if not unintended else 'FAIL'}
    print(yaml.safe_dump(report,sort_keys=False))
if __name__=='__main__': main()
