#!/usr/bin/env python
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.utils import choose_device, load_config, set_seed
from retracenet.final_modules import build_final_model
from train_v2_b0 import load_initial_weights
from train_v2_final_pipeline import (
    build_optimizer,
    clean_loss_config,
    dher_v2_aux_loss,
    load_initial_weights_repaired,
    resize_mask_for_supervision,
    set_trainable,
    tversky_loss,
)


def rms(x):
    return float(x.detach().float().square().mean().sqrt().cpu())


def grad_norm(model, prefix):
    vals=[]
    for n,p in model.named_parameters():
        if n.startswith(prefix) and p.grad is not None:
            vals.append(p.grad.detach().float().norm())
    if not vals:
        return 0.0
    return float(torch.stack(vals).norm().cpu())


def grad_rms(param):
    if param.grad is None:
        return 0.0
    return rms(param.grad)


def param_rms(param):
    return rms(param.detach())


def get_batch(cfg, split, device):
    data = {**cfg['data'], 'batch_size': 1, 'num_workers': 0}
    loader = build_loader(data, split, shuffle=True)
    fallback = None
    for batch in loader:
        if fallback is None:
            fallback = batch
        if float(batch['mask'].sum()) > 0:
            return batch
    return fallback


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--init', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--steps', type=int, default=5)
    args=ap.parse_args()
    cfg=load_config(args.config); set_seed(int(cfg.get('seed',3407))); device=choose_device(args.device)

    # tiny-mask supervision survival check
    synth=torch.zeros(1,1,513,517,device=device); synth[..., 12, 17]=1.0; synth[..., 250:252, 330:333]=1.0
    mask_checks={}
    for size in [(384,384),(256,256),(128,128),(64,64),(32,32),(17,19)]:
        y=resize_mask_for_supervision(synth, size, hard=True)
        mask_checks[f'{size[0]}x{size[1]}']={'sum':float(y.sum().cpu()), 'finite':bool(torch.isfinite(y).all().cpu())}

    parent_cfg=copy.deepcopy(cfg['model']); parent_cfg['architecture']='retracenet_b3_qtr_hcv2'
    child_cfg=copy.deepcopy(cfg['model']); child_cfg['architecture']='retracenet_b3_qtr_hcv2_dher_v21'
    parent=build_final_model(parent_cfg).to(device).eval()
    child=build_final_model(child_cfg).to(device).eval()
    reinit_keys = list(cfg.get('training', {}).get('reinitialize_keys') or [])
    reinit_prefixes = list(cfg.get('training', {}).get('reinitialize_prefixes') or [])
    construction = {
        'alpha_support': float(child.dher.alpha_support.detach().cpu()),
        'alpha_conflict': float(child.dher.alpha_conflict.detach().cpu()),
        'support_fuse_final_weight_rms': param_rms(child.dher.support_fuse[-1].weight),
        'support_fuse_final_bias_rms': param_rms(child.dher.support_fuse[-1].bias),
        'reexam_final_weight_rms': param_rms(child.dher.reexam[-1].weight),
        'reexam_final_bias_rms': param_rms(child.dher.reexam[-1].bias),
    }
    pr=load_initial_weights_repaired(parent,args.init,device,[],['dher.'],[])
    cr=load_initial_weights_repaired(child,args.init,device,[],reinit_prefixes,reinit_keys)
    after_load = {
        'alpha_support': float(child.dher.alpha_support.detach().cpu()),
        'alpha_conflict': float(child.dher.alpha_conflict.detach().cpu()),
        'support_fuse_final_weight_rms': param_rms(child.dher.support_fuse[-1].weight),
        'support_fuse_final_bias_rms': param_rms(child.dher.support_fuse[-1].bias),
        'reexam_final_weight_rms': param_rms(child.dher.reexam[-1].weight),
        'reexam_final_bias_rms': param_rms(child.dher.reexam[-1].bias),
    }
    allowed_missing = set(reinit_keys)
    bad_parent=[k for k in pr['missing'] if not k.startswith('dher.')]
    bad_child=[k for k in cr['missing'] if k not in allowed_missing and not any(k.startswith(p) for p in reinit_prefixes)]
    strict_ok=(not bad_parent and not bad_child and not pr['unexpected'] and not cr['unexpected'] and not pr['skipped_shape'] and not cr['skipped_shape'])

    batch=get_batch(cfg,'val',device); im=batch['image'].to(device); mask=batch['mask'].to(device); lab=batch['label'].to(device)
    with torch.inference_mode():
        po=parent(im); co=child(im)
        logits_diff=float((po['mask_logits']-co['mask_logits']).abs().max().cpu())
        hcv_diffs=[]
        for key in ['proposals','deltas','energies']:
            for a,b in zip(po['hcv2'][key], co['hcv2'][key]):
                hcv_diffs.append(float((a-b).abs().max().cpu()))
        hcv_max_diff=max(hcv_diffs) if hcv_diffs else 0.0
        support_write_rms0=rms(co['dher']['support_write'])
        conflict_write_rms0=rms(co['dher']['conflict_write'])
        d1_rms0=float(co['dher']['d1_rms'].cpu()) if 'd1_rms' in co['dher'] else 0.0
        sg=co['dher']['support_gate'].float(); cg=co['dher']['conflict_gate'].float()
        tgt=resize_mask_for_supervision(mask.float(), sg.shape[-2:], hard=True)>0.5
        if tgt.shape[1] != sg.shape[1]: tgt=tgt.expand_as(sg)
        support_gate_fg=float(sg[tgt].mean().cpu()) if tgt.any() else 0.0
        support_gate_bg=float(sg[~tgt].mean().cpu()) if (~tgt).any() else 0.0
        tgtc=resize_mask_for_supervision(mask.float(), cg.shape[-2:], hard=True)>0.5
        if tgtc.shape[1] != cg.shape[1]: tgtc=tgtc.expand_as(cg)
        conflict_gate_fg=float(cg[tgtc].mean().cpu()) if tgtc.any() else 0.0
        conflict_gate_bg=float(cg[~tgtc].mean().cpu()) if (~tgtc).any() else 0.0

    child.train(); set_trainable(child,'mechanism_repair')
    opt=build_optimizer(child,cfg,'mechanism_repair')
    seg_loss=ReTraceLoss.from_config(clean_loss_config(cfg['loss'])).to(device)
    rows=[]
    data={**cfg['data'], 'batch_size': 1, 'num_workers': 0}
    loader=iter(build_loader(data,'train',shuffle=True))
    for step in range(1,args.steps+1):
        try:
            b=next(loader)
        except StopIteration:
            loader=iter(build_loader(data,'train',shuffle=True)); b=next(loader)
        im=b['image'].to(device); mask=b['mask'].to(device); lab=b['label'].to(device)
        opt.zero_grad(set_to_none=True)
        out=child(im)
        losses=seg_loss(out,mask,lab); loss=losses['total']
        aux=dher_v2_aux_loss(out,mask,cfg); loss=loss+aux['dher_aux_total']
        tv_weight=float(cfg['loss']['weights'].get('tversky',0.0))
        if tv_weight>0:
            loss=loss+tv_weight*tversky_loss(out['mask_logits'],mask,float(cfg['loss'].get('tversky_alpha_fp',0.3)),float(cfg['loss'].get('tversky_beta_fn',0.7)))
        loss.backward()
        row={
            'step':step,
            'loss':float(loss.detach().cpu()),
            'alpha_support_before':float(child.dher.alpha_support.detach().cpu()),
            'alpha_conflict_before':float(child.dher.alpha_conflict.detach().cpu()),
            'grad_alpha_support':float(child.dher.alpha_support.grad.detach().cpu()) if child.dher.alpha_support.grad is not None else 0.0,
            'grad_alpha_conflict':float(child.dher.alpha_conflict.grad.detach().cpu()) if child.dher.alpha_conflict.grad is not None else 0.0,
            'grad_scale_heads':grad_norm(child,'dher.scale_heads'),
            'grad_support_fuse':grad_norm(child,'dher.support_fuse'),
            'grad_reexam':grad_norm(child,'dher.reexam'),
            'grad_support_fuse_final_weight_rms':grad_rms(child.dher.support_fuse[-1].weight),
            'grad_support_fuse_final_bias_rms':grad_rms(child.dher.support_fuse[-1].bias),
            'grad_reexam_final_weight_rms':grad_rms(child.dher.reexam[-1].weight),
            'grad_reexam_final_bias_rms':grad_rms(child.dher.reexam[-1].bias),
            'd1_rms_forward':float(out['dher']['d1_rms'].detach().cpu()) if 'd1_rms' in out['dher'] else 0.0,
            'support_write_rms_forward':rms(out['dher']['support_write']),
            'conflict_write_rms_forward':rms(out['dher']['conflict_write']),
        }
        opt.step()
        row.update({
            'alpha_support_after':float(child.dher.alpha_support.detach().cpu()),
            'alpha_conflict_after':float(child.dher.alpha_conflict.detach().cpu()),
        })
        rows.append(row)

    status={
        'construction': construction,
        'after_load': after_load,
        'strict_checkpoint_ok': strict_ok,
        'parent_missing_count': len(pr['missing']),
        'child_missing_count': len(cr['missing']),
        'child_missing_examples': cr['missing'][:12],
        'loaded_dher_keys_count': sum(1 for k in child.state_dict() if k.startswith('dher.') and k not in set(cr.get('missing', []))),
        'reinitialized_dher_keys_count': len([k for k in cr.get('skipped_reinitialized', []) if k.startswith('dher.')]),
        'reinitialized_key_names': cr.get('skipped_reinitialized', []),
        'zero_effect_logits_max_abs_diff': logits_diff,
        'zero_effect_hcv_max_abs_diff': hcv_max_diff,
        'step0_d1_rms': d1_rms0,
        'step0_support_write_rms': support_write_rms0,
        'step0_conflict_write_rms': conflict_write_rms0,
        'step0_support_write_rms_over_d1_rms': support_write_rms0/(d1_rms0+1e-12),
        'step0_conflict_write_rms_over_d1_rms': conflict_write_rms0/(d1_rms0+1e-12),
        'support_gate_fg': support_gate_fg,
        'support_gate_bg': support_gate_bg,
        'conflict_gate_fg': conflict_gate_fg,
        'conflict_gate_bg': conflict_gate_bg,
        'tiny_mask_resize_checks': mask_checks,
        'gradient_rows': rows,
    }
    status['pass']=(strict_ok and logits_diff <= 1e-6 and hcv_max_diff <= 1e-6 and support_write_rms0 == 0.0 and conflict_write_rms0 == 0.0 and abs(after_load['alpha_support']-0.25)<1e-6 and abs(after_load['alpha_conflict']-0.25)<1e-6 and support_gate_fg > support_gate_bg and conflict_gate_fg > conflict_gate_bg and rows[0]['grad_support_fuse_final_weight_rms']>0 and rows[0]['grad_reexam_final_weight_rms']>0 and (rows[1]['support_write_rms_forward']>0 or rows[1]['conflict_write_rms_forward']>0))
    print(yaml.safe_dump(status, sort_keys=False))

if __name__=='__main__':
    main()
