#!/usr/bin/env python
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import torch
import yaml
from PIL import Image
import numpy as np

ROOT=Path('/group4/zouwenxuan/iml_generalization/repos/ReTraceNet')
sys.path.insert(0, str(ROOT))
from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.utils import load_config, choose_device, set_seed
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import load_initial_weights_repaired, clean_loss_config, set_trainable, build_optimizer, forward_from_batch

base_cfg_path=ROOT/'configs/casia2_retracenet_score_sprint_c.yaml'
e3_cfg_path=ROOT/'configs/casia2_retracenet_score_sprint_c_e3_qtr_gate.yaml'
base_ckpt='/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/score_sprint_c_20260831/best.pt'
outdir=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/score_sprint_c_e3_qtr_gate_20260831')
outdir.mkdir(parents=True, exist_ok=True)

def split_report(manifest: str, image_size: int):
    by_split={}
    paths={}
    tiny_total=tiny_survive=0
    with open(manifest, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            sp=r['split']; by_split[sp]=by_split.get(sp,0)+1; paths.setdefault(sp,set()).add(r['image'])
            if sp=='train' and r.get('mask'):
                try:
                    m=np.array(Image.open(r['mask']).convert('L'))>0
                except Exception:
                    continue
                ratio=float(m.mean())
                if 0 < ratio < 0.005:
                    tiny_total += 1
                    im=Image.fromarray(m.astype('uint8')*255).resize((image_size,image_size), Image.NEAREST)
                    tiny_survive += int((np.array(im)>0).any())
    overlap={}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        overlap[f'{a}_{b}']=len(paths.get(a,set()) & paths.get(b,set()))
    return {'counts':by_split, 'overlap':overlap, 'train_tiny_lt_0p5_masks':tiny_total, 'train_tiny_survive_640':tiny_survive}

def grad_sum(named, substr):
    vals=[]
    for n,p in named:
        if substr in n:
            vals.append({'name':n,'requires_grad':p.requires_grad,'grad_abs_sum':None if p.grad is None else float(p.grad.detach().abs().sum().cpu())})
    return vals

set_seed(3407)
device=choose_device('cuda')
base_cfg=load_config(str(base_cfg_path)); e3_cfg=load_config(str(e3_cfg_path))
base=build_final_model(base_cfg['model']).to(device).eval()
e3=build_final_model(e3_cfg['model']).to(device)
base_init=load_initial_weights_repaired(base, base_ckpt, device)
e3_init=load_initial_weights_repaired(e3, base_ckpt, device)
loader=build_loader({**e3_cfg['data'],'batch_size':1}, 'val', shuffle=False)
batch=next(iter(loader))
with torch.no_grad():
    b0=forward_from_batch(base,batch,device)['mask_logits']
    g0=forward_from_batch(e3.eval(),batch,device)['mask_logits']
diff=(b0-g0).abs()
trainable=set_trainable(e3,'score_sprint_c')
opt=build_optimizer(e3,e3_cfg,'score_sprint_c')
loss_fn=ReTraceLoss.from_config(clean_loss_config(e3_cfg['loss'])).to(device)
e3.train(); opt.zero_grad(set_to_none=True)
out=forward_from_batch(e3,batch,device)
losses=loss_fn(out,batch['mask'].to(device),batch['label'].to(device))
loss=losses['total']
loss.backward()
qtr_trainable=[n for n,p in e3.named_parameters() if p.requires_grad and n.startswith('qtr.')]
report={
    'split': split_report(e3_cfg['data']['manifest'], int(e3_cfg['data']['image_size'])),
    'base_init': {'loaded':base_init['loaded'], 'missing_count':len(base_init['missing']), 'unexpected':base_init['unexpected'], 'skipped_shape':base_init['skipped_shape']},
    'e3_init': {'loaded':e3_init['loaded'], 'missing_count':len(e3_init['missing']), 'missing':e3_init['missing'], 'unexpected':e3_init['unexpected'], 'skipped_shape':e3_init['skipped_shape']},
    'step0_logits_vs_e1': {'max_abs_diff':float(diff.max().cpu()), 'mean_abs_diff':float(diff.mean().cpu())},
    'trainable_count': int(sum(p.numel() for p in e3.parameters() if p.requires_grad)),
    'qtr_trainable_count': int(sum(p.numel() for n,p in e3.named_parameters() if p.requires_grad and n.startswith('qtr.'))),
    'qtr_trainable_names': qtr_trainable,
    'optimizer_groups': [{'idx':i,'lr':g['lr'],'param_count':int(sum(p.numel() for p in g['params']))} for i,g in enumerate(opt.param_groups)],
    'sanity_loss': float(loss.detach().cpu()),
    'sanity_loss_finite': bool(torch.isfinite(loss.detach()).cpu()),
    'qtr_gate_grad': grad_sum(list(e3.named_parameters()), 'qtr.reliability_gates'),
}
(outdir/'preflight_e3_qtr_gate.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
