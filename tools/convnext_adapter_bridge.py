#!/usr/bin/env python
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.metrics import ForensicMetrics
from retrace.utils import choose_device, load_config, set_seed
from retracenet.final_modules import build_final_model
from retracenet.backbone import ConvNeXtTinyContentEncoder
from train_v2_final_pipeline import clean_loss_config, load_initial_weights_repaired


def norm_feat(x, eps=1e-6):
    m=x.mean((2,3), keepdim=True)
    s=x.std((2,3), keepdim=True, unbiased=False)
    return (x-m)/(s+eps)


class ConvNeXtAdapterBridge(nn.Module):
    def __init__(self, convnext_path: str):
        super().__init__()
        self.student = ConvNeXtTinyContentEncoder(pretrained_path=convnext_path, pretrained=False)
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(96,256,1,bias=False), nn.GroupNorm(32,256)),
            nn.Sequential(nn.Conv2d(192,512,1,bias=False), nn.GroupNorm(32,512)),
            nn.Sequential(nn.Conv2d(384,1024,1,bias=False), nn.GroupNorm(32,1024)),
            nn.Sequential(nn.Conv2d(768,2048,1,bias=False), nn.GroupNorm(32,2048)),
        ])
        for p in self.student.parameters():
            p.requires_grad_(False)

    def forward(self, image):
        feats=self.student(image)
        return [a(f) for a,f in zip(self.adapters, feats)]


class AdaptedModel(nn.Module):
    def __init__(self, parent, bridge):
        super().__init__()
        self.parent=parent
        self.bridge=bridge
    def forward(self, image, intervention_mask=None):
        _=intervention_mask
        content=self.bridge(image)
        detail=self.parent.forensic_stream(image)
        qtr=self.parent.qtr(image, content[:3], detail[:3]) if self.parent.qtr is not None else None
        hcv=self.parent.hcv2(content[:3], detail[:3], qtr) if self.parent.hcv2 is not None else None
        decoded=self.parent.decoder(content, detail)
        dher_diag=None
        if self.parent.dher is not None and hcv is not None:
            from retracenet.final_modules import DHERv2, DHERv21
            if isinstance(self.parent.dher, DHERv21):
                base_logits_d1=self.parent.mask_head(decoded[0]).detach()
                d1,dher_diag=self.parent.dher(decoded[0], detail[:2], hcv, qtr, base_logits_d1)
            elif isinstance(self.parent.dher, DHERv2):
                d1,dher_diag=self.parent.dher(decoded[0], detail[:2], hcv, qtr)
            else:
                d1,dher_diag=self.parent.dher(decoded[0], detail[0], hcv, qtr)
            decoded=[d1,*decoded[1:]]
        logits_low=self.parent.mask_head(decoded[0])
        logits=F.interpolate(logits_low, size=image.shape[-2:], mode='bilinear', align_corners=False)
        log_var=F.interpolate(self.parent.log_var_head(decoded[0]), size=image.shape[-2:], mode='bilinear', align_corners=False)
        aux_low=[head(f) for head,f in zip(self.parent.aux_heads, decoded[1:])]
        aux=[F.interpolate(a, size=image.shape[-2:], mode='bilinear', align_corners=False) for a in aux_low]
        out=self.parent._compat(logits, log_var, self.parent.image_head(decoded[0]), aux, content)
        out['aux_logits_low']=aux_low
        if qtr is not None: out['qtr']=qtr
        if hcv is not None: out['hcv2']=hcv
        if dher_diag is not None: out['dher']=dher_diag
        return out


def evaluate_subset(model, loader, device, n):
    model.eval(); metrics=ForensicMetrics(fixed_threshold=0.5); seen=0
    with torch.inference_mode():
        for b in loader:
            im=b['image'].to(device); mask=b['mask'].to(device); lab=b['label'].to(device)
            out=model(im)
            metrics.update(out, mask, lab)
            seen += im.shape[0]
            if seen >= n: break
    m=metrics.compute(101); m['samples']=seen
    return m


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--parent-checkpoint', required=True)
    ap.add_argument('--convnext-pretrained', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device')
    ap.add_argument('--iters', type=int, default=2500)
    ap.add_argument('--subset', type=int, default=160)
    args=ap.parse_args()
    set_seed(3407)
    outdir=Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    device=choose_device(args.device)
    cfg=load_config(args.config)
    parent=build_final_model(cfg['model']).to(device)
    rep=load_initial_weights_repaired(parent,args.parent_checkpoint,device,[],[],[])
    if rep['unexpected'] or rep['skipped_shape']:
        raise RuntimeError(rep)
    parent.eval()
    for p in parent.parameters(): p.requires_grad_(False)
    teacher=parent.content_encoder
    teacher.eval()
    bridge=ConvNeXtAdapterBridge(args.convnext_pretrained).to(device)
    trainable=[]
    for n,p in bridge.named_parameters():
        if n.startswith('adapters.'):
            p.requires_grad_(True); trainable.append(n)
        else:
            p.requires_grad_(False)
    opt=torch.optim.AdamW([p for p in bridge.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)
    loader=build_loader({**cfg['data'],'batch_size':1,'num_workers':4}, 'train', shuffle=True)
    fixed_val=build_loader({**cfg['data'],'batch_size':1,'num_workers':0}, 'val', shuffle=False)
    rows=[]; it=0
    while it < args.iters:
        for b in loader:
            im=b['image'].to(device)
            with torch.no_grad():
                tgt=[norm_feat(f) for f in teacher(im)]
            pred=[norm_feat(f) for f in bridge(im)]
            loss=0
            stage_losses=[]
            for p,t in zip(pred,tgt):
                sl=F.smooth_l1_loss(p,t) + 0.5*(1 - F.cosine_similarity(p.flatten(1), t.flatten(1), dim=1).mean())
                loss=loss+sl; stage_losses.append(float(sl.detach().cpu()))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            it+=1
            if it==1 or it%100==0 or it>=args.iters:
                row={'iter':it,'loss':float(loss.detach().cpu()),'stage_losses':stage_losses}
                rows.append(row); print(yaml.safe_dump(row, sort_keys=False).strip(), flush=True)
            if it>=args.iters: break
    torch.save({'bridge':bridge.state_dict(),'iters':it,'trainable':trainable}, outdir/'adapter_bridge.pt')
    adapted=AdaptedModel(parent, bridge).to(device)
    parent_metrics=evaluate_subset(parent, build_loader({**cfg['data'],'batch_size':1,'num_workers':0}, 'val', shuffle=False), device, args.subset)
    adapted_metrics=evaluate_subset(adapted, fixed_val, device, args.subset)
    decision='CLOSE_CONVNEXT_PERMANENTLY'
    if adapted_metrics.get('pixel_f1_oracle',0) >= 0.60:
        decision='HIGH_PRIORITY_CONTINUATION'
    elif adapted_metrics.get('pixel_f1_oracle',0) >= 0.55 and adapted_metrics.get('pixel_auc',0) >= 0.80:
        decision='ADAPTER_BRIDGE_PASSED'
    elif adapted_metrics.get('pixel_f1_oracle',0) >= 0.45:
        decision='STOP_CONVNEXT'
    report={'trainable':trainable,'trainable_param_count':sum(p.numel() for p in bridge.parameters() if p.requires_grad),'iters':it,'loss_rows':rows,'parent_subset':parent_metrics,'adapted_subset':adapted_metrics,'decision':decision}
    (outdir/'adapter_bridge_report.yaml').write_text(yaml.safe_dump(report, sort_keys=False), encoding='utf-8')
    print(yaml.safe_dump(report, sort_keys=False))

if __name__=='__main__': main()
