#!/usr/bin/env python
"""Training entry for the final overnight QTR/HCV/DHER pipeline."""
from __future__ import annotations

import argparse, copy, csv, json, math, shutil, sys, time
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrace.data import ManipulationDataset, read_manifest, build_loader
from retrace.engine import evaluate_model, train_one_epoch
from retrace.losses import ReTraceLoss, soft_dice_loss
from retrace.metrics import Confusion
from retrace.utils import atomic_torch_save, choose_device, load_checkpoint, load_config, set_seed
from retracenet.final_modules import build_final_model, dice_loss_from_prob
from train_v2_b0 import load_initial_weights


def forward_from_batch(model, batch, device, image=None):
    im = image if image is not None else batch['image'].to(device)
    kwargs = {}
    if 'dct' in batch:
        kwargs['dct'] = batch['dct'].to(device)
    if 'qtable' in batch:
        kwargs['qtable'] = batch['qtable'].to(device)
    return model(im, **kwargs)


def clean_loss_config(loss_cfg):
    cfg=dict(loss_cfg)
    cfg.pop('tversky_alpha_fp', None)
    cfg.pop('tversky_beta_fn', None)
    cfg.pop('weak_tiny_max_area_ratio', None)
    return cfg

BUCKETS=["lt_0p25","0p25_0p5","0p5_1","lt_0p5","lt_1","1_3","3_10","gte_10"]


def load_initial_weights_repaired(model, path, device, patterns=None, skip_prefixes=None, skip_keys=None, key_remaps=None):
    patterns = list(patterns or [])
    skip_prefixes = tuple(skip_prefixes or [])
    skip_keys = set(skip_keys or [])
    key_remaps = list(key_remaps or [])
    if not path:
        return {"path":"", "loaded":0, "loaded_keys":[], "missing":list(model.state_dict().keys()), "unexpected":[], "skipped_shape":[], "skipped_reinitialized":[]}
    checkpoint = load_checkpoint(path, torch.device('cpu'))
    source = checkpoint.get('model', checkpoint)
    target = model.state_dict()
    filtered = {}
    unexpected = []
    skipped_shape = []
    skipped_reinitialized = []
    for key, value in source.items():
        candidates = []
        if not patterns or any(pattern in key for pattern in patterns):
            candidates.append(key)
        for remap in key_remaps:
            src = str(remap.get("source", ""))
            dst = str(remap.get("target", ""))
            if src and key.startswith(src):
                mapped = dst + key[len(src):]
                if not patterns or any(pattern in key or pattern in mapped for pattern in patterns):
                    candidates.append(mapped)
        for target_key in candidates:
            if target_key in skip_keys or any(target_key.startswith(prefix) for prefix in skip_prefixes):
                if target_key in target:
                    skipped_reinitialized.append(target_key)
                continue
            if target_key not in target:
                unexpected.append(target_key)
                continue
            if tuple(value.shape) != tuple(target[target_key].shape):
                skipped_shape.append((target_key, tuple(value.shape), tuple(target[target_key].shape)))
                continue
            filtered[target_key] = value
    result = model.load_state_dict(filtered, strict=False)
    return {"path":path, "loaded":len(filtered), "loaded_keys":sorted(filtered), "missing":list(result.missing_keys), "unexpected":unexpected, "skipped_shape":skipped_shape, "skipped_reinitialized":skipped_reinitialized}


def overlay_checkpoint(model, path, patterns=None, skip_prefixes=None, skip_keys=None):
    patterns = list(patterns or [])
    skip_prefixes = tuple(skip_prefixes or [])
    skip_keys = set(skip_keys or [])
    if not path:
        return {"path":"", "loaded":0, "unexpected":[], "skipped_shape":[]}
    checkpoint = load_checkpoint(path, torch.device('cpu'))
    source = checkpoint.get('model', checkpoint)
    target = model.state_dict()
    filtered = {}
    unexpected = []
    skipped_shape = []
    for key, value in source.items():
        if patterns and not any(pattern in key for pattern in patterns):
            continue
        if key in skip_keys or any(key.startswith(prefix) for prefix in skip_prefixes):
            continue
        if key not in target:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            skipped_shape.append((key, tuple(value.shape), tuple(target[key].shape)))
            continue
        filtered[key] = value
    model.load_state_dict(filtered, strict=False)
    return {"path":path, "loaded":len(filtered), "unexpected":unexpected, "skipped_shape":skipped_shape}


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--phase',required=True,choices=['hcv2','qtr_hcv','dher','total','dher_v2','total_v2','dher_v21','total_v21','perf_sprint','mechanism_repair','convnext_clean','strong_backbone','dct_forensic','correspondence_probe','score_sprint_c','dher_writer_only','f11_forensic_plus','mechanism_full_adapt','mechanism_full_joint','region_refine_only','e15_hr_warmup','e15_hr_decoder_adapt']); p.add_argument('--init'); p.add_argument('--device'); return p.parse_args()

def bucket_names(r):
    if r<=0: return []
    if r<0.0025: names=['lt_0p25']
    elif r<0.005: names=['0p25_0p5']
    elif r<0.01: names=['0p5_1']
    elif r<0.03: names=['1_3']
    elif r<0.10: names=['3_10']
    else: names=['gte_10']
    if r<0.005: names.append('lt_0p5')
    if r<0.01: names.append('lt_1')
    return names

def score(conf):
    f1,iou,_=conf.scores(); p=conf.tp/max(1.0,conf.tp+conf.fp); r=conf.tp/max(1.0,conf.tp+conf.fn); return {'precision':p,'recall':r,'f1':f1,'iou':iou}


def resize_mask_for_supervision(target, size, hard=True):
    if tuple(target.shape[-2:]) == tuple(size):
        out = target.float()
    else:
        h, w = target.shape[-2:]
        oh, ow = int(size[0]), int(size[1])
        x = target.float()
        if h >= oh and w >= ow and h % oh == 0 and w % ow == 0:
            out = F.max_pool2d(x, kernel_size=(h // oh, w // ow), stride=(h // oh, w // ow))
        else:
            out = F.interpolate(x, size=(oh, ow), mode='area').clamp(0, 1)
    if hard:
        return (out > 0).float()
    return out.clamp(0, 1)


def spatial_margin_logit(pos_map, neg_map, eps=1e-6):
    margin = pos_map - neg_map
    m = margin.mean((2, 3), keepdim=True)
    sd = margin.std((2, 3), keepdim=True, unbiased=False)
    return (margin - m) / (sd + eps)

def write_size_profile(model, cfg, split, device, out_csv):
    loader=build_loader({**cfg['data'],'batch_size':cfg['evaluation']['batch_size'],'num_workers':0}, split, shuffle=False)
    th=float(cfg['evaluation'].get('threshold',0.5)); buckets={b:{'conf':Confusion(),'count':0,'ratio_sum':0.0} for b in BUCKETS}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            im=batch['image'].to(device); mask=batch['mask'].to(device).bool(); pred=forward_from_batch(model,batch,device,im)['mask_logits'].sigmoid()>=th
            for i in range(im.shape[0]):
                t=mask[i:i+1]; r=float(t.float().mean().cpu())
                if r<=0: continue
                for b in bucket_names(r):
                    e=buckets[b]; e['count']+=1; e['ratio_sum']+=r; p=pred[i:i+1]
                    e['conf'].tp += float((p&t).sum().cpu()); e['conf'].fp += float((p&~t).sum().cpu()); e['conf'].fn += float((~p&t).sum().cpu()); e['conf'].tn += float((~p&~t).sum().cpu())
    rows=[]
    for b,e in buckets.items():
        row={'bucket':b,'num_images':e['count'],'mean_area_ratio':e['ratio_sum']/max(1,e['count'])}; row.update(score(e['conf'])); rows.append(row)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['bucket','num_images','mean_area_ratio','precision','recall','f1','iou']); w.writeheader(); w.writerows(rows)
    return rows

@torch.inference_mode()
def evaluate_tampered_pooled(model, loader, device, amp, fixed_threshold=0.5, threshold_count=101, bins=2048):
    ths = torch.linspace(0.0, 1.0, int(threshold_count), dtype=torch.float32)
    tp = torch.zeros_like(ths, dtype=torch.float64)
    fp = torch.zeros_like(ths, dtype=torch.float64)
    fn = torch.zeros_like(ths, dtype=torch.float64)
    pos_hist = torch.zeros(int(bins), dtype=torch.float64)
    neg_hist = torch.zeros(int(bins), dtype=torch.float64)
    fixed_conf = Confusion()
    count = 0
    model.eval()
    for batch in loader:
        label = batch['label']
        if int(label.flatten()[0].item()) != 1:
            continue
        mask = batch['mask'].to(device)
        if float(mask.sum().detach().cpu()) <= 0:
            continue
        out = forward_from_batch(model, batch, device)
        prob = out['mask_logits'].sigmoid().detach().float().cpu()[0, 0]
        target = (mask.detach().cpu()[0, 0] > 0.5)
        count += 1
        pred_fixed = prob >= float(fixed_threshold)
        fixed_conf.tp += float((pred_fixed & target).sum())
        fixed_conf.fp += float((pred_fixed & ~target).sum())
        fixed_conf.fn += float((~pred_fixed & target).sum())
        fixed_conf.tn += float((~pred_fixed & ~target).sum())
        pred = prob[None, :, :] >= ths[:, None, None]
        tgt = target[None, :, :]
        tp += (pred & tgt).sum(dim=(1, 2)).double()
        fp += (pred & ~tgt).sum(dim=(1, 2)).double()
        fn += (~pred & tgt).sum(dim=(1, 2)).double()
        idx = (prob.flatten() * (bins - 1)).long().clamp(0, bins - 1)
        flat_t = target.flatten()
        pos_hist += torch.bincount(idx[flat_t], minlength=bins).double()
        neg_hist += torch.bincount(idx[~flat_t], minlength=bins).double()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    best_i = int(f1.argmax()) if f1.numel() else 0
    fixed_f1, fixed_iou, _ = fixed_conf.scores()
    fixed_p = fixed_conf.tp / max(1.0, fixed_conf.tp + fixed_conf.fp)
    fixed_r = fixed_conf.tp / max(1.0, fixed_conf.tp + fixed_conf.fn)
    if pos_hist.sum() == 0 or neg_hist.sum() == 0:
        auc = float('nan')
    else:
        htp = torch.flip(torch.cumsum(torch.flip(pos_hist, (0,)), 0), (0,))
        hfp = torch.flip(torch.cumsum(torch.flip(neg_hist, (0,)), 0), (0,))
        tpr = htp / pos_hist.sum().clamp_min(1.0)
        fpr = hfp / neg_hist.sum().clamp_min(1.0)
        auc = float(torch.trapz(torch.cat([tpr.new_zeros(1), torch.flip(tpr, (0,))]), torch.cat([fpr.new_zeros(1), torch.flip(fpr, (0,))])))
    return {
        'count': int(count),
        'f1_best': float(f1[best_i]) if count else float('nan'),
        'threshold': float(ths[best_i]) if count else float('nan'),
        'precision': float(precision[best_i]) if count else float('nan'),
        'recall': float(recall[best_i]) if count else float('nan'),
        'iou': float(tp[best_i] / (tp[best_i] + fp[best_i] + fn[best_i]).clamp_min(1.0)) if count else float('nan'),
        'auc': auc,
        'fixed_f1': fixed_f1,
        'fixed_precision': fixed_p,
        'fixed_recall': fixed_r,
        'fixed_iou': fixed_iou,
    }


def stat(): return {'sum':0.0,'sq':0.0,'count':0.0}
def add(s,x):
    x=x.detach().float().flatten();
    if x.numel()==0: return
    s['sum']+=float(x.sum().cpu()); s['sq']+=float(x.square().sum().cpu()); s['count']+=float(x.numel())
def mean(s): return s['sum']/max(1.0,s['count'])
def std(s):
    m=mean(s); return max(0.0,s['sq']/max(1.0,s['count'])-m*m)**0.5

@torch.inference_mode()
def hcv_diag(model, cfg, device, max_batches):
    vals=['proposal','local_proposal','energy','delta','abs_delta','positive_delta','inside_energy','outside_energy']
    buckets=['background','authentic']+BUCKETS
    stats={sc:{b:{v:stat() for v in vals} for b in buckets} for sc in range(3)}; counts={b:0 for b in buckets}; model.eval()
    loader=build_loader({**cfg['data'],'batch_size':1,'num_workers':0},'val',shuffle=False)
    for bi,batch in enumerate(loader):
        if bi>=max_batches: break
        im=batch['image'].to(device); mask=batch['mask'].to(device); label=batch['label'].to(device); out=forward_from_batch(model,batch,device,im); h=out.get('hcv2')
        if h is None: break
        ratio=float(mask.mean().cpu()); auth=bool(float(label[0].cpu())<=0.0)
        counts['background']+=1
        if auth: counts['authentic']+=1
        for bn in bucket_names(ratio): counts[bn]+=1
        for sc in range(3):
            prop=h['proposals'][sc].float(); loc=h['local_proposals'][sc].float(); en=h['energies'][sc].float(); de=h['deltas'][sc].float(); tgt=resize_mask_for_supervision(mask.float(), prop.shape[-2:], hard=True)>0.5; bg=~tgt
            d={'proposal':prop,'local_proposal':loc,'energy':en,'delta':de,'abs_delta':de.abs(),'positive_delta':(de>0).float(),'inside_energy':en,'outside_energy':en}
            for v,x in d.items():
                region = tgt if v=='inside_energy' else (bg if v=='outside_energy' else bg)
                add(stats[sc]['background'][v], x[region])
                if auth: add(stats[sc]['authentic'][v], x.flatten())
                for bn in bucket_names(ratio): add(stats[sc][bn][v], x[tgt])
    rows=[]
    for sc in range(3):
        bd=stats[sc]['background']['delta']; ba=stats[sc]['background']['abs_delta']
        for b in buckets:
            row={'scale':f's{sc+1}','bucket':b,'num_images':counts[b]}
            for v in vals: row[f'{v}_mean']=mean(stats[sc][b][v]); row[f'{v}_std']=std(stats[sc][b][v]); row[f'{v}_pixels']=int(stats[sc][b][v]['count'])
            row['sep_delta']=(mean(stats[sc][b]['delta'])-mean(bd))/(std(bd)+1e-8)
            row['sep_abs_delta']=(mean(stats[sc][b]['abs_delta'])-mean(ba))/(std(ba)+1e-8)
            row['inside_outside_energy_ratio']=mean(stats[sc][b]['inside_energy'])/(mean(stats[sc]['background']['outside_energy'])+1e-8)
            rows.append(row)
    return rows

def hcv_loss(out,target,label,cfg):
    h=out['hcv2']; total=target.new_zeros(()); parts={}
    w=cfg.get('hcv_loss',{})
    for sc in range(3):
        tgt=resize_mask_for_supervision(target, h['proposals'][sc].shape[-2:], hard=True)
        prop=h['proposals'][sc].clamp(1e-5,1-1e-5); pf=h['p_factual'][sc].clamp(1e-5,1-1e-5); pcf=h['p_counterfactual'][sc].clamp(1e-5,1-1e-5); delta=h['deltas'][sc]; en=h['energies'][sc]
        prop_loss=F.binary_cross_entropy(prop,tgt)+dice_loss_from_prob(prop,tgt)
        vf_loss=F.binary_cross_entropy(pf,tgt)+dice_loss_from_prob(pf,tgt)
        fg=tgt; bg=1-tgt
        cf_target = torch.zeros_like(pcf)
        cf_loss = (fg * F.binary_cross_entropy(pcf, cf_target, reduction='none')).sum() / fg.sum().clamp_min(1.0)
        withdrawal=(fg*F.softplus(float(w.get('withdrawal_margin',0.15))-delta)).sum()/fg.sum().clamp_min(1.0)
        stability=(bg*delta.abs()).sum()/bg.sum().clamp_min(1.0)
        locality=(bg*en).sum()/bg.sum().clamp_min(1.0)
        total=total+float(w.get('proposal',1.0))*prop_loss+float(w.get('verifier',1.0))*vf_loss+float(w.get('counterfactual',0.25))*cf_loss+float(w.get('withdrawal',0.6))*withdrawal+float(w.get('stability',0.25))*stability+float(w.get('locality',0.15))*locality
        parts[f's{sc+1}_proposal']=prop_loss.detach(); parts[f's{sc+1}_verifier']=vf_loss.detach(); parts[f's{sc+1}_counterfactual']=cf_loss.detach(); parts[f's{sc+1}_withdrawal']=withdrawal.detach(); parts[f's{sc+1}_stability']=stability.detach(); parts[f's{sc+1}_locality']=locality.detach()
    parts['total']=total
    return parts


def _norm_prob(x, eps=1e-6):
    m=x.mean((2,3),keepdim=True); sd=x.std((2,3),keepdim=True,unbiased=False)
    return torch.sigmoid((x-m)/(sd+eps))

def dher_v2_aux_loss(out,target,cfg):
    d=out.get('dher')
    if not d or not isinstance(d.get('support'), list):
        z=target.new_zeros(()); return {'dher_support_aux':z,'dher_refutation_aux':z,'dher_conflict_aux':z,'dher_gate_aux':z,'dher_aux_total':z}
    lw=cfg.get('dher_v2_loss',{})
    ws=float(lw.get('support',0.10)); wr=float(lw.get('refutation',0.05)); wc=float(lw.get('conflict',0.0)); wg=float(lw.get('gate',0.0))
    total=target.new_zeros(()); sp_total=target.new_zeros(()); rf_total=target.new_zeros(()); cf_total=target.new_zeros(()); gate_total=target.new_zeros(())
    for s_map,r_map in zip(d['support'][:2], d['refutation'][:2]):
        tgt=resize_mask_for_supervision(target, s_map.shape[-2:], hard=True)
        support_logit=spatial_margin_logit(s_map, r_map)
        refutation_logit=-support_logit
        sp=F.binary_cross_entropy_with_logits(support_logit,tgt)
        rf=F.binary_cross_entropy_with_logits(refutation_logit,1-tgt)
        sp_total=sp_total+sp; rf_total=rf_total+rf; total=total+ws*sp+wr*rf
    if wc > 0 and isinstance(d.get('conflict'), list):
        for k_map in d['conflict'][:2]:
            tgt=resize_mask_for_supervision(target, k_map.shape[-2:], hard=True)
            dil=F.max_pool2d(tgt, kernel_size=5, stride=1, padding=2)
            ero=1-F.max_pool2d(1-tgt, kernel_size=5, stride=1, padding=2)
            boundary=(dil-ero).clamp(0,1)
            cf=F.binary_cross_entropy(k_map.clamp(1e-5,1-1e-5), boundary)
            cf_total=cf_total+cf; total=total+wc*cf
    if wg > 0 and 'support_gate' in d:
        tgt=resize_mask_for_supervision(target, d['support_gate'].shape[-2:], hard=True)
        gate=d['support_gate']
        if gate.shape[1] != tgt.shape[1]:
            tgt=tgt.expand(-1, gate.shape[1], -1, -1)
        gs=F.binary_cross_entropy(gate.clamp(1e-5,1-1e-5), tgt)
        gate_total=gate_total+gs; total=total+wg*gs
    return {'dher_support_aux':sp_total.detach(),'dher_refutation_aux':rf_total.detach(),'dher_conflict_aux':cf_total.detach(),'dher_gate_aux':gate_total.detach(),'dher_aux_total':total}

def mechanism_full_loss(out,target,label,seg_loss,cfg):
    weights=cfg.get('mechanism_loss_weights',{})
    losses=seg_loss(out,target,label)
    total=float(weights.get('segmentation',1.0))*losses['total']
    merged={**losses}
    if out.get('hcv2') is not None and float(weights.get('hcv',1.0))>0:
        h=hcv_loss(out,target,label,cfg)
        total=total+float(weights.get('hcv',1.0))*h['total']
        merged.update({f'hcv_{k}':v for k,v in h.items() if k!='total'})
        merged['hcv_total']=h['total'].detach()
    if out.get('dher') is not None and float(weights.get('dher',1.0))>0:
        d=dher_v2_aux_loss(out,target,cfg)
        total=total+float(weights.get('dher',1.0))*d['dher_aux_total']
        merged.update(d)
    if cfg['training'].get('deep_supervision_weights'):
        wds=weighted_deep_supervision_loss(out,target,cfg); total=total+wds; merged['weighted_deep_supervision']=wds
    if float(cfg['loss']['weights'].get('tversky',0.0))>0:
        tv=tversky_loss(out['mask_logits'],target,float(cfg['loss'].get('tversky_alpha_fp',0.3)),float(cfg['loss'].get('tversky_beta_fn',0.7))); total=total+float(cfg['loss']['weights'].get('tversky'))*tv; merged['tversky']=tv
    if float(cfg['loss']['weights'].get('weak_tiny',0.0))>0:
        wt=weak_tiny_recall_loss(out['mask_logits'],target,float(cfg['loss'].get('weak_tiny_max_area_ratio',0.01))); total=total+float(cfg['loss']['weights'].get('weak_tiny'))*wt; merged['weak_tiny']=wt
    merged['total']=total
    return merged

@torch.inference_mode()
def dher_v2_diag(model,cfg,device,max_batches):
    vals=['support','refutation','conflict']
    stats={sc:{region:{v:stat() for v in vals} for region in ['fg','bg']} for sc in ['s1','s2']}
    gate_stats={k:{region:stat() for region in ['all','fg','bg']} for k in ['support_gate','conflict_gate']}
    write_stats={'support_write_abs':stat(),'conflict_write_abs':stat(),'d1_rms':stat(),'support_write_rms':stat(),'conflict_write_rms':stat()}
    alphas={'alpha_support':0.0,'alpha_conflict':0.0}
    model.eval(); loader=build_loader({**cfg['data'],'batch_size':1,'num_workers':0},'val',shuffle=False)
    def add_region(bucket,x,mask):
        if mask.shape[-2:] != x.shape[-2:]: mask=resize_mask_for_supervision(mask.float(), x.shape[-2:], hard=True)>0.5
        if mask.shape[1] != x.shape[1]: mask=mask.expand(-1,x.shape[1],-1,-1)
        add(bucket['fg'], x[mask]); add(bucket['bg'], x[~mask]); add(bucket['all'], x)
    for bi,batch in enumerate(loader):
        if bi>=max_batches: break
        im=batch['image'].to(device); mask=batch['mask'].to(device)>0.5; out=forward_from_batch(model,batch,device,im); d=out.get('dher')
        if not d or not isinstance(d.get('support'), list): break
        alphas['alpha_support']=float(d['alpha_support'].detach().cpu()); alphas['alpha_conflict']=float(d['alpha_conflict'].detach().cpu())
        for idx,sc in enumerate(['s1','s2']):
            s_map=d['support'][idx].float(); r_map=d['refutation'][idx].float(); k_map=d['conflict'][idx].float()
            tgt=resize_mask_for_supervision(mask.float(), s_map.shape[-2:], hard=True)>0.5; bg=~tgt
            for name,x in [('support',s_map),('refutation',r_map),('conflict',k_map)]:
                add(stats[sc]['fg'][name], x[tgt]); add(stats[sc]['bg'][name], x[bg])
        add_region(gate_stats['support_gate'], d['support_gate'].float(), mask)
        add_region(gate_stats['conflict_gate'], d['conflict_gate'].float(), mask)
        add(write_stats['support_write_abs'], d['support_write'].float().abs())
        add(write_stats['conflict_write_abs'], d['conflict_write'].float().abs())
        if 'd1_rms' in d: add(write_stats['d1_rms'], d['d1_rms'])
        if 'support_write_rms' in d: add(write_stats['support_write_rms'], d['support_write_rms'])
        if 'conflict_write_rms' in d: add(write_stats['conflict_write_rms'], d['conflict_write_rms'])
    rows=[]
    for sc in ['s1','s2']:
        row={'scale':sc, **alphas}
        for v in vals:
            row[f'{v}_fg_mean']=mean(stats[sc]['fg'][v]); row[f'{v}_bg_mean']=mean(stats[sc]['bg'][v]); row[f'{v}_fg_bg_gap']=row[f'{v}_fg_mean']-row[f'{v}_bg_mean']
        rows.append(row)
    gate_row={'scale':'gates', **alphas}
    for k,b in gate_stats.items():
        gate_row[f'{k}_mean']=mean(b['all']); gate_row[f'{k}_fg_mean']=mean(b['fg']); gate_row[f'{k}_bg_mean']=mean(b['bg'])
    gate_row.update({f'{k}_mean':mean(v) for k,v in write_stats.items()})
    d1r=gate_row.get('d1_rms_mean',0.0)
    gate_row['support_write_rms_over_d1_rms']=gate_row.get('support_write_rms_mean',0.0)/(d1r+1e-12)
    gate_row['conflict_write_rms_over_d1_rms']=gate_row.get('conflict_write_rms_mean',0.0)/(d1r+1e-12)
    rows.append(gate_row)
    return rows

def tversky_loss(logits,target,alpha_fp=0.3,beta_fn=0.7,eps=1.0):
    p=logits.sigmoid().flatten(1); t=target.flatten(1); tp=(p*t).sum(1); fp=(p*(1-t)).sum(1); fn=((1-p)*t).sum(1); return (1-(tp+eps)/(tp+alpha_fp*fp+beta_fn*fn+eps)).mean()

def weak_tiny_recall_loss(logits, target, max_area_ratio=0.01):
    target = target.float()
    area = target.flatten(1).mean(1)
    keep = (area > 0) & (area < float(max_area_ratio))
    if not bool(keep.any()):
        return logits.new_zeros(())
    lg = logits[keep]
    tg = target[keep]
    positive = (tg > 0.5).float()
    pos_loss = (positive * F.softplus(-lg)).sum(dim=(1, 2, 3)) / positive.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return pos_loss.mean() + 0.5 * soft_dice_loss(lg, tg)


def aux_seg_loss(logits,target):
    if target.shape[-2:] != logits.shape[-2:]:
        target=resize_mask_for_supervision(target, logits.shape[-2:], hard=True)
    return F.binary_cross_entropy_with_logits(logits,target)+soft_dice_loss(logits,target)

def weighted_deep_supervision_loss(out,target,cfg):
    weights=cfg.get('training',{}).get('deep_supervision_weights')
    if not weights:
        return target.new_zeros(())
    aux=out.get('aux_logits_low') or out.get('aux_logits') or []
    total=target.new_zeros(())
    for w,logits in zip(weights,aux):
        total=total+float(w)*aux_seg_loss(logits,target)
    return total


def boundary_target(target, kernel_size=5):
    pad=int(kernel_size)//2
    t=target.float()
    dil=F.max_pool2d(t, int(kernel_size), stride=1, padding=pad)
    ero=1-F.max_pool2d(1-t, int(kernel_size), stride=1, padding=pad)
    return (dil-ero).clamp(0,1)


def boundary_aux_loss(logits,target,cfg):
    if logits.shape[-2:] != target.shape[-2:]:
        target=F.interpolate(target.float(), size=logits.shape[-2:], mode='nearest')
    bt=boundary_target(target, int(cfg.get('loss',{}).get('boundary_kernel',5)))
    return F.binary_cross_entropy_with_logits(logits,bt)+soft_dice_loss(logits,bt)


def e15_hr_loss(out,target,label,seg_loss,cfg):
    losses=seg_loss(out,target,label)
    total=losses['total']
    weights=cfg.get('e15_hr_loss',{})
    wd1=float(weights.get('d1',0.2)); wd2=float(weights.get('d2',0.1)); wd3=float(weights.get('d3',0.05)); wb=float(weights.get('boundary',0.1))
    d1=aux_seg_loss(out['e15_d1_logits'],target)
    d2=aux_seg_loss(out['e15_d2_logits'],target)
    d3=aux_seg_loss(out['e15_d3_logits'],target)
    bd=boundary_aux_loss(out['e15_boundary_logits'],target,cfg)
    total=total+wd1*d1+wd2*d2+wd3*d3+wb*bd
    return {**losses,'e15_d1':d1.detach(),'e15_d2':d2.detach(),'e15_d3':d3.detach(),'e15_boundary':bd.detach(),'total':total}


@torch.inference_mode()
def evaluate_e15_boundary_f1(model, loader, device, amp, threshold=0.5):
    conf=Confusion(); kernel=5; count=0
    model.eval()
    for batch in loader:
        mask=batch['mask'].to(device)
        if float(mask.sum().detach().cpu()) <= 0:
            continue
        with torch.autocast(device_type=device.type,enabled=amp):
            out=forward_from_batch(model,batch,device)
        if 'e15_boundary_logits' not in out:
            continue
        logits=out['e15_boundary_logits']
        target=boundary_target(mask.float(), kernel)
        if target.shape[-2:] != logits.shape[-2:]:
            target=F.interpolate(target, size=logits.shape[-2:], mode='nearest')
        pred=logits.sigmoid()>=float(threshold); tgt=target>0.5
        conf.tp += float((pred&tgt).sum().detach().cpu())
        conf.fp += float((pred&~tgt).sum().detach().cpu())
        conf.fn += float((~pred&tgt).sum().detach().cpu())
        conf.tn += float((~pred&~tgt).sum().detach().cpu())
        count += int(mask.shape[0])
    f1,iou,_=conf.scores()
    precision=conf.tp/max(1.0,conf.tp+conf.fp)
    recall=conf.tp/max(1.0,conf.tp+conf.fn)
    return {'count':count,'f1':f1,'precision':precision,'recall':recall,'iou':iou,'threshold':float(threshold)}

def update_ema_model(ema_model, model, decay):
    with torch.no_grad():
        msd=model.state_dict(); esd=ema_model.state_dict()
        for k,v in esd.items():
            src=msd[k].detach()
            if torch.is_floating_point(v):
                v.mul_(decay).add_(src, alpha=1.0-decay)
            else:
                v.copy_(src)

def copy_model_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def train_loader_for(cfg):
    data={**cfg['data'],'batch_size':cfg['training']['batch_size']}
    use_tiny=bool(data.get('tiny_stratified_sampling',False))
    use_area=bool(data.get('area_stratified_sampling',False))
    use_type_area=bool(data.get('type_area_stratified_sampling',False))
    if not (use_tiny or use_area or use_type_area):
        return build_loader(data,'train')
    import numpy as np
    from PIL import Image
    records=read_manifest(data['manifest'], data['root'], 'train')
    if use_type_area:
        bucket_masses=data.get('type_area_sampling_bucket_masses') or {
            'Tp_S:lt_1':0.18,'Tp_S:1_5':0.17,'Tp_S:5_20':0.17,'Tp_S:gte_20':0.18,
            'Tp_D:lt_1':0.06,'Tp_D:1_5':0.08,'Tp_D:5_20':0.09,'Tp_D:gte_20':0.04,
            'authentic':0.03,
        }
        order=list(bucket_masses.keys())
    elif use_area:
        bucket_masses=data.get('area_sampling_bucket_masses') or {'lt_1':0.30,'1_5':0.35,'5_20':0.20,'gte_20':0.10,'authentic':0.05}
        order=['lt_1','1_5','5_20','gte_20','authentic']
    else:
        bucket_masses=data.get('tiny_sampling_bucket_masses') or {'lt_1':0.34,'1_3':0.25,'3_10':0.25,'gte_10':0.16,'authentic':0.02}
        order=['lt_1','1_3','3_10','gte_10','authentic']
    labels=[]; counts={k:0 for k in order}
    for r in records:
        if r.mask is None:
            lab='authentic'
        else:
            with Image.open(r.mask) as im:
                ratio=float((np.asarray(im.convert('L'))>0).mean())
            if ratio<=0:
                lab='authentic'
            elif use_type_area:
                name=r.image.name
                kind='Tp_S' if name.startswith('Tp_S') or '_S_' in name else ('Tp_D' if name.startswith('Tp_D') or '_D_' in name else 'Tp_U')
                if ratio<0.01: area='lt_1'
                elif ratio<0.05: area='1_5'
                elif ratio<0.20: area='5_20'
                else: area='gte_20'
                lab=f'{kind}:{area}'
                if lab not in counts:
                    counts[lab]=0
                    bucket_masses.setdefault(lab, 0.0)
                    order.append(lab)
            elif use_area:
                if ratio<0.01: lab='lt_1'
                elif ratio<0.05: lab='1_5'
                elif ratio<0.20: lab='5_20'
                else: lab='gte_20'
            else:
                if ratio<0.01: lab='lt_1'
                elif ratio<0.03: lab='1_3'
                elif ratio<0.10: lab='3_10'
                else: lab='gte_10'
        labels.append(lab)
        counts[lab]=counts.get(lab,0)+1
    weights=[]
    for lab in labels:
        weights.append(float(bucket_masses.get(lab,0.0))/max(1,counts.get(lab,0)))
    s=sum(weights)
    if s<=0:
        raise ValueError(f"Invalid sampling weights: masses={bucket_masses} counts={counts}")
    weights=[w/s for w in weights]
    mode='type_area_stratified' if use_type_area else ('area_stratified' if use_area else 'tiny_stratified')
    print(f"[sampler] mode={mode} counts={counts} masses={bucket_masses}", flush=True)
    ds=ManipulationDataset(records,image_size=data.get('image_size',512),training=True,jpeg_probability=data.get('jpeg_probability',0.0),jpeg_quality=data.get('jpeg_quality',(45,96)),blur_probability=data.get('blur_probability',0.0),color_jitter=data.get('color_jitter',0.0),positive_crop_probability=data.get('positive_crop_probability',0.0),boundary_crop_probability=data.get('boundary_crop_probability',0.0),min_positive_crop_ratio=data.get('min_positive_crop_ratio',0.0),crop_retries=data.get('crop_retries',10),dct_forensic=data.get('dct_forensic',False),dct_jpeg_quality=data.get('dct_jpeg_quality',100),dct_qtable_from_source=data.get('dct_qtable_from_source',True),dct_cat_root=data.get('dct_cat_root'),dct_source_mode=data.get('dct_source_mode','reencoded'))
    sampler=WeightedRandomSampler(weights,num_samples=len(records),replacement=True)
    return DataLoader(ds,batch_size=data.get('batch_size',1),sampler=sampler,num_workers=data.get('num_workers',4),pin_memory=torch.cuda.is_available(),drop_last=True,persistent_workers=data.get('num_workers',4)>0)

def set_trainable(model, phase):
    for n,p in model.named_parameters(): p.requires_grad_(False)
    if phase=='dher_writer_only':
        exact={
            'dher.alpha_support',
            'dher.alpha_conflict',
            'dher.support_fuse.4.weight',
            'dher.support_fuse.4.bias',
            'dher.reexam.4.weight',
            'dher.reexam.4.bias',
        }
        names=[]
        for n,p in model.named_parameters():
            if n in exact:
                p.requires_grad_(True)
                names.append(n)
        return names
    if phase=='hcv2': pats=['hcv2.']
    elif phase=='qtr_hcv': pats=['qtr.','hcv2.']
    elif phase=='dher': pats=['dher.','decoder.stage1','mask_head','hcv2.','qtr.']
    elif phase in ['dher_v2','dher_v21']: pats=['dher.','decoder.stage1','mask_head']
    elif phase=='total_v2': pats=['content_encoder','forensic_stream','dher.','decoder','mask_head','log_var_head','aux_heads','image_head']
    elif phase=='total_v21': pats=['dher.','decoder','mask_head','log_var_head','aux_heads','image_head']
    elif phase=='perf_sprint': pats=['content_encoder.layer4','forensic_stream.stage3','forensic_stream.stage4','dher.','decoder','mask_head','log_var_head','aux_heads','image_head']
    elif phase=='convnext_clean': pats=['content_encoder.features.4','content_encoder.features.5','content_encoder.features.6','content_encoder.features.7','forensic_stream.stage3','forensic_stream.stage4','decoder','mask_head','log_var_head','aux_heads','image_head']
    elif phase=='strong_backbone': pats=['content_encoder.layer2','content_encoder.layer3','content_encoder.layer4','forensic_stream.stage3','forensic_stream.stage4','decoder','mask_head','log_var_head','aux_heads','image_head']
    elif phase=='dct_forensic': pats=['content_encoder.layer4','forensic_stream.','decoder','mask_head','log_var_head','aux_heads','image_head']
    elif phase=='correspondence_probe': pats=['correspondence.','decoder.stage1','mask_head']
    elif phase=='mechanism_repair': pats=['dher.','decoder.stage1','mask_head']
    elif phase=='score_sprint_c': pats=['content_encoder','forensic_stream','decoder','p2_preserve','qtr.reliability_gates','content_context','mask_refine','mask_head','log_var_head','aux_heads','image_head']
    elif phase=='f11_forensic_plus': pats=['content_encoder.layer4','forensic_stream','qtr.','hcv2.','dher.','decoder','mask_head','log_var_head','aux_heads','image_head']
    elif phase=='mechanism_full_adapt': pats=['qtr.','hcv2.','dher.','decoder.stage1','mask_head','log_var_head','aux_heads','image_head']
    elif phase=='mechanism_full_joint': pats=['content_encoder.layer4','forensic_stream','qtr.','hcv2.','dher.','decoder','p2_preserve','content_context','mask_refine','mask_head','log_var_head','aux_heads','image_head']
    elif phase=='region_refine_only': pats=['region_refine.']
    elif phase=='e15_hr_warmup':
        pats=['e15_decoder.evidence_adapters','e15_decoder.spatial_stem','e15_decoder.up_to_half','e15_decoder.r2_','e15_decoder.up_to_full','e15_decoder.r1_','e15_decoder.region_head','e15_decoder.boundary_head','e15_decoder.aux_']
    elif phase=='e15_hr_decoder_adapt':
        pats=['content_encoder','hcv2.','dher.','decoder.stage1','decoder.stage2','decoder.stage3','mask_head','log_var_head','aux_heads','image_head','e15_decoder']
    else: pats=['content_encoder','forensic_stream','qtr.','hcv2.','dher.','decoder','mask_head','log_var_head','aux_heads','image_head']
    names=[]
    for n,p in model.named_parameters():
        if any(k in n for k in pats): p.requires_grad_(True); names.append(n)
    if phase in ['e15_hr_warmup','e15_hr_decoder_adapt']:
        names=[n for n in names if not n.startswith('e15_anchor_content_encoder.')]
        for n,p in model.named_parameters():
            if n.startswith('e15_anchor_content_encoder.'):
                p.requires_grad_(False)
    if phase=='score_sprint_c':
        names=[n for n in names if not n.startswith('p2_preserve.fuse.1.')]
        for n,p in model.named_parameters():
            if n.startswith('p2_preserve.fuse.1.'):
                p.requires_grad_(False)
    return names

def build_optimizer(model,cfg,phase):
    tr=cfg['training']; wd=float(tr.get('weight_decay',0.01))
    if phase=='hcv2': return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(tr.get('learning_rate',2e-5)),weight_decay=wd)
    specs=tr.get('parameter_groups') or []
    if not specs: return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(tr.get('learning_rate',2e-5)),weight_decay=wd)
    used=set(); groups=[]; named=list(model.named_parameters())
    for spec in specs:
        pats=spec.get('patterns',[]); ps=[]
        for n,p in named:
            if p.requires_grad and n not in used and any(k in n for k in pats): ps.append(p); used.add(n)
        if ps: groups.append({'params':ps,'lr':float(spec.get('lr',tr.get('learning_rate',2e-5))),'weight_decay':float(spec.get('weight_decay',wd))})
    rem=[p for n,p in named if p.requires_grad and n not in used]
    if rem: groups.append({'params':rem,'lr':float(tr.get('learning_rate',2e-5)),'weight_decay':wd})
    return torch.optim.AdamW(groups)

def main():
    args=parse_args(); cfg=load_config(args.config); set_seed(int(cfg.get('seed',3407))); device=choose_device(args.device); outdir=Path(cfg['output_dir']); outdir.mkdir(parents=True,exist_ok=True); shutil.copyfile(args.config,outdir/'config.yaml')
    model=build_final_model(cfg['model']).to(device); init=args.init or cfg['training'].get('init_checkpoint'); init_report=load_initial_weights_repaired(model,init,device,cfg['training'].get('init_load_patterns') or [],cfg['training'].get('reinitialize_prefixes') or [],cfg['training'].get('reinitialize_keys') or [],cfg['training'].get('init_key_remaps') or [])
    if bool(cfg['training'].get('strict_checkpoint', False)):
        allowed=tuple(cfg['training'].get('allowed_missing_prefixes') or [])
        allowed_keys=set(cfg['training'].get('allowed_missing_keys') or [])
        bad=[k for k in init_report['missing'] if k not in allowed_keys and not any(k.startswith(p) for p in allowed)]
    else:
        allowed=tuple(cfg['training'].get('allowed_missing_prefixes') or []); allowed_keys=set(cfg['training'].get('allowed_missing_keys') or []); bad=[k for k in init_report['missing'] if k not in allowed_keys and not (k.startswith('hcv2.') or k.startswith('qtr.') or k.startswith('dher.') or any(k.startswith(p) for p in allowed))]
    if bad or init_report['unexpected'] or init_report['skipped_shape']: raise RuntimeError(f'bad init missing={bad[:10]} unexpected={init_report["unexpected"][:10]} skipped={init_report["skipped_shape"][:10]}')
    secondary_init_report=None
    if cfg['training'].get('secondary_init_checkpoint'):
        secondary_init_report=overlay_checkpoint(
            model,
            cfg['training'].get('secondary_init_checkpoint'),
            cfg['training'].get('secondary_init_load_patterns') or [],
            cfg['training'].get('secondary_skip_prefixes') or [],
            cfg['training'].get('secondary_skip_keys') or [],
        )
        if secondary_init_report['unexpected'] or secondary_init_report['skipped_shape']:
            raise RuntimeError(f'bad secondary init unexpected={secondary_init_report["unexpected"][:10]} skipped={secondary_init_report["skipped_shape"][:10]}')
    value_overrides=cfg['training'].get('parameter_initial_values') or {}
    named_params=dict(model.named_parameters())
    applied_value_overrides=[]
    with torch.no_grad():
        for k,v in value_overrides.items():
            if k not in named_params:
                raise RuntimeError(f'parameter_initial_values key not found: {k}')
            named_params[k].fill_(float(v))
            applied_value_overrides.append(k)
    names=set_trainable(model,args.phase); print(yaml.safe_dump({'run':{'architecture':cfg['model'].get('architecture'),'parent':cfg['training'].get('parent','E5'),'phase':args.phase},'init':{'loaded':init_report['loaded'],'missing_count':len(init_report['missing']),'unexpected':len(init_report['unexpected']),'skipped':len(init_report['skipped_shape']),'skipped_reinitialized_count':len(init_report.get('skipped_reinitialized',[])),'skipped_reinitialized_examples':init_report.get('skipped_reinitialized',[])[:12]},'secondary_init':secondary_init_report,'parameter_initial_values':applied_value_overrides,'trainable_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'trainable_examples':names[:30]},sort_keys=False))
    train_loader=train_loader_for(cfg); val_loader=build_loader({**cfg['data'],'batch_size':cfg['evaluation']['batch_size']},'val',shuffle=False)
    seg_loss=ReTraceLoss.from_config(clean_loss_config(cfg['loss'])).to(device); opt=build_optimizer(model,cfg,args.phase); amp=bool(cfg['training'].get('amp',False) and device.type=='cuda'); scaler=torch.cuda.amp.GradScaler(enabled=amp)
    use_ema=bool(cfg['training'].get('ema',False)); ema_decay=float(cfg['training'].get('ema_decay',0.999)); ema_model=None
    if use_ema:
        ema_model=copy.deepcopy(model).to(device).eval()
        for p in ema_model.parameters(): p.requires_grad_(False)
    epochs=int(cfg['training']['epochs']); const=int(cfg['training'].get('constant_lr_epochs',epochs)); minr=float(cfg['training'].get('min_lr_ratio',1.0))
    def mult(ep):
        if ep<const: return 1.0
        prog=min(1.0,max(0.0,(ep-const+1)/max(1,epochs-const))); return minr+(1-minr)*0.5*(1+math.cos(math.pi*prog))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,mult)
    hist=outdir/'history.csv'; fields=['epoch','lr','train_total','val_f1','val_f1_oracle','val_precision_fixed','val_recall_fixed','val_iou_fixed','val_auc','oracle_threshold','selection_value','extra']
    with hist.open('w',newline='',encoding='utf-8') as f: csv.DictWriter(f,fieldnames=fields).writeheader()
    best=-1e9; stale=0; meaningful=float(cfg['training'].get('meaningful_delta',0.0)); patience=int(cfg['training'].get('early_stopping',epochs+1))
    selection_floor=cfg['training'].get('selection_floor'); floor_grace=int(cfg['training'].get('selection_floor_grace_epochs',epochs+1)); completed_epochs=0; stop_reason='max_epochs_completed'
    for ep in range(epochs):
        model.train(); av={}; count=0; opt.zero_grad(set_to_none=True); acc=max(1,int(cfg['training'].get('gradient_accumulation_steps',1)))
        epoch_started=time.perf_counter(); total_steps=len(train_loader)
        configured_log_every=int(cfg['training'].get('progress_log_interval_steps',0))
        log_every=configured_log_every if configured_log_every>0 else max(1,total_steps//20)
        print(f"[train] epoch={ep+1}/{epochs} step=0/{total_steps} (0.0%) lr={opt.param_groups[0]['lr']:.3e}",flush=True)
        for bi,b in enumerate(train_loader):
            im=b['image'].to(device); mask=b['mask'].to(device); lab=b['label'].to(device)
            with torch.autocast(device_type=device.type,enabled=amp):
                out=forward_from_batch(model,b,device,im)
                if args.phase in ['hcv2','qtr_hcv']:
                    losses=hcv_loss(out,mask,lab,cfg); loss=losses['total']
                elif args.phase in ['mechanism_full_adapt','mechanism_full_joint']:
                    losses=mechanism_full_loss(out,mask,lab,seg_loss,cfg); loss=losses['total']
                elif args.phase in ['e15_hr_warmup','e15_hr_decoder_adapt']:
                    losses=e15_hr_loss(out,mask,lab,seg_loss,cfg); loss=losses['total']
                else:
                    losses=seg_loss(out,mask,lab); loss=losses['total']
                    if ('dher_v2' in cfg['model']['architecture']) or ('dher_v21' in cfg['model']['architecture']):
                        aux_losses=dher_v2_aux_loss(out,mask,cfg); loss=loss+aux_losses['dher_aux_total']; losses={**losses,**aux_losses,'total':loss}
                    if cfg['training'].get('deep_supervision_weights'):
                        wds=weighted_deep_supervision_loss(out,mask,cfg); loss=loss+wds; losses={**losses,'weighted_deep_supervision':wds,'total':loss}
                    if float(cfg['loss']['weights'].get('tversky',0.0))>0:
                        tv=tversky_loss(out['mask_logits'],mask,float(cfg['loss'].get('tversky_alpha_fp',0.3)),float(cfg['loss'].get('tversky_beta_fn',0.7))); loss=loss+float(cfg['loss']['weights'].get('tversky'))*tv; losses={**losses,'tversky':tv,'total':loss}
                    if float(cfg['loss']['weights'].get('weak_tiny',0.0))>0:
                        wt=weak_tiny_recall_loss(out['mask_logits'],mask,float(cfg['loss'].get('weak_tiny_max_area_ratio',0.01))); loss=loss+float(cfg['loss']['weights'].get('weak_tiny'))*wt; losses={**losses,'weak_tiny':wt,'total':loss}
            scaler.scale(loss/acc).backward()
            if (bi+1)%acc==0 or bi+1==len(train_loader):
                if float(cfg['training'].get('grad_clip',0.0))>0:
                    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training'].get('grad_clip',1.0)))
                scaler.step(opt); scaler.update()
                if ema_model is not None: update_ema_model(ema_model, model, ema_decay)
                opt.zero_grad(set_to_none=True)
            for k,v in losses.items(): av[k]=av.get(k,0.0)+float(v.detach().cpu())*im.shape[0]
            count+=im.shape[0]
            completed=bi+1
            if completed==1 or completed%log_every==0 or completed==total_steps:
                elapsed=time.perf_counter()-epoch_started
                avg_loss=av.get('total',0.0)/max(1,count)
                rate=completed/max(elapsed,1e-9)
                eta=(total_steps-completed)/max(rate,1e-9)
                print(
                    f"[train] epoch={ep+1}/{epochs} step={completed}/{total_steps} "
                    f"({100.0*completed/max(1,total_steps):.1f}%) loss={avg_loss:.6f} "
                    f"lr={opt.param_groups[0]['lr']:.3e} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                    flush=True,
                )
        train_total=av.get('total',0.0)/max(1,count)
        print(f"[eval] epoch={ep+1}/{epochs} training_done loss={train_total:.6f}; starting validation",flush=True)
        fixed_th=float(cfg['evaluation'].get('threshold',0.5)); nth=int(cfg['evaluation'].get('thresholds',101))
        vl,raw_met=evaluate_model(model,seg_loss,val_loader,device,amp,fixed_th,nth)
        met=raw_met; eval_variant='raw'; save_model=model; ema_met=None
        selection_name=str(cfg['training'].get('selection_metric','pixel_f1_oracle'))
        raw_tampered=None; ema_tampered=None
        if selection_name == 'val_tampered_pooled_f1_best':
            raw_tampered=evaluate_tampered_pooled(model,val_loader,device,amp,fixed_th,nth)
        if ema_model is not None:
            _,ema_met=evaluate_model(ema_model,seg_loss,val_loader,device,amp,fixed_th,nth)
            if selection_name == 'val_tampered_pooled_f1_best':
                ema_tampered=evaluate_tampered_pooled(ema_model,val_loader,device,amp,fixed_th,nth)
                if float(ema_tampered.get('f1_best',float('nan'))) > float(raw_tampered.get('f1_best',float('nan'))):
                    met=ema_met; eval_variant='ema'; save_model=ema_model
            elif float(ema_met.get('pixel_f1_oracle',ema_met['pixel_f1_fixed'])) > float(raw_met.get('pixel_f1_oracle',raw_met['pixel_f1_fixed'])):
                met=ema_met; eval_variant='ema'; save_model=ema_model
        diag=[]; extra={'eval_variant':eval_variant}
        if raw_tampered is not None:
            extra['raw_val_tampered_pooled']=raw_tampered
        if ema_tampered is not None:
            extra['ema_val_tampered_pooled']=ema_tampered
        if ema_met is not None:
            extra['ema_oracle_f1']=ema_met.get('pixel_f1_oracle'); extra['ema_fixed_f1']=ema_met.get('pixel_f1_fixed')
        if 'e15_hr' in cfg['model']['architecture']:
            extra['boundary_f1']=evaluate_e15_boundary_f1(save_model,val_loader,device,amp,float(cfg['evaluation'].get('boundary_threshold',0.5)))
        if 'hcv2' in cfg['model']['architecture'] or 'qtr' in cfg['model']['architecture'] or 'dher' in cfg['model']['architecture']:
            diag=hcv_diag(model,cfg,device,int(cfg['training'].get('diagnostic_batches',512))); (outdir/f'hcv_diag_epoch{ep:03d}.yaml').write_text(yaml.safe_dump(diag,sort_keys=False),encoding='utf-8')
            for sc in ['s1','s2','s3']:
                r=next(x for x in diag if x['scale']==sc and x['bucket']=='lt_1')
                extra[f'{sc}_lt1_sep_delta']=r['sep_delta']; extra[f'{sc}_lt1_delta']=r['delta_mean']; extra[f'{sc}_lt1_prop']=r['proposal_mean']
        if ('dher_v2' in cfg['model']['architecture']) or ('dher_v21' in cfg['model']['architecture']):
            dd=dher_v2_diag(model,cfg,device,int(cfg['training'].get('diagnostic_batches',512))); (outdir/f'dher_v2_diag_epoch{ep:03d}.yaml').write_text(yaml.safe_dump(dd,sort_keys=False),encoding='utf-8')
            gate=next((x for x in dd if x.get('scale')=='gates'),{})
            extra['alpha_support']=gate.get('alpha_support',0.0); extra['alpha_conflict']=gate.get('alpha_conflict',0.0)
            extra['support_gate_mean']=gate.get('support_gate_mean',0.0); extra['support_gate_fg_mean']=gate.get('support_gate_fg_mean',0.0); extra['support_gate_bg_mean']=gate.get('support_gate_bg_mean',0.0)
            extra['d1_rms']=gate.get('d1_rms_mean',0.0); extra['support_write_rms']=gate.get('support_write_rms_mean',0.0); extra['conflict_write_rms']=gate.get('conflict_write_rms_mean',0.0)
            extra['support_write_rms_over_d1_rms']=gate.get('support_write_rms_over_d1_rms',0.0); extra['conflict_write_rms_over_d1_rms']=gate.get('conflict_write_rms_over_d1_rms',0.0)
            extra['conflict_gate_mean']=gate.get('conflict_gate_mean',0.0); extra['conflict_gate_fg_mean']=gate.get('conflict_gate_fg_mean',0.0); extra['conflict_gate_bg_mean']=gate.get('conflict_gate_bg_mean',0.0)
            for sc in ['s1','s2']:
                rr=next((x for x in dd if x.get('scale')==sc),{})
                extra[f'{sc}_support_fg_bg_gap']=rr.get('support_fg_bg_gap',0.0); extra[f'{sc}_refutation_fg_bg_gap']=rr.get('refutation_fg_bg_gap',0.0); extra[f'{sc}_conflict_fg_bg_gap']=rr.get('conflict_fg_bg_gap',0.0)
        size=write_size_profile(model,cfg,'val',device,outdir/f'val_size_epoch{ep:03d}.csv')
        for b in ['lt_0p5','lt_1']:
            r=next((x for x in size if x['bucket']==b),None)
            if r: extra[f'{b}_f1']=r['f1']; extra[f'{b}_recall']=r['recall']
        if selection_name == 'val_tampered_pooled_f1_best':
            selected_tampered = ema_tampered if eval_variant == 'ema' and ema_tampered is not None else raw_tampered
            cur = float(selected_tampered.get('f1_best', float('nan')))
        else:
            cur=float(met.get('pixel_f1_oracle',met['pixel_f1_fixed']))
        improved=cur>best+meaningful
        if cur>best: best=cur
        stale=0 if improved else stale+1
        state={'epoch':ep,'model':copy_model_state(model),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'best_metric':best,'config':cfg,'phase':args.phase,'init_report':init_report,'secondary_init_report':secondary_init_report,'eval_variant':'raw'}
        if ema_model is not None:
            state['ema_model']=copy_model_state(ema_model)
        atomic_torch_save(state,outdir/'last.pt')
        if cur>=best-1e-12:
            best_state=dict(state)
            best_state['model']=copy_model_state(save_model)
            best_state['eval_variant']=eval_variant
            atomic_torch_save(best_state,outdir/'best.pt')
        atomic_torch_save(state,outdir/f'milestone_eplus{ep+1:02d}.pt')
        row={'epoch':ep,'lr':opt.param_groups[0]['lr'],'train_total':train_total,'val_f1':met['pixel_f1_fixed'],'val_f1_oracle':met.get('pixel_f1_oracle'),'val_precision_fixed':met.get('pixel_precision_fixed'),'val_recall_fixed':met.get('pixel_recall_fixed'),'val_iou_fixed':met.get('pixel_iou_fixed'),'val_auc':met.get('pixel_auc'),'oracle_threshold':met.get('oracle_threshold'),'selection_value':cur,'extra':json.dumps(extra,sort_keys=True)}
        with hist.open('a',newline='',encoding='utf-8') as f: csv.DictWriter(f,fieldnames=fields).writerow(row)
        (outdir/'last_metrics.yaml').write_text(yaml.safe_dump({'metrics':met,'extra':extra,'epoch':ep},sort_keys=False),encoding='utf-8')
        print(yaml.safe_dump(row,sort_keys=False).strip(),flush=True)
        sched.step()
        completed_epochs=ep+1
        if selection_floor is not None and completed_epochs>=floor_grace and best<float(selection_floor):
            stop_reason=f"selection_floor_not_met: best_{selection_name}={best:.9f} < {float(selection_floor):.9f} after {completed_epochs} epochs"
            print(f"[stop] {stop_reason}",flush=True)
            break
        if stale>=patience:
            stop_reason=f"early_stopping: no_{selection_name}_improvement_for_{stale}_epochs"
            print(f"[stop] {stop_reason}",flush=True)
            break
    (outdir/'training_status.yaml').write_text(yaml.safe_dump({
        'phase':args.phase,
        'selection_metric':cfg['training'].get('selection_metric'),
        'best_selection_value':best,
        'completed_epochs':completed_epochs,
        'configured_max_epochs':epochs,
        'stop_reason':stop_reason,
        'best_checkpoint':str(outdir/'best.pt'),
    },sort_keys=False),encoding='utf-8')

if __name__=='__main__': main()
