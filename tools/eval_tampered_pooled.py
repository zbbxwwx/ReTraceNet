#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrace.data import build_loader
from retrace.utils import choose_device, load_checkpoint, load_config
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import forward_from_batch


def bucket_name(r: float) -> str:
    if r < 0.005: return 'lt_0p5'
    if r < 0.01: return '0p5_1'
    if r < 0.05: return '1_5'
    if r < 0.20: return '5_20'
    return 'gte_20'


def kind_from_path(path: str) -> str:
    name = Path(path).name
    if name.startswith('Tp_S') or '_S_' in name: return 'Tp_S'
    if name.startswith('Tp_D') or '_D_' in name: return 'Tp_D'
    return 'unknown'


def best_f1(scores: np.ndarray, labels: np.ndarray, thresholds: int = 101):
    if labels.size == 0 or labels.sum() == 0:
        return {'f1_best': float('nan'), 'threshold': float('nan'), 'precision': float('nan'), 'recall': float('nan'), 'iou': float('nan')}
    best = (-1.0, 0.0, 0.0, 0.0, 0.0)
    lab = labels.astype(bool)
    for t in np.linspace(0.0, 1.0, thresholds):
        pred = scores >= t
        tp = float(np.logical_and(pred, lab).sum())
        fp = float(np.logical_and(pred, ~lab).sum())
        fn = float(np.logical_and(~pred, lab).sum())
        p = tp / max(1.0, tp + fp)
        r = tp / max(1.0, tp + fn)
        f1 = 2 * p * r / max(1e-12, p + r)
        iou = tp / max(1.0, tp + fp + fn)
        if f1 > best[0]:
            best = (f1, t, p, r, iou)
    return {'f1_best': best[0], 'threshold': best[1], 'precision': best[2], 'recall': best[3], 'iou': best[4]}


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return float('nan')
    return float(roc_auc_score(labels.astype(np.uint8), scores.astype(np.float32)))


def summarize(items: list[tuple[np.ndarray, np.ndarray]], thresholds: int) -> dict:
    if not items:
        return {'count': 0, 'f1_best': float('nan'), 'auc': float('nan'), 'precision': float('nan'), 'recall': float('nan'), 'iou': float('nan'), 'threshold': float('nan')}
    s = np.concatenate([x for x,_ in items])
    y = np.concatenate([y for _,y in items])
    out = best_f1(s, y, thresholds)
    out['auc'] = auc(s, y)
    out['count'] = len(items)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--state', default='model', choices=['model','ema_model'])
    ap.add_argument('--split', default='val')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    cfg=load_config(args.config)
    cfg['data']['batch_size']=cfg.get('evaluation',{}).get('batch_size',1)
    device=choose_device(args.device)
    model=build_final_model(cfg['model']).to(device).eval()
    ckpt=load_checkpoint(args.checkpoint, torch.device('cpu'))
    if args.state not in ckpt:
        raise KeyError(f'{args.state} not in {args.checkpoint}; keys={list(ckpt)[:20]}')
    model.load_state_dict(ckpt[args.state], strict=True)
    loader=build_loader({**cfg['data'], 'batch_size':1}, args.split, shuffle=False)
    thresholds=int(cfg.get('evaluation',{}).get('thresholds',101))
    all_items=[]; by_kind={}; by_bucket={}; by_kind_bucket={}; auth_fp=0.0; auth_total=0.0
    fixed_tp=fixed_fp=fixed_fn=0.0
    with torch.no_grad():
        for b in loader:
            out=forward_from_batch(model,b,device)
            prob=out['mask_logits'].sigmoid().detach().cpu().numpy()[0,0].astype(np.float32)
            gt=(b['mask'].numpy()[0,0] > 0.5)
            pred=prob >= float(cfg.get('evaluation',{}).get('threshold',0.5))
            label=int(b['label'].item())
            path=b['path'][0] if isinstance(b.get('path'), list) else str(b.get('path',''))
            if label == 1 and gt.any():
                scores=prob.reshape(-1)
                labels=gt.reshape(-1).astype(np.uint8)
                all_items.append((scores, labels))
                kind=kind_from_path(path); bucket=bucket_name(float(gt.mean()))
                by_kind.setdefault(kind, []).append((scores, labels))
                by_bucket.setdefault(bucket, []).append((scores, labels))
                by_kind_bucket.setdefault(kind, {}).setdefault(bucket, []).append((scores, labels))
                fixed_tp += float(np.logical_and(pred, gt).sum())
                fixed_fp += float(np.logical_and(pred, ~gt).sum())
                fixed_fn += float(np.logical_and(~pred, gt).sum())
            else:
                auth_fp += float(pred.sum()); auth_total += float(pred.size)
    fixed_p=fixed_tp/max(1.0,fixed_tp+fixed_fp); fixed_r=fixed_tp/max(1.0,fixed_tp+fixed_fn)
    result={
        'checkpoint': args.checkpoint, 'state': args.state, 'split': args.split, 'num_images': len(loader.dataset),
        'tampered_images': len(all_items), 'authentic_images': int(auth_total/(cfg['data']['image_size']*cfg['data']['image_size'])) if auth_total else 0,
        'tampered_pooled': summarize(all_items, thresholds),
        'tampered_fixed_0p5': {'f1': 2*fixed_p*fixed_r/max(1e-12,fixed_p+fixed_r), 'precision': fixed_p, 'recall': fixed_r, 'iou': fixed_tp/max(1.0,fixed_tp+fixed_fp+fixed_fn)},
        'authentic_fpr_0p5': auth_fp/max(1.0,auth_total),
        'by_kind': {k:summarize(v, thresholds) for k,v in by_kind.items()},
        'by_bucket': {k:summarize(by_bucket.get(k,[]), thresholds) for k in ['lt_0p5','0p5_1','1_5','5_20','gte_20']},
        'by_kind_bucket': {k:{b:summarize(v.get(b,[]), thresholds) for b in ['lt_0p5','0p5_1','1_5','5_20','gte_20']} for k,v in by_kind_bucket.items()},
    }
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__=='__main__': main()
