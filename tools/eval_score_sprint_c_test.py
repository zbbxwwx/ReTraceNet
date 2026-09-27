#!/usr/bin/env python
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.metrics import ForensicMetrics
from retrace.utils import choose_device, load_checkpoint, load_config
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import clean_loss_config, forward_from_batch


def auc_np(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.reshape(-1).astype(np.float64)
    labels = labels.reshape(-1).astype(np.uint8)
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return float('nan')
    order = np.argsort(-scores, kind='mergesort')
    y = labels[order]
    s = scores[order]
    distinct = np.r_[np.where(np.diff(s))[0], y.size - 1]
    tp = np.cumsum(y)[distinct].astype(np.float64)
    fp = (1 + distinct - tp).astype(np.float64)
    tpr = np.r_[0.0, tp / max(1, pos)]
    fpr = np.r_[0.0, fp / max(1, neg)]
    return float(np.trapz(tpr, fpr))


def best_f1_np(scores: np.ndarray, labels: np.ndarray, threshold_count: int = 101) -> tuple[float, float, float, float, float]:
    scores = scores.reshape(-1).astype(np.float64)
    labels = labels.reshape(-1).astype(bool)
    if labels.sum() == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    best = (-1.0, 0.0, 0.0, 0.0, 0.0)
    for th in np.linspace(0.0, 1.0, threshold_count):
        pred = scores >= th
        tp = float(np.logical_and(pred, labels).sum())
        fp = float(np.logical_and(pred, ~labels).sum())
        fn = float(np.logical_and(~pred, labels).sum())
        p = tp / max(1.0, tp + fp)
        r = tp / max(1.0, tp + fn)
        f1 = 2 * p * r / max(1e-12, p + r)
        iou = tp / max(1.0, tp + fp + fn)
        if f1 > best[0]:
            best = (f1, p, r, iou, float(th))
    return best


def bucket_name(ratio: float) -> str:
    if ratio < 0.005:
        return 'lt_0p5'
    if ratio < 0.01:
        return '0p5_1'
    if ratio < 0.05:
        return '1_5'
    if ratio < 0.20:
        return '5_20'
    return 'gte_20'


def kind_from_path(path: str) -> str:
    name = Path(path).name
    if name.startswith('Tp_S') or '_S_' in name:
        return 'Tp_S'
    if name.startswith('Tp_D') or '_D_' in name:
        return 'Tp_D'
    return 'unknown'


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {'count': 0, 'f1': float('nan'), 'auc': float('nan'), 'recall': float('nan')}
    return {
        'count': len(rows),
        'f1': float(np.nanmean([r['f1'] for r in rows])),
        'auc': float(np.nanmean([r['auc'] for r in rows])),
        'recall': float(np.nanmean([r['recall'] for r in rows])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--state', default='model', choices=['model','ema_model'])
    ap.add_argument('--split', default='test')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg['data']['batch_size'] = cfg.get('evaluation', {}).get('batch_size', 1)
    cfg['data']['num_workers'] = 4
    device = choose_device(args.device)
    model = build_final_model(cfg['model']).to(device).eval()
    ckpt = load_checkpoint(args.checkpoint, torch.device('cpu'))
    if args.state not in ckpt:
        raise KeyError(f'{args.state} not found in {args.checkpoint}; keys={list(ckpt.keys())[:20]}')
    res = model.load_state_dict(ckpt[args.state], strict=True)
    loader = build_loader({**cfg['data'], 'batch_size': cfg.get('evaluation', {}).get('batch_size', 1)}, args.split, shuffle=False)
    seg_loss = ReTraceLoss.from_config(clean_loss_config(cfg['loss'])).to(device)
    metrics = ForensicMetrics(float(cfg.get('evaluation', {}).get('threshold', 0.5)))
    per_image = []
    with torch.no_grad():
        for b in loader:
            im = b['image'].to(device)
            mask = b['mask'].to(device)
            label = b['label'].to(device)
            out = forward_from_batch(model, b, device, im)
            metrics.update(out, mask, label)
            prob = out['mask_logits'].sigmoid().detach().cpu().numpy()
            gt = mask.detach().cpu().numpy()
            paths = b['path'] if isinstance(b['path'], list) else list(b['path'])
            for i, path in enumerate(paths):
                y = gt[i, 0] > 0.5
                if not y.any():
                    continue
                s = prob[i, 0]
                f1, prec, rec, iou, th = best_f1_np(s, y, int(cfg.get('evaluation', {}).get('thresholds', 101)))
                per_image.append({
                    'path': path,
                    'kind': kind_from_path(path),
                    'area_ratio': float(y.mean()),
                    'bucket': bucket_name(float(y.mean())),
                    'f1': f1,
                    'precision': prec,
                    'recall': rec,
                    'iou': iou,
                    'threshold': th,
                    'auc': auc_np(s, y),
                })
    pooled = metrics.compute(int(cfg.get('evaluation', {}).get('thresholds', 101)))
    by_kind = {k: summarize([r for r in per_image if r['kind'] == k]) for k in ['Tp_S','Tp_D','unknown']}
    by_bucket = {b: summarize([r for r in per_image if r['bucket'] == b]) for b in ['lt_0p5','0p5_1','1_5','5_20','gte_20']}
    by_kind_bucket = {}
    for k in ['Tp_S','Tp_D']:
        by_kind_bucket[k] = {b: summarize([r for r in per_image if r['kind'] == k and r['bucket'] == b]) for b in ['lt_0p5','0p5_1','1_5','5_20','gte_20']}
    result = {
        'checkpoint': args.checkpoint,
        'state': args.state,
        'split': args.split,
        'num_images': len(loader.dataset),
        'num_manipulated_used': len(per_image),
        'unified_pooled': pooled,
        'manipulated_macro': summarize(per_image),
        'by_kind': by_kind,
        'by_bucket': by_bucket,
        'by_kind_bucket': by_kind_bucket,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
