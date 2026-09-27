#!/usr/bin/env python
"""Validation/test stratification for ReTraceNet checkpoints."""
from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrace.data import build_loader
from retrace.metrics import Confusion
from retrace.utils import choose_device, load_checkpoint, load_config
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import forward_from_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--device')
    p.add_argument('--out', required=True)
    p.add_argument('--threshold', type=float, default=0.5)
    return p.parse_args()


def kind_from_path(path: str) -> str:
    stem = Path(path).stem.lower()
    parts = stem.split('_')
    if len(parts) > 1 and parts[0] == 'tp':
        if parts[1] == 's':
            return 'Tp_S_same-source_candidate'
        if parts[1] == 'd':
            return 'Tp_D_different-source_candidate'
    return 'authentic' if not stem.startswith('tp_') else 'unknown'


def area_bucket(r: float) -> str:
    if r <= 0: return 'authentic'
    if r < 0.005: return '<0.5%'
    if r < 0.01: return '0.5-1%'
    if r < 0.05: return '1-5%'
    if r < 0.20: return '5-20%'
    return '>=20%'


def add(conf: Confusion, pred: torch.Tensor, target: torch.Tensor):
    pred = pred.bool(); target = target.bool()
    conf.tp += float((pred & target).sum().cpu())
    conf.fp += float((pred & ~target).sum().cpu())
    conf.fn += float((~pred & target).sum().cpu())
    conf.tn += float((~pred & ~target).sum().cpu())


def scores(conf: Confusion) -> dict[str, float]:
    f1, iou, _ = conf.scores()
    p = conf.tp / max(1.0, conf.tp + conf.fp)
    r = conf.tp / max(1.0, conf.tp + conf.fn)
    return {'precision': p, 'recall': r, 'f1': f1, 'iou': iou, 'tp': conf.tp, 'fp': conf.fp, 'fn': conf.fn, 'tn': conf.tn}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = choose_device(args.device)
    loader = build_loader({**cfg['data'], 'batch_size': cfg['evaluation'].get('batch_size', 1), 'num_workers': 0}, args.split, shuffle=False)
    model = build_final_model(cfg['model']).to(device)
    ck = load_checkpoint(args.checkpoint, torch.device('cpu'))
    state = ck.get('ema_model') or ck.get('model') or ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(json.dumps({'missing': len(missing), 'unexpected': len(unexpected), 'missing_examples': list(missing)[:12], 'unexpected_examples': list(unexpected)[:12]}, indent=2))
    model.eval()
    groups = {'overall': {'count': 0, 'conf': Confusion()}}
    rows = []
    with torch.inference_mode():
        for batch in loader:
            out = forward_from_batch(model, batch, device)
            prob = out['mask_logits'].sigmoid().cpu()
            mask = batch['mask'].cpu().bool()
            pred = prob >= args.threshold
            paths = batch['path']
            for i, path in enumerate(paths):
                target = mask[i:i+1]
                p = pred[i:i+1]
                ratio = float(target.float().mean())
                kind = kind_from_path(path)
                bucket = area_bucket(ratio)
                for name in ['overall', kind, bucket, f'{kind}|{bucket}']:
                    groups.setdefault(name, {'count': 0, 'conf': Confusion()})
                    groups[name]['count'] += 1
                    add(groups[name]['conf'], p, target)
                c = Confusion(); add(c, p, target); sc = scores(c)
                rows.append({'path': path, 'kind': kind, 'area_bucket': bucket, 'gt_area_ratio': ratio, **sc})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out / 'per_image.csv').open('w', newline='') as f:
        fields = list(rows[0].keys()) if rows else ['path']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary = {name: {'count': g['count'], **scores(g['conf'])} for name, g in sorted(groups.items())}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
