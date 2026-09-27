#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from types import MethodType

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrace.data import build_loader
from retrace.engine import evaluate_model
from retrace.losses import ReTraceLoss
from retrace.metrics import Confusion
from retrace.utils import choose_device, load_config
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import clean_loss_config, load_initial_weights_repaired, score

BUCKETS = [
    ('lt_0p5', lambda r: 0 < r < 0.005),
    ('lt_1', lambda r: 0 < r < 0.01),
]

VARIANTS = [
    ('both_off', 0.0, 0.0),
    ('support_only', 1.0, 0.0),
    ('conflict_only', 0.0, 1.0),
    ('full', 1.0, 1.0),
    ('support_x2', 2.0, 1.0),
    ('support_x4', 4.0, 1.0),
    ('support_x8', 8.0, 1.0),
]


def install_dher_scaling(model, gamma_support: float, gamma_conflict: float):
    dher = model.dher
    if dher is None:
        raise RuntimeError('model has no DHER module')
    if not hasattr(dher, '_orig_forward_for_sweep'):
        dher._orig_forward_for_sweep = dher.forward
    orig = dher._orig_forward_for_sweep
    def scaled_forward(self, d1, *args, **kwargs):
        _out, diag = orig(d1, *args, **kwargs)
        out = d1 + float(gamma_support) * diag['support_write'] + float(gamma_conflict) * diag['conflict_write']
        return out, diag
    dher.forward = MethodType(scaled_forward, dher)


def eval_size_buckets(model, cfg, device, threshold):
    loader = build_loader({**cfg['data'], 'batch_size': cfg['evaluation']['batch_size'], 'num_workers': 0}, 'val', shuffle=False)
    confs = {name: Confusion() for name, _ in BUCKETS}
    counts = {name: 0 for name, _ in BUCKETS}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            im = batch['image'].to(device)
            mask = batch['mask'].to(device).bool()
            pred = model(im)['mask_logits'].sigmoid() >= threshold
            for i in range(im.shape[0]):
                t = mask[i:i+1]
                ratio = float(t.float().mean().cpu())
                if ratio <= 0:
                    continue
                p = pred[i:i+1]
                for name, fn in BUCKETS:
                    if not fn(ratio):
                        continue
                    counts[name] += 1
                    c = confs[name]
                    c.tp += float((p & t).sum().cpu())
                    c.fp += float((p & ~t).sum().cpu())
                    c.fn += float((~p & t).sum().cpu())
                    c.tn += float((~p & ~t).sum().cpu())
    out = {}
    for name, _ in BUCKETS:
        sc = score(confs[name])
        out[f'{name}_count'] = counts[name]
        out[f'{name}_f1'] = sc['f1']
        out[f'{name}_recall'] = sc['recall']
        out[f'{name}_precision'] = sc['precision']
        out[f'{name}_iou'] = sc['iou']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = choose_device(args.device)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    model = build_final_model(cfg['model']).to(device)
    report = load_initial_weights_repaired(model, args.checkpoint, device, [], [], [])
    bad = report['missing'] or report['unexpected'] or report['skipped_shape']
    if bad:
        raise RuntimeError(f'checkpoint not strict: {report}')
    criterion = ReTraceLoss.from_config(clean_loss_config(cfg['loss'])).to(device)
    loader = build_loader({**cfg['data'], 'batch_size': cfg['evaluation']['batch_size']}, 'val', shuffle=False)
    amp = bool(cfg['training'].get('amp', False) and device.type == 'cuda')
    threshold = float(cfg['evaluation'].get('threshold', 0.5))
    thresholds = int(cfg['evaluation'].get('thresholds', 101))
    rows = []
    for name, gs, gc in VARIANTS:
        install_dher_scaling(model, gs, gc)
        _, met = evaluate_model(model, criterion, loader, device, amp, threshold, thresholds)
        size = eval_size_buckets(model, cfg, device, threshold)
        row = {
            'variant': name,
            'gamma_support': gs,
            'gamma_conflict': gc,
            'fixed_f1': float(met['pixel_f1_fixed']),
            'oracle_f1': float(met.get('pixel_f1_oracle', met['pixel_f1_fixed'])),
            'precision': float(met['pixel_precision_fixed']),
            'recall': float(met['pixel_recall_fixed']),
            'iou': float(met['pixel_iou_fixed']),
            'auc': float(met.get('pixel_auc', float('nan'))),
            **size,
        }
        rows.append(row)
        print(yaml.safe_dump(row, sort_keys=False).strip(), flush=True)
    base = next(r for r in rows if r['variant'] == 'both_off')
    deltas = []
    for r in rows:
        if r is base:
            continue
        d = {'variant': r['variant']}
        for k in ['fixed_f1','oracle_f1','precision','recall','iou','auc','lt_0p5_f1','lt_0p5_recall','lt_1_f1','lt_1_recall']:
            d[f'delta_{k}'] = r[k] - base[k]
        deltas.append(d)
    with (outdir / 'dher_influence_sweep.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with (outdir / 'dher_influence_sweep_deltas.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(deltas[0].keys()))
        w.writeheader(); w.writerows(deltas)
    (outdir / 'dher_influence_sweep.yaml').write_text(yaml.safe_dump({'checkpoint_load': report, 'rows': rows, 'deltas_vs_both_off': deltas}, sort_keys=False), encoding='utf-8')
    print('SAVED', outdir / 'dher_influence_sweep.csv', outdir / 'dher_influence_sweep_deltas.csv', flush=True)

if __name__ == '__main__':
    main()
