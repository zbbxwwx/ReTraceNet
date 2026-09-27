#!/usr/bin/env python
from __future__ import annotations

import argparse, copy, csv, json, math, sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrace.data import build_loader
from retrace.engine import evaluate_model
from retrace.losses import ReTraceLoss
from retrace.utils import atomic_torch_save, choose_device, load_checkpoint, load_config, set_seed
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import (
    clean_loss_config,
    copy_model_state,
    dher_v2_aux_loss,
    hcv_diag,
    dher_v2_diag,
    set_trainable,
    build_optimizer,
    train_loader_for,
    tversky_loss,
    update_ema_model,
    weighted_deep_supervision_loss,
    write_size_profile,
)

FIELDS = ['epoch','lr','train_total','val_f1','val_f1_oracle','val_precision_fixed','val_recall_fixed','val_iou_fixed','val_auc','oracle_threshold','selection_value','extra']


def _lr_mult(epoch: int, epochs: int, const: int, minr: float) -> float:
    if epoch < const:
        return 1.0
    prog = min(1.0, max(0.0, (epoch - const + 1) / max(1, epochs - const)))
    return minr + (1 - minr) * 0.5 * (1 + math.cos(math.pi * prog))


def _set_epoch_lr(opt, cfg, epoch: int) -> None:
    tr = cfg['training']
    epochs = int(tr['epochs'])
    const = int(tr.get('constant_lr_epochs', epochs))
    minr = float(tr.get('min_lr_ratio', 1.0))
    specs = tr.get('parameter_groups') or []
    if specs:
        bases = [float(s.get('lr', tr.get('learning_rate', 2e-5))) for s in specs]
        if len(opt.param_groups) > len(bases):
            bases.append(float(tr.get('learning_rate', 2e-5)))
    else:
        bases = [float(tr.get('learning_rate', 2e-5))] * len(opt.param_groups)
    mult = _lr_mult(epoch, epochs, const, minr)
    for group, base in zip(opt.param_groups, bases):
        group['lr'] = base * mult


def _train_one(model, ema_model, seg_loss, train_loader, opt, cfg, phase, device, amp, scaler):
    model.train()
    av = {}
    count = 0
    opt.zero_grad(set_to_none=True)
    acc = max(1, int(cfg['training'].get('gradient_accumulation_steps', 1)))
    for bi, b in enumerate(train_loader):
        im = b['image'].to(device)
        mask = b['mask'].to(device)
        lab = b['label'].to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(im)
            losses = seg_loss(out, mask, lab)
            loss = losses['total']
            if ('dher_v2' in cfg['model']['architecture']) or ('dher_v21' in cfg['model']['architecture']):
                aux_losses = dher_v2_aux_loss(out, mask, cfg)
                loss = loss + aux_losses['dher_aux_total']
                losses = {**losses, **aux_losses, 'total': loss}
            if cfg['training'].get('deep_supervision_weights'):
                wds = weighted_deep_supervision_loss(out, mask, cfg)
                loss = loss + wds
                losses = {**losses, 'weighted_deep_supervision': wds, 'total': loss}
            if float(cfg['loss']['weights'].get('tversky', 0.0)) > 0:
                tv = tversky_loss(out['mask_logits'], mask, float(cfg['loss'].get('tversky_alpha_fp', 0.3)), float(cfg['loss'].get('tversky_beta_fn', 0.7)))
                loss = loss + float(cfg['loss']['weights'].get('tversky')) * tv
                losses = {**losses, 'tversky': tv, 'total': loss}
        scaler.scale(loss / acc).backward()
        if (bi + 1) % acc == 0 or bi + 1 == len(train_loader):
            if float(cfg['training'].get('grad_clip', 0.0)) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(cfg['training'].get('grad_clip', 1.0)))
            scaler.step(opt)
            scaler.update()
            if ema_model is not None:
                update_ema_model(ema_model, model, float(cfg['training'].get('ema_decay', 0.999)))
            opt.zero_grad(set_to_none=True)
        for k, v in losses.items():
            av[k] = av.get(k, 0.0) + float(v.detach().cpu()) * im.shape[0]
        count += im.shape[0]
    return av.get('total', 0.0) / max(1, count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--phase', default='perf_sprint')
    ap.add_argument('--device')
    ap.add_argument('--epochs-to-run', type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get('seed', 3407)))
    device = choose_device(args.device)
    outdir = Path(cfg['output_dir'])
    outdir.mkdir(parents=True, exist_ok=True)

    ck = load_checkpoint(args.checkpoint, torch.device('cpu'))
    start_epoch = int(ck['epoch']) + 1
    model = build_final_model(cfg['model']).to(device)
    missing, unexpected = model.load_state_dict(ck['model'], strict=False)
    if missing or unexpected:
        raise RuntimeError({'missing': missing[:20], 'unexpected': unexpected[:20]})

    names = set_trainable(model, args.phase)
    opt = build_optimizer(model, cfg, args.phase)
    opt.load_state_dict(ck['optimizer'])
    _set_epoch_lr(opt, cfg, start_epoch)

    amp = bool(cfg['training'].get('amp', False) and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    seg_loss = ReTraceLoss.from_config(clean_loss_config(cfg['loss'])).to(device)
    train_loader = train_loader_for(cfg)
    val_loader = build_loader({**cfg['data'], 'batch_size': cfg['evaluation']['batch_size']}, 'val', shuffle=False)

    ema_model = None
    if bool(cfg['training'].get('ema', False)):
        ema_model = copy.deepcopy(model).to(device).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        if 'ema_model' in ck:
            miss, unexp = ema_model.load_state_dict(ck['ema_model'], strict=False)
            if miss or unexp:
                raise RuntimeError({'ema_missing': miss[:20], 'ema_unexpected': unexp[:20]})

    hist = outdir / 'history.csv'
    if not hist.exists():
        with hist.open('w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    best = float(ck.get('best_metric', -1e9))

    print(yaml.safe_dump({
        'resume_from': args.checkpoint,
        'start_epoch': start_epoch,
        'epochs_to_run': args.epochs_to_run,
        'trainable_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'trainable_examples': names[:30],
        'lrs': [g['lr'] for g in opt.param_groups],
        'best_before': best,
    }, sort_keys=False), flush=True)

    for ep in range(start_epoch, start_epoch + args.epochs_to_run):
        train_total = _train_one(model, ema_model, seg_loss, train_loader, opt, cfg, args.phase, device, amp, scaler)
        _, met = evaluate_model(model, seg_loss, val_loader, device, amp, float(cfg['evaluation'].get('threshold', 0.5)), int(cfg['evaluation'].get('thresholds', 101)))
        eval_variant = 'raw'
        save_model = model
        ema_met = None
        if ema_model is not None:
            _, ema_met = evaluate_model(ema_model, seg_loss, val_loader, device, amp, float(cfg['evaluation'].get('threshold', 0.5)), int(cfg['evaluation'].get('thresholds', 101)))
            if float(ema_met.get('pixel_f1_oracle', ema_met['pixel_f1_fixed'])) > float(met.get('pixel_f1_oracle', met['pixel_f1_fixed'])):
                met = ema_met
                eval_variant = 'ema'
                save_model = ema_model
        extra = {'eval_variant': eval_variant}
        if ema_met is not None:
            extra['ema_oracle_f1'] = ema_met.get('pixel_f1_oracle')
            extra['ema_fixed_f1'] = ema_met.get('pixel_f1_fixed')
        if 'hcv2' in cfg['model']['architecture'] or 'qtr' in cfg['model']['architecture'] or 'dher' in cfg['model']['architecture']:
            diag = hcv_diag(model, cfg, device, int(cfg['training'].get('diagnostic_batches', 512)))
            (outdir / f'hcv_diag_epoch{ep:03d}.yaml').write_text(yaml.safe_dump(diag, sort_keys=False), encoding='utf-8')
            for sc in ['s1', 's2', 's3']:
                r = next(x for x in diag if x['scale'] == sc and x['bucket'] == 'lt_1')
                extra[f'{sc}_lt1_sep_delta'] = r['sep_delta']
                extra[f'{sc}_lt1_delta'] = r['delta_mean']
                extra[f'{sc}_lt1_prop'] = r['proposal_mean']
        if ('dher_v2' in cfg['model']['architecture']) or ('dher_v21' in cfg['model']['architecture']):
            dd = dher_v2_diag(model, cfg, device, int(cfg['training'].get('diagnostic_batches', 512)))
            (outdir / f'dher_v2_diag_epoch{ep:03d}.yaml').write_text(yaml.safe_dump(dd, sort_keys=False), encoding='utf-8')
            gate = next((x for x in dd if x.get('scale') == 'gates'), {})
            for k in ['alpha_support','alpha_conflict','support_gate_mean','support_gate_fg_mean','support_gate_bg_mean','d1_rms_mean','support_write_rms_mean','conflict_write_rms_mean','support_write_rms_over_d1_rms','conflict_write_rms_over_d1_rms','conflict_gate_mean','conflict_gate_fg_mean','conflict_gate_bg_mean']:
                if k in gate:
                    extra[k.replace('_mean_mean','_mean')] = gate[k]
            for sc in ['s1', 's2']:
                rr = next((x for x in dd if x.get('scale') == sc), {})
                extra[f'{sc}_support_fg_bg_gap'] = rr.get('support_fg_bg_gap', 0.0)
                extra[f'{sc}_refutation_fg_bg_gap'] = rr.get('refutation_fg_bg_gap', 0.0)
                extra[f'{sc}_conflict_fg_bg_gap'] = rr.get('conflict_fg_bg_gap', 0.0)
        size = write_size_profile(model, cfg, 'val', device, outdir / f'val_size_epoch{ep:03d}.csv')
        for b in ['lt_0p5', 'lt_1']:
            r = next((x for x in size if x['bucket'] == b), None)
            if r:
                extra[f'{b}_f1'] = r['f1']
                extra[f'{b}_recall'] = r['recall']

        cur = float(met.get('pixel_f1_oracle', met['pixel_f1_fixed']))
        if cur > best:
            best = cur
        state = {'epoch': ep, 'model': copy_model_state(model), 'optimizer': opt.state_dict(), 'scheduler': ck.get('scheduler', {}), 'best_metric': best, 'config': cfg, 'phase': args.phase, 'init_report': ck.get('init_report', {}), 'eval_variant': 'raw'}
        if ema_model is not None:
            state['ema_model'] = copy_model_state(ema_model)
        atomic_torch_save(state, outdir / 'last.pt')
        if cur >= best - 1e-12:
            best_state = dict(state)
            best_state['model'] = copy_model_state(save_model)
            best_state['eval_variant'] = eval_variant
            atomic_torch_save(best_state, outdir / 'best.pt')
        atomic_torch_save(state, outdir / f'milestone_eplus{ep+1:02d}.pt')
        row = {'epoch': ep, 'lr': opt.param_groups[0]['lr'], 'train_total': train_total, 'val_f1': met['pixel_f1_fixed'], 'val_f1_oracle': met.get('pixel_f1_oracle'), 'val_precision_fixed': met.get('pixel_precision_fixed'), 'val_recall_fixed': met.get('pixel_recall_fixed'), 'val_iou_fixed': met.get('pixel_iou_fixed'), 'val_auc': met.get('pixel_auc'), 'oracle_threshold': met.get('oracle_threshold'), 'selection_value': cur, 'extra': json.dumps(extra, sort_keys=True)}
        with hist.open('a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        (outdir / 'last_metrics.yaml').write_text(yaml.safe_dump({'metrics': met, 'extra': extra, 'epoch': ep}, sort_keys=False), encoding='utf-8')
        print(yaml.safe_dump(row, sort_keys=False).strip(), flush=True)

if __name__ == '__main__':
    main()
