"""Same E15 architecture: train every new layer against a frozen E5 path."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import math
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrace.data import ManipulationDataset, read_manifest, pil_to_tensor
from retrace.losses import soft_dice_loss, boundary_loss
from retrace.utils import load_config, set_seed, atomic_torch_save
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import load_initial_weights_repaired, train_loader_for, aux_seg_loss, boundary_aux_loss
from e15_original_metrics import ValidationStats, stratum


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(yaml.safe_dump(value, sort_keys=False), encoding='utf-8')
    tmp.replace(path)


class WholeImageDataset(ManipulationDataset):
    whole_probability = .6

    def _joint_geometry(self, image, mask):
        target = self.image_size
        if random.random() < self.whole_probability:
            self.last_geometry = 'whole'
            image = image.resize((target, target), Image.Resampling.BILINEAR)
            mask = mask.resize((target, target), Image.Resampling.NEAREST)
        else:
            self.last_geometry = 'crop'
            ratio = target * random.uniform(1., 1.5) / min(image.size)
            size = (max(target, round(image.width * ratio)), max(target, round(image.height * ratio)))
            image, mask = image.resize(size, Image.Resampling.BILINEAR), mask.resize(size, Image.Resampling.NEAREST)
            left, top = random.randint(0, size[0] - target), random.randint(0, size[1] - target)
            fg = np.argwhere(np.asarray(mask) > 127)
            if len(fg) and random.random() < self.positive_crop_probability:
                y, x = fg[random.randrange(len(fg))]
                left = max(0, min(int(x) - target // 2, size[0] - target))
                top = max(0, min(int(y) - target // 2, size[1] - target))
            box = (left, top, left + target, top + target)
            image, mask = image.crop(box), mask.crop(box)
        if random.random() < .5:
            image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < .1:
            image, mask = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return image, mask

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['geometry'] = self.last_geometry
        y = sample['mask'].numpy()
        sample['post_geometry_bucket'] = stratum(sample['path'], y) if y.any() else 'authentic_or_empty_crop'
        return sample


def loader_for(cfg):
    # Reuse the repository's type x area sampler, then replace geometry explicitly.
    base = train_loader_for(cfg)
    ds = base.dataset
    ds.__class__ = WholeImageDataset
    ds.whole_probability = float(cfg['data']['whole_image_probability'])
    return DataLoader(ds, batch_size=1, sampler=base.sampler, num_workers=4,
                      pin_memory=True, persistent_workers=True, drop_last=True)


def objective(out, target, cfg):
    logits = out['mask_logits']
    terms = {'bce': F.binary_cross_entropy_with_logits(logits, target),
             'dice': soft_dice_loss(logits, target),
             'mask_boundary': boundary_loss(logits, target, 5)}
    for key in ['d1', 'd2', 'd3']:
        terms[key] = aux_seg_loss(out['e15_' + key + '_logits'], target)
    terms['boundary_head'] = boundary_aux_loss(out['e15_boundary_logits'], target, cfg)
    terms['total'] = terms['bce'] + terms['dice'] + .05 * terms['mask_boundary'] + .2 * terms['d1'] + .1 * terms['d2'] + .05 * terms['d3'] + .1 * terms['boundary_head']
    return terms


def train_mode(model):
    # eval on every frozen module prevents BN/running-buffer drift of the E5 path.
    model.eval()
    model.content_encoder.proj.train()
    model.e15_decoder.train()


def make_optimizer(model, lr):
    groups = []
    for prefix in ['content_encoder.proj.', 'e15_decoder.']:
        params = [p for n, p in model.named_parameters() if n.startswith(prefix)]
        assert params
        groups.append({'params': params, 'lr': lr, 'name': prefix})
    allowed = {id(p) for g in groups for p in g['params']}
    for p in model.parameters():
        p.requires_grad_(id(p) in allowed)
    assert not any(p.requires_grad for p in model.forensic_stream.parameters())
    return torch.optim.AdamW(groups, weight_decay=.01)


@torch.inference_mode()
def evaluate(model, records, label, directory, include_old=False):
    model.eval()
    current = ValidationStats()
    old = ValidationStats() if include_old else None
    started = time.time()
    per_image = directory / (label + '_per_image.jsonl')
    with per_image.open('w', encoding='utf-8') as handle:
        for i, record in enumerate(records):
            with Image.open(record.image) as im:
                image = im.convert('RGB')
                size = image.size
                x = pil_to_tensor(image.resize((640, 640), Image.Resampling.BILINEAR))[None].cuda()
            if record.mask is None:
                y = np.zeros((size[1], size[0]), dtype=np.int16)
            else:
                with Image.open(record.mask) as mask:
                    y = (np.asarray(mask.convert('L')) > 127).astype(np.int16)
            assert y.shape == (size[1], size[0]), (record.image, y.shape, size)
            out = model(x)
            p = F.interpolate(out['mask_logits'].float().sigmoid(), size=y.shape, mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            current.add(p, y, record.label, record.image)
            row = {'path': str(record.image), 'label': record.label, 'bucket': stratum(record.image, y),
                   'tp': int(((p >= .5) & (y > 0)).sum()), 'fp': int(((p >= .5) & (y == 0)).sum()),
                   'fn': int(((p < .5) & (y > 0)).sum())}
            if include_old:
                po = F.interpolate(out['e15_old_logits'].float().sigmoid(), size=y.shape, mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
                old.add(po, y, record.label, record.image)
                row['corrected_pixels'] = int((((po >= .5) != (y > 0)) & ((p >= .5) == (y > 0))).sum())
                row['damaged_pixels'] = int((((po >= .5) == (y > 0)) & ((p >= .5) != (y > 0))).sum())
            handle.write(json.dumps(row) + '\n')
            del out, x
            if (i + 1) % 50 == 0 or i == 0 or i + 1 == len(records):
                print(f'[val:{label}] {i+1}/{len(records)} elapsed={(time.time()-started)/60:.1f}m', flush=True)
    result = {'protocol': 'original_GT_grid; probability_bilinear; valid_GT>=0; raw; thresholds=101; no_TTA',
              'split': 'val', 'metrics': current.result(), 'elapsed_seconds': time.time() - started}
    if old:
        result['old_path'] = old.result()
    dump(directory / (label + '.yaml'), result)
    return result


def preflight(model, optimizer, batch, cfg, directory):
    train_mode(model)
    x, y = batch['image'].cuda(), batch['mask'].cuda()
    initial = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    with torch.no_grad():
        out = model(x)
        diff = float((out['mask_logits'] - out['e15_old_logits']).abs().max())
        assert diff < 1e-6, diff
        old_before = out['e15_old_logits'].clone()
        assert all(float(a.beta) == 0 for a in model.e15_decoder.evidence_adapters)
        del out
    prefixes = ['content_encoder.proj.', *[f'e15_decoder.stage{i}.' for i in range(1, 5)],
                'e15_decoder.evidence_adapters.', 'e15_decoder.spatial_stem.', 'e15_decoder.r1_',
                'e15_decoder.r2_', 'e15_decoder.region_head.', 'e15_decoder.boundary_head.']
    gradients = {p: 0. for p in prefixes}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        losses = objective(out, y, cfg)
        assert torch.isfinite(losses['total'])
        losses['total'].backward()
        for prefix in prefixes:
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            gradients[prefix] = max(gradients[prefix], sum(float(g.detach().float().square().sum()) for g in grads) ** .5)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], .5, error_if_nonfinite=True)
        optimizer.step()
        del out, losses
    updates = {prefix: sum(float((p.detach().cpu() - initial[n]).abs().sum()) for n, p in model.named_parameters() if n.startswith(prefix)) for prefix in prefixes}
    assert all(v > 0 and math.isfinite(v) for v in gradients.values()), gradients
    assert all(v > 0 for v in updates.values()), updates
    frozen_delta = max(float((p.detach().cpu() - initial[n]).abs().max()) for n, p in model.named_parameters() if not p.requires_grad)
    assert frozen_delta == 0
    with torch.no_grad():
        out = model(x)
        old_diff = float((out['e15_old_logits'] - old_before).abs().max())
        assert old_diff < 1e-6, old_diff
        del out
    report = {'status': 'PASS', 'step0_identity_max_abs_diff': diff, 'old_path_drift_after_two_steps': old_diff,
              'frozen_parameter_max_delta': frozen_delta, 'gradient_norms': gradients, 'update_abs_sums': updates,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'optimizer': [{'name': g['name'], 'lr': g['lr'], 'parameters': sum(p.numel() for p in g['params'])} for g in optimizer.param_groups],
              'audit_steps_restored_before_training': True}
    dump(directory / 'preflight.yaml', report)
    print(yaml.safe_dump(report, sort_keys=False), flush=True)
    model.load_state_dict(initial, strict=True)
    optimizer.state.clear()
    del initial, x, y, old_before
    gc.collect()
    torch.cuda.empty_cache()


def main(args):
    cfg = load_config(args.config)
    directory = Path(cfg['output_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / 'overnight_summary.yaml'
    assert not (directory / 'last.pt').exists(), 'Refuse to overwrite existing training; use a new output directory.'
    summary = {'status': 'initializing', 'architecture': cfg['model']['architecture'], 'test_run': False,
               'phase': 'complete_new_layer_training', 'max_epochs': 8, 'minimum_epochs': 4, 'patience': 3,
               'selection': 'val_original_grid_fake_pooled_F1_best', 'auc_used_for_selection': False,
               'forensic_frozen': True, 'old_path_frozen': True, 'history': [], 'config': cfg,
               'catnet_validation': 'pending', 'stop_reason': None}
    dump(summary_path, summary)
    try:
        set_seed(int(cfg['seed']))
        model = build_final_model(cfg['model']).cuda()
        tr = cfg['training']
        report = load_initial_weights_repaired(model, tr['init_checkpoint'], torch.device('cuda'),
                  tr['init_load_patterns'], key_remaps=tr['init_key_remaps'])
        assert not report['unexpected'] and not report['skipped_shape']
        assert all(k.startswith(('content_encoder.', 'e15_decoder.')) for k in report['missing'])
        dump(directory / 'initialization.yaml', report)
        optimizer = make_optimizer(model, float(tr['learning_rate']))
        loader = loader_for(cfg)
        preflight(model, optimizer, next(iter(loader)), cfg, directory)
        if args.preflight_only:
            summary['status'] = 'preflight_passed'
            dump(summary_path, summary)
            return
        records = read_manifest(cfg['data']['manifest'], cfg['data']['root'], 'val')
        summary['status'] = 'baseline_validation'
        dump(summary_path, summary)
        model.cpu()
        torch.cuda.empty_cache()
        subprocess.run([sys.executable, str(ROOT / 'tools/eval_catnet_original_val.py'),
                        '--manifest', cfg['data']['manifest'], '--out', str(directory)], check=True)
        summary['catnet_validation'] = yaml.safe_load((directory / 'catnet_original_val.yaml').read_text())
        previous_path = cfg['reference_e15_checkpoint']
        previous_state = torch.load(previous_path, map_location='cpu')
        previous_model = build_final_model(previous_state['config']['model']).cuda().eval()
        previous_model.load_state_dict(previous_state['model'], strict=True)
        del previous_state
        summary['previous_e15_validation'] = evaluate(previous_model, records, 'previous_e15_original_val', directory)
        del previous_model
        gc.collect()
        torch.cuda.empty_cache()
        model.cuda()
        optimizer = make_optimizer(model, float(tr['learning_rate']))
        # Zero residual gives the frozen E5 prediction; same forward and metric path as every epoch.
        baseline = evaluate(model, records, 'e5_step0_original_val', directory)
        summary['e5_baseline'] = baseline
        best, stale = -math.inf, 0
        for ep in range(8):
            lr = float(tr['learning_rate']) * (.5 ** max(0, (ep - 3) / 4))
            for group in optimizer.param_groups:
                group['lr'] = lr
            train_mode(model)
            optimizer.zero_grad(set_to_none=True)
            start = time.time()
            totals, sampled = {}, {}
            summary.update(status='training', current_epoch=ep+1, lr=lr)
            dump(summary_path, summary)
            print(f'[train] epoch={ep+1}/8 step=0/{len(loader)} lr={lr:.3g}', flush=True)
            for bi, batch in enumerate(loader):
                x, y = batch['image'].cuda(), batch['mask'].cuda()
                out = model(x)
                terms = objective(out, y, cfg)
                if not torch.isfinite(terms['total']):
                    raise FloatingPointError(f'nonfinite loss epoch={ep+1}, batch={bi+1}')
                accumulation = min(4, len(loader) - (bi // 4) * 4)
                (terms['total'] / accumulation).backward()
                if (bi + 1) % 4 == 0 or bi + 1 == len(loader):
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], .5, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for key, value in terms.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach())
                for key in [batch['geometry'][0], batch['post_geometry_bucket'][0]]:
                    sampled[key] = sampled.get(key, 0) + 1
                del out, terms, x, y
                if bi == 0 or (bi + 1) % 25 == 0 or bi + 1 == len(loader):
                    elapsed = time.time() - start
                    progress = {'epoch': ep+1, 'step': bi+1, 'steps': len(loader), 'lr': lr,
                                'loss': totals['total'] / (bi+1), 'elapsed_minutes': elapsed/60,
                                'eta_minutes': elapsed / (bi+1) * (len(loader)-bi-1)/60}
                    dump(directory / 'progress.yaml', progress)
                    print('[train] ' + json.dumps(progress), flush=True)
            summary['status'] = 'epoch_validation'
            dump(summary_path, summary)
            result = evaluate(model, records, f'epoch_{ep+1:02d}_original_val', directory, include_old=True)
            cur = result['metrics']['fake']['oracle']['f1']
            assert math.isfinite(cur)
            assert abs(result['old_path']['fake']['oracle']['f1'] - baseline['metrics']['fake']['oracle']['f1']) < 1e-8, 'Frozen E5 path drifted'
            improved = cur > best + 1e-4
            stale = 0 if improved else stale + 1
            is_best = cur > best
            best = max(best, cur)
            row = {'epoch': ep+1, 'lr': lr, 'train_losses': {k: v/len(loader) for k,v in totals.items()},
                   'sampled_geometry_and_buckets': sampled, 'validation': result, 'selection_value': cur,
                   'delta_vs_e5': cur - baseline['metrics']['fake']['oracle']['f1'],
                   'alpha_support': float(model.dher.alpha_support.detach()), 'alpha_conflict': float(model.dher.alpha_conflict.detach()),
                   'betas': [float(a.beta.detach()) for a in model.e15_decoder.evidence_adapters],
                   'elapsed_seconds': time.time() - start}
            state = {'epoch': ep, 'model': {k:v.detach().cpu() for k,v in model.state_dict().items()},
                     'optimizer': optimizer.state_dict(), 'config': cfg, 'eval_variant': 'raw', 'best_metric': best,
                     'selection_threshold': result['metrics']['fake']['oracle']['threshold'], 'phase': 'e15_hr_corrected'}
            atomic_torch_save(state, directory / 'last.pt')
            if is_best:
                atomic_torch_save(state, directory / 'best.pt')
                summary.update(best_checkpoint=str(directory / 'best.pt'), best_epoch=ep+1, best_F1=best,
                               selected_threshold=result['metrics']['fake']['oracle']['threshold'])
            del state
            summary['history'].append(row)
            summary['completed_epochs'] = ep+1
            dump(summary_path, summary)
            print(f'[epoch_done] epoch={ep+1} F1={cur:.8f} best={best:.8f} delta_E5={row["delta_vs_e5"]:.8f}', flush=True)
            if ep+1 >= 4 and stale >= 3:
                summary['stop_reason'] = 'F1_patience_3_after_minimum_4_epochs'
                break
        summary['status'] = 'completed'
        summary['stop_reason'] = summary['stop_reason'] or 'maximum_8_epochs'
        summary['resource_gate_delta_0p005_met'] = best - baseline['metrics']['fake']['oracle']['f1'] >= .005
        dump(summary_path, summary)
        print('[complete] ' + summary['stop_reason'], flush=True)
    except Exception:
        summary.update(status='failed', stop_reason=traceback.format_exc())
        dump(summary_path, summary)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--preflight-only', action='store_true')
    main(p.parse_args())
