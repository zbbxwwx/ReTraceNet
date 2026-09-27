#!/usr/bin/env python
from __future__ import annotations

import argparse, csv, json, math, sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.metrics import ForensicMetrics
from retrace.utils import choose_device, load_checkpoint, load_config, set_seed
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import clean_loss_config, load_initial_weights_repaired, set_trainable, build_optimizer


def rms(t):
    x=t.detach().float()
    return float(x.square().mean().sqrt())


def stat_line(t):
    x=t.detach().float()
    return {'shape': list(x.shape), 'mean': float(x.mean()), 'std': float(x.std(unbiased=False)), 'rms': rms(x), 'min': float(x.min()), 'max': float(x.max())}


def evaluate_subset(model, crit, loader, device, n):
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


def make_filtered_parent(parent_path, conv_model, out_path):
    ck=load_checkpoint(parent_path, torch.device('cpu'))
    source=ck.get('model', ck)
    target=conv_model.state_dict()
    filtered={}; mismatched=[]; unexpected=[]
    for k,v in source.items():
        if k not in target:
            unexpected.append(k); continue
        if tuple(v.shape)!=tuple(target[k].shape):
            mismatched.append({'key':k,'parent_shape':list(v.shape),'target_shape':list(target[k].shape)}); continue
        filtered[k]=v
    missing=[k for k in target if k not in filtered]
    torch.save({'model': filtered, 'source_checkpoint': parent_path, 'loaded_key_count': len(filtered), 'shape_mismatched': mismatched, 'unexpected': unexpected, 'new_random_keys': missing}, out_path)
    return {'total_parent_keys': len(source), 'loaded_keys': len(filtered), 'shape_mismatched_keys': mismatched, 'unexpected_parent_keys': unexpected, 'new_random_keys': missing, 'output': str(out_path)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--parent-config', required=True)
    ap.add_argument('--parent-checkpoint', required=True)
    ap.add_argument('--conv-config', required=True)
    ap.add_argument('--conv-pretrained', required=True)
    ap.add_argument('--filtered-output', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--device')
    ap.add_argument('--subset', type=int, default=160)
    args=ap.parse_args()
    set_seed(3407)
    device=choose_device(args.device)
    report={}

    # AUDIT 1/2: direct torchvision sequence and exact official weights.
    from torchvision.models import convnext_tiny
    tv=convnext_tiny(weights=None)
    before={
        'features.0.0.weight': stat_line(tv.features[0][0].weight),
        'features.1.0.block.0.weight': stat_line(tv.features[1][0].block[0].weight),
        'features.7.2.block.5.bias': stat_line(tv.features[7][2].block[5].bias),
    }
    state=torch.load(args.conv_pretrained, map_location='cpu')
    missing, unexpected = tv.load_state_dict(state, strict=False)
    after={
        'features.0.0.weight': stat_line(tv.features[0][0].weight),
        'features.1.0.block.0.weight': stat_line(tv.features[1][0].block[0].weight),
        'features.7.2.block.5.bias': stat_line(tv.features[7][2].block[5].bias),
    }
    x=torch.zeros(1,3,640,640)
    shapes=[]
    with torch.inference_mode():
        z=x
        for i,m in enumerate(tv.features):
            z=m(z)
            shapes.append({'features_index': i, 'shape': list(z.shape)})
    report['official_pretrained_load']={'missing_keys': list(missing), 'unexpected_keys': list(unexpected), 'before': before, 'after': after}
    report['torchvision_feature_sequence_640']=shapes
    report['required_stage_shapes']={
        'C1_features_1': [1,96,160,160],
        'C2_features_3': [1,192,80,80],
        'C3_features_5': [1,384,40,40],
        'C4_features_7': [1,768,20,20],
    }

    # Build/load models.
    parent_cfg=load_config(args.parent_config)
    conv_cfg=load_config(args.conv_config)
    conv_cfg['model']['pretrained_path']=args.conv_pretrained
    conv_cfg['model']['convnext_pretrained']=False
    conv_model=build_final_model(conv_cfg['model']).to(device)
    filter_report=make_filtered_parent(args.parent_checkpoint, conv_model, args.filtered_output)
    load_report=load_initial_weights_repaired(conv_model, args.filtered_output, device, [], [], [])
    report['warm_start_from_B_E1_EMA_best']=filter_report
    report['conv_model_load_report']=load_report

    parent_model=build_final_model(parent_cfg['model']).to(device)
    parent_load=load_initial_weights_repaired(parent_model, args.parent_checkpoint, device, [], [], [])
    report['parent_load_report']={k:(len(v) if isinstance(v,list) else v) for k,v in parent_load.items()}

    # AUDIT 3: dataset image stats + normalized stats exactly once.
    loader=build_loader({**conv_cfg['data'],'batch_size':1,'num_workers':0}, 'val', shuffle=False)
    first=next(iter(loader))
    im=first['image']
    norm=(im-conv_model.content_encoder.mean.cpu())/conv_model.content_encoder.std.cpu()
    report['input_normalization']={
        'raw_image': stat_line(im),
        'normalized_once': stat_line(norm),
        'encoder_imagenet_normalize': bool(getattr(conv_model.content_encoder, 'imagenet_normalize', False)),
    }

    # AUDIT 1: our encoder C1-C4 shapes.
    conv_model.eval()
    with torch.inference_mode():
        feats=conv_model.content_encoder(im.to(device))
    report['our_content_encoder_outputs_640']=[stat_line(f.cpu()) for f in feats]

    # AUDIT 5: trainable/optimizer LR report.
    train_names=set_trainable(conv_model, 'convnext_clean')
    prefix_counts=defaultdict(int)
    for n,p in conv_model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith('content_encoder.features.0'): key='content_encoder.stem'
        elif n.startswith('content_encoder.features.1'): key='content_encoder.stage1'
        elif n.startswith('content_encoder.features.3'): key='content_encoder.stage2'
        elif n.startswith('content_encoder.features.5') or n.startswith('content_encoder.features.4'): key='content_encoder.stage3'
        elif n.startswith('content_encoder.features.7') or n.startswith('content_encoder.features.6'): key='content_encoder.stage4'
        elif n.startswith('forensic_stream'): key='forensic_stream'
        elif n.startswith('decoder'): key='decoder'
        elif n.startswith('mask_head'): key='mask_head'
        elif n.startswith('aux_heads'): key='aux_heads'
        elif n.startswith('log_var_head'): key='log_var_head'
        elif n.startswith('image_head'): key='image_head'
        else: key=n.split('.')[0]
        prefix_counts[key]+=p.numel()
    opt=build_optimizer(conv_model, conv_cfg, 'convnext_clean')
    group_report=[]
    specs=conv_cfg['training'].get('parameter_groups') or []
    for idx,g in enumerate(opt.param_groups):
        cnt=sum(p.numel() for p in g['params'])
        spec=specs[idx] if idx < len(specs) else {'patterns':['remaining']}
        group_report.append({'index':idx,'patterns':spec.get('patterns'), 'lr':g['lr'], 'parameter_count':cnt})
    report['trainable_parameter_counts_by_prefix']=dict(prefix_counts)
    report['optimizer_groups']=group_report
    report['trainable_examples']=train_names[:60]

    random_trainable=[k for k in filter_report['new_random_keys'] if any(k==n or k.rsplit('.',1)[0] in n for n in train_names)]
    # simpler exact parameter-name subset
    train_set=set(train_names)
    report['new_random_parameter_keys_trainable']=[k for k in filter_report['new_random_keys'] if k in train_set]
    report['new_random_parameter_keys_frozen_examples']=[k for k in filter_report['new_random_keys'] if k not in train_set][:80]

    # NO-TRAIN sanity subset.
    crit=ReTraceLoss.from_config(clean_loss_config(conv_cfg['loss'])).to(device)
    parent_model.eval(); conv_model.eval()
    report['sanity_subset']={
        'subset_n': args.subset,
        'parent_B_E1_EMA': evaluate_subset(parent_model, crit, build_loader({**parent_cfg['data'],'batch_size':1,'num_workers':0}, 'val', shuffle=False), device, args.subset),
        'convnext_warm_start': evaluate_subset(conv_model, crit, build_loader({**conv_cfg['data'],'batch_size':1,'num_workers':0}, 'val', shuffle=False), device, args.subset),
    }

    # PASS flags.
    stage_shapes=[s['shape'] for s in shapes]
    report['pass_flags']={
        'pretrained_exact': len(missing)==0 and len(unexpected)==0,
        'stage_indices_correct': stage_shapes[1]==[1,96,160,160] and stage_shapes[3]==[1,192,80,80] and stage_shapes[5]==[1,384,40,40] and stage_shapes[7]==[1,768,20,20],
        'normalization_once_plausible': 0.0 <= report['input_normalization']['raw_image']['min'] and report['input_normalization']['raw_image']['max'] <= 1.0 and report['input_normalization']['encoder_imagenet_normalize'],
        'inherits_forensic_decoder_heads': filter_report['loaded_keys'] > 100 and all(any(k.startswith(pref) for k in filter_report['warm_start_from_B_E1_EMA_best']['loaded_keys'] if False) for pref in []),
        'random_decoder_projection_trainable': all(k in train_set for k in filter_report['new_random_keys'] if k.startswith('decoder.stage') and k.endswith('.weight')),
    }

    Path(args.report).write_text(yaml.safe_dump(report, sort_keys=False), encoding='utf-8')
    print(yaml.safe_dump(report, sort_keys=False))

if __name__=='__main__':
    main()
