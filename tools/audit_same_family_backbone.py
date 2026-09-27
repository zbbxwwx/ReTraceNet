#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from collections import defaultdict
from pathlib import Path
import torch, yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.metrics import ForensicMetrics
from retrace.utils import choose_device, load_checkpoint, load_config, set_seed
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import clean_loss_config, load_initial_weights_repaired, set_trainable, build_optimizer


def stat(t):
    x=t.detach().float(); return {'shape':list(x.shape),'mean':float(x.mean()),'std':float(x.std(unbiased=False)),'rms':float(x.square().mean().sqrt()),'min':float(x.min()),'max':float(x.max())}

def eval_subset(model, loader, device, n):
    model.eval(); metrics=ForensicMetrics(fixed_threshold=0.5); seen=0
    with torch.inference_mode():
        for b in loader:
            im=b['image'].to(device); mask=b['mask'].to(device); lab=b['label'].to(device)
            out=model(im); metrics.update(out,mask,lab); seen+=im.shape[0]
            if seen>=n: break
    m=metrics.compute(101); m['samples']=seen; return m

def filter_parent(parent_path, target, out_path):
    ck=load_checkpoint(parent_path, torch.device('cpu')); src=ck.get('model', ck); tgt=target.state_dict()
    filt={}; skipped_content=[]; mism=[]; unexpected=[]
    for k,v in src.items():
        if k.startswith('content_encoder.'):
            skipped_content.append(k); continue
        if k not in tgt:
            unexpected.append(k); continue
        if tuple(v.shape)!=tuple(tgt[k].shape):
            mism.append({'key':k,'parent_shape':list(v.shape),'target_shape':list(tgt[k].shape)}); continue
        filt[k]=v
    missing=[k for k in tgt if k not in filt and not k.startswith('content_encoder.')]
    torch.save({'model':filt,'source_checkpoint':parent_path,'loaded_key_count':len(filt),'skipped_content_keys':skipped_content,'shape_mismatched_keys':mism,'unexpected_parent_keys':unexpected,'new_random_non_content_keys':missing}, out_path)
    return {'total_parent_keys':len(src),'loaded_keys':len(filt),'skipped_content_count':len(skipped_content),'shape_mismatched_keys':mism,'unexpected_parent_keys':unexpected,'new_random_non_content_keys':missing,'output':str(out_path)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--parent-config',required=True); ap.add_argument('--parent-checkpoint',required=True); ap.add_argument('--candidate-config',required=True); ap.add_argument('--pretrained',required=True); ap.add_argument('--backbone',required=True, choices=['resnext101_32x8d','resnet101']); ap.add_argument('--filtered-output',required=True); ap.add_argument('--report',required=True); ap.add_argument('--device'); ap.add_argument('--subset',type=int,default=160); args=ap.parse_args()
    set_seed(3407); device=choose_device(args.device); report={}
    if args.backbone=='resnext101_32x8d':
        from torchvision.models import resnext101_32x8d as make
    else:
        from torchvision.models import resnet101 as make
    raw=make(weights=None)
    before={'conv1.weight':stat(raw.conv1.weight),'layer4.2.conv3.weight':stat(raw.layer4[-1].conv3.weight)}
    state=torch.load(args.pretrained,map_location='cpu')
    missing,unexpected=raw.load_state_dict(state, strict=False)
    after={'conv1.weight':stat(raw.conv1.weight),'layer4.2.conv3.weight':stat(raw.layer4[-1].conv3.weight)}
    report['official_pretrained_load']={'missing_keys':list(missing),'unexpected_keys':list(unexpected),'before':before,'after':after}
    cand_cfg=load_config(args.candidate_config); par_cfg=load_config(args.parent_config)
    cand=build_final_model(cand_cfg['model']).to(device)
    filt=filter_parent(args.parent_checkpoint,cand,args.filtered_output)
    loadrep=load_initial_weights_repaired(cand,args.filtered_output,device,[],[],[])
    report['parent_warm_start']=filt; report['candidate_load_report']={k:(len(v) if isinstance(v,list) else v) for k,v in loadrep.items()}
    parent=build_final_model(par_cfg['model']).to(device); preport=load_initial_weights_repaired(parent,args.parent_checkpoint,device,[],[],[])
    report['parent_load_report']={k:(len(v) if isinstance(v,list) else v) for k,v in preport.items()}
    loader=build_loader({**cand_cfg['data'],'batch_size':1,'num_workers':0},'val',shuffle=False); first=next(iter(loader)); im=first['image']
    report['input_raw']=stat(im)
    with torch.inference_mode(): feats=cand.content_encoder(im.to(device))
    report['candidate_feature_shapes']=[stat(f.cpu()) for f in feats]
    names=set_trainable(cand,'strong_backbone'); counts=defaultdict(int)
    for n,p in cand.named_parameters():
        if not p.requires_grad: continue
        if n.startswith('content_encoder.stem'): key='content_encoder.stem'
        elif n.startswith('content_encoder.layer1'): key='content_encoder.layer1'
        elif n.startswith('content_encoder.layer2'): key='content_encoder.layer2'
        elif n.startswith('content_encoder.layer3'): key='content_encoder.layer3'
        elif n.startswith('content_encoder.layer4'): key='content_encoder.layer4'
        elif n.startswith('forensic_stream'): key='forensic_stream'
        elif n.startswith('decoder'): key='decoder'
        elif n.startswith('mask_head'): key='mask_head'
        elif n.startswith('aux_heads'): key='aux_heads'
        elif n.startswith('log_var_head'): key='log_var_head'
        elif n.startswith('image_head'): key='image_head'
        else: key=n.split('.')[0]
        counts[key]+=p.numel()
    opt=build_optimizer(cand,cand_cfg,'strong_backbone'); specs=cand_cfg['training'].get('parameter_groups') or []
    groups=[]
    for i,g in enumerate(opt.param_groups): groups.append({'index':i,'patterns':(specs[i].get('patterns') if i<len(specs) else ['remaining']),'lr':g['lr'],'parameter_count':sum(p.numel() for p in g['params'])})
    report['trainable_parameter_counts_by_prefix']=dict(counts); report['optimizer_groups']=groups; report['trainable_examples']=names[:60]
    crit=ReTraceLoss.from_config(clean_loss_config(cand_cfg['loss'])).to(device)
    report['sanity_subset']={
      'subset_n':args.subset,
      'parent_B_E1_EMA':eval_subset(parent,build_loader({**par_cfg['data'],'batch_size':1,'num_workers':0},'val',shuffle=False),device,args.subset),
      'candidate_no_train':eval_subset(cand,build_loader({**cand_cfg['data'],'batch_size':1,'num_workers':0},'val',shuffle=False),device,args.subset),
    }
    cm=report['sanity_subset']['candidate_no_train']; oracle=float(cm.get('pixel_f1_oracle',0)); auc=float(cm.get('pixel_auc',0))
    if oracle < 0.40: decision='STOP_LT_0P40'
    elif oracle >= 0.55 and auc >= 0.80: decision='LAUNCH_3_EPOCH_SCREEN'
    else: decision='STOP_SANITY_INSUFFICIENT'
    report['decision']=decision
    Path(args.report).write_text(yaml.safe_dump(report,sort_keys=False),encoding='utf-8')
    print(yaml.safe_dump(report,sort_keys=False))

if __name__=='__main__': main()
