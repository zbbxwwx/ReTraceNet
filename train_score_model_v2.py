#!/usr/bin/env python
from __future__ import annotations

import argparse, copy, csv, json, math, shutil, sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrace.data import build_loader
from retrace.engine import evaluate_model
from retrace.losses import ReTraceLoss
from retrace.utils import atomic_torch_save, choose_device, load_checkpoint, load_config, set_seed
from retracenet.score_model_v2 import build_score_model_v2
from train_v2_final_pipeline import clean_loss_config, train_loader_for, tversky_loss, update_ema_model

FIELDS = ['epoch','stage','lr','lambda_distill','train_total','val_f1','val_f1_oracle','val_precision_fixed','val_recall_fixed','val_iou_fixed','val_auc','oracle_threshold','selection_value','extra']


def copy_state(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items()}


def load_score_v2_parent(model, parent_path, device):
    ck=load_checkpoint(parent_path, torch.device('cpu'))
    src=ck.get('model', ck)
    tgt=model.state_dict()
    mapping={}
    for k,v in src.items():
        candidates=[]
        if k.startswith('decoder.'):
            candidates.append('teacher_decoder.'+k[len('decoder.'):])
        elif k.startswith('mask_head.'):
            candidates.append('teacher_mask_head.'+k[len('mask_head.'):])
        candidates.append(k)
        for nk in candidates:
            if nk in tgt and tuple(v.shape)==tuple(tgt[nk].shape):
                mapping[nk]=v
    missing=[k for k in tgt if k not in mapping]
    mismatched=[]
    unexpected=[]
    model.load_state_dict(mapping, strict=False)
    model.freeze_teacher()
    report={
        'parent_path': parent_path,
        'total_parent_keys': len(src),
        'loaded_keys': len(mapping),
        'missing_keys': missing,
        'shape_mismatch_keys': mismatched,
        'unexpected_keys': unexpected,
        'loaded_key_examples': sorted(mapping)[:40],
        'missing_key_examples': missing[:80],
    }
    return report


def set_trainable_score(model, stage):
    for n,p in model.named_parameters():
        p.requires_grad_(False)
    if stage == 'stage_a':
        pats=['fpn_decoder','mask_head','log_var_head','aux_heads','image_head']
    elif stage == 'stage_b':
        pats=['content_encoder.layer3','content_encoder.layer4','forensic_stream.stage3','forensic_stream.stage4','fpn_decoder','mask_head','log_var_head','aux_heads','image_head']
    else:
        raise ValueError(stage)
    names=[]
    for n,p in model.named_parameters():
        if any(k in n for k in pats) and not n.startswith('teacher_'):
            p.requires_grad_(True); names.append(n)
    model.freeze_teacher()
    return names


def build_optimizer_score(model,cfg,stage):
    tr=cfg['training']; wd=float(tr.get('weight_decay',0.01)); specs=tr.get('parameter_groups_'+stage) or tr.get('parameter_groups') or []
    named=list(model.named_parameters()); used=set(); groups=[]
    for spec in specs:
        pats=spec.get('patterns',[]); ps=[]
        for n,p in named:
            if p.requires_grad and n not in used and any(k in n for k in pats):
                ps.append(p); used.add(n)
        if ps:
            groups.append({'params':ps,'lr':float(spec.get('lr',tr.get('learning_rate',1e-5))),'weight_decay':float(spec.get('weight_decay',wd)),'name':spec.get('name','+'.join(pats))})
    rem=[p for n,p in named if p.requires_grad and n not in used]
    if rem: groups.append({'params':rem,'lr':float(tr.get('learning_rate',1e-5)),'weight_decay':wd,'name':'remaining'})
    return torch.optim.AdamW(groups)


def positive_preserving_resize(target, size):
    if target.shape[-2:] == size: return target
    h,w=target.shape[-2:]; oh,ow=size
    if oh <= h and ow <= w:
        return F.adaptive_max_pool2d(target, size)
    return F.interpolate(target, size=size, mode='nearest')


def dice_loss_logits(logits,target,eps=1.0):
    p=logits.sigmoid().flatten(1); t=target.flatten(1)
    inter=(p*t).sum(1); den=p.sum(1)+t.sum(1)
    return (1-(2*inter+eps)/(den+eps)).mean()


def aux_pp_loss(out,target,weights):
    loss=target.new_tensor(0.0)
    aux=out.get('aux_logits_low') or []
    for w,logits in zip(weights,aux):
        t=positive_preserving_resize(target, logits.shape[-2:])
        loss = loss + float(w)*(F.binary_cross_entropy_with_logits(logits,t)+dice_loss_logits(logits,t))
    return loss


def epoch_lr_mult(epoch, total_epochs, const=1, minr=0.1):
    if epoch < const: return 1.0
    prog=min(1.0,max(0.0,(epoch-const+1)/max(1,total_epochs-const)))
    return minr+(1-minr)*0.5*(1+math.cos(math.pi*prog))


def apply_lr(opt, bases, mult):
    for g,b in zip(opt.param_groups,bases): g['lr']=b*mult


def evaluate_with_ema(model, ema_model, crit, val_loader, device, amp, cfg):
    _,met=evaluate_model(model,crit,val_loader,device,amp,float(cfg['evaluation'].get('threshold',0.5)),int(cfg['evaluation'].get('thresholds',101)))
    variant='raw'; save_model=model; ema_met=None
    if ema_model is not None:
        _,ema_met=evaluate_model(ema_model,crit,val_loader,device,amp,float(cfg['evaluation'].get('threshold',0.5)),int(cfg['evaluation'].get('thresholds',101)))
        if float(ema_met.get('pixel_f1_oracle',ema_met['pixel_f1_fixed'])) > float(met.get('pixel_f1_oracle',met['pixel_f1_fixed'])):
            met=ema_met; variant='ema'; save_model=ema_model
    return met, variant, save_model, ema_met


def train_epoch(model, ema_model, crit, train_loader, opt, cfg, device, amp, scaler, lambda_distill):
    model.train(); model.freeze_teacher(); opt.zero_grad(set_to_none=True)
    acc=max(1,int(cfg['training'].get('gradient_accumulation_steps',1))); av={}; count=0
    aux_weights=cfg['training'].get('deep_supervision_weights') or [0.3,0.2,0.1]
    for bi,b in enumerate(train_loader):
        im=b['image'].to(device); mask=b['mask'].to(device); lab=b['label'].to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            out=model(im, return_teacher=lambda_distill>0)
            losses=crit(out,mask,lab); loss=losses['total']
            if float(cfg['loss']['weights'].get('tversky',0.0))>0:
                tv=tversky_loss(out['mask_logits'],mask,float(cfg['loss'].get('tversky_alpha_fp',0.3)),float(cfg['loss'].get('tversky_beta_fn',0.7)))
                loss=loss+float(cfg['loss']['weights'].get('tversky'))*tv; losses={**losses,'tversky':tv,'total':loss}
            aux_loss=aux_pp_loss(out,mask,aux_weights)
            loss=loss+aux_loss; losses={**losses,'pp_deep_supervision':aux_loss,'total':loss}
            if lambda_distill>0:
                teacher_prob=out['teacher_logits'].detach().sigmoid()
                kd=F.binary_cross_entropy_with_logits(out['mask_logits'],teacher_prob)
                loss=loss+float(lambda_distill)*kd; losses={**losses,'teacher_distill':kd,'total':loss}
        scaler.scale(loss/acc).backward()
        if (bi+1)%acc==0 or bi+1==len(train_loader):
            if float(cfg['training'].get('grad_clip',0.0))>0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training'].get('grad_clip',1.0)))
            scaler.step(opt); scaler.update()
            if ema_model is not None: update_ema_model(ema_model, model, float(cfg['training'].get('ema_decay',0.999)))
            opt.zero_grad(set_to_none=True)
        for k,v in losses.items(): av[k]=av.get(k,0.0)+float(v.detach().cpu())*im.shape[0]
        count+=im.shape[0]
    return {k:v/max(1,count) for k,v in av.items()}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--stage',choices=['stage_a','stage_b'],required=True); ap.add_argument('--init'); ap.add_argument('--device'); ap.add_argument('--epochs',type=int); ap.add_argument('--start-epoch',type=int,default=0); args=ap.parse_args()
    cfg=load_config(args.config); set_seed(int(cfg.get('seed',3407))); device=choose_device(args.device); outdir=Path(cfg['output_dir']); outdir.mkdir(parents=True,exist_ok=True); shutil.copyfile(args.config,outdir/f'config_{args.stage}.yaml')
    model=build_score_model_v2(cfg['model']).to(device)
    init=args.init or cfg['training'].get('init_checkpoint')
    if args.stage=='stage_a':
        init_report=load_score_v2_parent(model, init, device)
    else:
        ck=load_checkpoint(init, torch.device('cpu')); model.load_state_dict(ck['model'], strict=False); init_report={'resume_from':init,'epoch':ck.get('epoch'),'best_metric':ck.get('best_metric')}; model.freeze_teacher()
    names=set_trainable_score(model,args.stage); opt=build_optimizer_score(model,cfg,args.stage)
    bases=[g['lr'] for g in opt.param_groups]
    amp=bool(cfg['training'].get('amp',False) and device.type=='cuda'); scaler=torch.cuda.amp.GradScaler(enabled=amp)
    use_ema=bool(cfg['training'].get('ema',True)); ema_model=None
    if use_ema:
        ema_model=copy.deepcopy(model).to(device).eval();
        for p in ema_model.parameters(): p.requires_grad_(False)
        if args.stage=='stage_b' and init:
            ck=load_checkpoint(init, torch.device('cpu'))
            if 'ema_model' in ck: ema_model.load_state_dict(ck['ema_model'], strict=False)
    train_loader=train_loader_for(cfg); val_loader=build_loader({**cfg['data'],'batch_size':cfg['evaluation']['batch_size']},'val',shuffle=False)
    crit=ReTraceLoss.from_config(clean_loss_config(cfg['loss'])).to(device)
    total_epochs=int(args.epochs or cfg['training'].get(args.stage+'_epochs', cfg['training'].get('epochs',2)))
    hist=outdir/'history.csv'
    if not hist.exists():
        with hist.open('w',newline='',encoding='utf-8') as f: csv.DictWriter(f,fieldnames=FIELDS).writeheader()
    print(yaml.safe_dump({'stage':args.stage,'init_report':{k:(len(v) if isinstance(v,list) else v) for k,v in init_report.items()},'trainable_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'trainable_examples':names[:40],'optimizer_groups':[{'name':g.get('name'),'lr':g['lr'],'parameter_count':sum(p.numel() for p in g['params'])} for g in opt.param_groups]},sort_keys=False),flush=True)
    best=-1e9
    if args.stage=='stage_b' and init:
        ck=load_checkpoint(init, torch.device('cpu')); best=float(ck.get('best_metric',-1e9))
    for local_ep in range(total_epochs):
        epoch=args.start_epoch+local_ep
        mult=epoch_lr_mult(local_ep,total_epochs,int(cfg['training'].get('constant_lr_epochs',1)),float(cfg['training'].get('min_lr_ratio',0.1)))
        apply_lr(opt,bases,mult)
        lam=0.0
        if args.stage=='stage_a': lam=[0.30,0.15][local_ep] if local_ep<2 else 0.0
        train_losses=train_epoch(model,ema_model,crit,train_loader,opt,cfg,device,amp,scaler,lam)
        met,variant,save_model,ema_met=evaluate_with_ema(model,ema_model,crit,val_loader,device,amp,cfg)
        cur=float(met.get('pixel_f1_oracle',met['pixel_f1_fixed']))
        if cur>best: best=cur
        state={'epoch':epoch,'stage':args.stage,'model':copy_state(model),'best_metric':best,'config':cfg,'init_report':init_report,'eval_variant':'raw'}
        if ema_model is not None: state['ema_model']=copy_state(ema_model)
        atomic_torch_save(state,outdir/'last.pt')
        if cur>=best-1e-12:
            best_state=dict(state); best_state['model']=copy_state(save_model); best_state['eval_variant']=variant; atomic_torch_save(best_state,outdir/'best.pt')
        atomic_torch_save(state,outdir/f'milestone_eplus{epoch+1:02d}.pt')
        extra={'eval_variant':variant, 'ema_oracle_f1': (ema_met or {}).get('pixel_f1_oracle'), 'ema_fixed_f1': (ema_met or {}).get('pixel_f1_fixed'), **{f'train_{k}':v for k,v in train_losses.items()}}
        row={'epoch':epoch,'stage':args.stage,'lr':opt.param_groups[0]['lr'],'lambda_distill':lam,'train_total':train_losses.get('total',0.0),'val_f1':met['pixel_f1_fixed'],'val_f1_oracle':met.get('pixel_f1_oracle'),'val_precision_fixed':met.get('pixel_precision_fixed'),'val_recall_fixed':met.get('pixel_recall_fixed'),'val_iou_fixed':met.get('pixel_iou_fixed'),'val_auc':met.get('pixel_auc'),'oracle_threshold':met.get('oracle_threshold'),'selection_value':cur,'extra':json.dumps(extra,sort_keys=True)}
        with hist.open('a',newline='',encoding='utf-8') as f: csv.DictWriter(f,fieldnames=FIELDS).writerow(row)
        (outdir/'last_metrics.yaml').write_text(yaml.safe_dump({'metrics':met,'extra':extra,'epoch':epoch,'stage':args.stage},sort_keys=False),encoding='utf-8')
        print(yaml.safe_dump(row,sort_keys=False).strip(),flush=True)

if __name__=='__main__': main()
