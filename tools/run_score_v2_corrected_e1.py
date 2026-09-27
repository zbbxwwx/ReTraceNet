#!/usr/bin/env python
from __future__ import annotations
import argparse, copy, csv, json, sys
from pathlib import Path
import torch, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrace.data import build_loader
from retrace.losses import ReTraceLoss
from retrace.metrics import ForensicMetrics
from retrace.utils import atomic_torch_save, choose_device, load_checkpoint, load_config, set_seed
from retracenet.score_model_v2 import build_score_model_v2
from train_score_model_v2 import set_trainable_score, build_optimizer_score, train_epoch, copy_state
from train_v2_final_pipeline import clean_loss_config

FIELDS=['epoch','stage','lr','lambda_distill','train_total','val_f1_raw','val_f1_oracle_raw','val_auc_raw','val_f1_ema','val_f1_oracle_ema','val_auc_ema','selection_value','decision','extra']

def eval_metrics_and_calibration(model, loader, device, thresholds=101, cap=1000000):
    model.eval(); metrics=ForensicMetrics(fixed_threshold=0.5)
    fg_sum=bg_sum=0.0; fg_n=bg_n=0; pred_pos=gt_pos=tot=0; logit_sum=logit_sq=0.0
    fg_samp=[]; bg_samp=[]
    with torch.inference_mode():
        for b in loader:
            im=b['image'].to(device); mask=b['mask'].to(device); lab=b['label'].to(device)
            out=model(im); logits=out['mask_logits'].detach(); prob=logits.sigmoid(); m=mask>0.5
            metrics.update(out,mask,lab)
            pf=prob[m]; pb=prob[~m]
            if pf.numel():
                fg_sum+=float(pf.sum().cpu()); fg_n+=pf.numel()
                if sum(x.numel() for x in fg_samp)<cap: fg_samp.append(pf.flatten()[:max(0,cap-sum(x.numel() for x in fg_samp))].cpu())
            if pb.numel():
                bg_sum+=float(pb.sum().cpu()); bg_n+=pb.numel()
                if sum(x.numel() for x in bg_samp)<cap: bg_samp.append(pb.flatten()[:max(0,cap-sum(x.numel() for x in bg_samp))].cpu())
            pred_pos+=int((prob>=0.5).sum().cpu()); gt_pos+=int(m.sum().cpu()); tot+=prob.numel()
            lf=logits.float(); logit_sum+=float(lf.sum().cpu()); logit_sq+=float(lf.square().sum().cpu())
    met=metrics.compute(thresholds)
    fg=torch.cat(fg_samp) if fg_samp else torch.tensor([]); bg=torch.cat(bg_samp) if bg_samp else torch.tensor([])
    lm=logit_sum/max(1,tot); lv=max(0.0,logit_sq/max(1,tot)-lm*lm)
    cal={
        'foreground_probability_mean': fg_sum/max(1,fg_n),
        'foreground_probability_median_sampled': float(fg.median()) if fg.numel() else float('nan'),
        'background_probability_mean': bg_sum/max(1,bg_n),
        'background_probability_median_sampled': float(bg.median()) if bg.numel() else float('nan'),
        'predicted_positive_ratio_at_0p5': pred_pos/max(1,tot),
        'gt_positive_ratio': gt_pos/max(1,tot),
        'logit_mean': lm,
        'logit_std': lv**0.5,
        'median_sample_cap_per_fg_bg': cap,
    }
    return met, cal

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--device'); args=ap.parse_args()
    set_seed(3407); cfg=load_config(args.config); device=choose_device(args.device); outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    model=build_score_model_v2(cfg['model']).to(device); ck=load_checkpoint(args.checkpoint, torch.device('cpu')); model.load_state_dict(ck['model'], strict=False); model.freeze_teacher()
    names=set_trainable_score(model,'stage_a'); opt=build_optimizer_score(model,cfg,'stage_a')
    # Constant corrective-E1 LR: no scheduler decay.
    for g in opt.param_groups:
        if g.get('name')=='fpn_decoder': g['lr']=1e-5
        elif g.get('name')=='mask_head': g['lr']=2e-5
        else: g['lr']=1e-5
    ema_model=copy.deepcopy(model).to(device).eval()
    for p in ema_model.parameters(): p.requires_grad_(False)
    if 'ema_model' in ck: ema_model.load_state_dict(ck['ema_model'], strict=False)
    train_loader=train_score_loader=__import__('train_v2_final_pipeline').train_loader_for(cfg)
    val_loader=build_loader({**cfg['data'],'batch_size':cfg['evaluation']['batch_size']},'val',shuffle=False)
    crit=ReTraceLoss.from_config(clean_loss_config(cfg['loss'])).to(device)
    amp=bool(cfg['training'].get('amp',False) and device.type=='cuda'); scaler=torch.cuda.amp.GradScaler(enabled=amp)
    print(yaml.safe_dump({'resume_from':args.checkpoint,'source_epoch':ck.get('epoch'),'source_best_metric':ck.get('best_metric'),'lambda_distill':0.15,'trainable_count':sum(p.numel() for p in model.parameters() if p.requires_grad),'trainable_examples':names[:40],'optimizer_groups':[{'name':g.get('name'),'lr':g['lr'],'parameter_count':sum(p.numel() for p in g['params'])} for g in opt.param_groups]},sort_keys=False),flush=True)
    train_losses=train_epoch(model,ema_model,crit,train_loader,opt,cfg,device,amp,scaler,0.15)
    raw_met, raw_cal=eval_metrics_and_calibration(model,build_loader({**cfg['data'],'batch_size':cfg['evaluation']['batch_size'],'num_workers':0},'val',shuffle=False),device,int(cfg['evaluation'].get('thresholds',101)))
    ema_met, ema_cal=eval_metrics_and_calibration(ema_model,build_loader({**cfg['data'],'batch_size':cfg['evaluation']['batch_size'],'num_workers':0},'val',shuffle=False),device,int(cfg['evaluation'].get('thresholds',101)))
    if float(ema_met.get('pixel_f1_oracle',0)) > float(raw_met.get('pixel_f1_oracle',0)):
        selected='ema'; selected_met=ema_met; save_model=ema_model
    else:
        selected='raw'; selected_met=raw_met; save_model=model
    oracle=float(selected_met.get('pixel_f1_oracle',selected_met['pixel_f1_fixed'])); auc=float(selected_met.get('pixel_auc',0.0))
    if oracle>=0.745: decision='PASS_STAGE_A_ENTER_STAGE_B'
    elif oracle>=0.740: decision='PASS_STAGE_A_ENTER_STAGE_B'
    elif oracle>=0.735 and auc>=0.94: decision='PASS_STAGE_A_ENTER_STAGE_B'
    else: decision='STOP_DECODER_ROUTE'
    state={'epoch':1,'stage':'stage_a_corrected_e1','model':copy_state(model),'ema_model':copy_state(ema_model),'best_metric':oracle,'config':cfg,'source_checkpoint':args.checkpoint,'eval_variant':'raw'}
    atomic_torch_save(state,outdir/'last.pt')
    best_state=dict(state); best_state['model']=copy_state(save_model); best_state['eval_variant']=selected; atomic_torch_save(best_state,outdir/'best.pt')
    extra={'selected':selected,'raw_metrics':raw_met,'ema_metrics':ema_met,'raw_calibration':raw_cal,'ema_calibration':ema_cal, **{f'train_{k}':v for k,v in train_losses.items()}}
    row={'epoch':1,'stage':'stage_a_corrected_e1','lr':opt.param_groups[0]['lr'],'lambda_distill':0.15,'train_total':train_losses.get('total',0.0),'val_f1_raw':raw_met['pixel_f1_fixed'],'val_f1_oracle_raw':raw_met.get('pixel_f1_oracle'),'val_auc_raw':raw_met.get('pixel_auc'),'val_f1_ema':ema_met['pixel_f1_fixed'],'val_f1_oracle_ema':ema_met.get('pixel_f1_oracle'),'val_auc_ema':ema_met.get('pixel_auc'),'selection_value':oracle,'decision':decision,'extra':json.dumps(extra,sort_keys=True)}
    with (outdir/'history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerow(row)
    (outdir/'last_metrics.yaml').write_text(yaml.safe_dump({'row':row,'extra':extra},sort_keys=False),encoding='utf-8')
    (outdir/'decision.txt').write_text(f'oracle={oracle}\nauc={auc}\nselected={selected}\ndecision={decision}\n',encoding='utf-8')
    print(yaml.safe_dump({'row':row,'raw_calibration':raw_cal,'ema_calibration':ema_cal,'decision':decision},sort_keys=False),flush=True)
if __name__=='__main__': main()
