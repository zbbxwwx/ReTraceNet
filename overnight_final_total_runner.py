#!/usr/bin/env python
"""Independent overnight runner for final QTR -> HCV -> DHER -> Total pipeline."""
from __future__ import annotations

import argparse, csv, json, os, shutil, subprocess, sys, time
from pathlib import Path
from datetime import datetime
import yaml

ROOT = Path('/group4/zouwenxuan/iml_generalization/repos/ReTraceNet')
PY = Path('/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python')
B0_CKPT = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/b0_fulltrain_v2_from_e5/best.pt')
B0_ORACLE = 0.745215700809217
B0_FIXED = 0.7451277283393102
B0_RECALL = 0.7205228638026302
B0_PRECISION = 0.771472451191214
B0_IOU = 0.5937877066589519
B0_AUC = 0.9185181532631389
B0_LT05_F1 = 0.29934836075965415
B0_LT05_RECALL = 0.3977066012693194
B0_LT1_F1 = 0.4537111895109143
B0_LT1_RECALL = 0.4937353435991135


def now(): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def log(msg): print(f'[{now()}] {msg}', flush=True)

def base_cfg(arch, outdir, epochs, image_size=512, init=str(B0_CKPT), lr=2e-5):
    return {
        'seed': 3407,
        'experiment': outdir.name,
        'output_dir': str(outdir),
        'data': {
            'root': '/',
            'manifest': str(ROOT / 'data/casia2_721_manifest.csv'),
            'image_size': image_size,
            'num_workers': 4,
            'authentic_probability': 0.1,
            'positive_crop_probability': 0.7,
            'boundary_crop_probability': 0.0,
            'min_positive_crop_ratio': 0.005,
            'crop_retries': 20,
            'jpeg_probability': 0.15,
            'jpeg_quality': [55, 96],
            'blur_probability': 0.1,
            'color_jitter': 0.08,
        },
        'evaluation': {'batch_size': 1, 'threshold': 0.5, 'thresholds': 101, 'save_predictions': False},
        'model': {
            'architecture': arch,
            'decoder_channels': 128,
            'encoder_backbone': 'resnet50',
            'forensic_channels': [64,128,256,512],
            'freeze_encoder_bn': True,
            'pretrained_path': '/group4/zouwenxuan/iml_generalization/weights/resnet50-19c8e357.pth',
            'residual_kernels': 8,
        },
        'loss': {
            'boundary_kernel': 5,
            'counterfactual_margin': 0.5,
            'positive_weight_clip': [1.0, 1.0],
            'weights': {
                'localization': 1.0, 'dice': 1.0, 'deep_supervision': 0.4,
                'boundary': 0.0, 'image': 0.0, 'proposal': 0.0, 'counterfactual': 0.0,
                'degradation_consistency': 0.0, 'variance_regularization': 0.0,
                'orthogonality': 0.0, 'ledger': 0.0, 'intervention_locality': 0.0,
            }
        },
        'hcv_loss': {'proposal': 1.0, 'verifier': 1.0, 'withdrawal': 0.8, 'withdrawal_margin': 0.15, 'stability': 0.35, 'locality': 0.20},
        'training': {
            'init_checkpoint': init,
            'epochs': epochs,
            'batch_size': 1,
            'gradient_accumulation_steps': 4,
            'learning_rate': lr,
            'weight_decay': 0.01,
            'grad_clip': 0.5,
            'amp': False,
            'selection_metric': 'pixel_f1_oracle',
            'constant_lr_epochs': epochs,
            'min_lr_ratio': 1.0,
            'early_stopping': epochs + 1,
            'meaningful_delta': 0.0,
            'consistency_probability': 0.0,
            'intervention_teacher_probability': 0.0,
            'diagnostic_batches': 512,
        },
    }

def write_yaml(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding='utf-8')

def run(cmd, log_path, env=None, allow_fail=False):
    log(f'RUN {" ".join(str(x) for x in cmd)}')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as f:
        rc = subprocess.run([str(x) for x in cmd], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    log(f'EXIT rc={rc} log={log_path}')
    if rc != 0 and not allow_fail:
        raise RuntimeError(f'command failed rc={rc}: {cmd}; see {log_path}')
    return rc

def gpu_env(gpu='1'):
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=gpu; return env

def read_history(outdir):
    p=outdir/'history.csv'
    if not p.exists(): return []
    with p.open(encoding='utf-8') as f: return list(csv.DictReader(f))

def best_row(outdir):
    rows=read_history(outdir)
    if not rows: return None
    return max(rows, key=lambda r: float(r.get('val_f1_oracle') or -1))

def read_size(outdir, epoch):
    p=outdir/f'val_size_epoch{int(epoch):03d}.csv'
    if not p.exists(): return {}
    with p.open(encoding='utf-8') as f: return {r['bucket']:r for r in csv.DictReader(f)}

def read_diag(outdir, epoch):
    p=outdir/f'hcv_diag_epoch{int(epoch):03d}.yaml'
    if not p.exists(): return []
    return yaml.safe_load(p.read_text(encoding='utf-8')) or []

def row_float(row, key, default=0.0):
    try: return float(row.get(key, default))
    except Exception: return default

def hcv_pass(outdir):
    br=best_row(outdir)
    if not br: return False, {'reason':'no_history'}
    ep=int(br['epoch']); diag=read_diag(outdir,ep)
    def get(sc,b): return next((r for r in diag if r['scale']==sc and r['bucket']==b), {})
    conditions={}
    # 1: proposal not all-active and at least one/two scales foreground higher than authentic/background.
    good_prop=0
    for sc in ['s1','s2','s3']:
        fg=row_float(get(sc,'lt_1'),'proposal_mean'); bg=row_float(get(sc,'background'),'proposal_mean'); au=row_float(get(sc,'authentic'),'proposal_mean')
        if fg < 0.85 and (fg > bg + 0.03 or fg > au + 0.03): good_prop += 1
    conditions['proposal_calibrated'] = good_prop >= 2
    # 2: at least two scales tiny <1% positive Delta and Sep>=0.25.
    good_delta=sum(1 for sc in ['s1','s2','s3'] if row_float(get(sc,'lt_1'),'delta_mean')>0 and row_float(get(sc,'lt_1'),'sep_delta')>=0.25)
    conditions['tiny_positive_delta_sep'] = good_delta >= 2
    # 3: authentic/background absDelta lower than manipulated tiny.
    good_abs=sum(1 for sc in ['s1','s2','s3'] if row_float(get(sc,'lt_1'),'abs_delta_mean') > max(row_float(get(sc,'authentic'),'abs_delta_mean'), row_float(get(sc,'background'),'abs_delta_mean')) * 1.1)
    conditions['auth_bg_absdelta_lower'] = good_abs >= 2
    # 4: intervention inside/outside energy ratio.
    good_energy=sum(1 for sc in ['s1','s2','s3'] if row_float(get(sc,'lt_1'),'inside_outside_energy_ratio')>=1.5)
    conditions['energy_localized'] = good_energy >= 2
    oracle=row_float(br,'val_f1_oracle'); conditions['no_collapse'] = oracle >= B0_ORACLE - 0.01
    passed = sum(1 for k,v in conditions.items() if k!='no_collapse' and v) >= 3 and conditions['no_collapse']
    return passed, {'epoch':ep,'oracle':oracle,'conditions':conditions,'good_delta_scales':good_delta,'good_energy_scales':good_energy}

def qtr_pass(qtr_dir, hcv_dir):
    qb=best_row(qtr_dir); hb=best_row(hcv_dir)
    if not qb or not hb: return False, {'reason':'missing_history'}
    qdiag=read_diag(qtr_dir,int(qb['epoch'])); hdiag=read_diag(hcv_dir,int(hb['epoch']))
    def get(diag,sc,b): return next((r for r in diag if r['scale']==sc and r['bucket']==b), {})
    wins=0; details={}
    for sc in ['s1','s2','s3']:
        bg_down=row_float(get(qdiag,sc,'background'),'proposal_mean') < row_float(get(hdiag,sc,'background'),'proposal_mean') - 0.01
        auth_down=row_float(get(qdiag,sc,'authentic'),'proposal_mean') < row_float(get(hdiag,sc,'authentic'),'proposal_mean') - 0.01
        abs_down=row_float(get(qdiag,sc,'authentic'),'abs_delta_mean') < row_float(get(hdiag,sc,'authentic'),'abs_delta_mean') - 0.01
        sep_up=row_float(get(qdiag,sc,'lt_1'),'sep_delta') > row_float(get(hdiag,sc,'lt_1'),'sep_delta') + 0.05
        details[sc]={'bg_proposal_down':bg_down,'auth_proposal_down':auth_down,'auth_absdelta_down':abs_down,'tiny_sepdelta_up':sep_up}
        if (bg_down or auth_down) and abs_down and sep_up: wins += 1
    passed = wins >= 1 and row_float(qb,'val_f1_oracle') >= row_float(hb,'val_f1_oracle') - 0.003
    return passed, {'epoch':int(qb['epoch']),'oracle':row_float(qb,'val_f1_oracle'),'scale_details':details,'wins':wins}

def dher_pass(dher_dir, parent_dir):
    db=best_row(dher_dir); pb=best_row(parent_dir)
    if not db or not pb: return False, {'reason':'missing_history'}
    dsize=read_size(dher_dir,int(db['epoch'])); psize=read_size(parent_dir,int(pb['epoch']))
    overall=row_float(db,'val_f1_oracle') >= row_float(pb,'val_f1_oracle') + 0.002
    tiny05=float(dsize.get('lt_0p5',{}).get('f1',0)) >= float(psize.get('lt_0p5',{}).get('f1',0)) + 0.01
    tiny1=float(dsize.get('lt_1',{}).get('f1',0)) >= float(psize.get('lt_1',{}).get('f1',0)) + 0.01
    recall_ok=row_float(db,'val_recall_fixed') >= row_float(pb,'val_recall_fixed') - 0.01
    passed=(overall or tiny05 or tiny1) and recall_ok
    return passed, {'epoch':int(db['epoch']),'oracle':row_float(db,'val_f1_oracle'),'parent_oracle':row_float(pb,'val_f1_oracle'),'overall_gain':row_float(db,'val_f1_oracle')-row_float(pb,'val_f1_oracle'),'tiny05_gain':float(dsize.get('lt_0p5',{}).get('f1',0))-float(psize.get('lt_0p5',{}).get('f1',0)),'tiny1_gain':float(dsize.get('lt_1',{}).get('f1',0))-float(psize.get('lt_1',{}).get('f1',0)),'recall_delta':row_float(db,'val_recall_fixed')-row_float(pb,'val_recall_fixed')}

def model_row(name, res, mods, outdir, fallback=None):
    br=best_row(outdir) if outdir else None
    if not br and fallback: return fallback
    if not br: return {'Model':name,'Resolution':res,'Innovation modules':mods}
    size=read_size(outdir,int(br['epoch']))
    return {'Model':name,'Resolution':res,'Innovation modules':mods,'Fixed F1':row_float(br,'val_f1'),'Oracle F1':row_float(br,'val_f1_oracle'),'Precision':row_float(br,'val_precision_fixed'),'Recall':row_float(br,'val_recall_fixed'),'IoU':row_float(br,'val_iou_fixed'),'AUC':row_float(br,'val_auc'),'<0.5% F1':float(size.get('lt_0p5',{}).get('f1',0)),'<0.5% Recall':float(size.get('lt_0p5',{}).get('recall',0)),'<1% F1':float(size.get('lt_1',{}).get('f1',0)),'<1% Recall':float(size.get('lt_1',{}).get('recall',0)),'Checkpoint':str(outdir/'best.pt'),'Epoch':int(br['epoch'])}

def md_table(rows, keys):
    out=['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']
    for r in rows:
        vals=[]
        for k in keys:
            v=r.get(k,'')
            if isinstance(v,float): v=f'{v:.6f}'
            vals.append(str(v))
        out.append('| '+' | '.join(vals)+' |')
    return '\n'.join(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu',default='1'); ap.add_argument('--check-only',action='store_true'); args=ap.parse_args()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    root=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')/f'final_overnight_{stamp}'
    root.mkdir(parents=True, exist_ok=True)
    (Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')/'CURRENT_FINAL_OVERNIGHT.txt').write_text(str(root),encoding='utf-8')
    cfgdir=root/'configs'; logs=root/'logs'; checks=root/'checks'
    b0_cfg=base_cfg('retracenet_b0', root/'b0_reference_cfg', 0); b0_cfg_path=cfgdir/'b0_parent.yaml'; write_yaml(b0_cfg_path,b0_cfg)
    hcv_cfg=base_cfg('retracenet_b1_hcv2', root/'hcv2_calibrated', 5, lr=2e-5); hcv_cfg_path=cfgdir/'hcv2_calibrated.yaml'; write_yaml(hcv_cfg_path,hcv_cfg)
    qtr_cfg=base_cfg('retracenet_b2_qtr_hcv2', root/'qtr_hcv', 4, lr=2e-5); qtr_cfg['training']['parameter_groups']=[{'patterns':['qtr.'],'lr':2e-5},{'patterns':['hcv2.'],'lr':2e-6}]; qtr_cfg_path=cfgdir/'qtr_hcv.yaml'; write_yaml(qtr_cfg_path,qtr_cfg)
    log(f'created run root {root}')
    # Syntax/preflight before long launch.
    run([PY,'-m','py_compile','retracenet/final_modules.py','train_v2_final_pipeline.py','tools/check_final_pipeline.py'], logs/'py_compile.log')
    run([PY,'tools/check_final_pipeline.py','--parent-config',b0_cfg_path,'--child-config',hcv_cfg_path,'--checkpoint',B0_CKPT,'--phase','hcv2','--device','cuda:0','--output',checks/'hcv2_preflight.json'], logs/'hcv2_preflight.log', env=gpu_env(args.gpu))
    if args.check_only:
        log('check-only complete'); return
    stages={}; decisions={}
    # Phase A
    run([PY,'train_v2_final_pipeline.py','--config',hcv_cfg_path,'--phase','hcv2','--device','cuda:0'], logs/'hcv2_train.log', env=gpu_env(args.gpu))
    hp, hd = hcv_pass(hcv_cfg['output_dir'] and Path(hcv_cfg['output_dir']))
    stages['HCV-v2']=Path(hcv_cfg['output_dir']); decisions['HCV mechanism']=('PASS' if hp else 'FAIL', hd)
    parent_cfg_path=b0_cfg_path; parent_ckpt=B0_CKPT; parent_dir=None; parent_arch='retracenet_b0'; parent_mods='none'
    if hp:
        parent_cfg_path=hcv_cfg_path; parent_ckpt=Path(hcv_cfg['output_dir'])/'best.pt'; parent_dir=Path(hcv_cfg['output_dir']); parent_arch='retracenet_b1_hcv2'; parent_mods='HCV'
        # Phase B
        qtr_cfg['training']['init_checkpoint']=str(parent_ckpt); write_yaml(qtr_cfg_path,qtr_cfg)
        run([PY,'tools/check_final_pipeline.py','--parent-config',parent_cfg_path,'--child-config',qtr_cfg_path,'--checkpoint',parent_ckpt,'--phase','qtr_hcv','--device','cuda:0','--output',checks/'qtr_preflight.json'], logs/'qtr_preflight.log', env=gpu_env(args.gpu))
        run([PY,'train_v2_final_pipeline.py','--config',qtr_cfg_path,'--phase','qtr_hcv','--device','cuda:0'], logs/'qtr_train.log', env=gpu_env(args.gpu))
        qp, qd=qtr_pass(Path(qtr_cfg['output_dir']), Path(hcv_cfg['output_dir'])); stages['QTR+HCV']=Path(qtr_cfg['output_dir']); decisions['QTR']=('PASS' if qp else 'FAIL', qd)
        if qp:
            parent_cfg_path=qtr_cfg_path; parent_ckpt=Path(qtr_cfg['output_dir'])/'best.pt'; parent_dir=Path(qtr_cfg['output_dir']); parent_arch='retracenet_b2_qtr_hcv2'; parent_mods='QTR+HCV'
        # Phase C
        dher_arch='retracenet_b3_qtr_hcv2_dher' if qp else 'retracenet_b3_hcv2_dher'
        dher_cfg=base_cfg(dher_arch, root/'dher_parent', 6, init=str(parent_ckpt), lr=2e-5)
        dher_cfg['training'].update({'constant_lr_epochs':2,'min_lr_ratio':0.1,'early_stopping':7})
        dher_cfg['training']['parameter_groups']=[{'patterns':['dher.'],'lr':2e-5},{'patterns':['decoder.stage1','mask_head'],'lr':5e-6},{'patterns':['hcv2.','qtr.'],'lr':2e-6}]
        dher_cfg_path=cfgdir/'dher.yaml'; write_yaml(dher_cfg_path,dher_cfg)
        run([PY,'tools/check_final_pipeline.py','--parent-config',parent_cfg_path,'--child-config',dher_cfg_path,'--checkpoint',parent_ckpt,'--phase','dher','--device','cuda:0','--output',checks/'dher_preflight.json'], logs/'dher_preflight.log', env=gpu_env(args.gpu))
        run([PY,'train_v2_final_pipeline.py','--config',dher_cfg_path,'--phase','dher','--device','cuda:0'], logs/'dher_train.log', env=gpu_env(args.gpu))
        dp, dd=dher_pass(Path(dher_cfg['output_dir']), parent_dir); stages['DHER parent']=Path(dher_cfg['output_dir']); decisions['DHER']=('PASS' if dp else 'FAIL', dd)
        if dp:
            parent_cfg_path=dher_cfg_path; parent_ckpt=Path(dher_cfg['output_dir'])/'best.pt'; parent_dir=Path(dher_cfg['output_dir']); parent_arch=dher_arch; parent_mods='QTR+HCV+DHER' if qp else 'HCV+DHER'
    else:
        decisions['QTR']=('SKIP', {'reason':'HCV failed'}); decisions['DHER']=('SKIP', {'reason':'HCV failed'})
    # Phase D: final total, always execute. Try 640 then fallback 576 on OOM/crash.
    total_res=640
    total_cfg=base_cfg(parent_arch, root/f'final_total_{total_res}', 8, image_size=total_res, init=str(parent_ckpt), lr=5e-6)
    total_cfg['data']['tiny_stratified_sampling']=True
    total_cfg['loss']['weights']['tversky']=0.25; total_cfg['loss']['tversky_alpha_fp']=0.3; total_cfg['loss']['tversky_beta_fn']=0.7
    total_cfg['training'].update({'constant_lr_epochs':2,'min_lr_ratio':0.1,'early_stopping':4,'meaningful_delta':0.0005,'gradient_accumulation_steps':4})
    total_cfg['training']['parameter_groups']=[{'patterns':['content_encoder'],'lr':1e-6},{'patterns':['forensic_stream'],'lr':2e-6},{'patterns':['qtr.'],'lr':3e-6},{'patterns':['hcv2.'],'lr':2e-6},{'patterns':['dher.'],'lr':5e-6},{'patterns':['decoder','mask_head','log_var_head','aux_heads','image_head'],'lr':5e-6}]
    total_cfg_path=cfgdir/'final_total.yaml'; write_yaml(total_cfg_path,total_cfg)
    rc=run([PY,'train_v2_final_pipeline.py','--config',total_cfg_path,'--phase','total','--device','cuda:0'], logs/'final_total_640.log', env=gpu_env(args.gpu), allow_fail=True)
    if rc!=0:
        txt=(logs/'final_total_640.log').read_text(errors='ignore') if (logs/'final_total_640.log').exists() else ''
        if 'out of memory' in txt.lower() or 'cuda' in txt.lower():
            total_res=576; total_cfg['data']['image_size']=576; total_cfg['output_dir']=str(root/'final_total_576'); write_yaml(total_cfg_path,total_cfg)
            run([PY,'train_v2_final_pipeline.py','--config',total_cfg_path,'--phase','total','--device','cuda:0'], logs/'final_total_576.log', env=gpu_env(args.gpu))
        else:
            raise RuntimeError('Final Total failed for non-OOM reason')
    stages['Final Total']=Path(total_cfg['output_dir'])
    # Report
    rows=[{'Model':'B0','Resolution':512,'Innovation modules':'none','Fixed F1':B0_FIXED,'Oracle F1':B0_ORACLE,'Precision':B0_PRECISION,'Recall':B0_RECALL,'IoU':B0_IOU,'AUC':B0_AUC,'<0.5% F1':B0_LT05_F1,'<0.5% Recall':B0_LT05_RECALL,'<1% F1':B0_LT1_F1,'<1% Recall':B0_LT1_RECALL,'Checkpoint':str(B0_CKPT),'Epoch':2}]
    if 'HCV-v2' in stages: rows.append(model_row('HCV-v2',512,'HCV',stages['HCV-v2']))
    if 'QTR+HCV' in stages: rows.append(model_row('QTR+HCV',512,'QTR+HCV',stages['QTR+HCV']))
    if 'DHER parent' in stages: rows.append(model_row('DHER parent',512,'QTR+HCV+DHER' if decisions.get('QTR',('FAIL',))[0]=='PASS' else 'HCV+DHER',stages['DHER parent']))
    rows.append(model_row('Final Total',total_res,parent_mods+' + recipe',stages['Final Total']))
    best=max([r for r in rows if isinstance(r.get('Oracle F1'),float)], key=lambda r:r['Oracle F1'])
    best_recall=max([r for r in rows if isinstance(r.get('Recall'),float)], key=lambda r:r['Recall'])
    final=rows[-1]
    report=[]
    report.append('# Overnight Final Total Report\n')
    report.append(f'Run root: `{root}`\n')
    report.append(md_table(rows,['Model','Resolution','Innovation modules','Fixed F1','Oracle F1','Precision','Recall','IoU','AUC','<0.5% F1','<0.5% Recall','<1% F1','<1% Recall','Epoch','Checkpoint']))
    report.append('\n## Mechanism Decisions\n')
    for k,(status,detail) in decisions.items(): report.append(f'- {k}: **{status}** `{json.dumps(detail, sort_keys=True)}`')
    report.append('\n## Final Answers\n')
    report.append(f'1. Highest val oracle F1: **{best["Oracle F1"]:.6f}** from `{best["Model"]}`.')
    report.append(f'2. Gain over B0: **{best["Oracle F1"]-B0_ORACLE:+.6f}**.')
    report.append(f'3. Highest fixed-threshold Recall: **{best_recall["Recall"] if isinstance(best_recall.get("Recall"),float) else best_recall.get("Recall")}** from `{best_recall["Model"]}`.')
    report.append(f'4. Tiny metrics for Final Total: <0.5% F1 `{final.get("<0.5% F1")}`, <0.5% Recall `{final.get("<0.5% Recall")}`, <1% F1 `{final.get("<1% F1")}`, <1% Recall `{final.get("<1% Recall")}`.')
    report.append(f'5. Final Total innovations used: `{parent_mods}`; performance recipe: high resolution `{total_res}`, tiny-stratified sampling, Tversky recall objective.')
    deleted=['FGTP','HDRP','old CBR','direct-logit correction']
    if decisions.get('HCV mechanism',('FAIL',))[0] != 'PASS': deleted += ['HCV','QTR','DHER']
    elif decisions.get('QTR',('FAIL',))[0] != 'PASS': deleted += ['QTR']
    if decisions.get('DHER',('FAIL',))[0] != 'PASS': deleted += ['DHER']
    report.append(f'6. Innovations/modules deleted or excluded: `{", ".join(deleted)}`.')
    report.append(f'7. Checkpoint worth testing tomorrow: `{final.get("Checkpoint")}`.')
    md='\n'.join(report)+'\n'
    (root/'OVERNIGHT_FINAL_TOTAL_REPORT.md').write_text(md,encoding='utf-8')
    docs=ROOT/'docs'; docs.mkdir(exist_ok=True); (docs/'OVERNIGHT_FINAL_TOTAL_REPORT.md').write_text(md,encoding='utf-8')
    (root/'decisions.json').write_text(json.dumps(decisions,indent=2,sort_keys=True),encoding='utf-8')
    log(f'completed. report={root/"OVERNIGHT_FINAL_TOTAL_REPORT.md"}')

if __name__=='__main__': main()
