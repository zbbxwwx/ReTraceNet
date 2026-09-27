#!/usr/bin/env python
from __future__ import annotations
import csv, json, os, shutil, subprocess, sys, time
from pathlib import Path
import yaml

ROOT = Path('/group4/zouwenxuan/iml_generalization/repos/ReTraceNet')
PYTHON = '/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python'
BASE_CONFIG = ROOT / 'configs/casia2_retracenet_score_sprint_c.yaml'
BASE_CKPT = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/score_sprint_c_20260831/best.pt')
BASELINE_F1 = 0.749307492101594
BASELINE_AUC = 0.8923175983404259
RUN_ROOT = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/score_first_single_gpu_queue_20260831')
REPORT = RUN_ROOT / 'SCORE_FIRST_SINGLE_GPU_QUEUE_REPORT.md'
GPU = os.environ.get('SCORE_QUEUE_GPU', '0')
BUCKETS = ['lt_0p5','0p5_1','1_5','5_20','gte_20']


def log(msg: str) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with (RUN_ROOT / 'runner.log').open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def load_base() -> dict:
    return yaml.safe_load(BASE_CONFIG.read_text(encoding='utf-8'))


def scale_lrs(cfg: dict, factor: float = 0.2) -> None:
    tr = cfg['training']
    tr['learning_rate'] = float(tr.get('learning_rate', 1e-5)) * factor
    for spec in tr.get('parameter_groups', []):
        spec['lr'] = float(spec['lr']) * factor


def add_to_group(cfg: dict, base_pattern: str, new_pattern: str) -> None:
    for spec in cfg['training'].get('parameter_groups', []):
        pats = spec.get('patterns', [])
        if base_pattern in pats and new_pattern not in pats:
            pats.append(new_pattern)
            return
    cfg['training'].setdefault('parameter_groups', []).append({'patterns':[new_pattern], 'lr': float(cfg['training'].get('learning_rate', 1e-5)), 'weight_decay': float(cfg['training'].get('weight_decay',0.01))})


def make_cfg(run_name: str, segment: str, epochs: int, init_ckpt: Path, architecture: str, outdir: Path, tweak: str = '') -> Path:
    cfg = load_base()
    cfg['experiment'] = f'{run_name}_{segment}'
    cfg['output_dir'] = str(outdir)
    cfg['model']['architecture'] = architecture
    cfg['training']['init_checkpoint'] = str(init_ckpt)
    cfg['training']['epochs'] = int(epochs)
    cfg['training']['constant_lr_epochs'] = min(1, int(epochs))
    cfg['training']['early_stopping'] = int(epochs) + 5
    cfg['training']['strict_checkpoint'] = True
    cfg['training']['allowed_missing_prefixes'] = []
    scale_lrs(cfg, 0.2)
    if tweak == 'recall_loss':
        cfg['loss']['positive_weight_clip'] = [1.3, 1.3]
    if 'context_qtr' in architecture:
        cfg['training']['allowed_missing_prefixes'].append('content_context.')
        add_to_group(cfg, 'decoder', 'content_context')
    if 'mask_refine' in architecture:
        cfg['training']['allowed_missing_prefixes'].append('mask_refine.')
        add_to_group(cfg, 'decoder', 'mask_refine')
    outdir.mkdir(parents=True, exist_ok=True)
    cfg_dir = RUN_ROOT / 'generated_configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f'{run_name}_{segment}.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return path


def run_cmd(cmd: list[str], log_path: Path, cwd: Path = ROOT) -> None:
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = GPU
    with log_path.open('a', encoding='utf-8') as f:
        f.write('\n$ ' + ' '.join(cmd) + '\n')
        f.flush()
        p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f'command failed ({p.returncode}): {cmd}; see {log_path}')


def train_segment(run_name: str, segment: str, epochs: int, init_ckpt: Path, architecture: str, tweak: str = '', manifest: str | None = None) -> dict:
    outdir = RUN_ROOT / run_name / segment
    cfg_path = make_cfg(run_name, segment, epochs, init_ckpt, architecture, outdir, tweak)
    if manifest:
        cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
        cfg['data']['manifest'] = manifest
        cfg['data']['area_stratified_sampling'] = False
        cfg['data']['tiny_stratified_sampling'] = False
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    log(f'{run_name}/{segment}: train epochs={epochs} init={init_ckpt}')
    run_cmd([PYTHON, 'train_v2_final_pipeline.py', '--config', str(cfg_path), '--phase', 'score_sprint_c', '--device', 'cuda'], outdir / 'train.log')
    ckpt = outdir / 'best.pt'
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    eval_json = outdir / 'val_tampered_pooled.json'
    log(f'{run_name}/{segment}: evaluate val tampered pooled')
    run_cmd([PYTHON, 'tools/eval_tampered_pooled.py', '--config', str(cfg_path), '--checkpoint', str(ckpt), '--state', 'model', '--split', 'val', '--device', 'cuda', '--out', str(eval_json)], outdir / 'eval_val.log')
    res = json.loads(eval_json.read_text(encoding='utf-8'))
    res['config'] = str(cfg_path)
    res['checkpoint'] = str(ckpt)
    res['outdir'] = str(outdir)
    return res


def metric_line(res: dict) -> str:
    m = res['tampered_pooled']
    fixed = res.get('tampered_fixed_0p5', {})
    return f"F1best={m['f1_best']:.6f} AUC={m['auc']:.6f} P={m['precision']:.6f} R={m['recall']:.6f} IoU={m['iou']:.6f} th={m['threshold']:.2f} F1@0.5={fixed.get('f1', float('nan')):.6f}"


def decision_after_initial(name: str, res: dict) -> str:
    f1 = float(res['tampered_pooled']['f1_best']); auc = float(res['tampered_pooled']['auc'])
    if f1 >= 0.785:
        return 'near_target_continue'
    if f1 >= 0.770 and auc >= 0.895:
        return 'strong_continue'
    if f1 >= 0.760:
        return 'valuable_continue'
    if name == 'B' and auc >= 0.900:
        return 'auc_continue'
    if f1 < 0.755:
        return 'stop_invalid'
    if f1 <= BASELINE_F1 + 0.003 and auc <= BASELINE_AUC + (0.001 if name == 'A' else 0.0):
        return 'stop_no_early_gain'
    return 'stop_marginal'


def append_report(rows: list[dict], final: bool = False) -> None:
    lines = ['# Score-first single GPU queue', '', f'Baseline E1 tampered pooled F1-best = {BASELINE_F1:.6f}, AUC = {BASELINE_AUC:.6f}', '']
    lines += ['| Run | Segment | Val F1-best | Val AUC | P | R | IoU | Threshold | Decision | Checkpoint |', '|---|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        m=r['result']['tampered_pooled']
        lines.append(f"| {r['run']} | {r['segment']} | {m['f1_best']:.6f} | {m['auc']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['iou']:.6f} | {m['threshold']:.2f} | {r['decision']} | `{r['result']['checkpoint']}` |")
    if final:
        lines.append('\n## Final Test Evaluations\n')
        lines.append('| Run | Test F1-best | Test AUC | P | R | IoU | Threshold | Authentic FPR | |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---|')
        for r in rows:
            t=r.get('test_result')
            if not t: continue
            m=t['tampered_pooled']
            lines.append(f"| {r['run']}/{r['segment']} | {m['f1_best']:.6f} | {m['auc']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['iou']:.6f} | {m['threshold']:.2f} | {t['authentic_fpr_0p5']:.6f} | `{t['checkpoint']}` |")
    REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def build_hard_manifest() -> str:
    # Keep this simple and auditable: use an existing helper generated on demand.
    out = RUN_ROOT / 'hard_train_manifest.csv'
    if out.exists(): return str(out)
    log('D: build hard-example train manifest from E1 fixed validation-style train inference')
    script = RUN_ROOT / 'build_hard_manifest.py'
    script.write_text(BUILD_HARD_SCRIPT, encoding='utf-8')
    run_cmd([PYTHON, str(script), '--config', str(BASE_CONFIG), '--checkpoint', str(BASE_CKPT), '--out', str(out)], RUN_ROOT / 'build_hard_manifest.log')
    return str(out)

BUILD_HARD_SCRIPT = r'''
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/group4/zouwenxuan/iml_generalization/repos/ReTraceNet')
from retrace.data import build_loader
from retrace.utils import load_config, choose_device
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import load_initial_weights_repaired, forward_from_batch

def f1_best(s,y):
    best=0.0
    for t in np.linspace(0,1,51):
        p=s>=t; tp=float(np.logical_and(p,y).sum()); fp=float(np.logical_and(p,~y).sum()); fn=float(np.logical_and(~p,y).sum())
        pr=tp/max(1,tp+fp); re=tp/max(1,tp+fn); f=2*pr*re/max(1e-12,pr+re)
        best=max(best,f)
    return best
ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
cfg=load_config(args.config); cfg['data']['batch_size']=1; cfg['data']['num_workers']=4
device=choose_device('cuda'); model=build_final_model(cfg['model']).to(device).eval(); load_initial_weights_repaired(model,args.checkpoint,device)
loader=build_loader({**cfg['data'], 'batch_size':1, 'positive_crop_probability':0.0, 'jpeg_probability':0.0, 'blur_probability':0.0, 'color_jitter':0.0}, 'train', shuffle=False)
rows=[]
with torch.no_grad():
    for b in loader:
        if int(b['label'].item()) != 1: continue
        y=b['mask'].numpy()[0,0] > 0.5
        if not y.any(): continue
        p=forward_from_batch(model,b,device)['mask_logits'].sigmoid().detach().cpu().numpy()[0,0]
        rows.append((float(f1_best(p.reshape(-1), y.reshape(-1))), b['path'][0], b['mask_path'][0] if 'mask_path' in b else ''))
rows.sort(key=lambda x:x[0])
hard=set(x[1] for x in rows[:max(1,int(len(rows)*0.30))])
manifest=Path(cfg['data']['manifest'])
orig=list(csv.DictReader(manifest.open(encoding='utf-8-sig')))
fields=orig[0].keys()
train=[r for r in orig if r.get('split')=='train']; others=[r for r in orig if r.get('split')!='train']
hard_rows=[r for r in train if r['image'] in hard]
# 8 copies gives roughly half hard tampered exposure for the observed CASIA2 train composition.
aug=train + hard_rows*8 + others
with Path(args.out).open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(aug)
print({'train_original':len(train),'hard_count':len(hard_rows),'total_rows':len(aug),'out':args.out})
'''


def final_test(rows: list[dict]) -> None:
    seen=set()
    for r in rows:
        ckpt=r['result']['checkpoint']
        if ckpt in seen: continue
        seen.add(ckpt)
        outdir=Path(r['result']['outdir'])
        out=outdir/'test_tampered_pooled.json'
        log(f"{r['run']}/{r['segment']}: evaluate test tampered pooled")
        run_cmd([PYTHON, 'tools/eval_tampered_pooled.py', '--config', r['result']['config'], '--checkpoint', ckpt, '--state', 'model', '--split', 'test', '--device', 'cuda', '--out', str(out)], outdir/'eval_test.log')
        r['test_result']=json.loads(out.read_text(encoding='utf-8'))


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows=[]
    arch_base = load_base()['model']['architecture']
    runs = [
        {'name':'A', 'label':'recall_balanced', 'arch':arch_base, 'tweak':'recall_loss', 'initial':4, 'cont':8},
        {'name':'B', 'label':'context_strengthening', 'arch':arch_base + '_context_qtr', 'tweak':'', 'initial':4, 'cont':8},
        {'name':'C', 'label':'mask_refinement', 'arch':arch_base + '_mask_refine', 'tweak':'', 'initial':4, 'cont':6},
        {'name':'D', 'label':'hard_example', 'arch':arch_base, 'tweak':'', 'initial':5, 'cont':3},
    ]
    for spec in runs:
        manifest=None
        if spec['name']=='D':
            manifest=build_hard_manifest()
        res=train_segment(f"{spec['name']}_{spec['label']}", 'initial', spec['initial'], BASE_CKPT, spec['arch'], spec['tweak'], manifest)
        decision=decision_after_initial(spec['name'], res)
        rows.append({'run':spec['name'], 'segment':'initial', 'decision':decision, 'result':res})
        append_report(rows)
        log(f"{spec['name']} initial: {metric_line(res)} decision={decision}")
        if decision.startswith('stop'):
            continue
        cont_res=train_segment(f"{spec['name']}_{spec['label']}", 'continue', spec['cont'], Path(res['checkpoint']), spec['arch'], spec['tweak'], manifest)
        cont_dec=decision_after_initial(spec['name'], cont_res)
        rows.append({'run':spec['name'], 'segment':'continue', 'decision':cont_dec, 'result':cont_res})
        append_report(rows)
        log(f"{spec['name']} continue: {metric_line(cont_res)} decision={cont_dec}")
        if float(cont_res['tampered_pooled']['f1_best']) >= 0.785:
            log('Near-target branch found; stopping queue after continuation.')
            break
    final_test(rows)
    append_report(rows, final=True)
    log(f'DONE report={REPORT}')

if __name__=='__main__':
    main()
