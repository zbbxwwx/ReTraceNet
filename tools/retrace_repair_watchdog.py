#!/usr/bin/env python
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path('/group4/zouwenxuan/iml_generalization')
REPO = ROOT / 'repos/ReTraceNet'
OUT = ROOT / 'outputs/retracenet/fix_minimal_core_strong'
DIAG = ROOT / 'diagnostics/retracenet/fix_minimal_core_strong'
LOG = ROOT / 'logs/domain_casia2/retrace_repair_watchdog.log'
PYTHON = ROOT / 'envs/mesorch/bin/python'
CONFIG = REPO / 'configs/casia2_fix_minimal_core_strong_ampoff_resume.yaml'
PATTERN = 'casia2_fix_minimal_core_strong_ampoff_resume'
DIAG.mkdir(parents=True, exist_ok=True)

def log(msg: str) -> None:
    stamp = time.strftime('%F %T')
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f'[{stamp}] {msg}\n')
    print(f'[{stamp}] {msg}', flush=True)

def sh(cmd: str, timeout: int | None = None) -> subprocess.CompletedProcess:
    log('$ ' + cmd)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '1'
    return subprocess.run(cmd, cwd=REPO, shell=True, text=True, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout)

def running() -> bool:
    r = subprocess.run(f"pgrep -af '{PATTERN}'", shell=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return bool(r.stdout.strip())

def run_and_save(name: str, cmd: str, timeout: int | None = None) -> None:
    out_file = DIAG / f'{name}.log'
    try:
        r = sh(cmd, timeout=timeout)
        out_file.write_text(r.stdout, encoding='utf-8')
        log(f'{name} exit={r.returncode} log={out_file}')
    except Exception as e:
        out_file.write_text(repr(e), encoding='utf-8')
        log(f'{name} exception={e!r}')

def write_probe_script() -> Path:
    p = DIAG / 'probability_probe.py'
    p.write_text(r'''
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from retrace.data import build_loader
from retrace.model import build_model
from retrace.utils import load_checkpoint, load_config
from retrace.metrics import ForensicMetrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--output', required=True)
    args=ap.parse_args()
    cfg=load_config(args.config)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck=load_checkpoint(args.checkpoint, device)
    model=build_model(cfg['model']).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    loader_cfg={**cfg['data'], 'batch_size': cfg['evaluation']['batch_size']}
    loader=build_loader(loader_cfg, args.split, shuffle=False)
    metrics=ForensicMetrics(fixed_threshold=float(cfg['evaluation'].get('threshold',0.5)))
    pos_sum=neg_sum=pos_n=neg_n=fg05=fg02=pixels=0.0
    image_count=0
    with torch.inference_mode():
        for batch in loader:
            image=batch['image'].to(device)
            mask=batch['mask'].to(device)
            label=batch['label'].to(device)
            out=model(image)
            prob=out['mask_logits'].sigmoid().detach()
            metrics.update(out, mask, label)
            m=mask.bool()
            pos_sum += float(prob[m].sum()) if m.any() else 0.0
            pos_n += float(m.sum())
            neg_sum += float(prob[~m].sum())
            neg_n += float((~m).sum())
            fg05 += float((prob >= 0.5).sum())
            fg02 += float((prob >= 0.2).sum())
            pixels += prob.numel()
            image_count += image.shape[0]
            if args.max_samples and image_count >= args.max_samples:
                break
    res=metrics.compute()
    res.update({
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'split': args.split,
        'epoch': ck.get('epoch'),
        'best_f1_in_ckpt': ck.get('best_f1'),
        'samples': image_count,
        'pos_prob_mean': pos_sum / max(1.0, pos_n),
        'neg_prob_mean': neg_sum / max(1.0, neg_n),
        'pos_neg_gap': pos_sum / max(1.0, pos_n) - neg_sum / max(1.0, neg_n),
        'fg_ratio_0_5': fg05 / max(1.0, pixels),
        'fg_ratio_0_2': fg02 / max(1.0, pixels),
        'positive_pixels': pos_n,
        'negative_pixels': neg_n,
    })
    Path(args.output).write_text(json.dumps(res, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(res, indent=2, sort_keys=True))
if __name__ == '__main__':
    main()
''', encoding='utf-8')
    return p

def main():
    log('watchdog start')
    stable_done = False
    while running():
        time.sleep(30)
    log('training process finished or not running; starting diagnostics')
    hist = OUT / 'history.csv'
    if hist.exists():
        log('history:\n' + hist.read_text(encoding='utf-8'))
    probe = write_probe_script()
    for ck in ['best.pt', 'last.pt']:
        cp = OUT / ck
        if not cp.exists():
            log(f'missing checkpoint {cp}')
            continue
        for split in ['val', 'test']:
            run_and_save(
                f'evaluate_{ck}_{split}',
                f"{PYTHON} evaluate.py --config {CONFIG} --checkpoint {cp} --split {split} --output {DIAG}/evaluate_{ck}_{split}.json --device cuda",
                timeout=3600,
            )
            run_and_save(
                f'probe_{ck}_{split}',
                f"{PYTHON} {probe} --config {CONFIG} --checkpoint {cp} --split {split} --output {DIAG}/probe_{ck}_{split}.json",
                timeout=3600,
            )
    log('diagnostics complete; inspect JSON/logs before next repair step')

if __name__ == '__main__':
    main()
