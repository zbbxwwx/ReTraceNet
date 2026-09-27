#!/usr/bin/env python
"""Continue FINAL OVERNIGHT PIPELINE from existing HCV-v2 best.

This runner deliberately does not rerun HCV-v2. It resumes the failed overnight
pipeline at the QTR descriptor bug fix point and chains QTR -> DHER -> Final
Total using validation-only decisions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import yaml

from overnight_final_total_runner import (
    B0_AUC,
    B0_CKPT,
    B0_FIXED,
    B0_IOU,
    B0_LT05_F1,
    B0_LT05_RECALL,
    B0_LT1_F1,
    B0_LT1_RECALL,
    B0_ORACLE,
    B0_PRECISION,
    B0_RECALL,
    PY,
    ROOT,
    base_cfg,
    best_row,
    dher_pass,
    gpu_env,
    hcv_pass,
    md_table,
    model_row,
    qtr_pass,
    read_size,
    row_float,
    run,
    write_yaml,
)

RUN_ROOT = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/final_overnight_20260829_143051')
HCV_DIR = RUN_ROOT / 'hcv2_calibrated'
HCV_CFG = RUN_ROOT / 'configs' / 'hcv2_calibrated.yaml'
HCV_CKPT = HCV_DIR / 'best.pt'
QTR_CFG = RUN_ROOT / 'configs' / 'qtr_hcv.yaml'
LOGS = RUN_ROOT / 'logs'
CHECKS = RUN_ROOT / 'checks'
PID_JSON = RUN_ROOT / 'continue_runner_status.json'


def now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(msg: str) -> None:
    print(f'[{now()}] {msg}', flush=True)


def read_history(outdir: Path) -> list[dict[str, str]]:
    p = outdir / 'history.csv'
    if not p.exists():
        return []
    with p.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def stage_done(outdir: Path) -> bool:
    return (outdir / 'best.pt').exists() and bool(read_history(outdir))


def write_status(**kwargs) -> None:
    PID_JSON.write_text(json.dumps({'updated': now(), **kwargs}, indent=2, sort_keys=True), encoding='utf-8')


def key_prefix_check(checkpoint: Path, config_path: Path, allowed_prefix: str) -> dict:
    import sys
    sys.path.insert(0, str(ROOT))
    from retrace.utils import load_config
    from retracenet.final_modules import build_final_model

    cfg = load_config(str(config_path))
    model = build_final_model(cfg['model'])
    state = torch.load(checkpoint, map_location='cpu')
    sd = state.get('model', state)
    own = model.state_dict()
    missing = [k for k, v in own.items() if k not in sd or (k in sd and tuple(sd[k].shape) != tuple(v.shape))]
    unexpected = [k for k in sd if k not in own]
    bad_missing = [k for k in missing if not k.startswith(allowed_prefix)]
    return {
        'missing_count': len(missing),
        'unexpected_count': len(unexpected),
        'bad_missing_count': len(bad_missing),
        'missing_prefixes': sorted({k.split('.')[0] for k in missing}),
        'bad_missing_examples': bad_missing[:20],
    }


def train_stage(name: str, cfg_path: Path, phase: str, gpu: str, log_name: str) -> None:
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    outdir = Path(cfg['output_dir'])
    if stage_done(outdir):
        log(f'{name} already has best.pt/history.csv, skip training: {outdir}')
        return
    run([PY, 'train_v2_final_pipeline.py', '--config', cfg_path, '--phase', phase, '--device', 'cuda:0'], LOGS / log_name, env=gpu_env(gpu))


def make_report(decisions: dict, stages: dict[str, Path], final_res: int, final_mods: str) -> None:
    rows = [{
        'Model': 'B0', 'Resolution': 512, 'Innovation modules': 'none',
        'Fixed F1': B0_FIXED, 'Oracle F1': B0_ORACLE, 'Precision': B0_PRECISION,
        'Recall': B0_RECALL, 'IoU': B0_IOU, 'AUC': B0_AUC,
        '<0.5% F1': B0_LT05_F1, '<0.5% Recall': B0_LT05_RECALL,
        '<1% F1': B0_LT1_F1, '<1% Recall': B0_LT1_RECALL,
        'Checkpoint': str(B0_CKPT), 'Epoch': 2,
    }]
    rows.append(model_row('HCV-v2', 512, 'HCV', HCV_DIR))
    if 'QTR+HCV' in stages:
        rows.append(model_row('QTR+HCV', 512, 'QTR+HCV', stages['QTR+HCV']))
    if 'DHER parent' in stages:
        rows.append(model_row('DHER parent', 512, final_mods if 'DHER' in final_mods else 'candidate DHER', stages['DHER parent']))
    if 'Final Total' in stages:
        rows.append(model_row('Final Total', final_res, final_mods + ' + performance recipe', stages['Final Total']))

    numeric = [r for r in rows if isinstance(r.get('Oracle F1'), float)]
    best = max(numeric, key=lambda r: r['Oracle F1'])
    best_recall = max(numeric, key=lambda r: r['Recall'])
    best_tiny = max(numeric, key=lambda r: r.get('<1% F1', -1.0))
    final = rows[-1]

    deleted = ['FGTP', 'HDRP', 'CBR', 'direct-logit correction']
    if decisions.get('QTR', ('FAIL',))[0] != 'PASS':
        deleted.append('QTR')
    if decisions.get('DHER', ('FAIL',))[0] != 'PASS':
        deleted.append('DHER')

    report = []
    report.append('# Overnight Final Total Report\n')
    report.append(f'Run root: `{RUN_ROOT}`')
    report.append('Selection/evaluation: validation only; CASIA2 unified split; no test run.\n')
    report.append(md_table(rows, ['Model', 'Resolution', 'Innovation modules', 'Fixed F1', 'Oracle F1', 'Precision', 'Recall', 'IoU', 'AUC', '<0.5% F1', '<0.5% Recall', '<1% F1', '<1% Recall', 'Epoch', 'Checkpoint']))
    report.append('\n## Mechanism Decisions')
    for k, (status, detail) in decisions.items():
        report.append(f'- {k}: **{status}** `{json.dumps(detail, sort_keys=True)}`')
    report.append('\n## Required Summary')
    report.append(f'- Best overall val F1: **{best["Oracle F1"]:.6f}** from `{best["Model"]}`.')
    report.append(f'- Best recall: **{best_recall["Recall"]:.6f}** from `{best_recall["Model"]}`.')
    report.append(f'- Best tiny F1 (<1%): **{best_tiny.get("<1% F1", 0.0):.6f}** from `{best_tiny["Model"]}`.')
    report.append(f'- HCV PASS/FAIL: **{decisions.get("HCV-v2", ("PASS", {}))[0]}**.')
    report.append(f'- QTR PASS/FAIL: **{decisions.get("QTR", ("NOT RUN", {}))[0]}**.')
    report.append(f'- DHER PASS/FAIL: **{decisions.get("DHER", ("NOT RUN", {}))[0]}**.')
    report.append(f'- Final Total checkpoint: `{final.get("Checkpoint", "")}`.')
    report.append(f'- Final Total modules: `{final_mods}`; recipe: high-resolution `{final_res}`, tiny-aware sampling, recall-oriented objective.')
    report.append(f'- Excluded/deleted from final path: `{", ".join(deleted)}`.')
    md = '\n'.join(report) + '\n'
    (RUN_ROOT / 'OVERNIGHT_FINAL_TOTAL_REPORT.md').write_text(md, encoding='utf-8')
    docs = ROOT / 'docs'
    docs.mkdir(exist_ok=True)
    (docs / 'OVERNIGHT_FINAL_TOTAL_REPORT.md').write_text(md, encoding='utf-8')
    (RUN_ROOT / 'decisions_continue.json').write_text(json.dumps(decisions, indent=2, sort_keys=True), encoding='utf-8')
    log(f'report written: {RUN_ROOT / "OVERNIGHT_FINAL_TOTAL_REPORT.md"}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='1')
    ap.add_argument('--root', default=str(RUN_ROOT))
    args = ap.parse_args()
    if Path(args.root) != RUN_ROOT:
        raise SystemExit(f'this continuation runner is pinned to {RUN_ROOT}, got {args.root}')

    LOGS.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)
    write_status(stage='starting', root=str(RUN_ROOT), gpu=args.gpu)
    (Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2') / 'CURRENT_FINAL_OVERNIGHT.txt').write_text(str(RUN_ROOT), encoding='utf-8')

    if not HCV_CKPT.exists():
        raise FileNotFoundError(f'HCV-v2 best checkpoint missing: {HCV_CKPT}')
    if not HCV_CFG.exists():
        raise FileNotFoundError(f'HCV-v2 config missing: {HCV_CFG}')

    hcv_ok, hcv_detail = hcv_pass(HCV_DIR)
    decisions = {'HCV-v2': ('PASS' if hcv_ok else 'FAIL', hcv_detail)}
    if not hcv_ok:
        raise RuntimeError(f'HCV-v2 is expected PASS but judge failed: {hcv_detail}')

    run([PY, '-m', 'py_compile', 'retracenet/final_modules.py', 'train_v2_final_pipeline.py', 'tools/check_final_pipeline.py', 'overnight_final_total_continue_from_qtr.py'], LOGS / 'continue_py_compile.log')

    qtr_dir = RUN_ROOT / 'qtr_hcv'
    qtr_cfg = base_cfg('retracenet_b2_qtr_hcv2', qtr_dir, 4, lr=2e-5, init=str(HCV_CKPT))
    qtr_cfg['training']['parameter_groups'] = [
        {'patterns': ['qtr.'], 'lr': 2e-5},
        {'patterns': ['hcv2.'], 'lr': 2e-6},
    ]
    write_yaml(QTR_CFG, qtr_cfg)
    key_check = key_prefix_check(HCV_CKPT, QTR_CFG, 'qtr.')
    if key_check['bad_missing_count'] or key_check['unexpected_count']:
        raise RuntimeError(f'QTR key check failed: {key_check}')
    run([PY, 'tools/check_final_pipeline.py', '--parent-config', HCV_CFG, '--child-config', QTR_CFG, '--checkpoint', HCV_CKPT, '--phase', 'qtr_hcv', '--device', 'cuda:0', '--output', CHECKS / 'qtr_preflight_after_fix.json'], LOGS / 'qtr_preflight_after_fix.log', env=gpu_env(args.gpu))

    write_status(stage='qtr_training', root=str(RUN_ROOT), gpu=args.gpu, qtr_key_check=key_check)
    train_stage('QTR+HCV', QTR_CFG, 'qtr_hcv', args.gpu, 'qtr_train_continue.log')
    qtr_ok, qtr_detail = qtr_pass(qtr_dir, HCV_DIR)
    decisions['QTR'] = ('PASS' if qtr_ok else 'FAIL', qtr_detail)

    parent_cfg = QTR_CFG if qtr_ok else HCV_CFG
    parent_ckpt = (qtr_dir / 'best.pt') if qtr_ok else HCV_CKPT
    parent_dir = qtr_dir if qtr_ok else HCV_DIR
    parent_arch = 'retracenet_b2_qtr_hcv2' if qtr_ok else 'retracenet_b1_hcv2'
    parent_mods = 'QTR+HCV' if qtr_ok else 'HCV'
    stages = {'QTR+HCV': qtr_dir}

    dher_dir = RUN_ROOT / ('dher_qtr_hcv' if qtr_ok else 'dher_hcv')
    dher_arch = 'retracenet_b3_qtr_hcv2_dher' if qtr_ok else 'retracenet_b3_hcv2_dher'
    dher_cfg_path = RUN_ROOT / 'configs' / ('dher_qtr_hcv.yaml' if qtr_ok else 'dher_hcv.yaml')
    dher_cfg = base_cfg(dher_arch, dher_dir, 6, init=str(parent_ckpt), lr=2e-5)
    dher_cfg['training'].update({'constant_lr_epochs': 2, 'min_lr_ratio': 0.1, 'early_stopping': 7})
    dher_cfg['training']['parameter_groups'] = [
        {'patterns': ['dher.'], 'lr': 2e-5},
        {'patterns': ['decoder.stage1', 'mask_head'], 'lr': 5e-6},
        {'patterns': ['hcv2.', 'qtr.'], 'lr': 2e-6},
    ]
    write_yaml(dher_cfg_path, dher_cfg)
    run([PY, 'tools/check_final_pipeline.py', '--parent-config', parent_cfg, '--child-config', dher_cfg_path, '--checkpoint', parent_ckpt, '--phase', 'dher', '--device', 'cuda:0', '--output', CHECKS / 'dher_preflight.json'], LOGS / 'dher_preflight.log', env=gpu_env(args.gpu))

    write_status(stage='dher_training', root=str(RUN_ROOT), gpu=args.gpu, qtr=decisions['QTR'])
    train_stage('DHER', dher_cfg_path, 'dher', args.gpu, 'dher_train_continue.log')
    dher_ok, dher_detail = dher_pass(dher_dir, parent_dir)
    decisions['DHER'] = ('PASS' if dher_ok else 'FAIL', dher_detail)
    stages['DHER parent'] = dher_dir
    if dher_ok:
        parent_cfg = dher_cfg_path
        parent_ckpt = dher_dir / 'best.pt'
        parent_dir = dher_dir
        parent_arch = dher_arch
        parent_mods = ('QTR+HCV+DHER' if qtr_ok else 'HCV+DHER')

    final_res = 640
    final_dir = RUN_ROOT / 'final_total_640'
    final_cfg_path = RUN_ROOT / 'configs' / 'final_total.yaml'
    total_cfg = base_cfg(parent_arch, final_dir, 8, image_size=final_res, init=str(parent_ckpt), lr=5e-6)
    total_cfg['data']['tiny_stratified_sampling'] = True
    total_cfg['loss']['weights']['tversky'] = 0.25
    total_cfg['loss']['tversky_alpha_fp'] = 0.3
    total_cfg['loss']['tversky_beta_fn'] = 0.7
    total_cfg['training'].update({'constant_lr_epochs': 2, 'min_lr_ratio': 0.1, 'early_stopping': 4, 'meaningful_delta': 0.0005, 'gradient_accumulation_steps': 4})
    total_cfg['training']['parameter_groups'] = [
        {'patterns': ['content_encoder'], 'lr': 1e-6},
        {'patterns': ['forensic_stream'], 'lr': 2e-6},
        {'patterns': ['qtr.'], 'lr': 3e-6},
        {'patterns': ['hcv2.'], 'lr': 2e-6},
        {'patterns': ['dher.'], 'lr': 5e-6},
        {'patterns': ['decoder', 'mask_head', 'log_var_head', 'aux_heads', 'image_head'], 'lr': 5e-6},
    ]
    write_yaml(final_cfg_path, total_cfg)
    write_status(stage='final_total_640', root=str(RUN_ROOT), gpu=args.gpu, qtr=decisions['QTR'], dher=decisions['DHER'])
    rc = run([PY, 'train_v2_final_pipeline.py', '--config', final_cfg_path, '--phase', 'total', '--device', 'cuda:0'], LOGS / 'final_total_640_continue.log', env=gpu_env(args.gpu), allow_fail=True)
    if rc != 0:
        txt = (LOGS / 'final_total_640_continue.log').read_text(errors='ignore') if (LOGS / 'final_total_640_continue.log').exists() else ''
        if 'out of memory' not in txt.lower() and 'cuda' not in txt.lower():
            raise RuntimeError('Final Total 640 failed for a non-CUDA/OOM reason')
        final_res = 576
        final_dir = RUN_ROOT / 'final_total_576'
        total_cfg['data']['image_size'] = 576
        total_cfg['output_dir'] = str(final_dir)
        write_yaml(final_cfg_path, total_cfg)
        write_status(stage='final_total_576_fallback', root=str(RUN_ROOT), gpu=args.gpu)
        train_stage('Final Total 576', final_cfg_path, 'total', args.gpu, 'final_total_576_continue.log')
    stages['Final Total'] = Path(total_cfg['output_dir'])

    make_report(decisions, stages, final_res, parent_mods)
    write_status(stage='complete', root=str(RUN_ROOT), gpu=args.gpu, qtr=decisions['QTR'], dher=decisions['DHER'], report=str(RUN_ROOT / 'OVERNIGHT_FINAL_TOTAL_REPORT.md'))


if __name__ == '__main__':
    main()
