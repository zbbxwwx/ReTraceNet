"""Sections 1-2: recalibrate E24 only on CASIA2 val and compare external data.

New files are written below the requested final summary directory. Historical
metrics are preserved. This stage never starts a training job.
"""
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch
import run_cross_dataset_after_chain as cross
import run_fair_training_v1 as fair

BASE = Path('/group4/zouwenxuan/iml_generalization')
OUT = BASE / 'outputs/retracenet_v2/final_experiment_summary_20260912'
OLD = BASE / 'outputs/retracenet_v2/external_four_e24_same_protocol_20260912'
FAIR = BASE / 'outputs/retracenet_v2/fair_training_v1_full_20260911'
EXTERNAL = BASE / 'outputs/retracenet_v2/external_four_after_chain_20260911'
CHECKPOINT = BASE / 'outputs/retracenet_v2/e24_dher_calibrated_writeback_20260910/best.pt'
NAMES = ['CASIA1', 'Columbia']  # fast phase; NIST16/IMD2020 deferred


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    stage = OUT / '01_e24_recalibration'
    stage.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (stage / 'execution.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (stage / 'decision.json').exists():
        print('stage already complete; preserving results', flush=True)
        return
    state = {'status': 'waiting_for_e24_external', 'training_run': False,
             'external_used_for_threshold': False}
    atomic_json(stage / 'state.json', state)
    try:
        # Existing run must finish before new GPU work. A stopped failed run is
        # surfaced for engineering repair, not silently interpreted as a result.
        cross.wait_completed(OLD / 'summary.yaml', 30, 'E24 existing external')
        files = [Path(__file__), Path(cross.__file__), Path(fair.__file__),
                 Path(fair.FairDataset.__module__.replace('.', '/'))]
        files = [p for p in files if p.is_file()]
        snapshot = stage / 'code_snapshot'
        snapshot.mkdir(exist_ok=True)
        for p in files:
            shutil.copy2(p, snapshot / p.name)
        atomic_json(stage / 'provenance.json', {
            'command': sys.argv, 'pid': os.getpid(), 'checkpoint': str(CHECKPOINT),
            'checkpoint_sha256': sha(CHECKPOINT), 'files': {str(p): sha(p) for p in files},
            'manifest_sha256': sha(FAIR / 'clean_manifest.csv')})
        ck = torch.load(CHECKPOINT, map_location='cpu')
        cfg = ck['config']
        cross.save_yaml(stage / 'config.yaml', cfg)
        model = fair.e.build_final_model(cfg['model']).cuda().eval()
        model.load_state_dict(ck['model'], strict=True)
        model.disable_dct_evidence = False
        del ck
        state['status'] = 'calibrating_casia2_val'
        atomic_json(stage / 'state.json', state)
        calibration_path = stage / 'casia2_val.yaml'
        if calibration_path.exists():
            calibration = cross.load_yaml(calibration_path)
        else:
            records = fair.e.read_manifest(FAIR / 'clean_manifest.csv', cfg['data']['root'], 'val')
            ds = fair.FairDataset(records, FAIR / 'jpeg_cache', False)
            calibration = fair.evaluate(model, ds, cfg, stage, 'casia2_val')
        threshold = calibration['metrics']['fake']['oracle']['threshold']
        state.update(status='external_locked_evaluation', threshold=threshold,
                     recalibrated_val_F1=calibration['metrics']['fake']['oracle']['f1'])
        atomic_json(stage / 'state.json', state)
        rows = []
        for name in NAMES:
            dest = stage / (name + '.yaml')
            if dest.exists():
                result = cross.load_yaml(dest)
                assert result['locked']['threshold'] == threshold
            else:
                records = cross.parse_index(BASE / 'paper_compare/mvss_five_indices' / (name + '.txt'), name)
                result = cross.evaluate_dataset(model, records, stage / 'jpeg_cache' / name,
                                                threshold, name, stage)
            old = cross.load_yaml(EXTERNAL / (name + '.yaml'))
            e24m, fm = result['diagnostic_only']['fake'], old['diagnostic_only']['fake']
            row = {'dataset': name, 'e24_locked_f1': result['locked']['f1'],
                   'fair_locked_f1': old['locked']['f1'],
                   'e24_fixed05_f1': e24m['fixed']['f1'], 'fair_fixed05_f1': fm['fixed']['f1'],
                   'e24_oracle_f1': e24m['oracle']['f1'], 'fair_oracle_f1': fm['oracle']['f1'],
                   'e24_auc': e24m['auc'], 'fair_auc': fm['auc'],
                   'e24_precision': result['locked']['precision'], 'e24_recall': result['locked']['recall'],
                   'e24_iou': result['locked']['iou'],
                   'fair_precision': old['locked']['precision'], 'fair_recall': old['locked']['recall'],
                   'fair_iou': old['locked']['iou'],
                   'delta_fair_minus_e24': old['locked']['f1'] - result['locked']['f1']}
            rows.append(row)
            atomic_json(stage / 'partial_comparison.json', rows)
        # Operational definitions are explicit; no per-dataset threshold tuning.
        advantage = [r['dataset'] for r in rows
                     if (r['e24_auc'] >= r['fair_auc'] + .02 or
                         r['e24_oracle_f1'] >= r['fair_oracle_f1'] + .02)
                     and not (r['fair_auc'] >= r['e24_auc'] + .02 or
                              r['fair_oracle_f1'] >= r['e24_oracle_f1'] + .02)]
        fair_opposite = [r['dataset'] for r in rows
                         if r['fair_locked_f1'] >= r['e24_locked_f1'] + .02
                         and r['fair_oracle_f1'] >= r['e24_oracle_f1'] + .02]
        both_low = [r['dataset'] for r in rows
                    if max(r['e24_oracle_f1'], r['fair_oracle_f1']) < .5]
        if len(advantage) >= 3 and len(fair_opposite) < 2:
            route, reason = 'G-A', 'E24 consistent >=.02 AUC/oracle advantage on >=3 datasets without stable opposite Fair advantage'
        elif len(both_low) >= 2 and len(advantage) < 2 and len(fair_opposite) < 2:
            route, reason = 'G-B', 'Both oracle F1 below .5 on >=2 datasets, no clear consistent E24 advantage'
        else:
            route, reason = 'G-C', 'Mixed evidence: metadata diagnosis required before choosing a pilot'
        decision = {'status': 'completed', 'route': route, 'reason': reason,
                    'evidence': rows, 'consistent_e24_advantage': advantage,
                    'fair_opposite_advantage': fair_opposite, 'both_oracle_below_0p5': both_low,
                    'rejected_routes': ['automatic gate/region/decoder/boundary/prototype'],
                    'next': 'metadata diagnosis' if route == 'G-C' else 'implement and audit matched pilot',
                    'threshold_source': 'CASIA2 validation under current unified JPEG native-grid pipeline',
                    'e24_new_locked_threshold': threshold,
                    'recalibrated_casia2_val_F1': calibration['metrics']['fake']['oracle']['f1'],
                    'causal_limit': 'Training recipe and input distribution effects are not isolated; DCT-off is an inference intervention.'}
        atomic_json(OUT / 'generalization_route_decision.json', decision)
        cross.save_yaml(stage / 'final_summary.yaml', decision)
        # These are additional files requested by the user; old summary.yaml remains intact.
        cross.save_yaml(OLD / 'final_summary.yaml', decision)
        with (OUT / 'e24_vs_fair_external.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        md = '# E24 vs Fair: external cross-dataset evaluation\n\n'
        md += f'E24 threshold recalibrated on CASIA2 validation: {threshold}. Route: {route}.\n\n'
        md += '| Dataset | E24 locked | Fair locked | E24 oracle | Fair oracle | E24 AUC | Fair AUC |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for r in rows:
            md += '| ' + r['dataset'] + ' | ' + ' | '.join(f'{r[k]:.6f}' for k in
                ['e24_locked_f1','fair_locked_f1','e24_oracle_f1','fair_oracle_f1','e24_auc','fair_auc']) + ' |\n'
        md += '\n' + reason + '\n\nExternal oracle is diagnostic only. CASIA2 val values are not test values.\n'
        (OUT / '01_e24_vs_fair_external.md').write_text(md)
        (OLD / 'final_summary.md').write_text(md)
        (OUT / '02_generalization_route_decision.md').write_text('# Route decision\n\n' + reason + '\n\nNext: ' + decision['next'])
        atomic_json(stage / 'decision.json', decision)
        state.update(status='completed', route=route)
        atomic_json(stage / 'state.json', state)
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        atomic_json(stage / 'state.json', state)
        raise


if __name__ == '__main__':
    main()
