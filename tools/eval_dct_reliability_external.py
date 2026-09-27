"""Evaluate the custom DCT-reliability checkpoint on four external datasets."""
import argparse, copy, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tools'))
import run_imd_mit_full_v3  # noqa: F401
import run_cross_dataset_after_chain as cross
from run_casia2_dct_reliability_pilot import wrap, reset_diag, diag
from takeover_recalibrate import BASE, atomic_json, sha

def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.checkpoint, map_location='cpu')
    cfg = copy.deepcopy(ck['config'])
    threshold = float(ck['selection_threshold'])
    parent = torch.load(a.parent, map_location='cpu')
    image_threshold = float(parent.get('image_selection_threshold', threshold))
    model = cross.e.build_final_model(cfg['model']).cuda().eval()
    gates = wrap(model)
    model.load_state_dict(ck['model'], strict=True)
    rows = {}
    for name in ['CASIA1', 'Columbia', 'IMD2020', 'NIST16']:
        reset_diag(gates)
        records = cross.parse_index(BASE / 'paper_compare/mvss_five_indices' / (name + '.txt'), name)
        r = cross.evaluate_dataset(model, records, out / 'jpeg_cache' / name, threshold,
                                   image_threshold, name, out,
                                   evaluation_mode='cross_domain_locked',
                                   threshold_source='CASIA2_val')
        r['dct_reliability_gate_actual_forward_mean'] = diag(gates)
        cross.save_yaml(out / (name + '.yaml'), r)
        rows[name] = {
            'pixel_localization_fake_only': r['pixel_localization_fake_only'],
            'authentic_pixel_fpr': r['authentic_pixel_fpr'],
            'image_level_all': r['image_level_all'],
            'gate_mean': r['dct_reliability_gate_actual_forward_mean'],
        }
        atomic_json(out / 'partial.json', rows)
    result = {'status':'completed', 'checkpoint':a.checkpoint,
              'checkpoint_sha256':sha(a.checkpoint), 'threshold':threshold,
              'threshold_source':'CASIA2_val', 'image_threshold':image_threshold,
              'evaluation_mode':'cross_domain_locked', 'datasets':rows}
    cross.save_yaml(out / 'final_summary.yaml', result)
    atomic_json(out / 'summary.json', result)

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True)
    p.add_argument('--parent',required=True); p.add_argument('--out',required=True)
    main(p.parse_args())
