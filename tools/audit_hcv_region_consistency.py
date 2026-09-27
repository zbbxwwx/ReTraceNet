"""Preflight only. Synthetic shape/wiring checks do not measure localization quality."""
import argparse
import copy
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retracenet.final_modules import build_final_model
from retracenet.hcv_region_consistency import attach_region_consistency


def norm(t):
    return float(t.float().square().sum().sqrt()) if t is not None else 0.


def audit(path):
    torch.manual_seed(3407)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    assert checkpoint.get('eval_variant', 'raw') == 'raw'
    config = copy.deepcopy(checkpoint['config']['model'])
    assert not config.get('enable_splicing_region_affinity', False), 'Use E5/E21, not E23'
    assert not config.get('enable_region_consistency', False)
    model = build_final_model(config).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    image = torch.rand(1, 3, 128, 128)
    kwargs = {}
    if config.get('enable_dct_parallel', False):
        # Shape/wiring probe only; these are not JPEG evidence for accuracy evaluation.
        kwargs = dict(dct=torch.zeros(1, 21, 128, 128),
                      qtable=torch.ones(1, 1, 8, 8), dct_available=torch.ones(1, 1))
    # Use one real TRAIN splicing sample for the final preflight, never test data.
    from retrace.data import read_manifest, ManipulationDataset
    from PIL import Image
    import numpy as np
    data = checkpoint['config']['data']
    records = read_manifest(data['manifest'], data['root'], 'train')
    chosen = None
    for record in records:
        if 'tp_s' not in str(record.image).lower() or record.mask is None:
            continue
        with Image.open(record.mask) as mask:
            area = float((np.asarray(mask.convert('L')) > 127).mean())
        if area >= .2 and (not kwargs or record.image.suffix.lower() in ('.jpg', '.jpeg')):
            chosen = record
            break
    assert chosen is not None, 'No train Tp_S >=20% sample found'
    dataset = ManipulationDataset([chosen], image_size=int(data.get('image_size',640)), training=False,
                                  dct_forensic=bool(kwargs), dct_source_mode='original_jpeg')
    sample = dataset[0]
    image = sample['image'][None]
    if kwargs:
        kwargs = {k:sample[k][None] for k in ('dct','qtable','dct_available')}
        assert float(kwargs['dct_available'].sum()) > 0
    with torch.no_grad():
        baseline = model(image, **kwargs)
    attach_region_consistency(model.hcv2)
    model.eval()
    loaded = model.load_state_dict(checkpoint['model'], strict=False)
    assert not loaded.unexpected_keys
    assert all(k.startswith('hcv2.region_consistency.') for k in loaded.missing_keys)
    initial = {k: p.detach().clone() for k,p in model.hcv2.region_consistency.named_parameters()}
    with torch.no_grad():
        current = model(image, **kwargs)
    diff = float((current['mask_logits']-baseline['mask_logits']).abs().max())
    assert diff == 0., diff
    assert current['hcv2'].keys() == baseline['hcv2'].keys()
    shapes = {}
    for key, values in baseline['hcv2'].items():
        if isinstance(values, list):
            shapes[key] = [list(t.shape) for t in values]
            assert shapes[key] == [list(t.shape) for t in current['hcv2'][key]]
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith('hcv2.region_consistency.'))
    parameters = dict(model.hcv2.region_consistency.named_parameters())
    opt = torch.optim.Adam(list(parameters.values()), lr=1e-3)
    target = sample['mask'][None].float()
    steps = []
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        out = model(image, **kwargs)
        loss = F.binary_cross_entropy_with_logits(out['mask_logits'], target)
        loss.backward()
        grads = {k:norm(p.grad) for k,p in parameters.items()}
        assert all(torch.isfinite(p.grad).all() for p in parameters.values() if p.grad is not None)
        for i in range(2):
            assert grads[f'{i}.scale'] > 0, (i, grads)
            body = sum(v for k,v in grads.items() if k.startswith(f'{i}.refinement.'))
            assert (body == 0 if step == 0 else body > 0), (step, i, body)
        before = {k:p.detach().clone() for k,p in parameters.items()}
        opt.step()
        updates = {k:norm(p.detach()-before[k]) for k,p in parameters.items()}
        steps.append(dict(step=step, loss=float(loss.detach()), grad_norm=grads, parameter_delta=updates))
    # Switch must restore baseline even after the new parameters have updated.
    model.hcv2.enable_region_consistency = False
    with torch.no_grad():
        off = model(image, **kwargs)
    ablation_diff = float((off['mask_logits']-baseline['mask_logits']).abs().max())
    assert ablation_diff == 0
    model.hcv2.region_consistency.load_state_dict(initial)
    unchanged = [k for k,v in steps[-1]['parameter_delta'].items() if v == 0]
    return dict(checkpoint=path, status='PASS' if not unchanged else 'WIRING_PASS_UPDATE_INCOMPLETE',
                zero_update_parameters_step2=unchanged,
                probe_path=str(chosen.image), probe_split='train', probe_area=area,
                synthetic_probe_only=False,
                new_parameters=sum(p.numel() for p in parameters.values()),
                missing_keys=loaded.missing_keys, hcv_shapes=shapes,
                mask_shape=list(current['mask_logits'].shape), identity_max_abs_diff=diff,
                ablation_after_updates_max_abs_diff=ablation_diff, steps=steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = [audit(path) for path in args.checkpoint]
    Path(args.out).write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
