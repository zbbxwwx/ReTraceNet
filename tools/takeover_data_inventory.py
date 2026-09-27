"""Read-only dataset provenance inventory, resumable JSONL; does not split/train."""
import collections
import csv
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from takeover_recalibrate import OUT, BASE, FAIR, atomic_json, sha
from run_cross_dataset_after_chain import parse_index


def candidate_source(dataset, path, label):
    stem = path.stem
    if dataset.startswith('CASIA'):
        return re.findall(r'(?:ani|arc|art|cha|pla|nat|sec|txt)\d+', stem)
    if dataset == 'IMD2020':
        return [re.sub(r'_(?:fake|orig)(?:_.*)?$', '', stem)]
    if dataset == 'NIST16':
        return [stem]
    # Camera names or sub_NN indices do not establish Columbia source identity.
    return []


def main():
    dest = OUT / 'data_inventory'
    dest.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (dest / 'execution.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'running', 'training_run': False, 'split_created': False}
    atomic_json(dest / 'state.json', state)
    manifests = {}
    rows = list(csv.DictReader((FAIR / 'clean_manifest.csv').open()))
    # Use original config manifest to include the 46 removed training duplicates
    # in the inventory. Preserve the cleaned manifest for G training.
    import yaml
    cfg = yaml.safe_load((FAIR / 'config.yaml').read_text())
    historical_config = BASE / 'outputs/retracenet_v2/e24_dher_calibrated_writeback_20260910/config.yaml'
    if historical_config.exists():
        hc = yaml.safe_load(historical_config.read_text())
        if Path(hc['data']['manifest']).exists():
            rows = list(csv.DictReader(Path(hc['data']['manifest']).open()))
    manifests['CASIA2'] = [{'image': str(Path(cfg['data']['root']) / r['image']),
                           'mask': str(Path(cfg['data']['root']) / r['mask']) if r.get('mask') else None,
                           'label': int(r['label']), 'historical_split': r['split']} for r in rows]
    for name, folder in [('CASIA1','CASIA1.0'), ('Columbia','Columbia'), ('NIST16','NIST16'), ('IMD2020','IMD2020')]:
        recs = parse_index(BASE / 'paper_compare/mvss_five_indices' / (name + '.txt'), name)
        items = [{'image': str(r.image), 'mask': str(r.mask), 'label': r.label} for r in recs]
        root = BASE / 'data' / folder
        authentic_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.lower() == 'au']
        for au in authentic_dirs:
            items.extend({'image': str(p), 'mask': None, 'label': 0}
                         for p in sorted(au.iterdir()) if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'})
        manifests[name] = items
    master = {}
    for name, items in manifests.items():
        target = dest / (name + '.jsonl')
        done = {}
        if target.exists():
            for line in target.read_text().splitlines():
                item = json.loads(line)
                done[item['image']] = item
        assert len({r['image'] for r in items}) == len(items), f'duplicate index path in {name}'
        with target.open('a') as handle:
            for index, item in enumerate(items):
                path = Path(item['image'])
                if item['image'] in done:
                    continue
                with Image.open(path) as im:
                    rgb = np.asarray(im.convert('RGB'))
                    fmt = im.format
                    quant = getattr(im, 'quantization', None)
                h, w = rgb.shape[:2]
                byte_hash = sha(path)
                pixel_hash = hashlib.sha256(struct.pack('>II', h, w) + rgb.tobytes()).hexdigest()
                if item['mask']:
                    with Image.open(item['mask']) as im:
                        gt = np.asarray(im.convert('L'))
                    if gt.shape != (h,w):
                        raise ValueError(f'GT/image mismatch: {path}, {gt.shape}, {(h,w)}')
                    if not (gt > 127).any() or (gt > 127).all():
                        raise ValueError(f'empty/full fake GT requires inspection: {path}')
                    ratio = float((gt > 127).mean())
                    mask_hash = sha(item['mask'])
                else:
                    ratio, mask_hash = 0., None
                # Thumbnail hashes identify candidate near duplicates only.
                # They do not establish source relationships and are never used
                # to silently discard data or declare a source-disjoint split.
                thumb = np.asarray(Image.fromarray(rgb).resize((9,8)).convert('L'))
                dhash = int(''.join('1' if b else '0' for b in (thumb[:,1:] > thumb[:,:-1]).flat), 2)
                kind = 'unknown'
                if name.startswith('CASIA') and item['label']:
                    parts = path.stem.split('_')
                    if len(parts) > 1 and parts[1] in ['S','D']:
                        kind = parts[1]
                row = dict(item, dataset=name, height=h, width=w, format=fmt,
                           original_jpeg_available=(fmt == 'JPEG'), foreground_ratio=ratio,
                           area='authentic' if not item['label'] else 'lt1' if ratio < .01 else '1to5' if ratio < .05 else '5to20' if ratio < .2 else 'gte20',
                           manipulation_type=kind, sha256=byte_hash, rgb_sha256=pixel_hash,
                           mask_sha256=mask_hash, dhash=f'{dhash:016x}',
                           quantization_tables=quant,
                           source_tokens_candidate=candidate_source(name,path,item['label']),
                           source_relationship_status='candidate_needs_dataset_provenance_verification')
                handle.write(json.dumps(row) + '\n'); handle.flush()
                done[item['image']] = row
                if index % 100 == 0:
                    print('[inventory]', name, index+1, len(items), flush=True)
                    state.update(dataset=name, index=index+1, count=len(items))
                    atomic_json(dest / 'state.json', state)
        hashes = collections.defaultdict(list)
        for r in done.values():
            hashes[r['rgb_sha256']].append(r['image'])
        duplicates = [v for v in hashes.values() if len(v) > 1]
        bytegroups = collections.defaultdict(list)
        for r in done.values():
            bytegroups[r['sha256']].append(r['image'])
        result = {'count': len(done), 'labels': dict(collections.Counter(r['label'] for r in done.values())),
                  'formats': dict(collections.Counter(r['format'] for r in done.values())),
                  'area': dict(collections.Counter(r['area'] for r in done.values())),
                  'manipulation_type': dict(collections.Counter(r['manipulation_type'] for r in done.values())),
                  'decoded_duplicate_groups': duplicates,
                  'byte_duplicate_groups': [v for v in bytegroups.values() if len(v)>1],
                  'inventory_sha256': sha(target),
                  'source_isolation_verified': False,
                  'note': 'Source tokens and dHash are candidates. Source-disjoint split requires relation verification; no split generated yet.'}
        atomic_json(dest / (name + '_summary.json'), result)
        master[name] = result
    atomic_json(dest / 'inventory_summary.json', master)
    state.update(status='completed', next='verify_source_relations_after_G_then_group_aware_split')
    atomic_json(dest / 'state.json', state)
    atomic_json(dest / 'decision.json', {'status':'completed','route':'provenance_verification',
                'evidence': {k: {'count':v['count'], 'labels':v['labels']} for k,v in master.items()},
                'rejected_routes':['filename_random_split_without_group_audit'],
                'reason':'Exact duplicate inventory ready; source candidates must be verified before split.'})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        atomic_json(OUT / 'data_inventory/state.json', {'status':'failed', 'error':traceback.format_exc()})
        raise
