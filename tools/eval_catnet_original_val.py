"""Read-only CAT-Net validation reference; uses shared E15 original-grid metrics."""
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from e15_original_metrics import ValidationStats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    helper = Path('/group4/zouwenxuan/iml_generalization/outputs/domain_casia2_eval/catnet_unified1247_best_20260831_015845/export_catnet_unified1247.py')
    spec = importlib.util.spec_from_file_location('cat_reference', helper)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, encoding='utf-8-sig') as handle:
        rows = [r for r in csv.DictReader(handle) if r['split'] == 'val']
    assert len(rows) == 2494 and sum(int(r['label']) for r in rows) == 996
    ckpt_path = '/group4/zouwenxuan/iml_generalization/outputs/catnet/splicing_dataset/CAT_full/best.pth.tar'
    cfg_path = '/group4/zouwenxuan/iml_generalization/repos/CAT-Net/experiments/CAT_full.yaml'
    m.update_config(m.config, argparse.Namespace(cfg=cfg_path, opts=['TEST.MODEL_FILE', ckpt_path, 'TEST.FLIP_TEST', 'False']))
    ds = m.ManifestDataset(rows, directory / 'catnet_jpeg_cache')
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=2)
    net = getattr(m.models, m.config.MODEL.NAME).get_seg_model(m.config)
    state = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(state['state_dict'], strict=True)
    criterion = m.CrossEntropy(ignore_label=m.config.TRAIN.IGNORE_LABEL, weight=ds.class_weights).cuda()
    model = m.FullModel(net, criterion).cuda().eval()
    metrics = ValidationStats()
    start = time.time()
    with torch.inference_mode(), (directory / 'catnet_val_per_image.jsonl').open('w') as handle:
        for i, (image, label, qtable) in enumerate(loader):
            image, label = image.cuda(), label.long().cuda()
            if torch.is_tensor(qtable):
                qtable = qtable.cuda()
            _, pred = model(image, label, qtable)
            if pred.shape[-2:] != label.shape[-2:]:
                pred = F.interpolate(pred, size=label.shape[-2:], mode='bilinear', align_corners=False)
            scores = pred[0].float().softmax(0)[1].cpu().numpy()
            target = label[0].cpu().numpy()
            row = rows[i]
            metrics.add(scores, target, int(row['label']), row['image'])
            valid = target >= 0
            handle.write(json.dumps({'path': row['image'], 'tp': int(((scores >= .5) & (target > 0) & valid).sum()),
                'fp': int(((scores >= .5) & (target == 0) & valid).sum()), 'fn': int(((scores < .5) & (target > 0) & valid).sum())}) + '\n')
            if i == 0 or (i+1) % 50 == 0 or i+1 == len(rows):
                print(f'[val:CAT-Net] {i+1}/{len(rows)} elapsed={(time.time()-start)/60:.1f}m', flush=True)
    result = {'split': 'val', 'checkpoint': ckpt_path, 'metrics': metrics.result(),
              'protocol': 'original_GT_grid; valid_GT>=0; thresholds=101; 65536_bin_AUC; no_TTA',
              'input': 'official_RGB_DCT; nonJPEG_quality100; ignore_padding', 'elapsed_seconds': time.time()-start}
    (directory / 'catnet_original_val.yaml').write_text(yaml.safe_dump(result, sort_keys=False), encoding='utf-8')
    print('[val:CAT-Net] complete fake_F1=' + str(result['metrics']['fake']['oracle']['f1']), flush=True)


if __name__ == '__main__':
    main()
