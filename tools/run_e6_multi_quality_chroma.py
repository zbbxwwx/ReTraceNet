import copy
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path('/group4/zouwenxuan/iml_generalization/repos/ReTraceNet')
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import run_e5_paired_chroma_adaptation as e5
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from retrace.data import ManipulationDataset

E5 = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e5_paired_chroma_adaptation_20260923/epoch02.pt')
PARENT = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
MANIFEST = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
OUT = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e6_q90_q80_joint_adaptation_20260923')
SEED, EPOCHS, LR = 3407, 2, 1e-5


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    stable.dump(OUT / name, obj)


def jpeg(sample, quality, subsampling=2):
    vh, vw = map(int, sample['valid_hw'].tolist())
    fh, fw = sample['image'].shape[-2:]
    arr = (sample['image'][:, :vh, :vw].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype('uint8')
    import jpegio
    with tempfile.TemporaryDirectory(prefix='e6_') as directory:
        path = Path(directory) / 'x.jpg'
        Image.fromarray(arr).save(path, 'JPEG', quality=int(quality), subsampling=int(subsampling))
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB'), dtype='float32')
        parsed = jpegio.read(str(path))
        coef = parsed.coef_arrays[0]
        qtable = parsed.quant_tables[parsed.comp_info[0].quant_tbl_no].astype('float32')
    canvas = np.full((fh, fw, 3), 127.5, np.float32)
    canvas[:vh, :vw] = rgb
    coefficient_canvas = np.zeros((fh, fw), coef.dtype)
    h, w = min(fh, coef.shape[0]), min(fw, coef.shape[1])
    coefficient_canvas[:h, :w] = coef[:h, :w]
    out = dict(sample)
    out.update(
        image=torch.from_numpy(canvas).permute(2, 0, 1) / 255,
        dct=ManipulationDataset._dct_volume_from_coef(coefficient_canvas, fh, fw),
        qtable=torch.from_numpy(qtable[None]),
        dct_available=torch.ones(1),
    )
    return out


class MixedQuality(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        self.records = base.records

    def __len__(self):
        return len(self.base)

    def __getitem__(self, key):
        seed = int(key[1]) if isinstance(key, tuple) else int(key)
        sample = self.base[key]
        bucket = seed % 20
        if bucket < 6:
            return sample
        if bucket < 13:
            return jpeg(sample, 90, 2)
        return jpeg(sample, 80, 2)


class FixedQuality(torch.utils.data.Dataset):
    def __init__(self, base, quality=None, subsampling=2):
        self.base, self.quality, self.subsampling = base, quality, subsampling
        self.records = base.records

    def __len__(self):
        return len(self.base)

    def __getitem__(self, key):
        sample = self.base[key]
        return sample if self.quality is None else jpeg(sample, self.quality, self.subsampling)


def q80_audit(base, count=128):
    rows = []
    for index in range(min(count, len(base))):
        sample = base[index]
        a, b = jpeg(sample, 80, 0), jpeg(sample, 80, 2)
        dct_diff = (a['dct'].float() - b['dct'].float()).abs()
        qt_diff = (a['qtable'].float() - b['qtable'].float()).abs()
        rows.append((float(dct_diff.max()), float(qt_diff.max())))
    return {
        'pairs': len(rows),
        'luma_dct_max_abs_max': max(x[0] for x in rows),
        'qtable_max_abs_max': max(x[1] for x in rows),
        'luma_dct_identical_fraction': sum(x[0] == 0 for x in rows) / len(rows),
        'qtable_identical_fraction': sum(x[1] == 0 for x in rows) / len(rows),
    }


def main():
    summary = {
        'status': 'preparing',
        'experiment': 'E6_Q90_Q80_Joint_Adaptation',
        'checkpoint_init': str(E5),
        'architecture': 'unchanged ReTraceNet',
        'mixture': {'clean': 0.30, 'q90_420': 0.35, 'q80_420': 0.35},
        'trainable': ['content_encoder', 'forensic_stream', 'balanced_forensic_fusions'],
        'frozen': ['dct_branch', 'qtr', 'hcv2', 'decoder', 'dher', 'mask_head'],
        'teacher_distillation': False,
        'epochs': EPOCHS,
        'lr': LR,
        'test_read': False,
        'history': [],
    }
    dump('summary.yaml', summary)
    try:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        parent = torch.load(PARENT, map_location='cpu')
        checkpoint = torch.load(E5, map_location='cpu')
        cfg = copy.deepcopy(parent['config'])
        cfg['training']['dct_sample_dropout'] = 0.0
        train_records = fair.e.read_manifest(MANIFEST, cfg['data']['root'], 'train')
        val_records = fair.e.read_manifest(MANIFEST, cfg['data']['root'], 'val')
        model = fair.e.build_final_model(cfg['model']).cuda()
        model.load_state_dict(checkpoint['model'], strict=True)

        allowed = ('content_encoder.', 'forensic_stream.', 'balanced_forensic_fusions.')
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(allowed))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        summary['trainable_parameter_count'] = sum(parameter.numel() for parameter in trainable)
        summary['total_parameter_count'] = sum(parameter.numel() for parameter in model.parameters())

        optimizer = torch.optim.SGD(trainable, lr=LR, momentum=0.9, weight_decay=0.0005)
        loss_fn = stable.v3.e24.ReTraceLoss.from_config(stable.v3.e24.clean_loss_config(cfg['loss'])).cuda()
        train_base = FairDataset(train_records, OUT / 'cache_train', True)
        val_base = FairDataset(val_records, OUT / 'cache_val', False)
        summary['q80_preflight'] = q80_audit(val_base, 128)
        dump('summary.yaml', summary)

        fake = [i for i, record in enumerate(train_records) if record.label]
        authentic = [i for i, record in enumerate(train_records) if not record.label]
        balanced_count = min(len(fake), len(authentic))
        e5.OUT = OUT

        for epoch in range(1, EPOCHS + 1):
            generator = torch.Generator().manual_seed(SEED + epoch)
            indices = [fake[i] for i in torch.randperm(len(fake), generator=generator)[:balanced_count]]
            indices += [authentic[i] for i in torch.randperm(len(authentic), generator=generator)[:balanced_count]]
            indices = [indices[i] for i in torch.randperm(len(indices), generator=generator)]
            plan = [(idx, SEED * 1000000 + epoch * 100000 + j) for j, idx in enumerate(indices)]
            loader = DataLoader(MixedQuality(train_base), batch_size=4, sampler=plan, num_workers=2, pin_memory=True)

            model.eval()
            model.content_encoder.train()
            model.forensic_stream.train()
            model.balanced_forensic_fusions.train()
            stable.freeze_bn(model)
            total = 0.0
            started = time.time()
            for step, batch in enumerate(loader, 1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = stable.v3.fwd(model, batch, cfg, False)
                    loss = stable.v3.objective(output, batch, loss_fn, cfg)['total']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 5.0, error_if_nonfinite=True)
                optimizer.step()
                total += float(loss)
                if step == 1 or step % 50 == 0 or step == len(loader):
                    print('[e6]', epoch, step, len(loader), total / step, flush=True)

            checkpoint_path = OUT / f'epoch{epoch:02d}.pt'
            torch.save({'epoch': epoch, 'parent': str(E5), 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'experiment': summary['experiment']}, checkpoint_path)
            clean = e5.eval_ds(model, FixedQuality(val_base), cfg, f'ep{epoch:02d}_clean')
            threshold = clean['oracle']['threshold']
            q90 = e5.eval_ds(model, FixedQuality(val_base, 90, 2), cfg, f'ep{epoch:02d}_q90_420', threshold)
            q80 = e5.eval_ds(model, FixedQuality(val_base, 80, 2), cfg, f'ep{epoch:02d}_q80_420', threshold)
            row = {
                'epoch': epoch,
                'checkpoint': str(checkpoint_path),
                'train_total': total / len(loader),
                'clean': clean,
                'q90_420': q90,
                'q80_420': q80,
                'elapsed_minutes': (time.time() - started) / 60,
            }
            summary['history'].append(row)
            summary.update(status='epoch_completed', epoch=epoch)
            dump('summary.yaml', summary)

        last = summary['history'][-1]
        summary['gate'] = {
            'clean_f1_min': 0.925,
            'q90_420_f1_min': 0.73,
            'q80_420_f1_min': 0.45,
            'clean_f1': last['clean']['oracle']['f1'],
            'q90_420_f1_locked': last['q90_420']['clean_locked']['f1'],
            'q80_420_f1_locked': last['q80_420']['clean_locked']['f1'],
        }
        summary['gate']['passed'] = (
            summary['gate']['clean_f1'] >= 0.925
            and summary['gate']['q90_420_f1_locked'] >= 0.73
            and summary['gate']['q80_420_f1_locked'] >= 0.45
        )
        summary.update(status='completed_2ep_gate', decision='review_before_any_test')
        dump('summary.yaml', summary)
        print(yaml.safe_dump(summary, sort_keys=False), flush=True)
    except BaseException:
        summary.update(status='failed', traceback=traceback.format_exc())
        dump('summary.yaml', summary)
        raise


if __name__ == '__main__':
    main()
