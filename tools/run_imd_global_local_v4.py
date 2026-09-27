"""IMD V4 pilot: genuine whole-image + coordinate-aligned local content features.

This deliberately changes only the content path.  The forensic/QTR/HCV/decoder
chain and the Stable-BN training/evaluation protocol remain unchanged.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

import run_v3_stable_bn_indomain as stable


class GlobalFairDataset(stable.FairDataset):
    def __getitem__(self, key):
        item = super().__getitem__(key)
        with Image.open(item['jpeg_path']) as im:
            rgb = np.asarray(im.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32)
            width, height = im.size
        top, left = map(int, item['crop_origin'].tolist())
        valid_h, valid_w = map(int, item['valid_hw'].tolist())
        # Coordinates refer to the same JPEG used by local RGB/DCT/qtable.
        x0, y0 = left / width, top / height
        x1, y1 = min(left + valid_w, width) / width, min(top + valid_h, height) / height
        item['global_image'] = torch.from_numpy(rgb.copy()).permute(2, 0, 1) / 255.0
        item['global_roi'] = torch.tensor([x0, y0, x1, y1], dtype=torch.float32)
        return item


class GlobalLocalContentEncoder(nn.Module):
    def __init__(self, base, channels):
        super().__init__()
        self.base = base
        self.global_image = None
        self.global_roi = None
        self.fusions = nn.ModuleList()
        for c in channels:
            proj = nn.Conv2d(int(c) * 2, int(c), 1)
            nn.init.zeros_(proj.weight); nn.init.zeros_(proj.bias)
            self.fusions.append(proj)

    def set_context(self, global_image, global_roi):
        self.global_image = global_image
        self.global_roi = global_roi

    @staticmethod
    def roi_sample(feature, roi, out_hw):
        b = feature.shape[0]
        oh, ow = out_hw
        ys = torch.linspace(0, 1, oh, device=feature.device, dtype=feature.dtype)[None, :, None]
        xs = torch.linspace(0, 1, ow, device=feature.device, dtype=feature.dtype)[None, None, :]
        x0, y0, x1, y1 = [roi[:, i].to(feature.dtype) for i in range(4)]
        gx = x0[:, None, None] + xs * (x1 - x0)[:, None, None]
        gy = y0[:, None, None] + ys * (y1 - y0)[:, None, None]
        gx = gx.expand(b, oh, ow) * 2 - 1
        gy = gy.expand(b, oh, ow) * 2 - 1
        return F.grid_sample(feature, torch.stack([gx, gy], -1), mode='bilinear', padding_mode='border', align_corners=True)

    def forward(self, local_image):
        local = self.base(local_image)
        if self.global_image is None or self.global_roi is None:
            return local
        glob = self.base(self.global_image)
        out = []
        high_only = os.environ.get('V4_HIGH_LEVEL_ONLY') == '1'
        for index, (lf, gf, fusion) in enumerate(zip(local, glob, self.fusions)):
            if high_only and index < 2:
                out.append(lf)
                continue
            roi = self.roi_sample(gf, self.global_roi, lf.shape[-2:])
            out.append(lf + fusion(torch.cat([lf, roi], 1)))
        return out


_original_build = stable.v3.e.build_final_model
def build_global_local(config):
    model = _original_build(config)
    channels = tuple(config.get('content_channels', (256,512,1024,2048)))
    model.content_encoder = GlobalLocalContentEncoder(model.content_encoder, channels)
    init = os.environ.get('V4_INIT_CHECKPOINT')
    if init:
        checkpoint = torch.load(init, map_location='cpu')
        source = checkpoint['model']
        mapped = {}
        for key, value in source.items():
            if key.startswith('content_encoder.'):
                key = 'content_encoder.base.' + key[len('content_encoder.'):]
            mapped[key] = value
        incompatible = model.load_state_dict(mapped, strict=False)
        allowed_missing = {f'content_encoder.fusions.{i}.{name}' for i in range(4) for name in ('weight','bias')}
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError({'missing': incompatible.missing_keys, 'unexpected': incompatible.unexpected_keys})
        print('[V4] loaded Stable checkpoint; only global-local fusions are new:', init, flush=True)
    return model

def global_fwd(model, batch, cfg=None, train_dropout=False):
    enc = model.content_encoder
    enc.set_context(batch['global_image'].cuda(non_blocking=True), batch['global_roi'].cuda(non_blocking=True))
    x = {k: batch[k].cuda(non_blocking=True) for k in ['image','dct','qtable','dct_available']}
    if train_dropout and cfg is not None:
        p = float(cfg['training'].get('dct_sample_dropout', 0.0))
        x['dct_available'] *= (torch.rand_like(x['dct_available'].float()) >= p).to(x['dct_available'].dtype)
    return model(**x)

stable.FairDataset = GlobalFairDataset
stable.v3.e.build_final_model = build_global_local
stable.v3.fwd = global_fwd
stable.cross.fair.FairDataset = GlobalFairDataset
stable.cross.fair.fwd = global_fwd

# Optional phase-1 warm-up: keep the mature Stable100 network fixed and learn
# only the newly introduced global/local fusion projections.
if os.environ.get('V4_FUSION_ONLY') == '1':
    def fusion_only_train_mode(model):
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.content_encoder.fusions.parameters():
            parameter.requires_grad_(True)
        return stable.freeze_bn(model)
    stable.train_mode = fusion_only_train_mode

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True);p.add_argument('--split',required=True);p.add_argument('--template',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--seed',type=int,default=3407);p.add_argument('--resume',action='store_true')
    p.add_argument('--positive-crop-prob',type=float,default=0.0);p.add_argument('--positive-crop-retries',type=int,default=20)
    p.add_argument('--hcv-aux-scale',type=float,default=1.0);p.add_argument('--qtr-aux-scale',type=float,default=0.15);p.add_argument('--dher-aux-scale',type=float,default=0.05)
    p.add_argument('--forensic-mode',choices=('full','residual_only','dct_only'),default='full');p.add_argument('--hcv-semantic-bypass',action='store_true')
    p.add_argument('--verified-residual-decoder',action='store_true');p.add_argument('--decoder-ds-targeted',action='store_true');p.add_argument('--final-mask-only',action='store_true')
    p.add_argument('--splicing-region-affinity',action='store_true');p.add_argument('--region-affinity-weight',type=float,default=.10)
    p.add_argument('--early-gate-epoch',type=int,default=0);p.add_argument('--early-gate-min-f1',type=float,default=.5);p.add_argument('--skip-test',action='store_true')
    stable.main(p.parse_args())
