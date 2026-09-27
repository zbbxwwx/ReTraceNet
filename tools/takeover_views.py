"""Deterministic same-source RGB/DCT degradation views for CASIA2 pilots.

Every non-clean view is decoded from the exact JPEG used to extract its raw
coefficients and quantization table. Geometric transforms resize back to the
original grid, so the manipulation mask remains on its original pixel grid.
"""
import hashlib
import io
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from retrace.data import ManipulationDataset

PROXY_CONDITIONS = ['clean', 'jpeg100', 'jpeg90', 'jpeg80', 'jpeg70', 'jpeg60',
                    'recompression', 'subsampling444', 'subsampling420',
                    'resize075', 'resize125', 'blur', 'noise', 'brightness', 'contrast']


def stable_seed(path, extra=0):
    return (int(hashlib.sha256(str(path).encode()).hexdigest()[:8],16) + extra) % (2**32)


def make_view(sample, condition, seed):
    if condition == 'clean':
        return dict(sample)
    rng = np.random.default_rng(seed)
    # Only transform real pixels; do not recompress artificial padding as context.
    vh, vw = map(int, sample['valid_hw'].tolist())
    full_h, full_w = sample['image'].shape[-2:]
    array = (sample['image'][:, :vh, :vw].permute(1,2,0).numpy() * 255).round().clip(0,255).astype('uint8')
    im = Image.fromarray(array)
    quality, subsampling = 100, 0
    if condition == 'jpeg_random':
        quality = int(rng.integers(60,101))
    elif condition.startswith('jpeg'):
        quality = int(condition[4:])
    elif condition == 'recompression':
        buf = io.BytesIO(); im.save(buf, format='JPEG', quality=85, subsampling=2)
        buf.seek(0)
        with Image.open(buf) as intermediate:
            im = intermediate.convert('RGB')
        quality, subsampling = 70, 2
    elif condition in ['subsampling444', 'subsampling420']:
        quality, subsampling = 90, 0 if condition.endswith('444') else 2
    elif condition.startswith('resize'):
        factor = {'resize075': .75, 'resize125': 1.25}.get(condition)
        if factor is None:
            factor = float(rng.uniform(.75,1.25))
        im = im.resize((max(1,round(vw*factor)),max(1,round(vh*factor))), Image.Resampling.BILINEAR)
        im = im.resize((vw,vh), Image.Resampling.BILINEAR)
    elif condition == 'blur':
        im = im.filter(ImageFilter.GaussianBlur(.7))
    elif condition == 'noise':
        array = np.asarray(im).astype('float32') + rng.normal(0,2.0,(vh,vw,3))
        im = Image.fromarray(array.round().clip(0,255).astype('uint8'))
    elif condition == 'brightness':
        im = ImageEnhance.Brightness(im).enhance(1.1)
    elif condition == 'contrast':
        im = ImageEnhance.Contrast(im).enhance(1.1)
    else:
        raise ValueError(condition)
    import jpegio
    with tempfile.TemporaryDirectory(prefix='retrace_view_') as folder:
        path = Path(folder) / 'view.jpg'
        im.save(path, quality=quality, subsampling=subsampling)
        with Image.open(path) as decoded:
            pixels = np.asarray(decoded.convert('RGB'),dtype='float32')
        jpg = jpegio.read(str(path))
        coeff = jpg.coef_arrays[0]
        qtable = jpg.quant_tables[jpg.comp_info[0].quant_tbl_no].astype('float32')
    rgb = np.full((full_h,full_w,3),127.5,dtype='float32')
    rgb[:vh,:vw] = pixels
    coef = np.zeros((full_h,full_w),dtype=coeff.dtype)
    coef[:min(full_h,coeff.shape[0]),:min(full_w,coeff.shape[1])] = coeff[:full_h,:full_w]
    out = dict(sample)
    out.update(image=torch.from_numpy(rgb).permute(2,0,1)/255,
               dct=ManipulationDataset._dct_volume_from_coef(coef,full_h,full_w),
               qtable=torch.from_numpy(qtable[None]), dct_available=torch.ones(1))
    assert out['image'].shape == sample['image'].shape
    assert out['dct'].shape == sample['dct'].shape
    assert torch.equal(out['mask'], sample['mask'])
    return out


class ProxyDataset(torch.utils.data.Dataset):
    def __init__(self, base, condition):
        self.base, self.condition = base, condition
        self.records = base.records

    def __len__(self):
        return len(self.base)

    def __getitem__(self, key):
        sample = self.base[key]
        return make_view(sample, self.condition, stable_seed(sample['path'],9901))


class PairedDataset(torch.utils.data.Dataset):
    choices = ['jpeg_random','recompression','subsampling444','subsampling420',
               'resize_random','blur','noise','brightness','contrast']

    def __init__(self, base):
        self.base = base
        self.records = base.records

    def __len__(self):
        return len(self.base)

    def __getitem__(self, key):
        sample = self.base[key]
        seed = key[1] if isinstance(key,tuple) else stable_seed(sample['path'])
        rng = np.random.default_rng(seed)
        condition = self.choices[int(rng.integers(len(self.choices)))]
        return {'a':sample,'b':make_view(sample,condition,seed+301), 'condition':condition}
