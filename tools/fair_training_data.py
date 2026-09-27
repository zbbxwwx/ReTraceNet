"""Unified JPEG, original-scale block-aligned crops. Never resize coefficients."""
import hashlib
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from retrace.data import ManipulationDataset

class FairDataset(torch.utils.data.Dataset):
    def __init__(self,records,cache,training,positive_crop_prob=0.0,positive_crop_retries=20):
        self.records=records;self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.training=training
        self.positive_crop_prob=float(positive_crop_prob)
        self.positive_crop_retries=int(positive_crop_retries)
    def __len__(self):return len(self.records)
    def __getitem__(self,key):
        if isinstance(key,tuple):index,seed=key
        else:index,seed=key,0
        rng=random.Random(seed);r=self.records[index]
        jpeg=r.image;generated=jpeg.suffix.lower() not in {'.jpg','.jpeg'}
        if generated:
            jpeg=self.cache/(hashlib.sha256(str(r.image).encode()).hexdigest()+'.jpg')
            if not jpeg.exists():
                with Image.open(r.image) as im:im.convert('RGB').save(jpeg,quality=100,subsampling=0)
        import jpegio
        j=jpegio.read(str(jpeg));coef=j.coef_arrays[0]
        q=j.quant_tables[j.comp_info[0].quant_tbl_no].astype('float32')
        with Image.open(jpeg) as im:rgb=np.asarray(im.convert('RGB'),dtype=np.float32)
        h,w=rgb.shape[:2]
        gt=np.zeros((h,w),dtype=np.float32) if r.mask is None else (np.asarray(Image.open(r.mask).convert('L'))>127).astype('float32')
        assert gt.shape==(h,w),(r.image,gt.shape,rgb.shape)
        th,tw=(512,512) if self.training else (((h+31)//32)*32,((w+31)//32)*32)
        ph,pw=max(h,th),max(w,tw)
        canvas=np.full((ph,pw,3),127.5,dtype=np.float32);canvas[:h,:w]=rgb
        target=np.zeros((ph,pw),dtype=np.float32);target[:h,:w]=gt
        co=np.zeros((max(ph,coef.shape[0]),max(pw,coef.shape[1])),dtype=coef.dtype);co[:coef.shape[0],:coef.shape[1]]=coef
        top=(rng.randint(0,max(h-th,0))//8)*8 if self.training else 0
        left=(rng.randint(0,max(w-tw,0))//8)*8 if self.training else 0
        if self.training and r.mask is not None and gt.any() and rng.random()<self.positive_crop_prob:
            best=(top,left,-1.0)
            for _ in range(self.positive_crop_retries):
                candidate_top=(rng.randint(0,max(h-th,0))//8)*8
                candidate_left=(rng.randint(0,max(w-tw,0))//8)*8
                ratio=float(gt[candidate_top:min(candidate_top+th,h),candidate_left:min(candidate_left+tw,w)].mean())
                if ratio>best[2]:best=(candidate_top,candidate_left,ratio)
                if ratio>=0.001:break
            top,left=best[:2]
        assert top%8==0 and left%8==0
        vh,vw=min(h-top,th),min(w-left,tw)
        c=co[top:top+th,left:left+tw]
        mask=torch.from_numpy(target[top:top+th,left:left+tw].copy())[None]
        valid=torch.zeros_like(mask);valid[:,:vh,:vw]=1
        return {'image':torch.from_numpy(canvas[top:top+th,left:left+tw].copy()).permute(2,0,1)/255,
                'mask':mask,'valid':valid,'valid_hw':torch.tensor([vh,vw]),
                'label':torch.tensor([float(mask.any())]),'original_label':torch.tensor([float(r.label)]),
                'path':str(r.image),'jpeg_path':str(jpeg),'generated':torch.tensor([generated]),
                'dct':ManipulationDataset._dct_volume_from_coef(c,th,tw),
                'qtable':torch.from_numpy(q[None]),'dct_available':torch.ones(1),
                'crop_origin':torch.tensor([top,left])}

def crop_outputs(value,index,batch_size,valid_h,valid_w,full_h,full_w):
    """Exclude padded locations from existing per-image and multiscale losses."""
    if torch.is_tensor(value):
        if value.ndim and value.shape[0]==batch_size:value=value[index:index+1]
        if value.ndim==4:
            h,w=value.shape[-2:]
            ch=max(1,(valid_h*h+full_h-1)//full_h);cw=max(1,(valid_w*w+full_w-1)//full_w)
            value=value[...,:ch,:cw]
        return value
    if isinstance(value,dict):return {k:crop_outputs(v,index,batch_size,valid_h,valid_w,full_h,full_w) for k,v in value.items()}
    if isinstance(value,list):return [crop_outputs(v,index,batch_size,valid_h,valid_w,full_h,full_w) for v in value]
    if isinstance(value,tuple):return tuple(crop_outputs(v,index,batch_size,valid_h,valid_w,full_h,full_w) for v in value)
    return value
