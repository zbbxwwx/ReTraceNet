"""Replay Stable-BN IMD training sampler and measure actual foreground crop coverage."""
import json, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch

SPLIT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_IMD2020/split_721.json')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/imd2020_crop_coverage_audit_20260915.json')
SEED=1337
rows=json.loads(SPLIT.read_text())['train']
fake=[i for i,r in enumerate(rows) if int(r['label'])]
auth=[i for i,r in enumerate(rows) if not int(r['label'])]
nf=min(len(fake),len(auth))
plans={}
for ep in range(1,51):
 g=torch.Generator().manual_seed(SEED+ep)
 ids=[fake[i] for i in torch.randperm(len(fake),generator=g)[:nf]]
 for j,idx in enumerate(ids):
  r=rows[idx];rng=random.Random(SEED*1000000+ep*100000+j)
  with Image.open(r['image']) as im:w,h=im.size
  top=(rng.randint(0,max(h-512,0))//8)*8;left=(rng.randint(0,max(w-512,0))//8)*8
  plans.setdefault(idx,[]).append((ep,top,left,h,w))
stats=[]
for idx,items in plans.items():
 r=rows[idx];gt=(np.asarray(Image.open(r['mask']).convert('L'))>127);image_ratio=float(gt.mean())
 for ep,top,left,h,w in items:
  crop=gt[top:min(top+512,h),left:min(left+512,w)]
  stats.append({'epoch':ep,'positive':bool(crop.any()),'ratio':float(crop.mean()),'image_ratio':image_ratio})
pos=sum(x['positive'] for x in stats);rat=np.asarray([x['ratio'] for x in stats]);img=np.asarray([x['image_ratio'] for x in stats])
result={'dataset':'IMD2020','epochs':50,'fake_images_total':len(fake),'auth_images_total':len(auth),'fake_images_sampled_per_epoch':nf,'fake_crop_samples':len(stats),'positive_crop_count':pos,'empty_fake_crop_count':len(stats)-pos,'positive_crop_rate':pos/len(stats),'empty_fake_crop_rate':1-pos/len(stats),'crop_foreground_ratio_mean':float(rat.mean()),'crop_foreground_ratio_positive_median':float(np.median(rat[rat>0])) if pos else 0,'image_foreground_ratio_median':float(np.median(img))}
OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
