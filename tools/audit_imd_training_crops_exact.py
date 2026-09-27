"""Replay the exact IMD sampler seeds and report masks actually seen by training."""
import json,random
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image

BASE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')
SPLIT=BASE/'v3_offline_master_20260912/indomain_IMD2020/split_721.json'
OUT=BASE/'imd2020_exact_training_crop_audit_20260916';OUT.mkdir(parents=True,exist_ok=True)
SEED=1337;EPOCHS=20;TH=TW=512
rows=json.loads(SPLIT.read_text())['train'];fake=[i for i,r in enumerate(rows) if int(r['label'])];auth=[i for i,r in enumerate(rows) if not int(r['label'])];nf=min(len(fake),len(auth));ratios=[]
for ep in range(1,EPOCHS+1):
 g=torch.Generator().manual_seed(SEED+ep);ids=[fake[i] for i in torch.randperm(len(fake),generator=g)[:nf]]+[auth[i] for i in torch.randperm(len(auth),generator=g)[:nf]];ids=[ids[i] for i in torch.randperm(len(ids),generator=g)]
 for j,idx in enumerate(ids):
  r=rows[idx]
  if not int(r['label']):continue
  rng=random.Random(SEED*1000000+ep*100000+j)
  with Image.open(r['image']) as im:w,h=im.size
  gt=(np.asarray(Image.open(r['mask']).convert('L'))>127)
  top=(rng.randint(0,max(h-TH,0))//8)*8;left=(rng.randint(0,max(w-TW,0))//8)*8
  valid=gt[top:min(top+TH,h),left:min(left+TW,w)];ratios.append(float(valid.mean()) if valid.size else 0.0)
a=np.asarray(ratios);counts={'empty':int((a==0).sum()),'lt_0_1pct':int(((a>0)&(a<.001)).sum()),'0_1_to_1pct':int(((a>=.001)&(a<.01)).sum()),'1_to_5pct':int(((a>=.01)&(a<.05)).sum()),'gte_5pct':int((a>=.05).sum())}
result={'status':'completed','split':str(SPLIT),'seed':SEED,'epochs_replayed':EPOCHS,'positive_crop_prob':0.0,'fake_crop_samples':len(ratios),'counts':counts,'fractions':{k:v/len(ratios) for k,v in counts.items()},'foreground_ratio_all_median':float(np.median(a)),'foreground_ratio_positive_median':float(np.median(a[a>0])) if np.any(a>0) else 0.0,'note':'exact training sampler ordering and RNG seeds; valid unpadded mask region'}
(OUT/'summary.yaml').write_text(yaml.safe_dump(result,sort_keys=False));print(yaml.safe_dump(result,sort_keys=False))
