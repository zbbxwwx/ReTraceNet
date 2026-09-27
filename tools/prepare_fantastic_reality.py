"""Selectively extract FantasticReality localization data, convert masks, split by source graph."""
import argparse, json, re, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

def ids(stem): return re.findall(r'IMG_\d+',stem)
class UF:
 def __init__(self):self.p={}
 def find(self,x):
  self.p.setdefault(x,x)
  if self.p[x]!=x:self.p[x]=self.find(self.p[x])
  return self.p[x]
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[b]=a

def main(a):
 zpath=Path(a.zip);root=Path(a.root);root.mkdir(parents=True,exist_ok=True)
 keep=('FantasticReality_v1/dataset/ColorFakeImages/','FantasticReality_v1/dataset/ColorRealImages/','FantasticReality_v1/dataset/SegmentationFake/','FantasticReality_v1/dataset/image_db.pkl','FantasticReality_v1/dataset/classes.json')
 with zipfile.ZipFile(zpath) as z:
  members=[i for i in z.infolist() if any(i.filename.startswith(k) for k in keep) and not i.is_dir() and '/._' not in i.filename and not i.filename.endswith('.DS_Store')]
  for j,i in enumerate(members,1):
   rel='/'.join(i.filename.split('/')[2:]);dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True)
   if not dst.exists() or dst.stat().st_size!=i.file_size:
    with z.open(i) as src,dst.open('wb') as out:
     while True:
      b=src.read(8<<20)
      if not b:break
      out.write(b)
   if j%5000==0:print('[extract]',j,len(members),flush=True)
 fake=root/'ColorFakeImages';real=root/'ColorRealImages';npz=root/'SegmentationFake';masks=root/'SegmentationFakePNG';masks.mkdir(exist_ok=True)
 fake_rows=[]
 for j,img in enumerate(sorted(fake.glob('*.jpg')),1):
  src=npz/(img.stem+'.npz')
  if not src.exists():continue
  dst=masks/(img.stem+'.png')
  if not dst.exists():
   x=np.load(src);arr=x[x.files[0]];arr=np.asarray(arr).squeeze();arr=((arr>0)*255).astype('uint8');Image.fromarray(arr).save(dst)
  fake_rows.append((img,dst,ids(img.stem)))
  if j%5000==0:print('[mask]',j,flush=True)
 real_rows=[(p,None,ids(p.stem)) for p in sorted(real.glob('*.jpg'))]
 import random
 rng=random.Random(a.seed);data={k:[] for k in ('train','val','test')}
 def add_stratified(rows,label):
  rows=list(rows);rng.shuffle(rows);n=len(rows);n_train=int(.7*n);n_val=int(.2*n)
  chunks={'train':rows[:n_train],'val':rows[n_train:n_train+n_val],'test':rows[n_train+n_val:]}
  for split,chunk in chunks.items():
   for image,mask,_ in chunk:data[split].append({'image':str(image),'mask':str(mask) if mask else None,'label':label})
 add_stratified(fake_rows,1);add_stratified(real_rows,0)
 for rows in data.values():rng.shuffle(rows)
 out=root/'split_721_random_stratified.json';out.write_text(json.dumps(data,indent=2))
 report={'counts':{k:len(v) for k,v in data.items()},'labels':{k:{str(y):sum(r['label']==y for r in v) for y in (0,1)} for k,v in data.items()},'split_strategy':'label-stratified random 70/20/10; source grouping intentionally disabled by user','unmatched_fake_masks':len(list(fake.glob('*.jpg')))-len(fake_rows),'split':str(out)}
 (root/'preparation_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--root',required=True);p.add_argument('--seed',type=int,default=3407);main(p.parse_args())
