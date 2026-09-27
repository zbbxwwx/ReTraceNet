"""V3-compatible IMD reporting: pooled fake pixels, authentic FPR, all-image detection."""
import argparse,json
from pathlib import Path
import numpy as np,yaml
from PIL import Image
from sklearn.metrics import roc_auc_score

def prediction_index(pred_dir):
 directory=Path(pred_dir)
 manifest_path=directory/'manifest.json'
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text())
  # Preserve manifest order: a split may intentionally contain repeated
  # source paths (for example different manipulation records).  Matching by
  # unique image path would silently collapse those rows.
  return [Path(item['prediction']) for item in manifest]
 # Compatibility with older prediction folders that used image stems.
 return None

def prediction_path(row,pred_dir,index,position):
 if index is not None:
  if position >= len(index):raise FileNotFoundError(f'prediction missing at manifest row {position}')
  return index[position]
 return Path(pred_dir)/(Path(row['image']).stem+'.npz')

def load(rows,pred_dir):
 fake=[];auth=[];images=[]
 index=prediction_index(pred_dir)
 for position,r in enumerate(rows):
  z=np.load(prediction_path(r,pred_dir,index,position));p=np.asarray(z['map'],np.float32);score=float(z['score']);label=int(r['label']);images.append((score,label))
  if label:
   y=np.asarray(Image.open(r['mask']).convert('L'))>127
   if p.shape!=y.shape:raise RuntimeError((r['image'],p.shape,y.shape))
   fake.append((p,y))
  else:auth.append(p)
 return fake,auth,images
def counts(items,t):
 tp=fp=fn=0
 for p,y in items:
  z=p>=t;tp+=int((z&y).sum());fp+=int((z&~y).sum());fn+=int((~z&y).sum())
 return tp,fp,fn
def metric(c):
 tp,fp,fn=c;pr=tp/max(tp+fp,1);rc=tp/max(tp+fn,1);return {'tp':tp,'fp':fp,'fn':fn,'precision':pr,'recall':rc,'f1':2*pr*rc/max(pr+rc,1e-12)}
def select(fake):
 best=None
 for t in np.linspace(0,1,101):
  m=metric(counts(fake,float(t)));row={**m,'threshold':float(t)}
  if best is None or row['f1']>best['f1']:best=row
 return best
def image_select(items):
 best=None
 for t in np.linspace(0,1,101):
  c=counts([(np.array([s]),np.array([bool(y)])) for s,y in items],float(t));m=metric(c);row={**m,'threshold':float(t)}
  if best is None or row['f1']>best['f1']:best=row
 return best
def report(rows,pred_dir,pixel_t,image_t,mode,source):
 fake,auth,images=load(rows,pred_dir);pm=metric(counts(fake,pixel_t));fpr=sum(int((p>=pixel_t).sum()) for p in auth)/max(sum(p.size for p in auth),1);scores=np.array([s for s,_ in images]);labels=np.array([y for _,y in images]);auc=float(roc_auc_score(labels,scores)) if len(np.unique(labels))==2 else None;im=metric(counts([(np.array([s]),np.array([bool(y)])) for s,y in images],image_t))
 return {'evaluation_mode':mode,'threshold_source':source,'pixel_threshold':pixel_t,'image_threshold':image_t,'pixel_localization_fake_only_pooled_locked':pm,'authentic_pixel_fpr':fpr,'image_level_all':{**im,'auc':auc},'counts':{'images':len(rows),'fake':len(fake),'authentic':len(auth)}}
def main(a):
 d=json.loads(Path(a.split).read_text());vf,_,vi=load(d['val'],a.val_pred);pv=select(vf);iv=image_select(vi);val=report(d['val'],a.val_pred,pv['threshold'],iv['threshold'],'validation_select','IMD2020_val');test=report(d['test'],a.test_pred,pv['threshold'],iv['threshold'],'in_domain_locked','IMD2020_val');res={'status':'completed','protocol':'original_GT_grid; probability_bilinear; gt>127; fake-only pooled pixel localization; authentic FPR separate; all-image detection','validation_selection':{'pixel_f1_oracle_diagnostic':pv,'image_f1_oracle_diagnostic':iv},'validation':val,'test':test}
 Path(a.out).write_text(yaml.safe_dump(res,sort_keys=False));print(yaml.safe_dump(res,sort_keys=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--val-pred',required=True);p.add_argument('--test-pred',required=True);p.add_argument('--out',required=True);main(p.parse_args())
