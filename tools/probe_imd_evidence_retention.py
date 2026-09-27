"""Probe where foreground interior evidence is lost: N -> Q -> V -> D1."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch,yaml
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from retrace.data import Record
from fair_training_data import FairDataset
import run_balanced_residual_v3_full30 as v3

def recs(rows):return [Record(Path(r['image']),Path(r['mask']) if r.get('mask') else None,'val','IMD2020',int(r['label'])) for r in rows]
def sep(x,mask):
 y=F.interpolate(mask.float(),x.shape[-2:],mode='nearest')>0.5
 # Interior excludes a three-pixel feature-grid boundary; boundary is the removed ring.
 inv=(~y).float(); interior=~(F.max_pool2d(inv,7,1,3)>0); boundary=y&~interior; bg=~y
 z=x[0].float().permute(1,2,0)
 def score(a,b):
  aa=z[a[0,0]];bb=z[b[0,0]]
  if len(aa)<2 or len(bb)<2:return None
  d=(aa.mean(0)-bb.mean(0)).norm();den=aa.std(0).mean()+bb.std(0).mean()+1e-6
  return float(d/den)
 return {'interior_bg':score(interior,bg),'boundary_bg':score(boundary,bg),'interior_pixels':int(interior.sum()),'boundary_pixels':int(boundary.sum())}
def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);split=json.loads(Path(a.split).read_text());rows=recs(split['val']);ck=torch.load(a.checkpoint,map_location='cpu');cfg=ck['config'];m=v3.e.build_final_model(cfg['model']).cuda().eval();m.load_state_dict(ck['model'],strict=True);cap={}
 hs=[m.forensic_stream.register_forward_hook(lambda _m,_i,o:cap.__setitem__('N',o[0].detach())),m.qtr.register_forward_hook(lambda _m,_i,o:cap.__setitem__('Q',o['qualified_detail'][0].detach())),m.hcv2.register_forward_hook(lambda _m,_i,o:cap.__setitem__('V',o['verified_features'][0].detach())),m.dher.register_forward_pre_hook(lambda _m,x:cap.__setitem__('D1',x[0].detach()))]
 ds=FairDataset(rows,out/'cache',False);per=[]
 for i,r in enumerate(rows):
  if not r.label:continue
  with Image.open(r.image) as im:
   if max(im.size)>1024:continue
  b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)));ratio=float(b['mask'].mean())
  if ratio<.05:continue
  with torch.no_grad():v3.fwd(m,b,cfg,False)
  st={k:sep(v,b['mask'].cuda()) for k,v in cap.items()};per.append({'path':str(r.image),'ratio':ratio,'stages':st})
  if len(per)>=a.max_samples:break
 for h in hs:h.remove()
 agg={}
 for s in ('N','Q','V','D1'):
  agg[s]={k:float(np.mean([r['stages'][s][k] for r in per if r['stages'][s][k] is not None])) for k in ('interior_bg','boundary_bg')}
 result={'status':'completed','samples':len(per),'aggregate':agg,'per_image':per};(out/'summary.yaml').write_text(yaml.safe_dump(result,sort_keys=False));print(yaml.safe_dump({'samples':len(per),'aggregate':agg},sort_keys=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);p.add_argument('--max-samples',type=int,default=96);main(p.parse_args())
