"""TruFor inference on a locked JSON split with overlap-average sliding windows."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image

def starts(n,tile,stride):
 if n<=tile:return [0]
 xs=list(range(0,n-tile+1,stride))
 if xs[-1]!=n-tile:xs.append(n-tile)
 return xs

def run_tile(model,arr,device):
 h,w=arr.shape[:2];ph=(-h)%8;pw=(-w)%8;x=torch.from_numpy(arr.transpose(2,0,1)).float().unsqueeze(0).to(device)/256.0
 if ph or pw:x=F.pad(x,(0,pw,0,ph),mode='reflect')
 pred,_conf,det,_npp=model(x,save_np=False)
 if pred.shape[-2:]!=x.shape[-2:]:pred=F.interpolate(pred,size=x.shape[-2:],mode='bilinear',align_corners=False)
 p=F.softmax(pred,dim=1)[0,1,:h,:w].float().cpu().numpy();score=float(torch.sigmoid(det).item()) if det is not None else float(np.quantile(p,.995));return p,score

def main(a):
 repo=Path(a.repo);sys.path.insert(0,str(repo));from lib.config import config;from lib.utils import get_model
 config.defrost();config.merge_from_file(str(repo/'lib/config'/f'{a.experiment}.yaml'));config.freeze();device=torch.device(f'cuda:{a.gpu}')
 ck=torch.load(a.checkpoint,map_location='cpu');model=get_model(config);model.load_state_dict(ck['state_dict']);model.to(device).eval()
 split=json.loads(Path(a.split).read_text());rows=split[a.part];out=Path(a.out);out.mkdir(parents=True,exist_ok=True);seen=set();manifest=[]
 with torch.no_grad():
  for j,r in enumerate(rows,1):
   path=Path(r['image']);key=f'{j:06d}_{path.stem}_{hashlib.sha1(str(path).encode()).hexdigest()[:10]}'
   if key in seen:raise RuntimeError(f'duplicate stem: {key}')
   seen.add(key);arr=np.asarray(Image.open(path).convert('RGB'));h,w=arr.shape[:2]
   if max(h,w)<=a.full_max:
    prob,score=run_tile(model,arr,device);tiles=1
   else:
    acc=np.zeros((h,w),np.float32);cnt=np.zeros((h,w),np.float32);scores=[];tiles=0
    for y in starts(h,a.tile,a.stride):
     for x in starts(w,a.tile,a.stride):
      p,s=run_tile(model,arr[y:min(y+a.tile,h),x:min(x+a.tile,w)],device);hh,ww=p.shape;acc[y:y+hh,x:x+ww]+=p;cnt[y:y+hh,x:x+ww]+=1;scores.append(s);tiles+=1
    prob=acc/np.maximum(cnt,1);score=max(scores)
   np.savez_compressed(out/f'{key}.npz',map=prob.astype(np.float16),score=np.float32(score),imgsize=np.array([h,w]));manifest.append({'image':str(path),'prediction':str(out/f'{key}.npz'),'label':int(r['label']),'mask':r.get('mask'),'tiles':tiles})
   if j==1 or j%25==0 or j==len(rows):print('[infer]',a.part,j,len(rows),flush=True)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('[complete]',out)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--experiment',default='trufor_imd721');p.add_argument('--checkpoint',required=True);p.add_argument('--split',required=True);p.add_argument('--part',choices=('val','test'),required=True);p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--full-max',type=int,default=1536);p.add_argument('--tile',type=int,default=1024);p.add_argument('--stride',type=int,default=768);main(p.parse_args())
