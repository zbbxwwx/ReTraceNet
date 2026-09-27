import argparse,json,hashlib,sys,types
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
def key_for(path,i): return f"{i:06d}_{Path(path).stem}_{hashlib.sha1(str(path).encode()).hexdigest()[:10]}"
def main(a):
 pl=types.ModuleType('pytorch_lightning');pl.LightningModule=torch.nn.Module;sys.modules['pytorch_lightning']=pl
 repo=Path(a.repo);sys.path.insert(0,str(repo));from MantraNet.mantranet import pre_trained_model
 dev=torch.device(f'cuda:{a.gpu}');m=pre_trained_model(a.checkpoint,dev).to(dev).eval()
 rows=json.load(open(a.split))[a.part];out=Path(a.out);out.mkdir(parents=True,exist_ok=True);man=[]
 with torch.no_grad():
  for i,r in enumerate(rows):
   arr=np.asarray(Image.open(r['image']).convert('RGB'));h,w=arr.shape[:2]
   x=torch.from_numpy(arr.copy()).permute(2,0,1).float()[None].to(dev)
   p=m(x)[0,0].float().cpu().numpy()
   if p.shape!=(h,w): p=cv2.resize(p,(w,h),interpolation=cv2.INTER_LINEAR)
   p=np.clip(p,0,1).astype(np.float32);k=key_for(r['image'],i);q=out/f'{k}.npz'
   np.savez_compressed(q,map=p.astype(np.float16),score=np.float32(np.quantile(p,.99)),imgsize=np.array([h,w]))
   man.append({'image':r['image'],'prediction':str(q),'label':int(r['label']),'mask':r.get('mask')})
   if i==0 or (i+1)%100==0: print('[mantra]',a.part,i+1,len(rows),float(p.min()),float(p.max()),flush=True)
 (out/'manifest.json').write_text(json.dumps(man,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--split',required=True);p.add_argument('--part',required=True);p.add_argument('--out',required=True);p.add_argument('--gpu',type=int,default=0);main(p.parse_args())
