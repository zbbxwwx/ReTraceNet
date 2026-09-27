import argparse,json,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,precision_recall_curve
from pretrain_native_jpeg_local_paired_v2 import NativeJPEGForensicHRNet
from native_jpeg_fair_data import NativeJPEGFairDataset
def score(y,p):
 pr,rc,th=precision_recall_curve(y,p);f=2*pr*rc/(pr+rc+1e-12);k=int(f.argmax())
 return dict(f1=float(f[k]),auc=float(roc_auc_score(y,p)),precision=float(pr[k]),recall=float(rc[k]),threshold=float(th[min(k,len(th)-1)]))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',required=True);ap.add_argument('--global-ck',required=True);ap.add_argument('--local-ck',required=True);ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);split=json.load(open(a.split));results={}
 for mode,ck in [('random',None),('global',a.global_ck),('local',a.local_ck)]:
  torch.manual_seed(3407);np.random.seed(3407);random.seed(3407)
  enc=NativeJPEGForensicHRNet().cuda().eval()
  if ck:enc.load_state_dict(torch.load(ck,map_location='cpu')['encoder'],strict=True)
  enc.requires_grad_(False);arrays={}
  for role in ['train','val']:
   records=[SimpleNamespace(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') and r['mask']!='negative' else None,label=r['label']) for r in split[role]]
   if a.smoke:records=records[:4]
   ds=NativeJPEGFairDataset(records,out/'jpeg_cache',training=(role=='train'))
   xx=[];yy=[]
   for i in range(len(ds)):
    s=ds[(i,3407+i)];im=s['image'][None].cuda();d=s['dct'][None].cuda();q=s['qtable'][None].cuda()
    # Match paired JPEG-only pretraining. Residual sees zeros in ALL three controls.
    with torch.no_grad():
     feats=enc(torch.zeros_like(im),d,q,s['dct_available'][None].cuda())
     size=feats[0].shape[-2:];f=torch.cat([F.interpolate(v,size=size,mode='bilinear',align_corners=False) for v in feats],1)
     y=F.interpolate(s['mask'][None].cuda(),size=size,mode='nearest').flatten()
     valid=F.interpolate(s['valid'][None].cuda(),size=size,mode='nearest').flatten()>0
     idx=valid.nonzero().flatten().cpu();g=torch.Generator().manual_seed(100000+i)
     idx=idx[torch.randperm(len(idx),generator=g)[:2048]].cuda()
     xx.append(f[0].flatten(1).T[idx].cpu().half()); yy.append(y[idx].cpu())
    if i%100==0:print(mode,role,i,len(ds),flush=True)
   arrays[role]=(torch.cat(xx),torch.cat(yy))
  del enc;torch.cuda.empty_cache()
  x,y=arrays['train'];vx,vy=arrays['val'];mean=x.float().mean(0);std=x.float().std(0).clamp_min(1e-5)
  torch.manual_seed(123);head=torch.nn.Linear(x.shape[1],1).cuda();opt=torch.optim.Adam(head.parameters(),lr=.001);history=[];best=None
  for ep in range(1,11):
   g=torch.Generator().manual_seed(3407+ep);order=torch.randperm(len(y),generator=g)
   head.train()
   for idx in order.split(8192):
    z=head(((x[idx].float()-mean)/std).cuda()).flatten();loss=F.binary_cross_entropy_with_logits(z,y[idx].cuda());opt.zero_grad();loss.backward();opt.step()
   head.eval();pred=[]
   with torch.no_grad():
    for chunk in vx.split(8192):pred.append(head(((chunk.float()-mean)/std).cuda()).sigmoid().cpu().flatten())
   met=score(vy.numpy(),torch.cat(pred).numpy());history.append(dict(epoch=ep,**met))
   if best is None or met['f1']>best['f1']:best=dict(epoch=ep,**met)
  results[mode]=dict(best=best,history=history,train_pixels=len(y),val_pixels=len(vy))
  (out/'results.json').write_text(json.dumps(results,indent=2));print(mode,best,flush=True)
  del arrays,x,y,vx,vy,head,opt;torch.cuda.empty_cache()
 passed=results['local']['best']['f1']>max(results[m]['best']['f1'] for m in ['random','global']) and results['local']['best']['auc']>=max(results[m]['best']['auc'] for m in ['random','global'])
 (out/'decision.json').write_text(json.dumps(dict(pass_gate=passed,protocol='frozen JPEG-only encoder; four-scale linear pixel probe; deterministic 2048 pixels per image; CASIA1 train fit and val selection; no test',results=results),indent=2))
if __name__=='__main__':main()
