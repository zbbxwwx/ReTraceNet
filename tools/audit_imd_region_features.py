"""Read-only feature-separation audit on large IMD2020 validation manipulations."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,torch,yaml
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from retrace.data import Record
from fair_training_data import FairDataset
import run_balanced_residual_v3_full30 as v3

def records(rows):
 return [Record(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') else None,split='val',dataset='IMD2020',label=int(r['label'])) for r in rows]

def feature_stats(x,mask):
 mask=F.interpolate(mask.float(),size=x.shape[-2:],mode='nearest')[0,0]>0.5
 z=F.normalize(x[0].float().permute(1,2,0).reshape(-1,x.shape[1]),dim=1);m=mask.reshape(-1)
 fg=z[m];bg=z[~m]
 if len(fg)<2 or len(bg)<2:return None
 g=torch.Generator(device=fg.device).manual_seed(1337);nf=min(len(fg),512);nb=min(len(bg),512)
 fg=fg[torch.randperm(len(fg),generator=g,device=fg.device)[:nf]];bg=bg[torch.randperm(len(bg),generator=g,device=bg.device)[:nb]]
 fp=F.normalize(fg.mean(0),dim=0);bp=F.normalize(bg.mean(0),dim=0)
 within=(fg*fg.roll(1,0)).sum(1).mean();cross=(fg[:min(nf,nb)]*bg[:min(nf,nb)]).sum(1).mean()
 return {'within_fg_cos':float(within),'fg_bg_pair_cos':float(cross),'prototype_cos':float((fp*bp).sum()),'fg_to_fg_proto':float((fg*fp).sum(1).mean()),'bg_to_fg_proto':float((bg*fp).sum(1).mean())}

def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);split=json.loads(Path(a.split).read_text());rows=records(split['val']);ck=torch.load(a.checkpoint,map_location='cpu');cfg=ck['config'];model=v3.e.build_final_model(cfg['model']).cuda().eval();model.load_state_dict(ck['model'],strict=True)
 cap={}
 h1=model.dher.register_forward_pre_hook(lambda _m,args:cap.__setitem__('pre',args[0].detach()))
 h2=model.dher.register_forward_hook(lambda _m,_args,o:cap.__setitem__('post',o[0].detach()))
 ds=FairDataset(rows,out/'jpeg_cache',False);per=[]
 for i,r in enumerate(rows):
  if not r.label:continue
  with Image.open(r.image) as im:w,h=im.size
  if max(h,w)>1024:continue
  b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)));mask=b['mask'].cuda();ratio=float(mask.mean())
  if ratio<a.min_ratio:continue
  with torch.no_grad():v3.fwd(model,b,cfg,False)
  row={'path':str(r.image),'foreground_ratio':ratio,'pre':feature_stats(cap['pre'],mask),'post':feature_stats(cap['post'],mask)}
  if row['pre'] and row['post']:per.append(row)
  if len(per)>=a.max_samples:break
 h1.remove();h2.remove()
 def avg(stage,key):return float(np.mean([x[stage][key] for x in per])) if per else None
 keys=('within_fg_cos','fg_bg_pair_cos','prototype_cos','fg_to_fg_proto','bg_to_fg_proto')
 result={'status':'completed','checkpoint':a.checkpoint,'split':a.split,'selection':'fake val, original max dimension <=1024, foreground ratio >= '+str(a.min_ratio),'samples':len(per),'aggregate':{s:{k:avg(s,k) for k in keys} for s in ('pre','post')},'per_image':per,'interpretation_note':'Higher within-FG and larger (fg_to_fg_proto-bg_to_fg_proto) imply stronger region organization; this is diagnostic association, not causal proof.'}
 (out/'summary.yaml').write_text(yaml.safe_dump(result,sort_keys=False));print(yaml.safe_dump({k:v for k,v in result.items() if k!='per_image'},sort_keys=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);p.add_argument('--min-ratio',type=float,default=.20);p.add_argument('--max-samples',type=int,default=64);main(p.parse_args())
