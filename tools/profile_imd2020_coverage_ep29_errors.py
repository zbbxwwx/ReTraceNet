"""Read-only IMD2020 train/val error profile for the locked Stable-BN baseline."""
import csv,json,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,torch,yaml
from PIL import Image
from torch.utils.data import DataLoader
from retrace.data import Record,ManipulationDataset
import run_balanced_residual_v3_full30 as v3
import run_cross_dataset_after_chain as cross
from fair_training_data import FairDataset

import run_imd_mit_full_v3
BASE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')
RUN=BASE/'imd_tune_coverage_first_2k_20260920';OUT=BASE/'imd_coverage_ep29_error_profile_20260924';OUT.mkdir(parents=True,exist_ok=True)
SPLIT=BASE/'v3_offline_master_20260912/indomain_IMD2020/split_721.json'
def recs(rows,sp):return [Record(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') else None,split=sp,dataset='IMD2020',label=int(r['label'])) for r in rows]
def counts(p,y,t):
 z=p>=t; yy=y.astype(bool);tp=int((z&yy).sum());fp=int((z&~yy).sum());fn=int((~z&yy).sum())
 return tp,fp,fn
def metric(c):
 tp,fp,fn=c;return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1)}
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def pred(model,r,ds,i,cache):
 with Image.open(r.image) as im:w,h=im.size
 if max(h,w)>1024:return cross.predict_large_record(model,r,cache)[:2]
 b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)));vh,vw=map(int,b['valid_hw'][0]);p,_=cross.predict_probability(model,b);return p[:vh,:vw].numpy(),b['mask'][0,0,:vh,:vw].numpy().astype(np.int16)
def main():
 split=json.loads(SPLIT.read_text());val=recs(split['val'],'val');ck=torch.load(RUN/'best.pt',map_location='cpu');cfg=ck['config'];m=v3.e.build_final_model(cfg['model']).cuda().eval();m.load_state_dict(ck['model'],strict=True);th=float(ck['selection_threshold']);ds=FairDataset(val,OUT/'jpeg_cache',False)
 rows=[];groups={'overall':(0,0,0),'sliding':(0,0,0),'non_sliding':(0,0,0),'<1':(0,0,0),'1-5':(0,0,0),'5-20':(0,0,0),'>=20':(0,0,0)}
 for i,r in enumerate(val):
  if not r.label:continue
  p,y=pred(m,r,ds,i,OUT/'jpeg_cache');c=counts(p,y,th);ratio=float(y.mean());sliding=max(y.shape)>1024;key='<1' if ratio<.01 else '1-5' if ratio<.05 else '5-20' if ratio<.2 else '>=20'
  for g in ('overall','sliding' if sliding else 'non_sliding',key):groups[g]=add(groups[g],c)
  pred_ratio=float((p>=th).mean());rows.append({'path':str(r.image),'sliding':int(sliding),'height':y.shape[0],'width':y.shape[1],'gt_foreground_ratio':ratio,'predicted_foreground_ratio':pred_ratio,'pred_gt_area_ratio':pred_ratio/max(ratio,1e-12),**metric(c)})
  if len(rows)==1 or len(rows)%100==0:print('[profile]',len(rows),flush=True)
 with (OUT/'imd2020_error_profile.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 res={'status':'completed','checkpoint':str(RUN/'best.pt'),'threshold':th,'test_read':False,'fake_images':len(rows),'groups':{k:metric(v) for k,v in groups.items()},'area_ratio':{'median':float(np.median([r['pred_gt_area_ratio'] for r in rows])),'mean':float(np.mean([r['pred_gt_area_ratio'] for r in rows]))},'note':'tile center/border diagnostic is generated separately by Context-Core evaluator to avoid duplicated overlap pixels'}
 (OUT/'imd2020_error_profile.yaml').write_text(yaml.safe_dump(res,sort_keys=False));print(yaml.safe_dump(res,sort_keys=False))
if __name__=='__main__':main()
