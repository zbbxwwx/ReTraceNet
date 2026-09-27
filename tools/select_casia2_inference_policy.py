"""Select MiT cross-domain inference policy using CASIA2 validation only."""
import argparse,copy,json,sys,traceback
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_imd_mit_full_v3
import run_cross_dataset_after_chain as cross
import eval_imd_mit_context_policies as ctx
from e15_original_metrics import ValidationStats

WEIGHTS=(0.25,0.5,0.75);BASELINE=.9416009303242864

@torch.inference_mode()
def get(model,r,ds,i,cache,tile,stride):
 with Image.open(r.image) as im:w,h=im.size
 if max(h,w)>1024:return (*ctx.predict_large(model,r,cache,tile,stride),True)
 b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)));vh,vw=map(int,b['valid_hw'][0].tolist());o=cross.fair.fwd(model,b)
 return o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy(),b['mask'][0,0,:vh,:vw].numpy().astype('int16'),False

@torch.inference_mode()
def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);summary={'status':'loading','threshold_source':'CASIA2_val','baseline_val_f1':BASELINE,'test_run':False};cross.save_yaml(out/'summary.yaml',summary)
 try:
  ck=torch.load(a.checkpoint,map_location='cpu');m=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval();m.load_state_dict(ck['model'],strict=True)
  rows=json.loads(Path(a.split).read_text())['val'];recs=ctx.records(rows,'val','CASIA2');cache=out/'jpeg_cache';ds=cross.fair.FairDataset(recs,cache,False)
  ms={w:ValidationStats() for w in WEIGHTS};overlap=ValidationStats();large=0
  summary['status']='evaluating';cross.save_yaml(out/'summary.yaml',summary)
  for i,r in enumerate(recs):
   p1024,t,sl=get(m,r,ds,i,cache,1024,768);large+=int(sl)
   if sl:
    p1536,t2,_=get(m,r,ds,i,cache,1536,1152);p512,t3,_=get(m,r,ds,i,cache,1024,512)
   else:p1536,p512,t2,t3=p1024,p1024,t,t
   for w in WEIGHTS:ms[w].add(w*p1024+(1-w)*p1536,t,r.label,r.image)
   overlap.add(p512,t,r.label,r.image)
   if i==0 or (i+1)%100==0:print('[casia2-policy-val]',i+1,len(recs),'large',large,flush=True)
  multi={str(w):ms[w].result()['fake'] for w in WEIGHTS};ov=overlap.result()['fake'];best=max(WEIGHTS,key=lambda w:float(multi[str(w)]['oracle']['f1']));bestf=float(multi[str(best)]['oracle']['f1'])
  summary.update(status='completed',num_val=len(recs),large_count=large,multiscale=multi,selected_multiscale_weight_1024=best,selected_multiscale_f1=bestf,overlap_1024_stride512=ov,multiscale_gain=bestf-BASELINE,overlap_gain=float(ov['oracle']['f1'])-BASELINE)
  cross.save_yaml(out/'summary.yaml',summary)
 except BaseException:
  summary.update(status='failed',stop_reason=traceback.format_exc());cross.save_yaml(out/'summary.yaml',summary);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);main(p.parse_args())
