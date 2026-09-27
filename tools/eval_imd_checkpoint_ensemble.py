"""Val-gated ep86/ep100 probability ensemble over fixed 1024+1536 TTA."""
import argparse,copy,json,sys,traceback
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_imd_mit_full_v3
import run_cross_dataset_after_chain as cross
import eval_imd_mit_context_policies as ctx
from e15_original_metrics import ValidationStats
from v3_unified_metrics import AuthenticPixelFPR

ALPHAS=(0.75,0.5,0.25) # ep86 weight
FULL_MAX=1024
BASELINE_VAL=0.65581
GATE=0.005

def resize(s,t):
 if s.shape==t.shape:return s
 return torch.nn.functional.interpolate(torch.from_numpy(s)[None,None].float(),size=t.shape,mode='bilinear',align_corners=False)[0,0].numpy()

@torch.inference_mode()
def multiscale(model,r,ds,index,cache):
 with Image.open(r.image) as im:w,h=im.size
 if max(h,w)>FULL_MAX:
  a,t=ctx.predict_large(model,r,cache,1024,768);b,t2=ctx.predict_large(model,r,cache,1536,1152)
  if t.shape!=t2.shape or not np.array_equal(t,t2):raise RuntimeError('target mismatch')
  return 0.5*resize(a,t)+0.5*resize(b,t),t,True
 batch=next(iter(DataLoader(ds,batch_size=1,sampler=[index],num_workers=0)))
 vh,vw=map(int,batch['valid_hw'][0].tolist());o=cross.fair.fwd(model,batch)
 return o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy(),batch['mask'][0,0,:vh,:vw].numpy().astype(np.int16),False

def load_model(path):
 ck=torch.load(path,map_location='cpu');m=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval();m.load_state_dict(ck['model'],strict=True);return ck,m

@torch.inference_mode()
def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 summary={'status':'loading','ep86':a.ep86,'ep100':a.ep100,'alphas_ep86':list(ALPHAS),'baseline_val_f1':BASELINE_VAL,'required_gain':GATE,'test_run':False}
 cross.save_yaml(out/'summary.yaml',summary)
 try:
  ck86,m86=load_model(a.ep86);ck100,m100=load_model(a.ep100)
  split=json.loads(Path(a.split).read_text());val=ctx.records(split['val'],'val','IMD2020');cache=out/'jpeg_cache';ds=cross.fair.FairDataset(val,cache,False)
  stats={x:ValidationStats() for x in ALPHAS};sliding=0;summary['status']='evaluating_val';cross.save_yaml(out/'summary.yaml',summary)
  for i,r in enumerate(val):
   p86,t,s=multiscale(m86,r,ds,i,cache);p100,t2,_=multiscale(m100,r,ds,i,cache);sliding+=int(s)
   if t.shape!=t2.shape or not np.array_equal(t,t2):raise RuntimeError('checkpoint target mismatch')
   for x in ALPHAS:stats[x].add(x*p86+(1-x)*p100,t,r.label,r.image)
   if i==0 or (i+1)%25==0:print('[checkpoint-ensemble-val]',i+1,len(val),flush=True)
  vals={str(x):stats[x].result()['fake'] for x in ALPHAS};best=max(ALPHAS,key=lambda x:float(vals[str(x)]['oracle']['f1']));f1=float(vals[str(best)]['oracle']['f1']);thr=float(vals[str(best)]['oracle']['threshold'])
  passed=f1>=BASELINE_VAL+GATE
  summary.update(val=vals,val_sliding_count=sliding,selected_alpha_ep86=best,selected_val_f1=f1,locked_threshold=thr,gate_passed=passed,threshold_source='IMD2020_val_checkpoint_multiscale_ensemble')
  if not passed:
   summary.update(status='completed',stop_reason='val gain below +0.5pp gate; test not run');cross.save_yaml(out/'summary.yaml',summary);return
  recs=ctx.records(split['test'],'test','IMD2020');ds=cross.fair.FairDataset(recs,cache,False);diag=ValidationStats();locked=cross.LockedCounts(thr);auth=AuthenticPixelFPR();sliding=0;summary['status']='evaluating_test';cross.save_yaml(out/'summary.yaml',summary)
  for i,r in enumerate(recs):
   p86,t,s=multiscale(m86,r,ds,i,cache);p100,t2,_=multiscale(m100,r,ds,i,cache);sliding+=int(s);p=best*p86+(1-best)*p100
   diag.add(p,t,r.label,r.image)
   if r.label:locked.add(p,t)
   else:auth.add(p)
   if i==0 or (i+1)%25==0:print('[checkpoint-ensemble-test]',i+1,len(recs),flush=True)
  d=diag.result();result={'dataset':'IMD2020_test','policy':{'scales':[[1024,768],[1536,1152]],'scale_weights':[0.5,0.5],'checkpoint_alpha_ep86':best},'threshold_source':summary['threshold_source'],'threshold':thr,'fake_only_pooled':{'test_f1_locked':locked.result(),'auc':d['fake']['auc'],'test_f1_oracle_diagnostic':d['fake']['oracle']},'authentic_pixel_fpr':auth.result(thr),'sliding_count':sliding}
  cross.save_yaml(out/'result.yaml',result);summary.update(status='completed',test_run=True,test=result);cross.save_yaml(out/'summary.yaml',summary)
 except BaseException:
  summary.update(status='failed',stop_reason=traceback.format_exc());cross.save_yaml(out/'summary.yaml',summary);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ep86',required=True);p.add_argument('--ep100',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);main(p.parse_args())
