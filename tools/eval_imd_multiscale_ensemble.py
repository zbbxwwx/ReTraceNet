"""Val-selected 1024/1536 probability ensemble, followed by locked IMD test."""
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

WEIGHTS=(0.25,0.5,0.75) # weight assigned to 1024 prediction
FULL_MAX=1024

@torch.inference_mode()
def score_pair(model,r,ds,index,cache):
 with Image.open(r.image) as im:w,h=im.size
 if max(h,w)>FULL_MAX:
  a,target=ctx.predict_large(model,r,cache,1024,768)
  b,target2=ctx.predict_large(model,r,cache,1536,1152)
  if target.shape!=target2.shape or not np.array_equal(target,target2):raise RuntimeError('target mismatch')
  return a,b,target,True
 batch=next(iter(DataLoader(ds,batch_size=1,sampler=[index],num_workers=0)))
 vh,vw=map(int,batch['valid_hw'][0].tolist());o=cross.fair.fwd(model,batch)
 s=o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy()
 t=batch['mask'][0,0,:vh,:vw].numpy().astype(np.int16)
 return s,s,t,False

def resize(s,t):
 if s.shape==t.shape:return s
 return torch.nn.functional.interpolate(torch.from_numpy(s)[None,None].float(),size=t.shape,mode='bilinear',align_corners=False)[0,0].numpy()

@torch.inference_mode()
def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 summary={'status':'loading','checkpoint':a.checkpoint,'split':a.split,'weights_1024':list(WEIGHTS),'test_run':False}
 cross.save_yaml(out/'summary.yaml',summary)
 try:
  ck=torch.load(a.checkpoint,map_location='cpu');model=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval();model.load_state_dict(ck['model'],strict=True)
  split=json.loads(Path(a.split).read_text());val=ctx.records(split['val'],'val','IMD2020');cache=out/'jpeg_cache';ds=cross.fair.FairDataset(val,cache,False)
  stats={w:ValidationStats() for w in WEIGHTS};sliding=0
  summary['status']='evaluating_val';cross.save_yaml(out/'summary.yaml',summary)
  for i,r in enumerate(val):
   a1,a2,t,sl=score_pair(model,r,ds,i,cache);sliding+=int(sl);a1=resize(a1,t);a2=resize(a2,t)
   for w in WEIGHTS:stats[w].add(w*a1+(1-w)*a2,t,r.label,r.image)
   if i==0 or (i+1)%25==0:print('[ensemble-val]',i+1,len(val),flush=True)
  vals={str(w):stats[w].result()['fake'] for w in WEIGHTS}
  best_w=max(WEIGHTS,key=lambda w:float(vals[str(w)]['oracle']['f1']));threshold=float(vals[str(best_w)]['oracle']['threshold'])
  summary.update(val=vals,val_sliding_count=sliding,selected_weight_1024=best_w,locked_threshold=threshold,threshold_source='IMD2020_val_multiscale_ensemble',status='evaluating_test');cross.save_yaml(out/'summary.yaml',summary)
  recs=ctx.records(split['test'],'test','IMD2020');ds=cross.fair.FairDataset(recs,cache,False);diag=ValidationStats();locked=cross.LockedCounts(threshold);auth=AuthenticPixelFPR();sliding=0
  for i,r in enumerate(recs):
   a1,a2,t,sl=score_pair(model,r,ds,i,cache);sliding+=int(sl);s=best_w*resize(a1,t)+(1-best_w)*resize(a2,t)
   diag.add(s,t,r.label,r.image)
   if r.label:locked.add(s,t)
   else:auth.add(s)
   if i==0 or (i+1)%25==0:print('[ensemble-test]',i+1,len(recs),flush=True)
  d=diag.result();result={'dataset':'IMD2020_test','checkpoint_epoch':int(ck['epoch']),'policy':{'full_max_side':FULL_MAX,'tile_scales':[[1024,768],[1536,1152]],'weight_1024':best_w},'threshold_source':summary['threshold_source'],'threshold':threshold,'sliding_count':sliding,'fake_only_pooled':{'test_f1_locked':locked.result(),'auc':d['fake']['auc'],'test_f1_oracle_diagnostic':d['fake']['oracle']},'authentic_pixel_fpr':auth.result(threshold)}
  cross.save_yaml(out/'result.yaml',result);summary.update(status='completed',test_run=True,test=result);cross.save_yaml(out/'summary.yaml',summary)
 except BaseException:
  summary.update(status='failed',stop_reason=traceback.format_exc());cross.save_yaml(out/'summary.yaml',summary);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);main(p.parse_args())
