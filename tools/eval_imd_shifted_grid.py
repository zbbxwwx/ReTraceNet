"""Val-gated standard/shifted 1024-grid probability ensemble for IMD."""
import argparse,copy,hashlib,json,sys,traceback
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_imd_mit_full_v3
import run_cross_dataset_after_chain as cross
import eval_imd_mit_context_policies as ctx
from retrace.data import ManipulationDataset
from e15_original_metrics import ValidationStats
from v3_unified_metrics import AuthenticPixelFPR

TILE=1024;STRIDE=768;OFFSET=384;FULL_MAX=1024
BASELINE=0.65581;GATE=0.005

@torch.inference_mode()
def shifted(model,record,cache):
 cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);jpeg_path=record.image
 if jpeg_path.suffix.lower() not in {'.jpg','.jpeg'}:
  jpeg_path=cache/(hashlib.sha256(str(record.image).encode()).hexdigest()+'.jpg')
  if not jpeg_path.exists():
   with Image.open(record.image) as im:im.convert('RGB').save(jpeg_path,quality=100,subsampling=0)
 import jpegio
 jpeg=jpegio.read(str(jpeg_path));coef=jpeg.coef_arrays[0];q=jpeg.quant_tables[jpeg.comp_info[0].quant_tbl_no].astype('float32')
 with Image.open(jpeg_path) as im:rgb=np.asarray(im.convert('RGB'),dtype=np.float32)
 h,w=rgb.shape[:2]
 if record.mask is None:target=np.zeros((h,w),dtype=np.int16)
 else:
  with Image.open(record.mask) as im:target=(np.asarray(im.convert('L'))>127).astype(np.int16)
 if target.shape!=(h,w):raise ValueError(f'image/mask mismatch {record.image}')
 # Placing the original at a 384-pixel (48 JPEG-block) margin gives a grid
 # shifted by (-384,-384) in original coordinates without altering JPEG data.
 ph=max(TILE,((h+2*OFFSET+31)//32)*32);pw=max(TILE,((w+2*OFFSET+31)//32)*32)
 canvas=np.full((ph,pw,3),127.5,dtype=np.float32);canvas[OFFSET:OFFSET+h,OFFSET:OFFSET+w]=rgb
 cc=np.zeros((ph,pw),dtype=coef.dtype);cc[OFFSET:OFFSET+coef.shape[0],OFFSET:OFFSET+coef.shape[1]]=coef
 accum=torch.zeros((ph,pw),dtype=torch.float32);weight=torch.zeros((ph,pw),dtype=torch.float32)
 for top in ctx.tile_starts(ph,TILE,STRIDE):
  for left in ctx.tile_starts(pw,TILE,STRIDE):
   image=torch.from_numpy(canvas[top:top+TILE,left:left+TILE].copy()).permute(2,0,1)[None]/255
   cv=cc[top:top+TILE,left:left+TILE]
   batch={'image':image,'dct':ManipulationDataset._dct_volume_from_coef(cv,TILE,TILE)[None],'qtable':torch.from_numpy(q[None,None]),'dct_available':torch.ones((1,1))}
   prob=cross.fair.fwd(model,batch)['mask_logits'].float().sigmoid()[0,0].cpu()
   accum[top:top+TILE,left:left+TILE]+=prob;weight[top:top+TILE,left:left+TILE]+=1
 if not bool((weight>0).all()):raise RuntimeError('coverage gap')
 return (accum/weight)[OFFSET:OFFSET+h,OFFSET:OFFSET+w].numpy(),target

@torch.inference_mode()
def pair(model,r,ds,i,cache):
 with Image.open(r.image) as im:w,h=im.size
 if max(h,w)>FULL_MAX:
  a,t=ctx.predict_large(model,r,cache,TILE,STRIDE);b,t2=shifted(model,r,cache)
  if t.shape!=t2.shape or not np.array_equal(t,t2):raise RuntimeError('target mismatch')
  return a,b,t,True
 batch=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)));vh,vw=map(int,batch['valid_hw'][0].tolist());o=cross.fair.fwd(model,batch)
 s=o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy();t=batch['mask'][0,0,:vh,:vw].numpy().astype(np.int16);return s,s,t,False

@torch.inference_mode()
def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);summary={'status':'loading','policy':{'tile':TILE,'stride':STRIDE,'shift_offset':OFFSET,'grid_weights':[0.5,0.5]},'baseline_val_f1':BASELINE,'required_gain':GATE,'test_run':False};cross.save_yaml(out/'summary.yaml',summary)
 try:
  ck=torch.load(a.checkpoint,map_location='cpu');model=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval();model.load_state_dict(ck['model'],strict=True)
  split=json.loads(Path(a.split).read_text());val=ctx.records(split['val'],'val','IMD2020');cache=out/'jpeg_cache';ds=cross.fair.FairDataset(val,cache,False);stats=ValidationStats();sliding=0
  summary['status']='evaluating_val';cross.save_yaml(out/'summary.yaml',summary)
  for i,r in enumerate(val):
   x,y,t,s=pair(model,r,ds,i,cache);sliding+=int(s);stats.add(0.5*x+0.5*y,t,r.label,r.image)
   if i==0 or (i+1)%25==0:print('[shift-grid-val]',i+1,len(val),flush=True)
  vr=stats.result()['fake'];f1=float(vr['oracle']['f1']);thr=float(vr['oracle']['threshold']);passed=f1>=BASELINE+GATE
  summary.update(val=vr,val_sliding_count=sliding,locked_threshold=thr,selected_val_f1=f1,gate_passed=passed,threshold_source='IMD2020_val_shifted_grid')
  if not passed:
   summary.update(status='completed',stop_reason='val gain below +0.5pp gate; test not run');cross.save_yaml(out/'summary.yaml',summary);return
  recs=ctx.records(split['test'],'test','IMD2020');ds=cross.fair.FairDataset(recs,cache,False);diag=ValidationStats();locked=cross.LockedCounts(thr);auth=AuthenticPixelFPR();sliding=0;summary['status']='evaluating_test';cross.save_yaml(out/'summary.yaml',summary)
  for i,r in enumerate(recs):
   x,y,t,s=pair(model,r,ds,i,cache);sliding+=int(s);p=.5*x+.5*y;diag.add(p,t,r.label,r.image)
   if r.label:locked.add(p,t)
   else:auth.add(p)
   if i==0 or (i+1)%25==0:print('[shift-grid-test]',i+1,len(recs),flush=True)
  d=diag.result();result={'dataset':'IMD2020_test','policy':summary['policy'],'threshold_source':summary['threshold_source'],'threshold':thr,'fake_only_pooled':{'test_f1_locked':locked.result(),'auc':d['fake']['auc'],'test_f1_oracle_diagnostic':d['fake']['oracle']},'authentic_pixel_fpr':auth.result(thr),'sliding_count':sliding};cross.save_yaml(out/'result.yaml',result);summary.update(status='completed',test_run=True,test=result);cross.save_yaml(out/'summary.yaml',summary)
 except BaseException:
  summary.update(status='failed',stop_reason=traceback.format_exc());cross.save_yaml(out/'summary.yaml',summary);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);main(p.parse_args())
