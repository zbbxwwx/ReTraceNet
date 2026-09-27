"""Select a stable top-k checkpoint ensemble on deduplicated Columbia val, then test once."""
import argparse, copy, itertools, json, math, sys, traceback
from pathlib import Path
import numpy as np, torch, yaml
from PIL import Image
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_columbia_mit_seed
import run_cross_dataset_after_chain as cross
import eval_imd_mit_context_policies as ctx
from e15_original_metrics import ValidationStats
from v3_unified_metrics import AuthenticPixelFPR
from retrace.data import Record
TILE=1024;STRIDE=768;FULL_MAX=1024

def dedupe(rows):
 seen=set();out=[]
 for r in rows:
  k=r.get('rgb_sha256') or r['image']
  if k not in seen:seen.add(k);out.append(r)
 return out

def records(rows,part):
 return [Record(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') else None,split=part,dataset='Columbia',label=int(r['label'])) for r in rows]

def candidate_checkpoints(run):
 last=torch.load(run/'last.pt',map_location='cpu');rows={int(r['epoch']):r for r in last['history']}
 by_epoch={}
 for p in sorted((run/'checkpoints').glob('epoch*.pt')):
  ep=int(p.stem.replace('epoch',''));r=rows[ep];by_epoch[ep]=(p,ep,float(r['val_fake_f1']),float(r['val_fake_auc']))
 best_ep=int(last.get('best_epoch',max(rows,key=lambda e:float(rows[e]['val_fake_f1']))))
 if (run/'best.pt').exists():
  r=rows[best_ep];by_epoch[best_ep]=(run/'best.pt',best_ep,float(r['val_fake_f1']),float(r['val_fake_auc']))
 all_candidates=sorted(by_epoch.values(),key=lambda x:x[1])
 # Keep a small, high-quality pool. Final membership is selected using ensembled val probabilities.
 pool=sorted(all_candidates,key=lambda x:(x[2]+.2*x[3],x[2]),reverse=True)[:6]
 return pool,all_candidates

@torch.inference_mode()
def predict_one(model,r,ds,index,cache):
 with Image.open(r.image) as im:w,h=im.size
 if max(h,w)>FULL_MAX:return (*ctx.predict_large(model,r,cache,TILE,STRIDE),True)
 b=next(iter(DataLoader(ds,batch_size=1,sampler=[index],num_workers=0)));vh,vw=map(int,b['valid_hw'][0].tolist());o=cross.fair.fwd(model,b)
 return o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy(),b['mask'][0,0,:vh,:vw].numpy().astype(np.int16),False

def cache_predictions(checkpoints,recs,part,out):
 ds=cross.fair.FairDataset(recs,out/'jpeg_cache',False);dirs=[]
 for path,ep,_,_ in checkpoints:
  d=out/f'pred_{part}_ep{ep:03d}';d.mkdir(parents=True,exist_ok=True);dirs.append(d)
  ck=torch.load(path,map_location='cpu');m=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval();m.load_state_dict(ck['model'],strict=True)
  for i,r in enumerate(recs):
   p,t,_=predict_one(m,r,ds,i,out/'jpeg_cache');np.save(d/f'{i:04d}.npy',p.astype(np.float16));
   if not (out/f'target_{part}_{i:04d}.npy').exists():np.save(out/f'target_{part}_{i:04d}.npy',t.astype(np.int8))
  del m;torch.cuda.empty_cache()
 return dirs

def aggregate(recs,dirs,part,out,threshold=None):
 stats=ValidationStats();locked=cross.LockedCounts(threshold) if threshold is not None else None;auth=AuthenticPixelFPR()
 for i,r in enumerate(recs):
  p=np.mean([np.load(d/f'{i:04d}.npy').astype(np.float32) for d in dirs],axis=0);t=np.load(out/f'target_{part}_{i:04d}.npy').astype(np.int16);stats.add(p,t,r.label,r.image)
  if threshold is not None:
   if r.label:locked.add(p,t)
   else:auth.add(p)
 diag=stats.result()
 if threshold is None:return diag
 f=locked.result();ar=auth.result(threshold);tp,fn=int(f['tp']),int(f['fn']);fp=int(f['fp'])+int(ar['false_positive_pixels']);tn=int(f['tn'])+int(ar['authentic_pixels'])-int(ar['false_positive_pixels'])
 return {'fake_only_pooled':{'test_f1_locked':f,'auc':diag['fake']['auc'],'test_f1_oracle_diagnostic':diag['fake']['oracle']},'authentic_pixel_fpr':ar,'all_images_pooled':{'test_f1_locked':{'threshold':threshold,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn)},'auc':diag['all']['auc']}}

def choose_val_ensemble(recs,candidates,dirs,out):
 trials=[]
 for size in range(1,min(3,len(candidates))+1):
  for ids in itertools.combinations(range(len(candidates)),size):
   chosen_dirs=[dirs[i] for i in ids];diag=aggregate(recs,chosen_dirs,'val',out)
   f1=float(diag['fake']['oracle']['f1']);auc=float(diag['fake']['auc'])
   trials.append({'indices':list(ids),'epochs':[candidates[i][1] for i in ids],'f1':f1,'auc':auc,'threshold':float(diag['fake']['oracle']['threshold'])})
 best=max(trials,key=lambda x:(x['f1'],x['auc'],-len(x['indices'])))
 return best,trials

def main(a):
 run=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);summary={'status':'loading','run':str(run),'test_run':False}
 try:
  pool,allc=candidate_checkpoints(run);summary['all_saved_candidates']=[{'path':str(p),'epoch':e,'f1':f,'auc':u} for p,e,f,u in allc];summary['candidate_pool']=[{'path':str(p),'epoch':e,'f1':f,'auc':u} for p,e,f,u in pool];(out/'summary.yaml').write_text(yaml.safe_dump(summary,sort_keys=False))
  split=json.loads(Path(a.split).read_text());val=records(dedupe(split['val']),'val');vd=cache_predictions(pool,val,'val',out);selection,trials=choose_val_ensemble(val,pool,vd,out);chosen=[pool[i] for i in selection['indices']];chosen_vd=[vd[i] for i in selection['indices']];vr=aggregate(val,chosen_vd,'val',out);thr=float(selection['threshold']);summary.update(status='evaluating_test',val_trials=trials,val_selection=selection,val=vr,selected=[{'path':str(p),'epoch':e,'f1':f,'auc':u} for p,e,f,u in chosen],locked_threshold=thr,threshold_source='Columbia_dedup_val_probability_ensemble');(out/'summary.yaml').write_text(yaml.safe_dump(summary,sort_keys=False))
  test=records(dedupe(split['test']),'test');td=cache_predictions(chosen,test,'test',out);result=aggregate(test,td,'test',out,thr);result.update(dataset='Columbia',threshold=thr,threshold_source=summary['threshold_source'],checkpoint_epochs=[x[1] for x in chosen],policy={'tile':TILE,'stride':STRIDE,'full_max_side':FULL_MAX});(out/'result.yaml').write_text(yaml.safe_dump(result,sort_keys=False));summary.update(status='completed',test_run=True,test=result);(out/'summary.yaml').write_text(yaml.safe_dump(summary,sort_keys=False))
 except BaseException:
  summary.update(status='failed',error=traceback.format_exc());(out/'summary.yaml').write_text(yaml.safe_dump(summary,sort_keys=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);main(p.parse_args())
