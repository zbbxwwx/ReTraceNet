"""Val-only A: DCT selection upper bounds. B: paired CAT-Net error overlap."""
import sys,json,time,traceback,importlib.util,argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import run_e21_jpeg_history_branch as e
OUT=Path(sys.argv[1]); OUT.mkdir(parents=True,exist_ok=True)
BASE=Path('/group4/zouwenxuan/iml_generalization')
CK=BASE/'outputs/retracenet_v2/e24_dher_calibrated_writeback_20260910/best.pt'
summary={'status':'running_A','test_run':False,'threshold':0.5,'oracle_warning':'GT-assisted diagnostic upper bounds, not deployable model scores','block_sizes_original_pixels':[16,32,64]}
e.dump(OUT/'summary.yaml',summary)
def counts(p,y):
 return np.array([(p&y).sum(),(p&~y).sum(),(~p&y).sum()],dtype=np.int64)
def f1(c):
 t,p,n=map(int,c); return 2*t/max(1,2*t+p+n)
def blocks(a,b,y,size):
 rows=[]
 for r in range(0,y.shape[0],size):
  for c in range(0,y.shape[1],size):
   sl=np.s_[r:r+size,c:c+size]; rows.append([counts(a[sl],y[sl]),counts(b[sl],y[sl])])
 return np.asarray(rows)
def optimum(parts):
 x=np.concatenate(parts,axis=0); q=0.
 for _ in range(100):
  score=2*x[:,:,0]-q*(2*x[:,:,0]+x[:,:,1]+x[:,:,2]); idx=score.argmax(1)
  c=x[np.arange(len(x)),idx].sum(0); nq=f1(c)
  if abs(nq-q)<1e-12: break
  q=nq
 return {'f1':nq,'tp':int(c[0]),'fp':int(c[1]),'fn':int(c[2]),'selection':'group-specific pooled-F1 optimal GT-assisted selection'}
def save(): e.dump(OUT/'summary.yaml',summary)
try:
 torch.set_num_threads(4)
 state=torch.load(CK,map_location='cpu'); cfg=state['config']; model=e.build_final_model(cfg['model']).cuda().eval(); model.load_state_dict(state['model'],strict=True)
 records=e.read_manifest(cfg['data']['manifest'],cfg['data']['root'],'val'); assert len(records)==2494
 ds=e.ManipulationDataset(records,image_size=int(cfg['data'].get('image_size',640)),training=False,dct_forensic=True,dct_source_mode='original_jpeg')
 cache=OUT/'paired';cache.mkdir(exist_ok=True)
 sums={}; options={}; start=time.time()
 with torch.inference_mode(),open(OUT/'A_per_image.jsonl','w') as log:
  for i,r in enumerate(records):
   with Image.open(r.image) as im: w,h=im.size
   y=np.zeros((h,w),dtype=bool) if r.mask is None else np.asarray(Image.open(r.mask).convert('L'))>127
   s=ds[i]; batch={k:s[k][None] for k in ['image','dct','qtable','dct_available']}; preds=[]
   for off in [False,True]:
    model.disable_dct_evidence=off; o=e.forward(model,batch); assert bool(float(o['dct_evidence']['disabled']))==off
    preds.append(F.interpolate(o['mask_logits'].float().sigmoid(),size=(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy())
   a,b=[p>=.5 for p in preds]; oracle=np.where(a==y,a,b)
   bucket=e.stratum(r.image,y); groups=['all']+(['fake',bucket] if r.label else [])
   cs={'on':counts(a,y),'off':counts(b,y),'pixel_oracle':counts(oracle,y)}
   opts={'image':np.asarray([[cs['on'],cs['off']]])}
   for size in [16,32,64]: opts[str(size)]=blocks(a,b,y,size)
   for g in groups:
    sums.setdefault(g,{k:np.zeros(3,dtype=np.int64) for k in cs});options.setdefault(g,{k:[] for k in opts})
    for k in cs:sums[g][k]+=cs[k]
    for k in opts:options[g][k].append(opts[k])
   np.savez_compressed(cache/f'{i:04d}.npz',on=preds[0],off=preds[1],gt=y,path=str(r.image))
   log.write(json.dumps({'path':str(r.image),'bucket':bucket,'dct_available':float(s['dct_available'].mean()),'counts':{k:v.tolist() for k,v in cs.items()}})+'\n')
   if i%50==0: print('A',i+1,len(records),'minutes',round((time.time()-start)/60,2),flush=True)
 summary['A']={g:{**{k:{'f1':f1(v),'counts':v.tolist()} for k,v in c.items()},**{k+'_oracle':optimum(options[g][k]) for k in options[g]}} for g,c in sums.items()}
 summary['status']='running_B';save(); print('A_COMPLETE',json.dumps(summary['A']['fake']),flush=True)
 del model,state,ds,options;torch.cuda.empty_cache()
 helper=BASE/'outputs/domain_casia2_eval/catnet_unified1247_best_20260831_015845/export_catnet_unified1247.py'
 spec=importlib.util.spec_from_file_location('cat_reference',helper); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 catck=BASE/'outputs/catnet/splicing_dataset/CAT_full/best.pth.tar'
 m.update_config(m.config,argparse.Namespace(cfg=str(BASE/'repos/CAT-Net/experiments/CAT_full.yaml'),opts=['TEST.MODEL_FILE',str(catck),'TEST.FLIP_TEST','False']))
 rows=[{'image':str(r.image),'mask':str(r.mask) if r.mask else '', 'label':int(r.label)} for r in records]
 ds=m.ManifestDataset(rows,OUT/'cat_jpeg_cache');net=getattr(m.models,m.config.MODEL.NAME).get_seg_model(m.config)
 net.load_state_dict(torch.load(catck,map_location='cpu')['state_dict'],strict=True)
 model=m.FullModel(net,m.CrossEntropy(ignore_label=m.config.TRAIN.IGNORE_LABEL,weight=ds.class_weights).cuda()).cuda().eval()
 overlap={}; catstats=e.ValidationStats()
 with torch.inference_mode(),open(OUT/'B_per_image.jsonl','w') as log:
  for i,r in enumerate(records):
   z=np.load(cache/f'{i:04d}.npz'); y=z['gt'];a=z['on']>=.5;b=z['off']>=.5; h,w=y.shape
   image,label,qt=ds[i];image=image[None].cuda();label=label[None].long().cuda();qt=qt[None].cuda() if torch.is_tensor(qt) else qt
   _,pred=model(image,label,qt)
   # Probability interpolation, then remove official bottom/right padding.
   prob=F.interpolate(pred.float().softmax(1)[:,1:2],size=label.shape[-2:],mode='bilinear',align_corners=False)[0,0].cpu().numpy()[:h,:w]
   target=label[0].cpu().numpy()[:h,:w];assert target.shape==y.shape and np.array_equal(target>0,y),'GT grid mismatch'
   c=prob>=.5;catstats.add(prob,y.astype(np.int16),r.label,r.image)
   ac=a==y;bc=b==y;cc=c==y; bucket=e.stratum(r.image,y)
   masks={'cat_correct_both_e24_wrong':cc&~ac&~bc,'cat_correct_e24_disagree':cc&(ac^bc),'e24_on_correct_cat_wrong':ac&~cc,'cat_correct_e24_on_wrong':cc&~ac,'both_correct':cc&ac,'both_wrong':~cc&~ac}
   row={'path':str(r.image),'bucket':bucket,'cat_counts':counts(c,y).tolist(),'e24_on_counts':counts(a,y).tolist(),'cat_minus_e24_f1':f1(counts(c,y))-f1(counts(a,y))}
   groups=['all']+(['fake',bucket] if r.label else [])
   for k,v in masks.items():
    for cls,region in [('foreground',y),('background',~y)]:
     key=k+'_'+cls;n=int((v&region).sum());row[key]=n
     for g in groups:overlap.setdefault(g,{});overlap[g][key]=overlap[g].get(key,0)+n
   log.write(json.dumps(row)+'\n')
   if i%50==0:print('B',i+1,len(records),flush=True)
 summary.update(status='completed',B=overlap,cat_metrics=catstats.result(),cat_checkpoint=str(catck),cat_input='official RGB/DCT with quality100 conversion for nonJPEG; differs from E24 availability masking',elapsed_minutes=(time.time()-start)/60);save();print('COMPLETE',flush=True)
except BaseException:
 summary.update(status='failed',error=traceback.format_exc());save();raise
