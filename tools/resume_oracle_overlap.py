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
unused={'status':'running_A','test_run':False,'threshold':0.5,'oracle_warning':'GT-assisted diagnostic upper bounds, not deployable model scores','block_sizes_original_pixels':[16,32,64]}
import yaml
summary=yaml.safe_load((OUT/'summary.yaml').read_text())
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
def save(): import yaml
summary=yaml.safe_load((OUT/'summary.yaml').read_text())
try:
 torch.set_num_threads(4)
 state=torch.load(CK,map_location='cpu');cfg=state['config'];del state
 records=e.read_manifest(cfg['data']['manifest'],cfg['data']['root'],'val');cache=OUT/'paired';start=time.time()
 summary['status']='resuming_B';summary['previous_error']=summary.pop('error',None);save()
 helper=BASE/'outputs/domain_casia2_eval/catnet_unified1247_best_20260831_015845/export_catnet_unified1247.py'
 spec=importlib.util.spec_from_file_location('cat_reference',helper); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 catck=BASE/'outputs/catnet/splicing_dataset/CAT_full/best.pth.tar'
 m.update_config(m.config,argparse.Namespace(cfg=str(BASE/'repos/CAT-Net/experiments/CAT_full.yaml'),opts=['TEST.MODEL_FILE',str(catck),'TEST.FLIP_TEST','False']))
 rows=[{'image':str(r.image),'mask':str(r.mask) if r.mask else '', 'label':int(r.label)} for r in records]
 class SharedGT(m.ManifestDataset):
  def get_tamp(self,index):
   r=self.rows[index];z=np.load(cache/f'{index:04d}.npz');jpg=m.make_jpeg_for_catnet(r['image'],self.jpg_cache_dir)
   return self._create_tensor(jpg,z['gt'].astype(np.uint8))
 ds=SharedGT(rows,OUT/'cat_jpeg_cache');net=getattr(m.models,m.config.MODEL.NAME).get_seg_model(m.config)
 net.load_state_dict(torch.load(catck,map_location='cpu')['state_dict'],strict=True)
 model=m.FullModel(net,m.CrossEntropy(ignore_label=m.config.TRAIN.IGNORE_LABEL,weight=ds.class_weights).cuda()).cuda().eval()
 overlap={}; catcounts={}
 prior=[json.loads(x) for x in (OUT/'B_per_image.jsonl').read_text().splitlines()]
 def accumulate(row,r):
  groups=['all']+(['fake',row['bucket']] if r.label else [])
  for g in groups:
   catcounts.setdefault(g,np.zeros(3,dtype=np.int64));catcounts[g]+=row['cat_counts']
   for k,v in row.items():
    if k.endswith('_foreground') or k.endswith('_background'):overlap.setdefault(g,{});overlap[g][k]=overlap[g].get(k,0)+v
 for i,row in enumerate(prior):
  assert row['path']==str(records[i].image);accumulate(row,records[i])
 print('RESUME_B',len(prior),flush=True)
 with torch.inference_mode(),open(OUT/'B_per_image.jsonl','a') as log:
  for i,r in enumerate(records):
   if i<len(prior):continue
   z=np.load(cache/f'{i:04d}.npz'); y=z['gt'];a=z['on']>=.5;b=z['off']>=.5; h,w=y.shape
   image,label,qt=ds[i];image=image[None].cuda();label=label[None].long().cuda();qt=qt[None].cuda() if torch.is_tensor(qt) else qt
   _,pred=model(image,label,qt)
   # Probability interpolation, then remove official bottom/right padding.
   prob=F.interpolate(pred.float().softmax(1)[:,1:2],size=label.shape[-2:],mode='bilinear',align_corners=False)[0,0].cpu().numpy()[:h,:w]
   target=label[0].cpu().numpy()[:h,:w];assert target.shape==y.shape and np.array_equal(target>0,y),'GT grid mismatch'
   c=prob>=.5
   ac=a==y;bc=b==y;cc=c==y; bucket=e.stratum(r.image,y)
   masks={'cat_correct_both_e24_wrong':cc&~ac&~bc,'cat_correct_e24_disagree':cc&(ac^bc),'e24_on_correct_cat_wrong':ac&~cc,'cat_correct_e24_on_wrong':cc&~ac,'both_correct':cc&ac,'both_wrong':~cc&~ac}
   row={'path':str(r.image),'bucket':bucket,'cat_counts':counts(c,y).tolist(),'e24_on_counts':counts(a,y).tolist(),'cat_minus_e24_f1':f1(counts(c,y))-f1(counts(a,y))}
   groups=['all']+(['fake',bucket] if r.label else [])
   for k,v in masks.items():
    for cls,region in [('foreground',y),('background',~y)]:
     key=k+'_'+cls;n=int((v&region).sum());row[key]=n
   accumulate(row,r)
   log.write(json.dumps(row)+'\n')
   if i%50==0:print('B',i+1,len(records),flush=True)
 summary.update(status='completed',B=overlap,cat_fixed_metrics={g:{'f1':f1(c),'counts':c.tolist()} for g,c in catcounts.items()},auc_note='AUC not reconstructed from resumed confusion counts; B uses fixed threshold overlap',cat_checkpoint=str(catck),cat_input='official RGB/DCT with quality100 conversion for nonJPEG; differs from E24 availability masking',elapsed_minutes=(time.time()-start)/60);save();print('COMPLETE',flush=True)
except BaseException:
 summary.update(status='failed',error=traceback.format_exc());save();raise
