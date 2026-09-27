import copy,sys,tempfile,traceback
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import run_imd_mit_full_v3
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from retrace.data import ManipulationDataset
PARENT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
E3=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e3_fullmodel_jpeg_adaptation1_20260922/qtr_localnorm_epoch01.pt')
E5=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e5_paired_chroma_adaptation_20260923/epoch02.pt')
MANIFEST=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e5_locked_casia2_test_20260923');TH=.31
MODELS=[('E5_epoch02',E5),('Parent_epoch76',PARENT),('E3_epoch01',E3)]
CONDS=[('clean',None,None),('q90_420',90,2),('q80_420',80,2),('q70_420',70,2),('q90_444',90,0),('q80_444',80,0),('q70_444',70,0)]
def save(x):OUT.mkdir(parents=True,exist_ok=True);stable.dump(OUT/'summary.yaml',x)
def jpeg(sample,q,sub):
 vh,vw=map(int,sample['valid_hw'].tolist());fh,fw=sample['image'].shape[-2:];arr=(sample['image'][:,:vh,:vw].permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')
 import jpegio
 with tempfile.TemporaryDirectory(prefix='lockedtest_') as d:
  p=Path(d)/'x.jpg';Image.fromarray(arr).save(p,'JPEG',quality=q,subsampling=sub)
  with Image.open(p) as im:rgb=np.asarray(im.convert('RGB'),dtype='float32')
  j=jpegio.read(str(p));coef=j.coef_arrays[0];qt=j.quant_tables[j.comp_info[0].quant_tbl_no].astype('float32')
 canvas=np.full((fh,fw,3),127.5,np.float32);canvas[:vh,:vw]=rgb;cc=np.zeros((fh,fw),coef.dtype);h=min(fh,coef.shape[0]);w=min(fw,coef.shape[1]);cc[:h,:w]=coef[:h,:w]
 o=dict(sample);o.update(image=torch.from_numpy(canvas).permute(2,0,1)/255,dct=ManipulationDataset._dct_volume_from_coef(cc,fh,fw),qtable=torch.from_numpy(qt[None]),dct_available=torch.ones(1));return o
class View(torch.utils.data.Dataset):
 def __init__(self,b,q,sub):self.base=b;self.q=q;self.sub=sub;self.records=b.records
 def __len__(self):return len(self.base)
 def __getitem__(self,k):return self.base[k] if self.q is None else jpeg(self.base[k],self.q,self.sub)
def counts_init():return {'tp':0,'fp':0,'fn':0,'tn':0}
def add(c,p,y):
 v=y>=0;yy=y[v]>0;pp=p[v]>=TH;c['tp']+=int((pp&yy).sum());c['fp']+=int((pp&~yy).sum());c['fn']+=int((~pp&yy).sum());c['tn']+=int((~pp&~yy).sum())
def metric(c):
 tp,fp,fn,tn=[c[k] for k in ('tp','fp','fn','tn')];return {**c,'threshold':TH,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),'fpr':fp/max(fp+tn,1)}
def evaluate(model,ds,cfg,label):
 model.eval();stats=fair.e.ValidationStats();cnt={'all':counts_init(),'fake':counts_init(),'authentic':counts_init()}
 with torch.no_grad():
  for i,r in enumerate(ds.records):
   b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)));o=stable.v3.fwd(model,b,cfg,False);vh,vw=map(int,b['valid_hw'][0].tolist());p=o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy();y=b['mask'][0,0,:vh,:vw].numpy().astype(np.int16);stats.add(p,y,r.label,r.image);add(cnt['all'],p,y);add(cnt['fake' if r.label else 'authentic'],p,y)
   if i==0 or (i+1)%200==0:print('[test]',label,i+1,len(ds),flush=True)
 raw=stats.result();return {'n_images':len(ds),'locked':{k:metric(v) for k,v in cnt.items()},'auc':{'all_images_pooled':raw['all']['auc'],'fake_only_pooled':raw['fake']['auc']},'oracle_diagnostic':{'all_images_pooled':raw['all']['oracle'],'fake_only_pooled':raw['fake']['oracle']}}
def main():
 par=torch.load(PARENT,map_location='cpu');cfg=copy.deepcopy(par['config']);rows=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'test');base=FairDataset(rows,OUT/'cache_test',False)
 summary={'status':'running','protocol':'locked_one_shot_CASIA2_test','threshold':TH,'threshold_source':'E5_epoch02_CASIA2_clean_val','manifest':str(MANIFEST),'test_images':len(rows),'main_conditions':['clean','q90_420','q80_420','q70_420'],'supplementary_conditions':['q90_444','q80_444','q70_444'],'models':{},'selection_after_test':False}
 if (OUT/'summary.yaml').exists():
  old=yaml.safe_load((OUT/'summary.yaml').read_text());summary['models']=old.get('models',{})
 save(summary)
 try:
  states={'Parent_epoch76':par['model'],'E3_epoch01':torch.load(E3,map_location='cpu')['model'],'E5_epoch02':torch.load(E5,map_location='cpu')['model']}
  for mn,_ in MODELS:
   summary['models'].setdefault(mn,{})
   model=fair.e.build_final_model(cfg['model']).cuda();model.load_state_dict(states[mn],strict=True)
   for cn,q,sub in CONDS:
    if cn in summary['models'][mn]:continue
    z=evaluate(model,View(base,q,sub),cfg,mn+'_'+cn);summary['models'][mn][cn]=z;save(summary)
   del model;torch.cuda.empty_cache()
  summary['status']='completed_locked_no_further_selection';save(summary);print(yaml.safe_dump(summary,sort_keys=False),flush=True)
 except BaseException:
  summary['status']='failed';summary['traceback']=traceback.format_exc();save(summary);raise
if __name__=='__main__':main()
