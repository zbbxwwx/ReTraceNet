"""IMD2020 Context-Core pilot: 768 context input, center 512 supervision/output."""
import argparse,copy,hashlib,json,math,random,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,torch,yaml
from PIL import Image
from torch.utils.data import DataLoader
from retrace.data import Record,ManipulationDataset
import run_balanced_residual_v3_full30 as v3
import run_cross_dataset_after_chain as cross
from fair_training_data import crop_outputs

BASE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')
def dump(p,x):v3.e.dump(Path(p),x)
def recs(rows,sp):return [Record(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') else None,split=sp,dataset='IMD2020',label=int(r['label'])) for r in rows]
def freeze_bn(m):
 n=0
 for branch in (m.content_encoder,m.dct_branch):
  for x in branch.modules():
   if isinstance(x,torch.nn.modules.batchnorm._BatchNorm):x.eval();n+=1
 return n
def train_mode(m):m.train();return freeze_bn(m)

class ContextCoreDataset(torch.utils.data.Dataset):
 def __init__(self,records,cache,context=768,core=512):self.records=records;self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True);self.context=context;self.core=core;self.margin=(context-core)//2;assert context%32==0 and core%32==0 and self.margin%8==0
 def __len__(self):return len(self.records)
 def __getitem__(self,key):
  if isinstance(key,tuple):idx,seed=key
  else:idx,seed=key,0
  rng=random.Random(seed);r=self.records[idx];jpeg=r.image;generated=jpeg.suffix.lower() not in {'.jpg','.jpeg'}
  if generated:
   jpeg=self.cache/(hashlib.sha256(str(r.image).encode()).hexdigest()+'.jpg')
   if not jpeg.exists():
    with Image.open(r.image) as im:im.convert('RGB').save(jpeg,quality=100,subsampling=0)
  import jpegio;j=jpegio.read(str(jpeg));coef=j.coef_arrays[0];q=j.quant_tables[j.comp_info[0].quant_tbl_no].astype('float32')
  with Image.open(jpeg) as im:rgb=np.asarray(im.convert('RGB'),dtype=np.float32)
  h,w=rgb.shape[:2];gt=np.zeros((h,w),np.float32) if r.mask is None else (np.asarray(Image.open(r.mask).convert('L'))>127).astype('float32');assert gt.shape==(h,w)
  top=(rng.randint(0,max(h-self.core,0))//8)*8;left=(rng.randint(0,max(w-self.core,0))//8)*8;m=self.margin
  canvas=np.full((h+2*m,w+2*m,3),127.5,np.float32);canvas[m:m+h,m:m+w]=rgb
  target=np.zeros((h+2*m,w+2*m),np.float32);target[m:m+h,m:m+w]=gt
  cc=np.zeros((max(h,coef.shape[0])+2*m,max(w,coef.shape[1])+2*m),dtype=coef.dtype);cc[m:m+coef.shape[0],m:m+coef.shape[1]]=coef
  image=canvas[top:top+self.context,left:left+self.context];mask=target[top:top+self.context,left:left+self.context];co=cc[top:top+self.context,left:left+self.context]
  # Arrays can be short only for tiny images; right/bottom pad without changing block phase.
  def pad(a,shape,val=0):
   z=np.full(shape,val,dtype=a.dtype);z[:a.shape[0],:a.shape[1],...]=a;return z
  if image.shape[:2]!=(self.context,self.context):image=pad(image,(self.context,self.context,3),127.5)
  if mask.shape!=(self.context,self.context):mask=pad(mask,(self.context,self.context),0)
  if co.shape!=(self.context,self.context):co=pad(co,(self.context,self.context),0)
  vh,vw=min(max(h-top,0),self.core),min(max(w-left,0),self.core);coremask=mask[m:m+self.core,m:m+self.core]
  return {'image':torch.from_numpy(image.copy()).permute(2,0,1)/255,'mask':torch.from_numpy(mask.copy())[None],'valid_hw':torch.tensor([self.context,self.context]),'core_valid_hw':torch.tensor([vh,vw]),'label':torch.tensor([float(coremask.any())]),'original_label':torch.tensor([float(r.label)]),'path':str(r.image),'dct':ManipulationDataset._dct_volume_from_coef(co,self.context,self.context),'qtable':torch.from_numpy(q[None]),'dct_available':torch.ones(1),'crop_origin':torch.tensor([top,left]),'core_margin':torch.tensor(m)}

def center_crop(v,bs,context,core):
 if torch.is_tensor(v):
  if v.ndim and v.shape[0]==bs:v=v
  if v.ndim==4:
   h,w=v.shape[-2:];m=(context-core)//2;y0=m*h//context;y1=(m+core)*h//context;x0=m*w//context;x1=(m+core)*w//context;return v[...,y0:y1,x0:x1]
  return v
 if isinstance(v,dict):return {k:center_crop(x,bs,context,core) for k,x in v.items()}
 if isinstance(v,list):return [center_crop(x,bs,context,core) for x in v]
 if isinstance(v,tuple):return tuple(center_crop(x,bs,context,core) for x in v)
 return v
def objective(out,b,seg,cfg,context,core):
 bs=b['image'].shape[0];oo=center_crop(out,bs,context,core);m=(context-core)//2
 bb=dict(b);bb['image']=b['image'][...,m:m+core,m:m+core];bb['mask']=b['mask'][...,m:m+core,m:m+core];bb['valid_hw']=b['core_valid_hw'];return v3.objective(oo,bb,seg,cfg)

def starts(n,core=512,stride=384):
 if n<=core:return [0]
 x=list(range(0,n-core+1,stride));last=((n-core)//8)*8
 if x[-1]!=last:x.append(last)
 return x
@torch.inference_mode()
def predict(model,r,cache,context=768,core=512):
 m=(context-core)//2;jpeg=r.image;cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
 if jpeg.suffix.lower() not in {'.jpg','.jpeg'}:
  jpeg=cache/(hashlib.sha256(str(r.image).encode()).hexdigest()+'.jpg')
  if not jpeg.exists():
   with Image.open(r.image) as im:im.convert('RGB').save(jpeg,quality=100,subsampling=0)
 import jpegio;j=jpegio.read(str(jpeg));coef=j.coef_arrays[0];q=j.quant_tables[j.comp_info[0].quant_tbl_no].astype('float32')
 with Image.open(jpeg) as im:rgb=np.asarray(im.convert('RGB'),np.float32)
 h,w=rgb.shape[:2];gt=np.zeros((h,w),np.int16) if r.mask is None else (np.asarray(Image.open(r.mask).convert('L'))>127).astype(np.int16);assert gt.shape==(h,w)
 ph=max(core,((h+31)//32)*32);pw=max(core,((w+31)//32)*32);canvas=np.full((ph+2*m,pw+2*m,3),127.5,np.float32);canvas[m:m+h,m:m+w]=rgb;cc=np.zeros((ph+2*m,pw+2*m),coef.dtype);cc[m:m+coef.shape[0],m:m+coef.shape[1]]=coef
 acc=torch.zeros((ph,pw));weight=torch.zeros((ph,pw));scores=[]
 for top in starts(ph,core):
  for left in starts(pw,core):
   im=canvas[top:top+context,left:left+context];co=cc[top:top+context,left:left+context];b={'image':torch.from_numpy(im.copy()).permute(2,0,1)[None]/255,'dct':ManipulationDataset._dct_volume_from_coef(co,context,context)[None],'qtable':torch.from_numpy(q[None,None]),'dct_available':torch.ones((1,1))};o=v3.fwd(model,b);p=o['mask_logits'].float().sigmoid()[0,0,m:m+core,m:m+core].cpu();acc[top:top+core,left:left+core]+=p;weight[top:top+core,left:left+core]+=1;scores.append(float(o['image_logit'].float().sigmoid().mean()))
 return (acc/weight.clamp_min(1))[:h,:w].numpy(),gt,float(np.mean(sorted(scores,reverse=True)[:max(1,math.ceil(.2*len(scores)))]))
@torch.inference_mode()
def evaluate(model,records,cache,out,context,core,locked_threshold=None):
 model.eval();out=Path(out);out.mkdir(parents=True,exist_ok=True);st=cross.ValidationStats();locked=cross.LockedCounts(locked_threshold) if locked_threshold is not None else None
 for i,r in enumerate(records):
  p,y,_=predict(model,r,cache,context,core);st.add(p,y,r.label,r.image)
  if r.label and locked is not None:locked.add(p,y)
  if i==0 or (i+1)%100==0:print('[ctx-val]',i+1,len(records),flush=True)
 z=st.result();result={'diagnostic':z,'pixel_localization_fake_only':{'test_f1_locked':locked.result() if locked is not None else dict(z['fake']['oracle']),'auc':z['fake']['auc'],'test_f1_oracle_diagnostic':z['fake']['oracle']},'threshold_source':'IMD2020_val' if locked is not None else 'IMD2020_val_current','test_read':locked is not None};dump(Path(out)/'IMD2020.yaml',result);return result

def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);summary={'status':'preparing','experiment':'IMD2020 Context-Core','context':a.context,'supervised_core':512,'test_run':False};dump(out/'summary.yaml',summary)
 try:
  split=json.loads(Path(a.split).read_text());tr,va=(recs(split[k],k) for k in ('train','val'));template=torch.load(a.template,map_location='cpu');cfg=copy.deepcopy(template['config']);cfg['training']['dct_sample_dropout']=.2
  torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);model=v3.e.build_final_model(cfg['model']).cuda();bn=freeze_bn(model);seg=v3.e24.ReTraceLoss.from_config(v3.e24.clean_loss_config(cfg['loss'])).cuda();opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=.0005)
  ds=ContextCoreDataset(tr,out/'jpeg_cache_train',a.context,512);fake=[i for i,r in enumerate(tr) if r.label];auth=[i for i,r in enumerate(tr) if not r.label];nf=min(len(fake),len(auth));steps=math.ceil(2*nf/a.batch_size/a.accum);total=a.epochs*steps;gs=0;history=[];best=-1;bestep=0
  summary.update(status='training',epochs=a.epochs,batch_size=a.batch_size,gradient_accumulation=a.accum,effective_batch=a.batch_size*a.accum,bn_stats_frozen=True,bn_module_count=bn,history=history,best_F1=best);dump(out/'summary.yaml',summary)
  for ep in range(1,a.epochs+1):
   g=torch.Generator().manual_seed(a.seed+ep);ids=[fake[i] for i in torch.randperm(len(fake),generator=g)[:nf]]+[auth[i] for i in torch.randperm(len(auth),generator=g)[:nf]];ids=[ids[i] for i in torch.randperm(len(ids),generator=g)];plan=[(x,a.seed*1000000+ep*100000+j) for j,x in enumerate(ids)];ld=DataLoader(ds,batch_size=a.batch_size,sampler=plan,num_workers=2,pin_memory=True);train_mode(model);opt.zero_grad(set_to_none=True);tot=0
   for st,b in enumerate(ld,1):
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=objective(v3.fwd(model,b,cfg,True),b,seg,cfg,a.context,512)['total']/a.accum
    loss.backward();tot+=float(loss)*a.accum
    if st%a.accum==0 or st==len(ld):
     lr=.001*(1-min(gs/max(total,1),1))**.9
     for q in opt.param_groups:q['lr']=lr
     torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);gs+=1
    if st==1 or st%25==0:print('[ctx-train]',ep,st,len(ld),tot/st,flush=True)
   vr=evaluate(model,va,out/'jpeg_cache_val',out/f'val_epoch{ep:03d}',a.context,512);q=vr['diagnostic']['fake']['oracle'];score=float(q['f1']);row={'epoch':ep,'val_fake_f1':score,'val_fake_auc':vr['diagnostic']['fake']['auc'],'precision':q['precision'],'recall':q['recall'],'threshold':q['threshold']};history.append(row)
   if score>best:best=score;bestep=ep;torch.save({'model':v3.e.copy_state(model),'config':cfg,'epoch':ep,'selection_threshold':q['threshold']},out/'best.pt')
   torch.save({'model':v3.e.copy_state(model),'optimizer':opt.state_dict(),'config':cfg,'epoch':ep,'history':history},out/'last.pt');summary.update(history=history,best_F1=best,best_epoch=bestep);dump(out/'summary.yaml',summary)
   if ep>=10 and max(x['val_fake_f1'] for x in history[-3:])<0.4933455107:summary['stop_reason']='gate_epoch10_delta_under_1pp';break
  if a.test and summary.get('stop_reason')!='gate_epoch10_delta_under_1pp':
   bc=torch.load(out/'best.pt',map_location='cpu');model.load_state_dict(bc['model'],strict=True);te=recs(split['test'],'test');summary['test']=evaluate(model,te,out/'jpeg_cache_test',out/'test_once',a.context,512,float(bc['selection_threshold']));summary['test_run']=True
  summary.update(status='completed',stop_reason=summary.get('stop_reason','fixed_budget_complete'));dump(out/'summary.yaml',summary)
 except BaseException:summary.update(status='failed',stop_reason=traceback.format_exc());dump(out/'summary.yaml',summary);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--template',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=15);p.add_argument('--context',type=int,default=768);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--accum',type=int,default=2);p.add_argument('--seed',type=int,default=1337);p.add_argument('--test',action='store_true');main(p.parse_args())
