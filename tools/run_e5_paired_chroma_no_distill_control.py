import copy,json,random,sys,tempfile,time,traceback
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,yaml
from PIL import Image
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import run_imd_mit_full_v3
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from retrace.data import ManipulationDataset
E3=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e3_fullmodel_jpeg_adaptation1_20260922/qtr_localnorm_epoch01.pt')
PARENT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
MANIFEST=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e5_paired_chroma_no_distill_control_20260924')
SEED=3407;EPOCHS=2;LR=1e-5;LAMBDA=0.0;CONF=.7
def dump(n,x):OUT.mkdir(parents=True,exist_ok=True);stable.dump(OUT/n,x)
def jpeg(sample,sub):
 vh,vw=map(int,sample['valid_hw'].tolist());fh,fw=sample['image'].shape[-2:];arr=(sample['image'][:,:vh,:vw].permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')
 import jpegio
 with tempfile.TemporaryDirectory(prefix='e5_') as d:
  p=Path(d)/'x.jpg';Image.fromarray(arr).save(p,'JPEG',quality=90,subsampling=sub)
  with Image.open(p) as im:rgb=np.asarray(im.convert('RGB'),dtype='float32')
  j=jpegio.read(str(p));coef=j.coef_arrays[0];qt=j.quant_tables[j.comp_info[0].quant_tbl_no].astype('float32')
 canvas=np.full((fh,fw,3),127.5,np.float32);canvas[:vh,:vw]=rgb;cc=np.zeros((fh,fw),coef.dtype);h=min(fh,coef.shape[0]);w=min(fw,coef.shape[1]);cc[:h,:w]=coef[:h,:w]
 o=dict(sample);o.update(image=torch.from_numpy(canvas).permute(2,0,1)/255,dct=ManipulationDataset._dct_volume_from_coef(cc,fh,fw),qtable=torch.from_numpy(qt[None]),dct_available=torch.ones(1));return o
class Pair(torch.utils.data.Dataset):
 def __init__(self,b):self.base=b;self.records=b.records
 def __len__(self):return len(self.base)
 def __getitem__(self,k):
  s=self.base[k];stu=jpeg(s,2)
  return stu
class Fixed(torch.utils.data.Dataset):
 def __init__(self,b,sub=None):self.base=b;self.sub=sub;self.records=b.records
 def __len__(self):return len(self.base)
 def __getitem__(self,k):return self.base[k] if self.sub is None else jpeg(self.base[k],self.sub)
def teacher_batch(b):
 x=dict(b)
 for n in ('image','dct','qtable','dct_available'):x[n]=b['teacher_'+n]
 return x
def eval_ds(model,ds,cfg,label,locked=None):
 model.eval();stats=fair.e.ValidationStats();tp=fp=fn=tn=0
 with torch.no_grad():
  for i,r in enumerate(ds.records):
   b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)));o=stable.v3.fwd(model,b,cfg,False);vh,vw=map(int,b['valid_hw'][0].tolist());p=o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy();y=b['mask'][0,0,:vh,:vw].numpy().astype(np.int16);stats.add(p,y,r.label,r.image)
   if locked is not None and r.label==1 and (y>0).any():
    v=y>=0;yy=y[v]>0;pp=p[v]>=locked;tp+=int((pp&yy).sum());fp+=int((pp&~yy).sum());fn+=int((~pp&yy).sum());tn+=int((~pp&~yy).sum())
   if i==0 or (i+1)%200==0:print('[val]',label,i+1,len(ds),flush=True)
 m=stats.result()['fake'];out={'auc':float(m['auc']),'oracle':{k:float(m['oracle'][k]) for k in ('threshold','precision','recall','f1')}}
 if locked is not None:out['clean_locked']={'threshold':float(locked),'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),'tp':tp,'fp':fp,'fn':fn,'tn':tn}
 dump(label+'.yaml',out);return out
def main():
 summary={'status':'preparing','experiment':'E5_Paired_Chroma_NoDistill_Control','checkpoint_init':str(E3),'epochs':EPOCHS,'lr':LR,'teacher':'disabled','student':'E3 Q90 4:2:0','distill':'disabled matched-budget control','distill_weight':LAMBDA,'dct_branch':'original/native per view; student DCT branch frozen','test_read':False,'history':[]};dump('summary.yaml',summary)
 try:
  random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);par=torch.load(PARENT,map_location='cpu');ck=torch.load(E3,map_location='cpu');cfg=copy.deepcopy(par['config']);cfg['training']['dct_sample_dropout']=0.
  tr=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'train');va=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'val');student=fair.e.build_final_model(cfg['model']).cuda();student.load_state_dict(ck['model'],strict=True)
  for n,p in student.named_parameters():p.requires_grad_(not n.startswith('dct_stream'))
  trainable=[p for p in student.parameters() if p.requires_grad];opt=torch.optim.SGD(trainable,lr=LR,momentum=.9,weight_decay=.0005);seg=stable.v3.e24.ReTraceLoss.from_config(stable.v3.e24.clean_loss_config(cfg['loss'])).cuda()
  trainbase=FairDataset(tr,OUT/'cache_train',True);valbase=FairDataset(va,OUT/'cache_val',False);fake=[i for i,r in enumerate(tr) if r.label];auth=[i for i,r in enumerate(tr) if not r.label];nf=min(len(fake),len(auth))
  for ep in range(1,EPOCHS+1):
   g=torch.Generator().manual_seed(SEED+ep);ids=[fake[i] for i in torch.randperm(len(fake),generator=g)[:nf]]+[auth[i] for i in torch.randperm(len(auth),generator=g)[:nf]];ids=[ids[i] for i in torch.randperm(len(ids),generator=g)];plan=[(idx,SEED*1000000+ep*100000+j) for j,idx in enumerate(ids)];loader=DataLoader(Pair(trainbase),batch_size=4,sampler=plan,num_workers=2,pin_memory=True)
   stable.train_mode(student)
   for n,p in student.named_parameters():p.requires_grad_(not n.startswith('dct_stream'))
   total=mainloss=distloss=maskfrac=0.;start=time.time()
   for st,b in enumerate(loader,1):
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):so=stable.v3.fwd(student,b,cfg,False);lm=stable.v3.objective(so,b,seg,cfg)['total']
    ld=so['mask_logits'].new_zeros(());loss=lm
    loss.backward();torch.nn.utils.clip_grad_norm_(trainable,5.,error_if_nonfinite=True);opt.step();total+=float(loss);mainloss+=float(lm)
    if st==1 or st%50==0 or st==len(loader):print('[no-distill]',ep,st,len(loader),total/st,mainloss/st,flush=True)
   cp=OUT/f'epoch{ep:02d}.pt';torch.save({'epoch':ep,'parent':'E3','model':student.state_dict(),'optimizer':opt.state_dict(),'experiment':'E5_Paired_Chroma_NoDistill_Control'},cp)
   clean=eval_ds(student,Fixed(valbase,None),cfg,f'ep{ep:02d}_clean');thr=clean['oracle']['threshold'];q444=eval_ds(student,Fixed(valbase,0),cfg,f'ep{ep:02d}_q90_444',thr);q420=eval_ds(student,Fixed(valbase,2),cfg,f'ep{ep:02d}_q90_420',thr)
   row={'epoch':ep,'checkpoint':str(cp),'train_total':total/len(loader),'train_main':mainloss/len(loader),'train_distill':distloss/len(loader),'reliable_pixel_fraction':maskfrac/len(loader),'clean':clean,'q90_444':q444,'q90_420':q420,'elapsed_minutes':(time.time()-start)/60};summary['history'].append(row);summary.update(status='epoch_completed',epoch=ep);dump('summary.yaml',summary)
  summary.update(status='completed_2ep_gate',decision='review_before_continue_or_control');dump('summary.yaml',summary);print(yaml.safe_dump(summary,sort_keys=False),flush=True)
 except BaseException:
  summary.update(status='failed',traceback=traceback.format_exc());dump('summary.yaml',summary);raise
if __name__=='__main__':main()
