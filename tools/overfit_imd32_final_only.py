"""IMD 32-image final-mask-only overfit diagnostic."""
import copy,json,math,random,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch,torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from retrace.data import Record
import run_balanced_residual_v3_full30 as v3
import run_cross_dataset_after_chain as cross
from fair_training_data import FairDataset
BASE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2');SRC=BASE/'imd2020_stable_bn_v1_full50_20260913';OUT=BASE/'imd2020_overfit32_final_only_20260915';OUT.mkdir(parents=True,exist_ok=True)
def freeze_bn(m):
 for branch in (m.content_encoder,m.dct_branch):
  for x in branch.modules():
   if isinstance(x,torch.nn.modules.batchnorm._BatchNorm):x.eval()
def train_mode(m):m.train();freeze_bn(m)
def batch(ds,ids,seed):return next(iter(DataLoader(ds,batch_size=len(ids),sampler=[(x,seed+i) for i,x in enumerate(ids)],num_workers=0)))
state={'status':'preparing','loss':'final mask BCE + Dice only','initialization':'public Content/DCT; localization chain random'};v3.e.dump(OUT/'summary.yaml',state)
try:
 split=json.loads((BASE/'v3_offline_master_20260912/indomain_IMD2020/split_721.json').read_text());rows=[r for r in split['train'] if int(r['label'])];eligible=[]
 for r in rows:
  with Image.open(r['image']) as im:w,h=im.size
  if h<=512 and w<=512:eligible.append((r,h,w))
 random.Random(3407).shuffle(eligible);chosen=eligible[:32];assert len(chosen)==32
 recs=[Record(image=Path(r['image']),mask=Path(r['mask']),split='train',dataset='IMD2020',label=1) for r,_,_ in chosen];(OUT/'manifest.json').write_text(json.dumps([{'image':r['image'],'mask':r['mask'],'h':h,'w':w} for r,h,w in chosen],indent=2))
 template=torch.load(SRC/'best.pt',map_location='cpu');cfg=copy.deepcopy(template['config']);torch.manual_seed(3407);np.random.seed(3407);random.seed(3407);model=v3.e.build_final_model(cfg['model']).cuda();train_mode(model);opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=.0005);ds=FairDataset(recs,OUT/'jpeg_cache',True)
 hist=[]
 for step in range(1,1201):
  ids=[((step-1)*4+j)%32 for j in range(4)];b=batch(ds,ids,900000+step*10);opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   out=v3.fwd(model,b,cfg,False);loss=0.
   for i in range(4):
    vh,vw=map(int,b['valid_hw'][i].tolist());z=out['mask_logits'][i:i+1,:,:vh,:vw].float();y=b['mask'][i:i+1,:,:vh,:vw].float().cuda();bce=F.binary_cross_entropy_with_logits(z,y);p=z.sigmoid();inter=(p*y).sum();dice=1-(2*inter+1)/(p.sum()+y.sum()+1);loss=loss+(bce+dice)/4
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
  if step==1 or step%25==0:v3.e.dump(OUT/'progress.yaml',{'step':step,'loss':float(loss)});print('[imd-overfit32-final]',step,float(loss),flush=True)
  if step%100==0:
   model.eval();res=cross.evaluate_dataset(model,recs,OUT/'jpeg_cache',None,None,'IMD_OVERFIT32',OUT/f'eval_{step:04d}',evaluation_mode='diagnostic_final_only',threshold_source='oracle_diagnostic');q=res['pixel_localization_fake_only']['test_f1_oracle_diagnostic'];row={'step':step,'loss':float(loss),'f1':q['f1'],'precision':q['precision'],'recall':q['recall'],'auc':res['pixel_localization_fake_only']['auc'],'threshold':q['threshold']};hist.append(row);state.update(status='running',history=hist);v3.e.dump(OUT/'summary.yaml',state)
   if q['f1']>=.9:state.update(status='completed',overfit_pass=True,stop_reason='F1 reached 0.90');v3.e.dump(OUT/'summary.yaml',state);break
   train_mode(model)
 else:state.update(status='completed',overfit_pass=False,stop_reason='1200 steps without F1 0.90');v3.e.dump(OUT/'summary.yaml',state)
except BaseException:state.update(status='failed',stop_reason=traceback.format_exc());v3.e.dump(OUT/'summary.yaml',state);raise
