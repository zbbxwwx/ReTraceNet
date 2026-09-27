"""IMD in-domain diagnosis: full train-fake eval plus weighted loss audit."""
import json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch,yaml
from PIL import Image
from torch.utils.data import DataLoader
from retrace.data import Record
import run_balanced_residual_v3_full30 as v3
import run_cross_dataset_after_chain as cross
from fair_training_data import FairDataset

BASE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')
RUN=BASE/'imd2020_stable_bn_v1_full50_20260913';OUT=BASE/'imd2020_indomain_diagnosis_20260915';OUT.mkdir(parents=True,exist_ok=True)
SPLIT=BASE/'v3_offline_master_20260912/indomain_IMD2020/split_721.json'
def recs(rows,sp):return [Record(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') else None,split=sp,dataset='IMD2020',label=int(r['label'])) for r in rows]
def freeze_bn(model):
 for branch in (model.content_encoder,model.dct_branch):
  for m in branch.modules():
   if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
def metadata(rows):
 out={'count':len(rows),'fake':sum(int(x['label']) for x in rows),'authentic':sum(not int(x['label']) for x in rows),'jpeg_fake':0,'nonjpeg_fake':0,'sliding_fake':0,'ratios':[],'resolutions':[]}
 for r in rows:
  if not int(r['label']):continue
  ext=Path(r['image']).suffix.lower();out['jpeg_fake']+=ext in ('.jpg','.jpeg');out['nonjpeg_fake']+=ext not in ('.jpg','.jpeg')
  with Image.open(r['image']) as im:w,h=im.size
  out['sliding_fake']+=max(h,w)>1024;out['resolutions'].append(h*w)
  gt=np.asarray(Image.open(r['mask']).convert('L'))>127;out['ratios'].append(float(gt.mean()))
 rr=np.asarray(out.pop('ratios'));px=np.asarray(out.pop('resolutions'));out.update(foreground_ratio_mean=float(rr.mean()),foreground_ratio_median=float(np.median(rr)),foreground_ratio_bins={'lt1':int((rr<.01).sum()),'1_5':int(((rr>=.01)&(rr<.05)).sum()),'5_20':int(((rr>=.05)&(rr<.2)).sum()),'gte20':int((rr>=.2).sum())},resolution_pixels_median=float(np.median(px)),resolution_bins={'le1MP':int((px<=1_000_000).sum()),'1_4MP':int(((px>1_000_000)&(px<=4_000_000)).sum()),'gt4MP':int((px>4_000_000).sum())});return out

split=json.loads(SPLIT.read_text());tr,va,te=(recs(split[k],k) for k in ('train','val','test'))
ck=torch.load(RUN/'best.pt',map_location='cpu');model=v3.e.build_final_model(ck['config']['model']).cuda();model.load_state_dict(ck['model'],strict=True);model.eval()
summary={'status':'evaluating_train_fake','checkpoint_epoch':int(ck['epoch']),'val_locked_threshold':float(ck['selection_threshold']),'counts':{k:metadata(split[k]) for k in ('train','val','test')}};v3.e.dump(OUT/'summary.yaml',summary)
train_fake=[r for r in tr if r.label]
train_result=cross.evaluate_dataset(model,train_fake,OUT/'jpeg_cache_train',None,None,'IMD2020_train_fake',OUT/'train_fake_eval',evaluation_mode='diagnostic_train_fake_eval_forced',threshold_source='diagnostic_oracle_only')
summary['train_fake']=train_result['pixel_localization_fake_only'];summary['status']='auditing_losses';v3.e.dump(OUT/'summary.yaml',summary)
ds=FairDataset(tr,OUT/'jpeg_cache_loss',True);fake=[i for i,r in enumerate(tr) if r.label];auth=[i for i,r in enumerate(tr) if not r.label];nf=min(len(fake),len(auth));g=torch.Generator().manual_seed(1337+49);ids=[fake[i] for i in torch.randperm(len(fake),generator=g)[:nf]]+[auth[i] for i in torch.randperm(len(auth),generator=g)[:nf]];ids=[ids[i] for i in torch.randperm(len(ids),generator=g)][:80];plan=[(idx,1337*1000000+49*100000+j) for j,idx in enumerate(ids)];loader=DataLoader(ds,batch_size=4,sampler=plan,num_workers=2,pin_memory=True)
model.train();freeze_bn(model);seg=v3.e24.ReTraceLoss.from_config(v3.e24.clean_loss_config(ck['config']['loss'])).cuda();sums={};n=0
for b in loader:
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):terms=v3.objective(v3.fwd(model,b,ck['config'],True),b,seg,ck['config'])
 for k,v in terms.items():sums[k]=sums.get(k,0.)+float(v)
 n+=1
summary['weighted_loss_components_20_batches']={k:v/n for k,v in sums.items()};summary['status']='completed';v3.e.dump(OUT/'summary.yaml',summary);print(yaml.safe_dump(summary,sort_keys=False),flush=True)
