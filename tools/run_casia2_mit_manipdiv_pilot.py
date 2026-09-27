"""CASIA2-train-only manipulation-diversity pilot on the current MiT-V3.

Exactly half of training accesses remain original CASIA2 samples.  The other
half synthesize splicing/copy-move from CASIA2 train authentic images only,
with balanced manipulation-area buckets.  Validation remains untouched.
"""
import json, math, random, sys, tempfile, traceback
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_imd_mit_full_v3 as mit  # patches current MiT-V3 builder
import run_v3_stable_bn_indomain as stable
from fair_training_data import FairDataset as BaseFairDataset
from retrace.data import ManipulationDataset

OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_mit_manipdiv_2k_pilot_v3_20260921')
SPLIT=OUT/'casia2_manifest_split.json'
MANIFEST=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
TEMPLATE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
AREA=(('lt_1',.003,.010),('1_5',.010,.050),('5_20',.050,.200),('gte_20',.200,.380))

def to_pil(t,h,w):return Image.fromarray((t[:,:h,:w].permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8'))
def jpeg_inputs(sample,quality,subsampling):
 vh,vw=map(int,sample['valid_hw'].tolist());fh,fw=sample['image'].shape[-2:];im=to_pil(sample['image'],vh,vw)
 import jpegio
 with tempfile.TemporaryDirectory(prefix='manipdiv_') as d:
  p=Path(d)/'x.jpg';im.save(p,quality=int(quality),subsampling=int(subsampling));
  with Image.open(p) as dec:pix=np.asarray(dec.convert('RGB'),dtype='float32')
  jpg=jpegio.read(str(p));coef0=jpg.coef_arrays[0];qt=jpg.quant_tables[jpg.comp_info[0].quant_tbl_no].astype('float32')
 rgb=np.full((fh,fw,3),127.5,dtype='float32');rgb[:vh,:vw]=pix;coef=np.zeros((fh,fw),dtype=coef0.dtype);coef[:min(fh,coef0.shape[0]),:min(fw,coef0.shape[1])]=coef0[:fh,:fw]
 out=dict(sample);out.update(image=torch.from_numpy(rgb).permute(2,0,1)/255,dct=ManipulationDataset._dct_volume_from_coef(coef,fh,fw),qtable=torch.from_numpy(qt[None]),dct_available=torch.ones(1));return out

class ManipDiversityDataset(torch.utils.data.Dataset):
 def __init__(self,records,cache,train,**kw):
  self.base=BaseFairDataset(records,cache,train,**kw);self.records=self.base.records;self.train=train;self.auth=[i for i,r in enumerate(records) if not r.label];assert self.auth
 def __len__(self):return len(self.base)
 def __getitem__(self,key):
  if not self.train:return self.base[key]
  idx,seed=key if isinstance(key,tuple) else (int(key),3407+int(key));j=int(seed)%100000
  if j%2==0:return self.base[key]  # exact 50% original accesses
  rng=np.random.default_rng(int(seed));bucket=AREA[(j//2)%4];kind='splicing' if ((j//2)//4)%2==0 else 'copy_move'
  bg_i=self.auth[int(rng.integers(len(self.auth)))];bg=self.base[(bg_i,int(seed)+11)];vh,vw=map(int,bg['valid_hw'].tolist());bgim=to_pil(bg['image'],vh,vw)
  if kind=='copy_move':donim=bgim.copy()
  else:
   di=self.auth[int(rng.integers(len(self.auth)))];don=self.base[(di,int(seed)+29)];dh,dw=map(int,don['valid_hw'].tolist());donim=to_pil(don['image'],dh,dw)
  ratio=float(rng.uniform(bucket[1],bucket[2]));area=max(64,ratio*vh*vw);aspect=float(np.exp(rng.uniform(np.log(.75),np.log(1.33))));pw=max(8,min(vw,int(round(math.sqrt(area*aspect)))));ph=max(8,min(vh,int(round(area/max(pw,1)))))
  if donim.width<pw or donim.height<ph:donim=donim.resize((max(pw,donim.width),max(ph,donim.height)),Image.Resampling.BILINEAR)
  sx=int(rng.integers(max(1,donim.width-pw+1)));sy=int(rng.integers(max(1,donim.height-ph+1)));patch=donim.crop((sx,sy,sx+pw,sy+ph))
  if rng.random()<.55:
   angle=float(rng.uniform(-12,12));patch=patch.rotate(angle,Image.Resampling.BILINEAR,expand=False)
  if rng.random()<.65:
   patch=ImageEnhance.Brightness(patch).enhance(float(rng.uniform(.85,1.15)));patch=ImageEnhance.Contrast(patch).enhance(float(rng.uniform(.85,1.15)))
  # V1 uses a strict rectangular hard mask so the actual GT area, rather than
  # only the intended bounding-box area, stays inside its balanced bucket.
  # Boundary appearance is still randomized by hard-vs-feather blending.
  alpha=Image.new('L',(pw,ph),0);dr=ImageDraw.Draw(alpha);shape=0
  if shape==0:dr.rectangle((0,0,pw-1,ph-1),fill=255)
  elif shape==1:dr.ellipse((0,0,pw-1,ph-1),fill=255)
  else:
   pts=[];n=12
   for k in range(n):
    a=2*math.pi*k/n;r=float(rng.uniform(.65,1.0));pts.append((pw/2+math.cos(a)*pw*.48*r,ph/2+math.sin(a)*ph*.48*r))
   dr.polygon(pts,fill=255)
  feather=bool(rng.integers(2));blend=alpha.filter(ImageFilter.GaussianBlur(float(rng.uniform(.6,2.0)))) if feather else alpha
  x=int(rng.integers(max(1,vw-pw+1)));y=int(rng.integers(max(1,vh-ph+1)));forged=bgim.copy();forged.paste(patch,(x,y),blend)
  if rng.random()<.25:forged=forged.filter(ImageFilter.GaussianBlur(float(rng.uniform(.2,.8))))
  mask=Image.new('L',(vw,vh),0);mask.paste(alpha.point(lambda z:255 if z>16 else 0),(x,y));m=torch.from_numpy((np.asarray(mask)>127).astype('float32'))[None]
  out=dict(bg);canvas=torch.full_like(bg['image'],.5);canvas[:,:vh,:vw]=torch.from_numpy(np.asarray(forged,dtype='float32').copy()).permute(2,0,1)/255;mc=torch.zeros_like(bg['mask']);mc[:,:vh,:vw]=m
  actual=float(m.mean());out.update(image=canvas,mask=mc,label=torch.ones_like(bg['label']),path=f'synthetic_{kind}_{bucket[0]}_{seed}.jpg',synthetic=True,synthetic_kind=kind,synthetic_bucket=bucket[0],synthetic_feather=feather,synthetic_area=actual)
  assert bucket[1]*.95 <= actual <= bucket[2]*1.05,(bucket,actual,pw,ph,vw,vh)
  # Always form RGB/DCT/qtable from this exact synthesized image; q100 means no extra heavy postprocessing.
  q=(100,90,70)[int(rng.integers(3))] if rng.random()<.5 else 100
  return jpeg_inputs(out,q,0 if rng.random()<.5 else 2)

def preflight(split):
 recs=stable.records(split['train'],'train','CASIA2');ds=ManipDiversityDataset(recs,OUT/'jpeg_cache_preflight',True);rows=[]
 for j in range(32):
  s=ds[(j,3407000000+100000+j)];rows.append({'synthetic':bool(s.get('synthetic',False)),'kind':s.get('synthetic_kind'),'bucket':s.get('synthetic_bucket'),'area':s.get('synthetic_area')})
 syn=[r for r in rows if r['synthetic']];assert len(syn)==16,(len(syn),rows);assert {r['kind'] for r in syn}=={'splicing','copy_move'};assert {r['bucket'] for r in syn}=={x[0] for x in AREA}
 OUT.mkdir(parents=True,exist_ok=True);stable.dump(OUT/'preflight.yaml',{'status':'passed','samples':32,'original':16,'synthetic':16,'synthetic_kinds':{k:sum(r['kind']==k for r in syn) for k in ('splicing','copy_move')},'area_buckets':{k:sum(r['bucket']==k for r in syn) for k,_,_ in AREA},'rows':rows})

def ensure_split():
 import csv
 groups={'train':[],'val':[],'test':[]}
 with MANIFEST.open(newline='',encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):
   row={'image':r['image'],'mask':r['mask'] or None,'label':int(r['label'])};groups[r['split']].append(row)
 # CASIA2 formal training uses the existing manifest split; test may be empty in this
 # source-training manifest, so validation is reused only as a placeholder because
 # this pilot is --skip-test and never reads the placeholder test population.
 if not groups['test']:groups['test']=[dict(groups['val'][0])]
 OUT.mkdir(parents=True,exist_ok=True);SPLIT.write_text(json.dumps(groups));return groups

def main():
 split=ensure_split();preflight(split);stable.FairDataset=ManipDiversityDataset
 class A:pass
 a=A();a.dataset='CASIA2';a.split=str(SPLIT);a.template=str(TEMPLATE);a.out=str(OUT);a.epochs=1;a.batch_size=4;a.learning_rate=.001;a.seed=3407;a.resume=False;a.positive_crop_prob=0.;a.positive_crop_retries=20;a.hcv_aux_scale=1.;a.qtr_aux_scale=.15;a.dher_aux_scale=.05;a.forensic_mode='full';a.hcv_semantic_bypass=False;a.verified_residual_decoder=False;a.decoder_ds_targeted=False;a.final_mask_only=False;a.splicing_region_affinity=False;a.region_affinity_weight=.1;a.early_gate_epoch=0;a.early_gate_min_f1=.5;a.skip_test=True;a.variant='baseline';a.train_crop=512;a.schedule_total_steps=1744
 stable.main(a)
if __name__=='__main__':main()
