"""CASIA2-val robustness, two fixed levels per operation, threshold locked at 0.34."""
import hashlib,sys,tempfile,json
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image,ImageEnhance,ImageFilter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_imd_mit_full_v3  # patch exact MiT model builder
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from retrace.data import ManipulationDataset
from takeover_proxy import evaluate

CK=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_mit_jpeg_only_dg_probe1_20260920/epoch01.pt')
MAN=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_mit_jpeg_only_dg_epoch1_robustness_20260920')
CONDS={'jpeg_q90':('jpeg',90),'jpeg_q70':('jpeg',70),'resize_090':('resize',.9),'resize_075':('resize',.75),
 'blur_05':('blur',.5),'blur_10':('blur',1.),'noise_2':('noise',2.),'noise_5':('noise',5.),
 'brightness_090':('brightness',.9),'brightness_110':('brightness',1.1),
 'contrast_090':('contrast',.9),'contrast_110':('contrast',1.1),
 'subsampling_444':('subsampling',0),'subsampling_420':('subsampling',2)}

def transform(sample,kind,value,seed):
 vh,vw=map(int,sample['valid_hw'].tolist());fh,fw=sample['image'].shape[-2:]
 arr=(sample['image'][:,:vh,:vw].permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8');im=Image.fromarray(arr);q,sub=100,0
 rng=np.random.default_rng(seed)
 if kind=='jpeg':q,sub=int(value),2
 elif kind=='subsampling':q,sub=90,int(value)
 elif kind=='resize':
  nw,nh=max(1,round(vw*value)),max(1,round(vh*value));im=im.resize((nw,nh),Image.Resampling.BILINEAR).resize((vw,vh),Image.Resampling.BILINEAR)
 elif kind=='blur':im=im.filter(ImageFilter.GaussianBlur(float(value)))
 elif kind=='noise':im=Image.fromarray((np.asarray(im).astype('float32')+rng.normal(0,float(value),(vh,vw,3))).round().clip(0,255).astype('uint8'))
 elif kind=='brightness':im=ImageEnhance.Brightness(im).enhance(float(value))
 elif kind=='contrast':im=ImageEnhance.Contrast(im).enhance(float(value))
 import jpegio
 with tempfile.TemporaryDirectory(prefix='robust_') as d:
  p=Path(d)/'x.jpg';im.save(p,quality=q,subsampling=sub)
  with Image.open(p) as dec:pix=np.asarray(dec.convert('RGB'),dtype='float32')
  jpg=jpegio.read(str(p));coef0=jpg.coef_arrays[0];qt=jpg.quant_tables[jpg.comp_info[0].quant_tbl_no].astype('float32')
 rgb=np.full((fh,fw,3),127.5,dtype='float32');rgb[:vh,:vw]=pix
 coef=np.zeros((fh,fw),dtype=coef0.dtype);h=min(fh,coef0.shape[0]);w=min(fw,coef0.shape[1]);coef[:h,:w]=coef0[:h,:w]
 out=dict(sample);out.update(image=torch.from_numpy(rgb).permute(2,0,1)/255,dct=ManipulationDataset._dct_volume_from_coef(coef,fh,fw),qtable=torch.from_numpy(qt[None]),dct_available=torch.ones(1));return out

class DS(torch.utils.data.Dataset):
 def __init__(self,base,cond):self.base=base;self.records=base.records;self.cond=cond
 def __len__(self):return len(self.base)
 def __getitem__(self,key):
  s=self.base[key];seed=int(hashlib.sha256((str(s['path'])+self.cond).encode()).hexdigest()[:8],16);return transform(s,*CONDS[self.cond],seed)

def main():
 OUT.mkdir(parents=True,exist_ok=True);ck=torch.load(CK,map_location='cpu');cfg=ck['config'];thr=0.34
 assert abs(float(ck['selection_threshold'])-thr)<1e-9
 model=fair.e.build_final_model(cfg['model']).cuda().eval();model.load_state_dict(ck['model'],strict=True)
 recs=fair.e.read_manifest(MAN,cfg['data']['root'],'val');base=FairDataset(recs,OUT/'jpeg_cache',False);table={}
 for name in CONDS:
  r=evaluate(model,DS(base,name),thr);table[name]={'fake_locked':r['fake_locked'],'fake_oracle_diagnostic':r['metrics']['fake']['oracle'],'pixel_auc':r['metrics']['fake']['auc'],'authentic_fpr':r['authentic_locked']['fpr']}
  (OUT/f'{name}.json').write_text(json.dumps(r));(OUT/'summary.yaml').write_text(yaml.safe_dump({'status':'running','threshold':thr,'conditions':table},sort_keys=False))
 result={'status':'completed','checkpoint':str(CK),'threshold':thr,'threshold_source':'CASIA2_clean_val','dataset':'CASIA2_val','external_test_read':False,'conditions':table}
 (OUT/'summary.yaml').write_text(yaml.safe_dump(result,sort_keys=False));print(yaml.safe_dump(result,sort_keys=False))
if __name__=='__main__':main()
