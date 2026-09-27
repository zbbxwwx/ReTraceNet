import copy,sys,tempfile,traceback
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_imd_mit_full_v3  # register MiT content builder
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from retrace.data import ManipulationDataset
import run_v3_stable_bn_indomain as stable
PARENT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
E3=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e3_fullmodel_jpeg_adaptation1_20260922/qtr_localnorm_epoch01.pt')
MANIFEST=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e3_q90_subsampling_audit_20260923')
def jpeg(sample,sub):
 vh,vw=map(int,sample['valid_hw'].tolist());fh,fw=sample['image'].shape[-2:];arr=(sample['image'][:,:vh,:vw].permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')
 import jpegio
 with tempfile.TemporaryDirectory(prefix='subaudit_') as d:
  p=Path(d)/'x.jpg';Image.fromarray(arr).save(p,'JPEG',quality=90,subsampling=sub)
  with Image.open(p) as im:rgb=np.asarray(im.convert('RGB'),dtype='float32')
  j=jpegio.read(str(p));coef=j.coef_arrays[0];qt=j.quant_tables[j.comp_info[0].quant_tbl_no].astype('float32')
 canvas=np.full((fh,fw,3),127.5,np.float32);canvas[:vh,:vw]=rgb;cc=np.zeros((fh,fw),coef.dtype);h=min(fh,coef.shape[0]);w=min(fw,coef.shape[1]);cc[:h,:w]=coef[:h,:w]
 o=dict(sample);o.update(image=torch.from_numpy(canvas).permute(2,0,1)/255,dct=ManipulationDataset._dct_volume_from_coef(cc,fh,fw),qtable=torch.from_numpy(qt[None]),dct_available=torch.ones(1));return o
class View(torch.utils.data.Dataset):
 def __init__(self,b,s):self.base=b;self.s=s;self.records=b.records
 def __len__(self):return len(self.base)
 def __getitem__(self,i):return jpeg(self.base[i],self.s)
def pathof(r):
 for k in ('image','image_path','path','rgb_path'):
  v=getattr(r,k,None)
  if v:return str(v)
 return str(r)
try:
 OUT.mkdir(parents=True,exist_ok=True);par=torch.load(PARENT,map_location='cpu');ck=torch.load(E3,map_location='cpu');cfg=copy.deepcopy(par['config']);model=fair.e.build_final_model(cfg['model']).cuda();model.load_state_dict(ck['model'],strict=True);model.eval()
 va=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'val');vj=[r for r in va if Path(pathof(r)).suffix.lower() in {'.jpg','.jpeg','.jpe'}]
 groups={'all':va,'original_jpeg':vj};res={}
 for gn,rows in groups.items():
  base=FairDataset(rows,OUT/f'cache_{gn}',False);res[gn]={'n':len(rows)}
  for label,sub in [('q90_444',0),('q90_420',2)]:
   z=fair.evaluate(model,View(base,sub),cfg,OUT,f'{gn}_{label}');m=z['metrics']['fake'];res[gn][label]={'auc':float(m['auc']),'fixed_05_f1':float(m['fixed']['f1']),'fixed_05_precision':float(m['fixed']['precision']),'fixed_05_recall':float(m['fixed']['recall']),'oracle_f1':float(m['oracle']['f1']),'oracle_threshold':float(m['oracle']['threshold']),'oracle_precision':float(m['oracle']['precision']),'oracle_recall':float(m['oracle']['recall'])}
 out={'status':'completed','checkpoint':str(E3),'quality':90,'subsampling':{'q90_444':0,'q90_420':2},'results':res,'test_read':False};(OUT/'summary.yaml').write_text(yaml.safe_dump(out,sort_keys=False));print(yaml.safe_dump(out,sort_keys=False),flush=True)
except BaseException:
 out={'status':'failed','traceback':traceback.format_exc()};OUT.mkdir(parents=True,exist_ok=True);(OUT/'summary.yaml').write_text(yaml.safe_dump(out,sort_keys=False));raise
