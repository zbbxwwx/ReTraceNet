import copy,sys,tempfile,traceback
from pathlib import Path
import numpy as np,torch,yaml
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import run_imd_mit_full_v3
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from retrace.data import ManipulationDataset
PARENT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
E3=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e3_fullmodel_jpeg_adaptation1_20260922/qtr_localnorm_epoch01.pt')
MANIFEST=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e3_q90_rgb_dct_crossover_20260923')
def encode(sample,sub,raw=False):
 vh,vw=map(int,sample['valid_hw'].tolist());fh,fw=sample['image'].shape[-2:]
 arr=(sample['image'][:,:vh,:vw].permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')
 import jpegio
 with tempfile.TemporaryDirectory(prefix='cross_') as d:
  p=Path(d)/'x.jpg';Image.fromarray(arr).save(p,'JPEG',quality=90,subsampling=sub)
  with Image.open(p) as im: rgb=np.asarray(im.convert('RGB'),dtype='float32')
  j=jpegio.read(str(p));coef=j.coef_arrays[0].copy();qt=j.quant_tables[j.comp_info[0].quant_tbl_no].astype('float32').copy()
 canvas=np.full((fh,fw,3),127.5,np.float32);canvas[:vh,:vw]=rgb
 cc=np.zeros((fh,fw),coef.dtype);h=min(fh,coef.shape[0]);w=min(fw,coef.shape[1]);cc[:h,:w]=coef[:h,:w]
 o=dict(sample);o.update(image=torch.from_numpy(canvas).permute(2,0,1)/255,dct=ManipulationDataset._dct_volume_from_coef(cc,fh,fw),qtable=torch.from_numpy(qt[None]),dct_available=torch.ones(1))
 return (o,coef,qt) if raw else o
class Hybrid(torch.utils.data.Dataset):
 def __init__(self,b,rgbsub,dctsub):self.base=b;self.rgbsub=rgbsub;self.dctsub=dctsub;self.records=b.records
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  s=self.base[i];a=encode(s,self.rgbsub);b=encode(s,self.dctsub)
  a['dct']=b['dct'];a['qtable']=b['qtable'];a['dct_available']=b['dct_available'];return a
def pathof(r):
 for k in ('image','image_path','path','rgb_path'):
  v=getattr(r,k,None)
  if v:return str(v)
 return str(r)
try:
 OUT.mkdir(parents=True,exist_ok=True)
 par=torch.load(PARENT,map_location='cpu');ck=torch.load(E3,map_location='cpu');cfg=copy.deepcopy(par['config'])
 model=fair.e.build_final_model(cfg['model']).cuda();model.load_state_dict(ck['model'],strict=True);model.eval()
 rows=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'val');base=FairDataset(rows,OUT/'cache_all',False)
 # 128-pair raw luminance coefficient/qtable audit
 audit=[]
 for i in range(min(128,len(base))):
  s=base[i];_,c444,q444=encode(s,0,True);_,c420,q420=encode(s,2,True)
  h=min(c444.shape[0],c420.shape[0]);w=min(c444.shape[1],c420.shape[1]);x=c444[:h,:w].astype(np.float64);y=c420[:h,:w].astype(np.float64)
  audit.append(dict(coef_exact=bool(np.array_equal(x,y)),coef_mae=float(np.abs(x-y).mean()),coef_max_abs=float(np.abs(x-y).max()),qtable_exact=bool(np.array_equal(q444,q420)),qtable_max_abs=float(np.abs(q444-q420).max())))
 coef_exact_rate=float(np.mean([x['coef_exact'] for x in audit]));q_exact_rate=float(np.mean([x['qtable_exact'] for x in audit]))
 pre={'n':len(audit),'luma_coef_exact_rate':coef_exact_rate,'luma_coef_mae_mean':float(np.mean([x['coef_mae'] for x in audit])),'luma_coef_max_abs_max':float(np.max([x['coef_max_abs'] for x in audit])),'qtable_exact_rate':q_exact_rate,'qtable_max_abs_max':float(np.max([x['qtable_max_abs'] for x in audit]))}
 (OUT/'precheck.yaml').write_text(yaml.safe_dump(pre,sort_keys=False));print('PRECHECK',yaml.safe_dump(pre,sort_keys=False),flush=True)
 # Crossed diagnostic combinations. Existing A/B results remain in the sibling audit.
 combos={'C_rgb444_dct420':(0,2),'D_rgb420_dct444':(2,0)};results={}
 for name,(rs,ds) in combos.items():
  z=fair.evaluate(model,Hybrid(base,rs,ds),cfg,OUT,name);m=z['metrics']['fake']
  results[name]={'auc':float(m['auc']),'fixed_05':{k:float(m['fixed'][k]) for k in ('precision','recall','f1')},'oracle':{k:float(m['oracle'][k]) for k in ('threshold','precision','recall','f1')}}
 out={'status':'completed','checkpoint':str(E3),'quality':90,'precheck':pre,'combinations':{'A':'RGB444+DCT444 (prior audit)','B':'RGB420+DCT420 (prior audit)','C':'RGB444+DCT420','D':'RGB420+DCT444'},'results':results,'prior_audit':str(OUT.parent/'e3_q90_subsampling_audit_20260923'/'summary.yaml'),'test_read':False}
 (OUT/'summary.yaml').write_text(yaml.safe_dump(out,sort_keys=False));print(yaml.safe_dump(out,sort_keys=False),flush=True)
except BaseException:
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'summary.yaml').write_text(yaml.safe_dump({'status':'failed','traceback':traceback.format_exc()},sort_keys=False));raise
