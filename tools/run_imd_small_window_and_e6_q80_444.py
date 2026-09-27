import argparse,copy,json,sys,yaml,torch
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import eval_imd_mit_context_policies as ctx
import run_cross_dataset_after_chain as cross
import run_e6_multi_quality_chroma as e6
import run_e5_paired_chroma_adaptation as e5

def save(p,x): Path(p).write_text(yaml.safe_dump(x,sort_keys=False))

def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 # fixed deterministic subset: fake val images requiring sliding, first N in split order
 ck=torch.load(a.imd_checkpoint,map_location='cpu')
 model=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval()
 model.load_state_dict(ck['model'],strict=True)
 sp=json.load(open(a.imd_split)); rr=ctx.records(sp['val'])
 recs=[]
 for r in rr:
  if r.label!=1: continue
  with Image.open(r.image) as im:w,h=im.size
  if max(h,w)>1024:recs.append(r)
  if len(recs)>=a.n:break
 payload={'status':'running','subset':'first fake val images with max_side>1024','n':len(recs),'policies':{}}
 save(out/'imd_small_summary.yaml',payload)
 for name,tile,stride in [('tile512_stride384',512,384),('tile768_stride512',768,512),('tile1024_stride768',1024,768),('tile1536_stride1152',1536,1152)]:
  z=ctx.evaluate_policy(model,recs,out/'jpeg_cache',tile,stride,out,name,full_max=0)
  payload['policies'][name]=z;save(out/'imd_small_summary.yaml',payload)
 payload['status']='completed';save(out/'imd_small_summary.yaml',payload)
 del model;torch.cuda.empty_cache()
 # E6 epoch02 Q80 4:4:4 val; clean threshold comes from epoch02 summary
 ep=torch.load(a.e6_checkpoint,map_location='cpu')
 cfg=copy.deepcopy(ep['config']);m=e5.build_model(cfg).cuda().eval();m.load_state_dict(ep['model'],strict=True)
 base=e5.base_records('val'); threshold=float(yaml.safe_load(open(a.e6_summary))['history'][-1]['clean']['oracle']['threshold'])
 q=e5.eval_ds(m,e5.FixedQuality(base,80,0),cfg,'e6_ep02_q80_444',threshold)
 res={'status':'completed','checkpoint':a.e6_checkpoint,'population':'CASIA2 val','jpeg_quality':80,'subsampling':'4:4:4','clean_locked_threshold':threshold,'metrics':q,'test_read':False}
 save(out/'e6_ep02_q80_444_val.yaml',res);print(yaml.safe_dump(res,sort_keys=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--imd-checkpoint',required=True);p.add_argument('--imd-split',required=True);p.add_argument('--e6-checkpoint',required=True);p.add_argument('--e6-summary',required=True);p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=48);main(p.parse_args())
