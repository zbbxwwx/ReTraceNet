"""Columbia val sliding-fake diagnostic; mirrors fixed IMD small-window protocol."""
import argparse,copy,json,sys,yaml,torch
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_imd_mit_full_v3
import run_cross_dataset_after_chain as cross
import eval_imd_mit_context_policies as ctx
from retrace.data import Record

def records(rows):return [Record(image=Path(x['image']),mask=Path(x['mask']) if x.get('mask') else None,split='val',dataset='Columbia',label=int(x['label'])) for x in rows]
def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ck=torch.load(a.checkpoint,map_location='cpu');m=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval();m.load_state_dict(ck['model'],strict=True)
 rr=records(json.loads(Path(a.split).read_text())['val']);recs=[]
 for r in rr:
  if r.label!=1:continue
  with Image.open(r.image) as im:w,h=im.size
  if max(w,h)>1024:recs.append(r)
  if len(recs)>=a.n:break
 s={'status':'running','dataset':'Columbia_val','checkpoint':a.checkpoint,'selection':'first up to 48 fake val images with max_side>1024','requested_n':a.n,'actual_n':len(recs),'test_read':False,'policies':{}};(out/'summary.yaml').write_text(yaml.safe_dump(s,sort_keys=False))
 if not recs:raise RuntimeError('No Columbia val fake image requires sliding under max_side>1024 rule')
 for name,tile,stride in [('tile512_stride384',512,384),('tile768_stride512',768,512),('tile1024_stride768',1024,768),('tile1536_stride1152',1536,1152)]:
  z=ctx.evaluate_policy(m,recs,out/'jpeg_cache',tile,stride,out,name,full_max=0);s['policies'][name]=z;(out/'summary.yaml').write_text(yaml.safe_dump(s,sort_keys=False))
 s['status']='completed';s['ranking']=sorted(({'policy':k,'f1':v['primary']['f1'],'auc':v['fake_only_pooled']['auc'],'threshold':v['primary']['threshold']} for k,v in s['policies'].items()),key=lambda x:x['f1'],reverse=True);(out/'summary.yaml').write_text(yaml.safe_dump(s,sort_keys=False));print(yaml.safe_dump(s,sort_keys=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=48);main(p.parse_args())
