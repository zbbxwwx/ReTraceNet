"""Full IMD2020 val comparison: current 1024/768 vs candidate 768/512.
Selects one fake-only pooled threshold per policy, then reports all populations and strata at that threshold. Never reads test.
"""
import argparse, copy, json, sys, traceback
from pathlib import Path
import numpy as np, torch, yaml
from PIL import Image
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_imd_mit_full_v3
import run_cross_dataset_after_chain as cross
import eval_imd_mit_context_policies as ctx
from e15_original_metrics import PixelStats
from retrace.data import Record

def recs(rows):
 return [Record(image=Path(x['image']),mask=Path(x['mask']) if x.get('mask') else None,split='val',dataset='IMD2020',label=int(x['label'])) for x in rows]
def at(s,t):
 r=s.result(); i=int(np.abs(s.thresholds-float(t)).argmin());tp=s.pos[::-1].cumsum()[::-1];fp=s.neg[::-1].cumsum()[::-1];fn=s.pos.sum()-tp;tn=s.neg.sum()-fp
 return {'threshold':float(s.thresholds[i]),'precision':float(tp[i]/max(tp[i]+fp[i],1)),'recall':float(tp[i]/max(tp[i]+fn[i],1)),'f1':float(2*tp[i]/max(2*tp[i]+fp[i]+fn[i],1)),'tp':int(tp[i]),'fp':int(fp[i]),'fn':int(fn[i]),'tn':int(tn[i]),'auc':r['auc'],'count':r['count']}
@torch.inference_mode()
def evaluate(model,records,out,name,tile,stride,full_max=1024):
 cache=out/'jpeg_cache';ds=cross.fair.FairDataset(records,cache,False)
 groups={k:PixelStats() for k in ('all_images','fake_only','authentic','sliding_fake','non_sliding_fake','area_lt1','area_1_5','area_5_20','area_gte20')}
 direct=sliding=0
 for i,r in enumerate(records):
  with Image.open(r.image) as im:w,h=im.size
  is_slide=max(h,w)>full_max
  if is_slide:
   p,y=ctx.predict_large(model,r,cache,tile,stride);sliding+=1
  else:
   b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)));vh,vw=map(int,b['valid_hw'][0].tolist());o=cross.fair.fwd(model,b);p=o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy();y=b['mask'][0,0,:vh,:vw].numpy().astype(np.int16);direct+=1
  if p.shape!=y.shape:p=torch.nn.functional.interpolate(torch.from_numpy(p)[None,None].float(),size=y.shape,mode='bilinear',align_corners=False)[0,0].numpy()
  groups['all_images'].add(p,y)
  if r.label==1 and (y>0).any():
   groups['fake_only'].add(p,y);groups['sliding_fake' if is_slide else 'non_sliding_fake'].add(p,y);ratio=float((y>0).mean());key='area_lt1' if ratio<.01 else 'area_1_5' if ratio<.05 else 'area_5_20' if ratio<.2 else 'area_gte20';groups[key].add(p,y)
  elif r.label==0:groups['authentic'].add(p,y)
  if i==0 or (i+1)%25==0:print(f'[{name}] {i+1}/{len(records)} direct={direct} sliding={sliding}',flush=True)
 threshold=float(groups['fake_only'].result()['oracle']['threshold'])
 metrics={k:at(v,threshold) for k,v in groups.items()}
 result={'policy':name,'tile':tile,'stride':stride,'full_max_side':full_max,'num_images':len(records),'direct_count':direct,'sliding_count':sliding,'threshold_source':'complete_IMD2020_val_fake_only_pooled','selected_threshold':threshold,'fake_only_pooled_best':groups['fake_only'].result()['oracle'],'all_images_pooled_at_fake_threshold':metrics['all_images'],'fake_only_at_selected_threshold':metrics['fake_only'],'authentic_pixel_fpr':{'threshold':threshold,'fpr':metrics['authentic']['f1'] if False else metrics['authentic']['fp']/max(metrics['authentic']['fp']+metrics['authentic']['tn'],1),'false_positive_pixels':metrics['authentic']['fp'],'authentic_pixels':metrics['authentic']['fp']+metrics['authentic']['tn']},'groups_at_selected_threshold':{k:metrics[k] for k in ('sliding_fake','non_sliding_fake','area_lt1','area_1_5','area_5_20','area_gte20')},'all_images_pooled_best_diagnostic':groups['all_images'].result()['oracle'],'test_read':False}
 (out/f'{name}.yaml').write_text(yaml.safe_dump(result,sort_keys=False));return result

def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);summary={'status':'loading','checkpoint':a.checkpoint,'split':a.split,'test_read':False,'policies':{}};(out/'summary.yaml').write_text(yaml.safe_dump(summary,sort_keys=False))
 try:
  ck=torch.load(a.checkpoint,map_location='cpu');m=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval();m.load_state_dict(ck['model'],strict=True);rows=recs(json.loads(Path(a.split).read_text())['val'])
  for name,tile,stride in [('current_tile1024_stride768',1024,768),('candidate_tile768_stride512',768,512)]:
   summary['status']='evaluating_'+name;(out/'summary.yaml').write_text(yaml.safe_dump(summary,sort_keys=False));z=evaluate(m,rows,out,name,tile,stride);summary['policies'][name]=z;(out/'summary.yaml').write_text(yaml.safe_dump(summary,sort_keys=False))
  x=summary['policies']['current_tile1024_stride768'];y=summary['policies']['candidate_tile768_stride512'];df=y['fake_only_pooled_best']['f1']-x['fake_only_pooled_best']['f1'];da=y['fake_only_at_selected_threshold']['auc']-x['fake_only_at_selected_threshold']['auc'];fpr=y['authentic_pixel_fpr']['fpr']-x['authentic_pixel_fpr']['fpr'];adopt=df>0 and fpr<=.005
  summary.update(status='completed',delta_fake_f1=df,delta_fake_f1_pp=100*df,delta_fake_auc=da,delta_authentic_fpr=fpr,decision='adopt_768_512' if adopt else 'keep_1024_768');(out/'summary.yaml').write_text(yaml.safe_dump(summary,sort_keys=False));print(yaml.safe_dump(summary,sort_keys=False))
 except BaseException:
  summary.update(status='failed',stop_reason=traceback.format_exc());(out/'summary.yaml').write_text(yaml.safe_dump(summary,sort_keys=False));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);main(p.parse_args())
