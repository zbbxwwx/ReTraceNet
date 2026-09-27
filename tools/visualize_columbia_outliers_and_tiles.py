"""Retrospective visualization of fixed Columbia test predictions and one-image tile audit."""
import copy,hashlib,json,sys,yaml
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from scipy.ndimage import sobel,binary_dilation,binary_erosion
ROOT=Path('/group4/zouwenxuan/iml_generalization');REPO=ROOT/'repos/ReTraceNet';sys.path[:0]=[str(REPO),str(REPO/'tools')]
import run_imd_mit_full_v3
import run_cross_dataset_after_chain as cross
import eval_imd_mit_context_policies as ctx
from retrace.data import ManipulationDataset,Record
RUN=ROOT/'outputs/retracenet_v2/columbia_mit_dedup_topk_20260921';OUT=RUN/'retrospective_error_audit_20260924/visuals';OUT.mkdir(parents=True,exist_ok=True);SEEDS=(3407,2026,42)
def dedupe(rows):
 s=set();o=[]
 for r in rows:
  k=r.get('rgb_sha256') or r['image']
  if k not in s:s.add(k);o.append(r)
 return o
def roots(seed):
 r=RUN/f'seed_{seed}'/'topk_ensemble_1024_768';d=yaml.safe_load((r/'summary.yaml').read_text());return r,[r/f"pred_test_ep{int(x['epoch']):03d}" for x in d['selected']],float(d['locked_threshold']),[Path(x['path']) for x in d['selected']]
def prob(ds,i):return np.mean([np.load(d/f'{i:04d}.npy').astype(np.float32) for d in ds],0)
def counts(ps,ys,t,skip=None):
 tp=fp=fn=tn=0
 for i,(p,y) in enumerate(zip(ps,ys)):
  if i==skip:continue
  z=p>=t;tp+=int((z&y).sum());fp+=int((z&~y).sum());fn+=int((~z&y).sum());tn+=int((~z&~y).sum())
 return {'threshold':t,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1)}
def color_heat(p):
 p=np.clip(p,0,1);r=np.clip(1.5*p,0,1);b=np.clip(1.5*(1-p),0,1);g=np.clip(1.5-3*np.abs(p-.5),0,1);return (255*np.stack([r,g,b],2)).astype(np.uint8)
def fit(a,w=520):
 im=Image.fromarray(a);h=max(1,round(im.height*w/im.width));return np.asarray(im.resize((w,h),Image.Resampling.BILINEAR))
def panel(arrs,labels,path):
 ims=[Image.fromarray(fit(a)) for a in arrs];w=max(x.width for x in ims);hh=max(x.height for x in ims)+34;can=Image.new('RGB',(w*len(ims),hh),'white');dr=ImageDraw.Draw(can)
 for j,(im,l) in enumerate(zip(ims,labels)):can.paste(im,(j*w,34));dr.text((j*w+8,8),l,fill='black')
 can.save(path)
def jpeg_parts(rec,cache):
 cache.mkdir(parents=True,exist_ok=True);jp=rec.image
 if jp.suffix.lower() not in {'.jpg','.jpeg'}:
  jp=cache/(hashlib.sha256(str(rec.image).encode()).hexdigest()+'.jpg')
  if not jp.exists():Image.open(rec.image).convert('RGB').save(jp,quality=100,subsampling=0)
 import jpegio;j=jpegio.read(str(jp));coef=j.coef_arrays[0];qt=j.quant_tables[j.comp_info[0].quant_tbl_no].astype('float32');rgb=np.asarray(Image.open(jp).convert('RGB'),dtype=np.float32);return rgb,coef,qt
@torch.inference_mode()
def model_tiles(ckpath,rec,cache,tile=1024,stride=768):
 ck=torch.load(ckpath,map_location='cpu');m=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval();m.load_state_dict(ck['model'],strict=True);rgb,coef,qt=jpeg_parts(rec,cache);h,w=rgb.shape[:2];ph=max(tile,((h+31)//32)*32);pw=max(tile,((w+31)//32)*32);can=np.full((ph,pw,3),127.5,np.float32);can[:h,:w]=rgb;cc=np.zeros((ph,pw),coef.dtype);cc[:coef.shape[0],:coef.shape[1]]=coef;out=[]
 for top in ctx.tile_starts(ph,tile,stride):
  for left in ctx.tile_starts(pw,tile,stride):
   x=torch.from_numpy(can[top:top+tile,left:left+tile].copy()).permute(2,0,1)[None]/255;b={'image':x,'dct':ManipulationDataset._dct_volume_from_coef(cc[top:top+tile,left:left+tile],tile,tile)[None],'qtable':torch.from_numpy(qt[None,None]),'dct_available':torch.ones((1,1))};p=cross.fair.fwd(m,b)['mask_logits'].float().sigmoid()[0,0].cpu().numpy();out.append({'top':top,'left':left,'prob':p})
 del m;torch.cuda.empty_cache();return out,(h,w),(ph,pw)
def main():
 split=json.loads((ROOT/'outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json').read_text());rows=dedupe(split['test']);rr=[roots(s) for s in SEEDS];ys=[np.load(rr[0][0]/f'target_test_{i:04d}.npy').astype(bool) for i in range(len(rows))];seedps={s:[prob(ds,i) for i in range(len(rows))] for s,(_,ds,_,_) in zip(SEEDS,rr)};fake_ids=[i for i,r in enumerate(rows) if int(r['label'])];hard=next(i for i,r in enumerate(rows) if Path(r['image']).name=='canonxt_kodakdcs330_sub_22.tif');auth=next(i for i,r in enumerate(rows) if Path(r['image']).name=='kodakdcs330_03_sub_01.tif')
 report={'scope':'retrospective; no test-based selection','hard_fake_index':hard,'authentic_outlier_index':auth,'fake_only_pooled':{}}
 for s,(_,_,t,_) in zip(SEEDS,rr):
  ps=[seedps[s][i] for i in fake_ids];yy=[ys[i] for i in fake_ids];j=fake_ids.index(hard);report['fake_only_pooled'][str(s)]={'included':counts(ps,yy,t),'excluding_hard_fake_diagnostic':counts(ps,yy,t,j)}
 cps=[.75*seedps[3407][i]+.25*seedps[2026][i] for i in fake_ids];yy=[ys[i] for i in fake_ids];report['fake_only_pooled']['cross_seed_0p75_0p25_0']={'included':counts(cps,yy,.58),'excluding_hard_fake_diagnostic':counts(cps,yy,.58,fake_ids.index(hard))}
 for idx,tag in [(hard,'hard_fake'),(auth,'authentic_fp')]:
  rgb=np.asarray(Image.open(rows[idx]['image']).convert('RGB'));y=ys[idx];arr=[rgb,np.repeat((255*y).astype(np.uint8)[:,:,None],3,axis=2)];lab=['RGB','GT']
  masks=[]
  for s,(_,_,t,_) in zip(SEEDS,rr):p=seedps[s][idx];arr += [color_heat(p),np.repeat((255*(p>=t)).astype(np.uint8)[:,:,None],3,axis=2)];lab += [f'seed{s} prob',f'seed{s} mask t={t:.2f}'];masks.append(p>=t)
  z=np.mean(np.stack(masks),0)>=.5;err=np.zeros((*y.shape,3),np.uint8);err[(~z)&y]=[0,0,255];err[z&~y]=[255,0,0];err[z&y]=[0,255,0];arr.append(err);lab.append('error: green TP red FP blue FN');panel(arr,lab,OUT/f'{tag}_overview.png')
 # Tile audit: average corresponding tiles across each seed's val-selected checkpoints, then stitch and compare saved map.
 rec=Record(image=Path(rows[hard]['image']),mask=Path(rows[hard]['mask']),split='test',dataset='Columbia',label=1);tile_report={}
 for s,(root,dirs,t,cks) in zip(SEEDS,rr):
  per=[model_tiles(c,rec,OUT/'jpeg_cache') for c in cks];n=len(per[0][0]);tiles=[]
  for j in range(n):tiles.append({'top':per[0][0][j]['top'],'left':per[0][0][j]['left'],'prob':np.mean([x[0][j]['prob'] for x in per],0)})
  h,w=per[0][1];ph,pw=per[0][2];acc=np.zeros((ph,pw),np.float32);wei=np.zeros((ph,pw),np.float32)
  imgs=[];labs=[]
  for j,q in enumerate(tiles):acc[q['top']:q['top']+1024,q['left']:q['left']+1024]+=q['prob'];wei[q['top']:q['top']+1024,q['left']:q['left']+1024]+=1;imgs.append(color_heat(q['prob']));labs.append(f'tile{j} top={q["top"]} left={q["left"]}')
  stitched=(acc/wei)[:h,:w];saved=seedps[s][hard];diff=float(np.max(np.abs(stitched-saved)));tile_report[str(s)]={'tile_count':n,'origins':[{'top':q['top'],'left':q['left']} for q in tiles],'max_abs_stitch_vs_saved':diff,'each_tile_positive_fraction':[float((q['prob']>=t).mean()) for q in tiles],'stitched_positive_fraction':float((stitched>=t).mean()),'gt_ratio':float(ys[hard].mean())};imgs += [color_heat(stitched),color_heat(saved)];labs += ['reconstructed stitch','saved probability'];panel(imgs,labs,OUT/f'hard_fake_seed{s}_tiles.png')
 report['tile_audit']=tile_report
 # Authentic edge overlap descriptive plot
 rgb=np.asarray(Image.open(rows[auth]['image']).convert('RGB'));gray=rgb.mean(2).astype(np.float32);g=np.hypot(sobel(gray,0),sobel(gray,1));g=(255*g/max(g.max(),1e-9)).astype(np.uint8);panel([rgb,np.repeat(g[:,:,None],3,2),color_heat(np.mean([seedps[s][auth] for s in SEEDS],0))],['auth RGB','gradient','mean seed probability'],OUT/'authentic_fp_edge_texture.png')
 (OUT/'report.yaml').write_text(yaml.safe_dump(report,sort_keys=False));print(yaml.safe_dump(report,sort_keys=False))
if __name__=='__main__':main()
