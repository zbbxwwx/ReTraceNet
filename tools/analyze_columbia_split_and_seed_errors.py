"""Retrospective Columbia split/error audit. No training, no model selection, no test tuning."""
import json,csv,re,sys
from pathlib import Path
import numpy as np,yaml
from PIL import Image
from scipy.ndimage import binary_erosion,binary_dilation,sobel
ROOT=Path('/group4/zouwenxuan/iml_generalization')
RUN=ROOT/'outputs/retracenet_v2/columbia_mit_dedup_topk_20260921'
SPLIT=ROOT/'outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json'
OUT=RUN/'retrospective_error_audit_20260924';OUT.mkdir(parents=True,exist_ok=True)
SEEDS=(3407,2026,42)
def dedupe(rows):
 seen=set();o=[]
 for r in rows:
  k=r.get('rgb_sha256') or r['image']
  if k not in seen:seen.add(k);o.append(r)
 return o
def family(path):
 s=Path(path).stem.lower();s=re.sub(r'_sub_?\d+$','',s);s=re.sub(r'_edgemask$','',s);return s
def stats(rows):
 dims=[];rat=[];formats={};fams={};fake=auth=0
 for r in rows:
  with Image.open(r['image']) as im:w,h=im.size
  dims.append((w,h,w*h,max(w,h)));formats[Path(r['image']).suffix.lower()]=formats.get(Path(r['image']).suffix.lower(),0)+1;fams[family(r['image'])]=fams.get(family(r['image']),0)+1
  if int(r['label']):
   fake+=1
   with Image.open(r['mask']) as im:y=np.asarray(im.convert('L'))>127
   rat.append(float(y.mean()))
  else:auth+=1
 a=np.asarray(dims)
 return {'count':len(rows),'fake':fake,'authentic':auth,'width_median':float(np.median(a[:,0])),'height_median':float(np.median(a[:,1])),'pixels_median':float(np.median(a[:,2])),'max_side_median':float(np.median(a[:,3])),'max_side_gt1024':int((a[:,3]>1024).sum()),'formats':formats,'manipulation_ratio':None if not rat else {'median':float(np.median(rat)),'mean':float(np.mean(rat)),'lt1':int(sum(x<.01 for x in rat)),'1_5':int(sum(.01<=x<.05 for x in rat)),'5_20':int(sum(.05<=x<.2 for x in rat)),'gte20':int(sum(x>=.2 for x in rat))},'top_filename_families':sorted(({'family':k,'count':v} for k,v in fams.items()),key=lambda x:x['count'],reverse=True)[:12]}
def seed_dirs(seed,part):
 root=RUN/f'seed_{seed}'/'topk_ensemble_1024_768';s=yaml.safe_load((root/'summary.yaml').read_text());dirs=[root/f"pred_{part}_ep{int(x['epoch']):03d}" for x in s['selected']];res=yaml.safe_load((root/'result.yaml').read_text());return root,dirs,float(res['threshold'])
def prob(dirs,i):return np.mean([np.load(d/f'{i:04d}.npy').astype(np.float32) for d in dirs],axis=0)
def met(p,y,t):
 z=p>=t;tp=int((z&y).sum());fp=int((z&~y).sum());fn=int((~z&y).sum());tn=int((~z&~y).sum());return tp,fp,fn,tn,2*tp/max(2*tp+fp+fn,1)
def main():
 raw=json.loads(SPLIT.read_text());sp={k:dedupe(v) for k,v in raw.items()};report={'status':'completed','scope':'retrospective descriptive analysis only; test not used for model or threshold selection','split_stats':{k:stats(v) for k,v in sp.items()}}
 # Exact RGB hash overlap across partitions.
 hs={k:set(x.get('rgb_sha256') for x in v if x.get('rgb_sha256')) for k,v in sp.items()};report['cross_split_exact_rgb_overlap']={f'{a}_{b}':len(hs[a]&hs[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
 recs=sp['test'];roots=[]
 for s in SEEDS:roots.append((s,*seed_dirs(s,'test')))
 rows=[]
 for i,r in enumerate(recs):
  y=np.load(roots[0][1]/f'target_test_{i:04d}.npy').astype(bool)
  with Image.open(r['image']) as im:rgb=np.asarray(im.convert('RGB'),dtype=np.float32)/255
  gray=rgb.mean(2);grad=np.hypot(sobel(gray,0),sobel(gray,1));entry={'index':i,'path':r['image'],'name':Path(r['image']).name,'label':int(r['label']),'width':int(y.shape[1]),'height':int(y.shape[0]),'gt_ratio':float(y.mean())}
  f1s=[];fps=[];fns=[]
  for seed,root,dirs,thr in roots:
   p=prob(dirs,i);tp,fp,fn,tn,f1=met(p,y,thr);entry[f'f1_{seed}']=f1;entry[f'fp_{seed}']=fp;entry[f'fn_{seed}']=fn;f1s.append(f1);fps.append(fp);fns.append(fn)
  entry['mean_f1']=float(np.mean(f1s));entry['std_f1']=float(np.std(f1s));entry['all_three_f1_lt_0p5']=bool(all(x<.5 for x in f1s));entry['all_three_f1_lt_0p7']=bool(all(x<.7 for x in f1s));entry['dominant_error']='FN' if np.mean(fns)>np.mean(fps) else 'FP'
  if int(r['label']):
   # Average seed probability, descriptive only. Boundary is 5px inner band; far FP excludes 5px outer band.
   ps=np.mean([prob(x[2],i) for x in roots],axis=0);zs=[ps>=x[3] for x in roots];z=np.mean(np.stack(zs),axis=0)>=.5
   er=binary_erosion(y,iterations=5);boundary=y&~er;dil=binary_dilation(y,iterations=5);entry['interior_recall']=float((z&er).sum()/max(er.sum(),1));entry['boundary_recall']=float((z&boundary).sum()/max(boundary.sum(),1));far=z&~dil;near=z&~y&dil;entry['fp_far_fraction']=float(far.sum()/max((z&~y).sum(),1));entry['fp_near_boundary_fraction']=float(near.sum()/max((z&~y).sum(),1));entry['fp_gradient_mean']=float(grad[z&~y].mean()) if (z&~y).any() else 0.;entry['tn_gradient_mean']=float(grad[(~z)&(~y)].mean()) if ((~z)&(~y)).any() else 0.
  rows.append(entry)
 fields=sorted(set().union(*(x.keys() for x in rows)));f=OUT/'per_image_test.csv'
 with f.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
 fake=[r for r in rows if r['label']==1];auth=[r for r in rows if r['label']==0]
 report['seed_error_overlap']={'test_images':len(rows),'fake_images':len(fake),'authentic_images':len(auth),'all_three_fake_f1_lt_0p5':sum(r['all_three_f1_lt_0p5'] for r in fake),'all_three_fake_f1_lt_0p7':sum(r['all_three_f1_lt_0p7'] for r in fake),'pairwise_f1_correlation':{}}
 for ai,a in enumerate(SEEDS):
  for b in SEEDS[ai+1:]:report['seed_error_overlap']['pairwise_f1_correlation'][f'{a}_{b}']=float(np.corrcoef([r[f'f1_{a}'] for r in fake],[r[f'f1_{b}'] for r in fake])[0,1])
 report['error_geometry']={'fake_mean_interior_recall':float(np.mean([r['interior_recall'] for r in fake])),'fake_mean_boundary_recall':float(np.mean([r['boundary_recall'] for r in fake])),'fake_mean_fp_far_fraction':float(np.mean([r['fp_far_fraction'] for r in fake])),'fake_mean_fp_near_boundary_fraction':float(np.mean([r['fp_near_boundary_fraction'] for r in fake])),'fake_fp_gradient_over_tn_gradient':float(np.mean([r['fp_gradient_mean']/max(r['tn_gradient_mean'],1e-9) for r in fake]))}
 report['common_hard_fake_images']=sorted(({k:r[k] for k in ('index','name','gt_ratio','mean_f1','std_f1','dominant_error','interior_recall','boundary_recall','fp_far_fraction')} for r in fake),key=lambda x:x['mean_f1'])[:15]
 report['highest_authentic_fp_images']=sorted(({k:r[k] for k in ('index','name','mean_f1','dominant_error') }|{'mean_fp_pixels':float(np.mean([r[f'fp_{s}'] for s in SEEDS]))} for r in auth),key=lambda x:x['mean_fp_pixels'],reverse=True)[:10]
 (OUT/'report.yaml').write_text(yaml.safe_dump(report,sort_keys=False));print(yaml.safe_dump(report,sort_keys=False))
if __name__=='__main__':main()
