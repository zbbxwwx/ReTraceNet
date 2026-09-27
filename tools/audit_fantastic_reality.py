import argparse, hashlib, json, random, re
from pathlib import Path
import numpy as np
from PIL import Image

def src_ids(path): return set(re.findall(r'IMG_\d+',Path(path).stem))
def main(a):
 root=Path(a.root);split=json.loads(Path(a.split).read_text());errors=[]
 expected={'fake':19423,'real':16592};rows=[r for s in split.values() for r in s]
 images=[r['image'] for r in rows]
 if len(images)!=len(set(images)):errors.append('duplicate image path across split')
 if sum(r['label']==1 for r in rows)!=expected['fake']:errors.append('fake count mismatch')
 if sum(r['label']==0 for r in rows)!=expected['real']:errors.append('real count mismatch')
 source_sets={s:set().union(*(src_ids(r['image']) for r in rs)) for s,rs in split.items()}
 leakage={f'{a}-{b}':sorted(source_sets[a]&source_sets[b])[:20] for a,b in [('train','val'),('train','test'),('val','test')] if source_sets[a]&source_sets[b]}
 if leakage and not a.allow_source_overlap:errors.append('source leakage')
 mask_fg=[];shape_bad=[];empty=[];missing=[];auth_mask=[];decode_bad=[]
 for j,r in enumerate(rows,1):
  ip=Path(r['image']);mp=Path(r['mask']) if r.get('mask') else None
  if not ip.is_file():missing.append(str(ip));continue
  try:
   with Image.open(ip) as im:isz=im.size;im.verify()
  except Exception as e:decode_bad.append([str(ip),str(e)]);continue
  if r['label']==0:
   if mp is not None:auth_mask.append(str(ip))
   continue
  if mp is None or not mp.is_file():missing.append(str(mp));continue
  with Image.open(mp) as m:
   if m.size!=isz:shape_bad.append([str(ip),isz,str(mp),m.size])
   arr=np.asarray(m.convert('L'))
  fg=int((arr>127).sum());mask_fg.append(fg/arr.size)
  if fg==0:empty.append(str(mp))
  if j%5000==0:print('[audit]',j,len(rows),flush=True)
 # Recheck converted masks exactly against deterministic NPZ samples.
 fakes=[r for r in rows if r['label']==1];random.Random(3407).shuffle(fakes);conversion_bad=[]
 for r in fakes[:min(1000,len(fakes))]:
  stem=Path(r['image']).stem;npz=root/'SegmentationFake'/(stem+'.npz')
  x=np.load(npz);src=(np.asarray(x[x.files[0]]).squeeze()>0)
  dst=np.asarray(Image.open(r['mask']).convert('L'))>127
  if src.shape!=dst.shape or not np.array_equal(src,dst):conversion_bad.append(stem)
 if shape_bad:errors.append('image-mask shape mismatch')
 if empty:errors.append('empty fake masks')
 if missing:errors.append('missing paths')
 if auth_mask:errors.append('authentic rows unexpectedly have mask')
 if decode_bad:errors.append('image decode failures')
 if conversion_bad:errors.append('NPZ-PNG conversion mismatch')
 ratios={k:len(v)/max(len(rows),1) for k,v in split.items()}
 report={'status':'passed' if not errors else 'failed','errors':errors,'counts':{k:len(v) for k,v in split.items()},'ratios':ratios,
  'labels':{k:{str(y):sum(r['label']==y for r in v) for y in (0,1)} for k,v in split.items()},'source_ids':{k:len(v) for k,v in source_sets.items()},'source_leakage':leakage,
  'mask_foreground_ratio':{'count':len(mask_fg),'min':float(np.min(mask_fg)) if mask_fg else None,'median':float(np.median(mask_fg)) if mask_fg else None,'mean':float(np.mean(mask_fg)) if mask_fg else None,'max':float(np.max(mask_fg)) if mask_fg else None},
  'missing_count':len(missing),'shape_bad_count':len(shape_bad),'empty_mask_count':len(empty),'decode_bad_count':len(decode_bad),'conversion_samples':min(1000,len(fakes)),'conversion_bad_count':len(conversion_bad),
  'split_sha256':hashlib.sha256(Path(a.split).read_bytes()).hexdigest()}
 out=Path(a.out);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 if errors:raise SystemExit(2)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);p.add_argument('--allow-source-overlap',action='store_true');main(p.parse_args())
