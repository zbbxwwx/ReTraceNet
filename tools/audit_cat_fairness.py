"""Read-only dataset/checkpoint/protocol provenance audit. No inference/test run."""
import csv, hashlib, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch

BASE=Path('/group4/zouwenxuan/iml_generalization')
CAT=BASE/'repos/CAT-Net';RE=BASE/'repos/ReTraceNet'
sys.path.insert(0,str(RE))
from retrace.data import read_manifest
OUT=BASE/'outputs/retracenet_v2/cat_fairness_audit_20260911'
OUT.mkdir(exist_ok=True)
report={'status':'running','test_run':False,'training_run':False,'historical_manifest_caveat':'Current lists compared; checkpoint does not necessarily bind immutable list hashes.'}
def save(): (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def metadata(path):
    path=Path(path)
    if not path.exists():return {'path':str(path),'exists':False}
    return {'path':str(path),'size':path.stat().st_size,'mtime':path.stat().st_mtime,'sha256':sha(path)}
source=BASE/'outputs/retracenet_v2/e24_dher_calibrated_writeback_20260910/best.pt'
state=torch.load(source,map_location='cpu');cfg=state['config']
report['e24']={'checkpoint':metadata(source),'keys':list(state),'config':cfg,'best_metric':state.get('best_metric'),'epoch':state.get('epoch')}
del state
catck=BASE/'outputs/catnet/splicing_dataset/CAT_full/best.pth.tar'
state=torch.load(catck,map_location='cpu')
report['cat']={'checkpoint':metadata(catck),'keys':list(state),
 'scalar_metadata':{k:v for k,v in state.items() if isinstance(v,(str,int,float,bool,type(None)))},
 'state_tensor_elements':sum(t.numel() for t in state['state_dict'].values() if torch.is_tensor(t))}
del state
records={split:read_manifest(cfg['data']['manifest'],cfg['data']['root'],split) for split in ['train','val']}
report['e24']['manifest']=metadata(cfg['data']['manifest'])
catrows={};lists={}
for split,term in [('train','train'),('val','valid')]:
    catrows[split]=[]
    for auth in ['', 'auth_']:
        f=CAT/'Splicing/data'/f'CASIA_v2_{auth}{term}_list_filtered.txt'
        rows=list(csv.reader(f.read_text().splitlines()))
        catrows[split]+=rows;lists[f.name]={**metadata(f),'rows':len(rows)}
report['cat']['lists']=lists
def name(p):return Path(p).name.lower()
sets={split:{name(r.image) for r in rs} for split,rs in records.items()}
catsets={split:{name(r[0]) for r in rs} for split,rs in catrows.items()}
report['split_comparison']={split:{'e24_rows':len(records[split]),'cat_rows':len(catrows[split]),
 'e24_label_counts':dict(Counter(int(r.label) for r in records[split])),
 'identical_name_set':sets[split]==catsets[split],
 'only_e24':sorted(sets[split]-catsets[split]),'only_cat':sorted(catsets[split]-sets[split])} for split in sets}
report['direct_name_overlap']={'cat_train_e24_val':sorted(catsets['train']&sets['val']),
 'cat_train_cat_val':sorted(catsets['train']&catsets['val']),
 'e24_train_e24_val':sorted(sets['train']&sets['val'])}
save();print('LISTS',report['split_comparison'],flush=True)

# Compare original source bytes and GT, not their recompressed derivatives.
catroot=BASE/'data/CATNET_CASIA';lookup={name(r.image):r for rs in records.values() for r in rs}
mismatches=[];gt_changes=[];missing=[];digest_cache={};pixel_cache={}
def hashes(path):
    key=str(path.resolve())
    if key not in digest_cache:
        digest_cache[key]=sha(path)
        with Image.open(path) as im:
            arr=np.asarray(im.convert('RGB'))
        pixel_cache[key]=hashlib.sha256(str(arr.shape).encode()+arr.tobytes()).hexdigest()
    return digest_cache[key],pixel_cache[key]
hash_splits={s:{'bytes':defaultdict(list),'pixels':defaultdict(list)} for s in records}
total=sum(len(v) for v in catrows.values());done=0
for split,rs in catrows.items():
    for row in rs:
        done+=1;original=catroot/row[0];r=lookup.get(name(row[0]))
        if r is None or not original.exists():missing.append(row[0]);continue
        hb,hp=hashes(r.image);cb,cp=hashes(original)
        hash_splits[split]['bytes'][hb].append(str(r.image));hash_splits[split]['pixels'][hp].append(str(r.image))
        if hb!=cb:mismatches.append({'name':name(row[0]),'pixels_equal':hp==cp,'cat_original':str(original),'e24_original':str(r.image)})
        if r.mask is not None and row[1]!='None':
            cm=catroot/row[1]
            if not cm.exists():missing.append(str(cm));continue
            ca=np.asarray(Image.open(cm).convert('L'));ea=np.asarray(Image.open(r.mask).convert('L'))
            if ca.shape!=ea.shape:
                gt_changes.append({'name':name(row[0]),'split':split,'shape_mismatch':[list(ca.shape),list(ea.shape)]})
            else:
                diff=int(np.count_nonzero((ca>0)!=(ea>127)))
                if diff:gt_changes.append({'name':name(row[0]),'split':split,'different_training_label_pixels':diff,
                                          'same_threshold127_difference':int(np.count_nonzero((ca>127)!=(ea>127)))})
        if done%1000==0:print('HASH',done,total,flush=True)
report['source_comparison']={'checked':done,'missing':missing,'byte_mismatches':mismatches,'training_GT_differences':gt_changes}
report['content_cross_split_overlap']={kind:[{'train':hash_splits['train'][kind][h],'val':hash_splits['val'][kind][h]}
    for h in sorted(hash_splits['train'][kind].keys()&hash_splits['val'][kind].keys())] for kind in ['bytes','pixels']}
report['within_split_duplicate_pixels']={s:[v for v in hash_splits[s]['pixels'].values() if len(v)>1] for s in records}
pretrained=[CAT/'pretrained_models/DCT_djpeg.pth.tar',CAT/'pretrained_models/hrnetv2_w48_imagenet_pretrained.pth']
for k,v in cfg['model'].items():
    if isinstance(v,str) and ('pretrain' in k or 'checkpoint' in k) and Path(v).is_file():pretrained.append(Path(v))
report['pretraining_artifacts']=[metadata(p) for p in dict.fromkeys(pretrained)]
report['pretraining_caveat']='Local file names, hashes and logs identify artifacts, not exhaustive original pretraining sample membership.'
report['evidence_files']=[metadata(p) for p in [CAT/'tools/train.py',CAT/'Splicing/data/data_core.py',CAT/'Splicing/data/AbstractDataset.py',CAT/'Splicing/data/dataset_CASIA.py',CAT/'experiments/CAT_full.yaml',
 BASE/'outputs/catnet/splicing_dataset/CAT_full/CAT_full_2026-08-19-07-59_train.log',BASE/'outputs/catnet/splicing_dataset/CAT_full/CAT_full_2026-08-20-05-27_train.log']]
report['status']='completed';save();print('COMPLETE',flush=True)
