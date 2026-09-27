import copy, json, math, random, sys, traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from run_qtr_localnorm_jpeg_pilot import Fixed

PARENT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
MANIFEST=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/jpeg_representation_drift_probe_20260922')
N_PER_GROUP=64;SEED=3407;EPS=1e-8

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))

def image_path(r):
    for k in ('image','image_path','path','rgb_path'):
        v=getattr(r,k,None)
        if v:return str(v)
    if isinstance(r,dict):
        for k in ('image','image_path','path','rgb_path'):
            if r.get(k):return str(r[k])
    return str(r)

def original_group(r):
    return 'original_jpeg' if Path(image_path(r)).suffix.lower() in {'.jpg','.jpeg','.jpe'} else 'original_nonjpeg'

class Capture:
    def __init__(self,model):
        self.cur={};self.handles=[]
        self.handles.append(model.content_encoder.register_forward_hook(lambda m,i,o:self._list('C',o)))
        self.handles.append(model.forensic_stream.register_forward_hook(lambda m,i,o:self._list('R',o)))
        self.handles.append(model.dct_branch.register_forward_hook(lambda m,i,o:self._list('D',o)))
        for j,m in enumerate(model.balanced_forensic_fusions):
            self.handles.append(m.register_forward_hook(lambda m,i,o,j=j:self._one('N',j,o[0] if isinstance(o,(tuple,list)) else o)))
        self.handles.append(model.qtr.register_forward_hook(lambda m,i,o:self._list('Q',o['qualified_detail'])))
    def _list(self,prefix,out):
        for j,x in enumerate(out):self._one(prefix,j,x)
    def _one(self,prefix,j,x):self.cur[f'{prefix}{j+1}']=x.detach().float().cpu()
    def reset(self):self.cur={}
    def close(self):
        for h in self.handles:h.remove()

def batchify(s):
    out={}
    for k,v in s.items():
        if torch.is_tensor(v):out[k]=v.unsqueeze(0).cuda(non_blocking=True)
        else:out[k]=[v]
    return out

def run_one(model,capture,sample,cfg):
    capture.reset()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        stable.v3.fwd(model,batchify(sample),cfg,False)
    return {k:v.clone() for k,v in capture.cur.items()}

def metrics(a,b):
    x=a.double().reshape(-1);y=b.double().reshape(-1)
    nx=torch.linalg.vector_norm(x);ny=torch.linalg.vector_norm(y)
    cos=float(torch.dot(x,y)/(nx*ny+EPS));rel=float(torch.linalg.vector_norm(y-x)/(nx+EPS));ratio=float(ny/(nx+EPS))
    return {'cosine':cos,'relative_l2':rel,'norm_ratio_q90_over_clean':ratio}

def summarize(rows):
    out={}
    groups=sorted({r['group'] for r in rows})
    layers=sorted({k for r in rows for k in r['layers']},key=lambda s:(s[0],int(s[1:])))
    for g in groups:
        rr=[r for r in rows if r['group']==g];out[g]={'n':len(rr),'layers':{}}
        for layer in layers:
            vals=[r['layers'][layer] for r in rr if layer in r['layers']]
            out[g]['layers'][layer]={}
            for key in ('cosine','relative_l2','norm_ratio_q90_over_clean'):
                z=np.asarray([v[key] for v in vals],dtype=np.float64)
                out[g]['layers'][layer][key]={'mean':float(z.mean()),'median':float(np.median(z)),'std':float(z.std()),'p10':float(np.quantile(z,.1)),'p90':float(np.quantile(z,.9))}
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True);status={'status':'preparing','parent':str(PARENT),'training':False,'pair':'same val image clean vs JPEG Q90','n_per_group':N_PER_GROUP,'requested_features':['C1-C4','R1-R4','D1-D4','N1-N4','Q1-Q4'],'actual_qtr_scales':['Q1','Q2','Q3'],'Q4':'unavailable_by_architecture'};dump('summary.json',status)
    try:
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
        ck=torch.load(PARENT,map_location='cpu');assert int(ck['epoch'])==76
        cfg=copy.deepcopy(ck['config'])
        model=fair.e.build_final_model(cfg['model']).cuda();model.load_state_dict(ck['model'],strict=True);model.eval()
        va=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'val');base=FairDataset(va,OUT/'clean_cache',False);q90=Fixed(base,90)
        by=defaultdict(list)
        for i,r in enumerate(va):by[original_group(r)].append(i)
        rng=random.Random(SEED);selected=[]
        for g in ('original_jpeg','original_nonjpeg'):
            ids=by[g][:];rng.shuffle(ids);selected += [(g,i) for i in ids[:N_PER_GROUP]]
        if any(sum(g==x for x,_ in selected)==0 for g in ('original_jpeg','original_nonjpeg')):raise RuntimeError({k:len(v) for k,v in by.items()})
        status.update(status='running',available_counts={k:len(v) for k,v in by.items()},selected_counts={g:sum(x==g for x,_ in selected) for g in by});dump('summary.json',status)
        cap=Capture(model);rows=[]
        for n,(g,i) in enumerate(selected,1):
            a=run_one(model,cap,base[i],cfg);b=run_one(model,cap,q90[i],cfg)
            common=sorted(set(a)&set(b),key=lambda s:(s[0],int(s[1:])))
            rows.append({'index':i,'path':image_path(va[i]),'group':g,'layers':{k:metrics(a[k],b[k]) for k in common}})
            if n==1 or n%10==0 or n==len(selected):dump('progress.json',{'done':n,'total':len(selected),'current_group':g,'current_path':image_path(va[i])})
        cap.close();dump('per_image.json',rows);agg=summarize(rows)
        status.update(status='completed',n=len(rows),aggregate=agg,artifacts={'per_image':str(OUT/'per_image.json'),'summary':str(OUT/'summary.json')});dump('summary.json',status)
        print(json.dumps(status,indent=2),flush=True)
    except BaseException:
        status.update(status='failed',traceback=traceback.format_exc());dump('summary.json',status);raise
if __name__=='__main__':main()
