import copy,json,random,sys,traceback
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from run_qtr_localnorm_jpeg_pilot import Fixed

PARENT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
MANIFEST=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/dct_dequant_signedlog_probe_20260922')
N_PER_GROUP=64;SEED=3407;EPS=1e-8

def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def pathof(r):
    for k in ('image','image_path','path','rgb_path'):
        v=getattr(r,k,None)
        if v:return str(v)
    return str(r)
def group(r):return 'original_jpeg' if Path(pathof(r)).suffix.lower() in {'.jpg','.jpeg','.jpe'} else 'original_nonjpeg'

def stabilize_dct_volume(dct,qtable):
    """21-bin quantized coefficient volume -> dequantized signed-log -> soft 21-bin volume."""
    if dct.shape[1]!=21:raise ValueError(tuple(dct.shape))
    q=qtable
    if q.ndim==3:q=q.unsqueeze(1)
    h,w=dct.shape[-2:];qh=(h+7)//8;qw=(w+7)//8
    qmap=q.repeat(1,1,qh,qw)[...,:h,:w].to(device=dct.device,dtype=torch.float32)
    bins=torch.arange(-10,11,device=dct.device,dtype=torch.float32).view(1,21,1,1)
    cq=(dct.float()*bins).sum(1,keepdim=True)
    deq=cq*qmap
    z=torch.sign(deq)*torch.log1p(deq.abs())
    z=z.clamp(-10,10)
    lo=torch.floor(z);hi=torch.ceil(z);whi=z-lo;wlo=1-whi
    idxlo=(lo+10).long();idxhi=(hi+10).long()
    out=torch.zeros((dct.shape[0],21,h,w),device=dct.device,dtype=torch.float32)
    out.scatter_add_(1,idxlo,wlo);out.scatter_add_(1,idxhi,whi)
    return out

def batch(s):return {k:(v.unsqueeze(0).cuda() if torch.is_tensor(v) else [v]) for k,v in s.items()}
def feat(branch,s):
    b=batch(s);dv=stabilize_dct_volume(b['dct'],b['qtable'])
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):o=branch(dv,b['qtable'])
    return [x.detach().float().cpu() for x in o]
def met(a,b):
    x=a.double().reshape(-1);y=b.double().reshape(-1);nx=torch.linalg.vector_norm(x);ny=torch.linalg.vector_norm(y)
    return {'cosine':float(torch.dot(x,y)/(nx*ny+EPS)),'relative_l2':float(torch.linalg.vector_norm(y-x)/(nx+EPS)),'norm_ratio_q90_over_clean':float(ny/(nx+EPS))}
def aggregate(rows):
    out={}
    for g in ('original_jpeg','original_nonjpeg'):
        rr=[r for r in rows if r['group']==g];out[g]={'n':len(rr),'layers':{}}
        for j in range(4):
            vals=[r['layers'][f'D{j+1}'] for r in rr];out[g]['layers'][f'D{j+1}']={}
            for k in ('cosine','relative_l2','norm_ratio_q90_over_clean'):
                z=np.asarray([v[k] for v in vals]);out[g]['layers'][f'D{j+1}'][k]={'mean':float(z.mean()),'median':float(np.median(z)),'std':float(z.std())}
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True);s={'status':'preparing','training':False,'operation':'Q-aware dequantization + signed-log + soft re-encoding to original 21-bin interface','n_per_group':N_PER_GROUP,'gate':{'mean_d1_d4_cosine_original_nonjpeg_min':.50}};dump('summary.json',s)
    try:
        ck=torch.load(PARENT,map_location='cpu');assert int(ck['epoch'])==76;cfg=copy.deepcopy(ck['config'])
        model=fair.e.build_final_model(cfg['model']).cuda();model.load_state_dict(ck['model'],strict=True);model.eval();branch=model.dct_branch
        va=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'val');base=FairDataset(va,OUT/'clean_cache',False);q90=Fixed(base,90)
        by=defaultdict(list)
        for i,r in enumerate(va):by[group(r)].append(i)
        rng=random.Random(SEED);sel=[]
        for g in ('original_jpeg','original_nonjpeg'):
            ids=by[g][:];rng.shuffle(ids);sel += [(g,i) for i in ids[:N_PER_GROUP]]
        rows=[]
        for n,(g,i) in enumerate(sel,1):
            a=feat(branch,base[i]);b=feat(branch,q90[i]);rows.append({'index':i,'path':pathof(va[i]),'group':g,'layers':{f'D{j+1}':met(a[j],b[j]) for j in range(4)}})
            if n==1 or n%10==0 or n==len(sel):dump('progress.json',{'done':n,'total':len(sel),'group':g})
        agg=aggregate(rows);score=float(np.mean([agg['original_nonjpeg']['layers'][f'D{i}']['cosine']['mean'] for i in range(1,5)]));passed=score>.50
        dump('per_image.json',rows);s.update(status='completed',aggregate=agg,original_nonjpeg_mean_d1_d4_cosine=score,decision='PASS_train_E4' if passed else 'REJECT_no_training',artifacts={'per_image':str(OUT/'per_image.json')});dump('summary.json',s);print(json.dumps(s,indent=2))
    except BaseException:s.update(status='failed',traceback=traceback.format_exc());dump('summary.json',s);raise
if __name__=='__main__':main()
