"""Frozen Fair-E22 architecture ablation audit. No training and no external test."""
import copy, json, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import run_fair_training_v1 as fair
from run_cross_dataset_after_chain import LockedCounts

BASE=Path('/group4/zouwenxuan/iml_generalization')
PARENT=BASE/'outputs/retracenet_v2/fair_training_v1_full_20260911'
CKPT=PARENT/'best.pt'
OUT=BASE/'outputs/retracenet_v2/fair_e22_architecture_ablation_20260912'
VARIANTS=['full','rgb_content_off','forensic_residual_off','qtr_bypass','hcv_evidence_chain_bypass','dher_bypass','dct_off']
LOCKED_THRESHOLD=0.45

class ZeroLikeOutput(nn.Module):
    def __init__(self, inner):
        super().__init__(); self.inner=inner
    def forward(self,*args,**kwargs):
        out=self.inner(*args,**kwargs)
        return [torch.zeros_like(x) for x in out] if isinstance(out,(list,tuple)) else torch.zeros_like(out)

def dump(path,obj):
    fair.e.dump(Path(path),obj)

def build(variant):
    ck=torch.load(CKPT,map_location='cpu')
    cfg=copy.deepcopy(ck['config'])
    model=fair.e.build_final_model(cfg['model']).cuda().eval()
    model.load_state_dict(ck['model'],strict=True)
    if variant=='rgb_content_off':
        model.content_encoder=ZeroLikeOutput(model.content_encoder)
    elif variant=='forensic_residual_off':
        model.forensic_stream=ZeroLikeOutput(model.forensic_stream)
    elif variant=='qtr_bypass':
        model.qtr=None
    elif variant=='hcv_evidence_chain_bypass':
        model.hcv2=None
    elif variant=='dher_bypass':
        model.dher=None
    elif variant=='dct_off':
        model.disable_dct_evidence=True
    return cfg,model,ck

@torch.inference_mode()
def run_variant(variant,ds,cfg,model):
    stats=fair.e.ValidationStats(); locked=LockedCounts(LOCKED_THRESHOLD)
    for i,r in enumerate(ds.records):
        b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)))
        o=fair.fwd(model,b); vh,vw=map(int,b['valid_hw'][0].tolist())
        p=o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy()
        y=b['mask'][0,0,:vh,:vw].numpy().astype(np.int16)
        stats.add(p,y,r.label,r.image); locked.add(p,y)
        if i==0 or (i+1)%200==0:
            print(f'[audit:{variant}] {i+1}/{len(ds)}',flush=True)
    metrics=stats.result()
    return {'variant':variant,'training_run':False,'dataset':'CASIA2 validation',
            'locked_threshold_source':'Fair E22 validation selection','locked':locked.result(),
            'metrics':metrics}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    state={'status':'running','checkpoint':str(CKPT),'training_run':False,
           'protocol':'same Fair-E22 CASIA2 validation inputs; raw; original GT grid',
           'locked_threshold':LOCKED_THRESHOLD,'variants':VARIANTS,'completed':[]}
    prior=(OUT/'state.json')
    if prior.exists():
        try: state.update(json.loads(prior.read_text()))
        except Exception: pass
    state['status']='running'; dump(OUT/'state.yaml',state); prior.write_text(json.dumps(state,indent=2))
    try:
        ck=torch.load(CKPT,map_location='cpu'); cfg=copy.deepcopy(ck['config']); del ck
        records=fair.e.read_manifest(PARENT/'clean_manifest.csv',cfg['data']['root'],'val')
        ds=fair.FairDataset(records,PARENT/'jpeg_cache',False)
        # One-batch switch audit before the long run.
        b=next(iter(DataLoader(ds,batch_size=1,sampler=[0],num_workers=0)))
        _,full,_=build('full'); base=fair.fwd(full,b)['mask_logits'].float()
        sanity={}
        del full
        for v in VARIANTS[1:]:
            _,m,_=build(v); pred=fair.fwd(m,b)['mask_logits'].float()
            sanity[v]={'mean_abs_logit_diff':float((pred-base).abs().mean()),
                       'max_abs_logit_diff':float((pred-base).abs().max())}
            del m; torch.cuda.empty_cache()
        dump(OUT/'switch_sanity.yaml',sanity)
        results={}
        for v in VARIANTS:
            dest=OUT/(v+'.yaml')
            if dest.exists():
                import yaml; result=yaml.safe_load(dest.read_text())
            else:
                cfg,m,_=build(v); result=run_variant(v,ds,cfg,m); dump(dest,result)
                del m; torch.cuda.empty_cache()
            results[v]=result
            state['completed']=list(results); prior.write_text(json.dumps(state,indent=2)); dump(OUT/'state.yaml',state)
        full=results['full']
        rows=[]
        for v in VARIANTS:
            r=results[v]
            rows.append({'variant':v,'locked_f1':r['locked']['f1'],
                         'locked_delta':r['locked']['f1']-full['locked']['f1'],
                         'oracle_fake_f1':r['metrics']['fake']['oracle']['f1'],
                         'oracle_delta':r['metrics']['fake']['oracle']['f1']-full['metrics']['fake']['oracle']['f1'],
                         'auc_fake':r['metrics']['fake']['auc']})
        summary={'status':'completed','training_run':False,'checkpoint':str(CKPT),
                 'locked_threshold':LOCKED_THRESHOLD,'switch_sanity':sanity,'table':rows,
                 'interpretation_limit':'hcv_evidence_chain_bypass also disables dependent DHER execution; it is not a pure HCV-only causal effect'}
        dump(OUT/'summary.yaml',summary); state.update(status='completed',table=rows)
        prior.write_text(json.dumps(state,indent=2)); dump(OUT/'state.yaml',state)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc())
        prior.write_text(json.dumps(state,indent=2)); dump(OUT/'state.yaml',state); raise
if __name__=='__main__': main()
