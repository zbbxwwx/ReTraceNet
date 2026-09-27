"""CASIA2-only MiT-V3 DG post-training with synchronized JPEG views."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
import torch
import torch.nn.functional as F

# Importing this patches the shared model builder to the exact MiT-B2 V3 model.
import run_imd_mit_full_v3  # noqa: F401
import run_v3_stable_bn_indomain as stable
import takeover_views as views
import takeover_train as train

class DGPairedDataset(torch.utils.data.Dataset):
    # Discrete requested qualities plus explicit recompression/subsampling.
    choices=['jpeg50','jpeg60','jpeg70','jpeg80','jpeg90','jpeg100',
             'recompression','subsampling444','subsampling420']
    def __init__(self,base): self.base=base; self.records=base.records
    def __len__(self): return len(self.base)
    def __getitem__(self,key):
        sample=self.base[key]
        seed=key[1] if isinstance(key,tuple) else views.stable_seed(sample['path'])
        rng=np.random.default_rng(seed)
        cond=self.choices[int(rng.integers(len(self.choices)))]
        return {'a':sample,'b':views.make_view(sample,cond,seed+301),'condition':cond}

def dg_step(model,batch,seg,cfg,strategy,cons_weight,rng,optimizer):
    assert strategy=='paired'
    optimizer.zero_grad(set_to_none=True)
    a,b=batch['a'],dict(batch['b'])
    # Native teacher always has its available DCT; degraded student independently
    # drops DCT for 30% of samples so residual/content must remain useful.
    keep=(torch.rand(b['dct_available'].shape,generator=rng)>=0.30).to(b['dct_available'].dtype)
    b['dct_available']=b['dct_available']*keep
    with torch.autocast('cuda',dtype=torch.bfloat16):
        oa=stable.v3.fwd(model,a,cfg,False)
        la=stable.v3.objective(oa,a,seg,cfg)['total']
        pa=oa['mask_logits'].detach().float().sigmoid()
    if not torch.isfinite(la): raise FloatingPointError('nonfinite teacher loss')
    la.backward(); la_value=float(la.detach()); del oa,la
    with torch.autocast('cuda',dtype=torch.bfloat16):
        ob=stable.v3.fwd(model,b,cfg,False)
        lb=stable.v3.objective(ob,b,seg,cfg)['total']
    pb=ob['mask_logits'].float().sigmoid(); valid=b['valid'].cuda().bool()
    confident=(pa>=0.8)|(pa<=0.2); selected=valid&confident
    consistency=F.mse_loss(pb[selected],pa[selected]) if bool(selected.any()) else pb.sum()*0.0
    loss=lb+cons_weight*consistency
    if not torch.isfinite(loss): raise FloatingPointError('nonfinite student loss')
    loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    optimizer.step()
    return la_value+float(loss.detach()),float(consistency.detach()),float(grad)

train.PairedDataset=DGPairedDataset
train.training_step=dg_step

if __name__=='__main__':
    p=train.argparse.ArgumentParser()
    p.add_argument('--out',required=True); p.add_argument('--parent',required=True)
    p.add_argument('--manifest',required=True); p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--workers',type=int,default=2); p.add_argument('--lambda-cons',type=float,default=.05)
    p.add_argument('--public-init',action='store_true')
    a=p.parse_args(); a.strategy='paired'
    train.main(a)
