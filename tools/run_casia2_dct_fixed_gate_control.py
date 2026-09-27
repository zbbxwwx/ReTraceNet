"""Residual-anchored, feature-conditioned DCT reliability pilot on CASIA2."""
import copy, json, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_casia2_mit_jpeg_only_probe as jp
import run_imd_mit_full_v3  # noqa: F401; patches model builder to MiT-V3
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset

PARENT=jp.PARENT
MANIFEST=jp.MANIFEST
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_dct_fixed_gate_control1_20260921')
SEED=3407

def dump(name,obj): stable.dump(OUT/name,obj)

class AdaptiveDCTReliability(nn.Module):
    """Preserve the trained fusion, but conditionally attenuate its DCT input."""
    def __init__(self, base):
        super().__init__(); self.base=base
        self.mlp=nn.Sequential(nn.Linear(4,16),nn.GELU(),nn.Linear(16,1))
        nn.init.zeros_(self.mlp[-1].weight);nn.init.constant_(self.mlp[-1].bias,4.0)
        self.register_buffer('alpha_sum',torch.zeros(()),persistent=False)
        self.register_buffer('alpha_count',torch.zeros(()),persistent=False)
    def reset_diagnostics(self): self.alpha_sum.zero_();self.alpha_count.zero_()
    def alpha_mean(self): return float((self.alpha_sum/self.alpha_count.clamp_min(1)).cpu())
    def forward(self,residual,dct,available):
        if dct.shape[-2:]!=residual.shape[-2:]:
            dct=torch.nn.functional.interpolate(dct,size=residual.shape[-2:],mode='bilinear',align_corners=False)
        r=self.base.residual(residual);d=self.base.dct(dct)
        rr=r.float().square().mean((1,2,3)).sqrt();dd=d.float().square().mean((1,2,3)).sqrt()
        corr=(r.float()*d.float()).mean((1,2,3))/(rr*dd+1e-6)
        gap=(r.float()-d.float()).abs().mean((1,2,3))
        stats=torch.stack([rr,dd,dd/(rr+1e-6),corr-gap],1).to(dtype=r.dtype)
        alpha=torch.sigmoid(self.mlp(stats))*available.to(dtype=r.dtype)
        self.alpha_sum.add_(alpha.detach().float().sum());self.alpha_count.add_(alpha.numel())
        out=self.base.fuse(torch.cat([r,d*alpha[:,:,None,None]],1))
        return out,r,d*alpha[:,:,None,None]

def wrap(model):
    device=next(model.parameters()).device
    wrapped=[]
    for old in model.balanced_forensic_fusions:
        wrapped.append(AdaptiveDCTReliability(old).to(device))
    model.balanced_forensic_fusions=nn.ModuleList(wrapped)
    return wrapped

def reset_diag(gates):
    for g in gates:g.reset_diagnostics()

def diag(gates): return [g.alpha_mean() for g in gates]

def evaluate(model,ds,cfg,tag,gates):
    reset_diag(gates);res=fair.evaluate(model,ds,cfg,OUT,tag);return res,diag(gates)

def metric(res):
    q=res['metrics']['fake']['oracle'];return {'f1':float(q['f1']),'precision':float(q['precision']),'recall':float(q['recall']),'auc':float(res['metrics']['fake']['auc']),'threshold':float(q['threshold'])}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary={'status':'preparing','experiment':'fixed_alpha_dct_control','parent':str(PARENT),'gpu':1,'epochs':1,'test_run':False,'external_datasets_read':False,'gate_lr':0.0,'base_lr':1e-5,'initial_alpha':float(torch.sigmoid(torch.tensor(4.0)))};dump('summary.yaml',summary)
    try:
        torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
        ck=torch.load(PARENT,map_location='cpu');cfg=copy.deepcopy(ck['config']);cfg['training']['dct_sample_dropout']=0.;cfg['training']['learning_rate']=1e-5;cfg['training']['epochs']=1
        tr=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'train');vr=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'val')
        fake=[i for i,r in enumerate(tr) if r.label];auth=[i for i,r in enumerate(tr) if not r.label];nf=min(len(fake),len(auth))
        base=FairDataset(tr,OUT/'jpeg_cache_train',True);train=jp.TrainJPEG50(base)
        clean=FairDataset(vr,OUT/'jpeg_cache_val',False);shift=jp.ShiftVal(clean)
        model=fair.e.build_final_model(cfg['model']).cuda();model.load_state_dict(ck['model'],strict=True);gates=wrap(model);stable.train_mode(model)
        parent_clean,a0c=evaluate(model,clean,cfg,'parent_clean',gates);parent_shift,a0s=evaluate(model,shift,cfg,'parent_shift',gates)
        seg=stable.v3.e24.ReTraceLoss.from_config(stable.v3.e24.clean_loss_config(cfg['loss'])).cuda()
        gate_params=[p for g in gates for p in g.mlp.parameters()]
        for p in gate_params: p.requires_grad_(False)
        gate_ids={id(p) for p in gate_params};base_params=[p for p in model.parameters() if id(p) not in gate_ids]
        opt=torch.optim.SGD([{'params':base_params,'lr':1e-5}],momentum=.9,weight_decay=.0005)
        gen=torch.Generator().manual_seed(SEED+1);ids=[fake[i] for i in torch.randperm(len(fake),generator=gen)[:nf]]+[auth[i] for i in torch.randperm(len(auth),generator=gen)[:nf]];ids=[ids[i] for i in torch.randperm(len(ids),generator=gen)];plan=[(idx,SEED*1000000+100000+j) for j,idx in enumerate(ids)]
        loader=DataLoader(train,batch_size=4,sampler=plan,num_workers=2,pin_memory=True);total=0.;started=time.time();reset_diag(gates)
        for st,b in enumerate(loader,1):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=stable.v3.fwd(model,b,cfg,False);terms=stable.v3.objective(out,b,seg,cfg);loss=terms['total']
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();total+=float(loss)
            if st==1 or st%25==0 or st==len(loader):
                dump('progress.yaml',{'batch':st,'batches':len(loader),'loss':total/st,'grad_norm':float(gn),'train_gate_mean':diag(gates),'elapsed_minutes':(time.time()-started)/60});print('[dct-reliability]',st,len(loader),total/st,flush=True)
        clean_res,ac=evaluate(model,clean,cfg,'clean_val',gates);shift_res,ash=evaluate(model,shift,cfg,'shift_val',gates)
        cm,sm=metric(clean_res),metric(shift_res);pcm,psm=metric(parent_clean),metric(parent_shift)
        state={'model':fair.e.copy_state(model),'optimizer':opt.state_dict(),'config':cfg,'epoch':1,'parent_epoch':76,'selection_threshold':cm['threshold'],'best_metric':cm['f1'],'custom_module':'AdaptiveDCTReliability_fixed_constant_alpha'};fair.e24.atomic_torch_save(state,OUT/'epoch01.pt')
        summary.update(status='completed',train_loss=total/len(loader),parent_clean=pcm,parent_shift=psm,clean_val=cm,shift_val=sm,parent_gate_clean=a0c,parent_gate_shift=a0s,trained_gate_clean=ac,trained_gate_shift=ash,delta_clean_f1=cm['f1']-pcm['f1'],delta_shift_f1=sm['f1']-psm['f1'],clean_gate_pass=cm['f1']>=.936,decision='run_external_probe' if cm['f1']>=.936 and sm['f1']>=psm['f1']+.03 else 'reject_or_revise');dump('summary.yaml',summary);print(json.dumps(summary,indent=2),flush=True)
    except BaseException:
        summary.update(status='failed',stop_reason=traceback.format_exc());dump('summary.yaml',summary);raise
if __name__=='__main__':main()
