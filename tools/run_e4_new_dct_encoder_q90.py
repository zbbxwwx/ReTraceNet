"""E4: random new compression-stabilized DCT encoder + fusion, parent epoch76 frozen."""
import copy, json, random, time, traceback, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from retracenet.cat_dct_stream import DCTCompressionForensicStream
from run_e3_fullmodel_jpeg_adaptation import Fixed, jpeg_view
from run_new_dct_representation_probe import make_dct_matrix, AC63

PARENT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
MANIFEST=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e4_rgbdct_qtablefree_encoder_q90_5ep_20260923')
SEED=3407; EPOCHS=5

def dump(name,obj): stable.dump(OUT/name,obj)
class ConvBlock(torch.nn.Sequential):
    def __init__(self,ci,co,stride=1):
        g=max(1,min(32,co//4))
        while co%g:g-=1
        super().__init__(torch.nn.Conv2d(ci,co,3,stride,1,bias=False),torch.nn.GroupNorm(g,co),torch.nn.GELU(),torch.nn.Conv2d(co,co,3,1,1,bias=False),torch.nn.GroupNorm(g,co),torch.nn.GELU())
class RGBDCTEncoder(torch.nn.Module):
    # Input grid is H/8 x W/8. Outputs align to residual pyramid /4,/8,/16,/32.
    def __init__(self,channels):
        super().__init__(); c1,c2,c3,c4=channels
        self.stem=ConvBlock(63,96); self.p1=torch.nn.Conv2d(96,c1,1); self.p2=torch.nn.Conv2d(96,c2,1)
        self.s3=ConvBlock(96,192,2); self.p3=torch.nn.Conv2d(192,c3,1)
        self.s4=ConvBlock(192,384,2); self.p4=torch.nn.Conv2d(384,c4,1)
    def forward(self,dct,qtable=None):
        x=self.stem(dct.float()); x3=self.s3(x); x4=self.s4(x3)
        return [self.p1(torch.nn.functional.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)),self.p2(x),self.p3(x3),self.p4(x4)]
def rgb_dct(sample):
    image=sample['image'].float(); _,h,w=image.shape; h=(h//8)*8; w=(w//8)*8
    y=.299*image[0,:h,:w]+.587*image[1,:h,:w]+.114*image[2,:h,:w]
    blocks=y.unfold(0,8,8).unfold(1,8,8).contiguous(); C=make_dct_matrix(blocks.device,blocks.dtype)
    tmp=torch.einsum('ab,ijbc->ijac',C,blocks); d=torch.einsum('ijab,cb->ijac',tmp,C)
    freq=d.permute(2,3,0,1).reshape(64,h//8,w//8); idx=[u*8+v for u,v in AC63]
    z=freq[idx]; z=torch.sign(z)*torch.log1p(z.abs()); mu=z.mean((-2,-1),keepdim=True); var=(z-mu).square().mean((-2,-1),keepdim=True)
    out=dict(sample); out['dct']=(z-mu)*torch.rsqrt(var+1e-6); out['qtable']=torch.ones((1,8,8)); out['dct_available']=torch.ones(1); return out
class RGBDCTDataset(torch.utils.data.Dataset):
    def __init__(self,base): self.base=base; self.records=base.records
    def __len__(self): return len(self.base)
    def __getitem__(self,key): return rgb_dct(self.base[key])
class CleanQ90(torch.utils.data.Dataset):
    def __init__(self,base): self.base=base; self.records=base.records
    def __len__(self): return len(self.base)
    def __getitem__(self,key):
        s=self.base[key]; seed=key[1] if isinstance(key,tuple) else SEED+int(key)
        s=s if int(np.random.default_rng(seed).integers(2))==0 else jpeg_view(s,90)
        return rgb_dct(s)

def set_mode(model):
    model.eval(); model.dct_branch.train(); model.balanced_forensic_fusions.train()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary={'status':'preparing','experiment':'E4_RGB_DCT_qtable_free_encoder_Q90','parent':str(PARENT),'epochs':EPOCHS,
      'trainable':['dct_branch','balanced_forensic_fusions'],'frozen':['content','residual','QTR','HCV','decoder','DHER','mask_head'],
      'representation':'decoded RGB -> Y -> orthonormal 8x8 DCT -> AC63 -> signed-log -> per-frequency norm; qtable-free','train_views':'50% clean + 50% Q90','targets':{'clean_f1_reference':.937,'q90_f1_reference':.523},'test_read':False}
    dump('summary.yaml',summary)
    try:
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        ck=torch.load(PARENT,map_location='cpu'); assert int(ck['epoch'])==76
        cfg=copy.deepcopy(ck['config']); cfg['training']['dct_sample_dropout']=0.
        tr=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'train'); va=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'val')
        model=fair.e.build_final_model(cfg['model']).cuda(); model.load_state_dict(ck['model'],strict=True)
        channels=tuple(cfg['model'].get('forensic_channels',(64,128,256,512)))
        model.dct_branch=RGBDCTEncoder(channels).cuda()
        for p in model.parameters(): p.requires_grad_(False)
        for p in model.dct_branch.parameters(): p.requires_grad_(True)
        for p in model.balanced_forensic_fusions.parameters(): p.requires_grad_(True)
        trainable=[p for p in model.parameters() if p.requires_grad]
        groups=[{'params':model.dct_branch.parameters(),'lr':1e-4,'weight_decay':1e-4},
                {'params':model.balanced_forensic_fusions.parameters(),'lr':5e-5,'weight_decay':1e-4}]
        opt=torch.optim.AdamW(groups)
        seg=stable.v3.e24.ReTraceLoss.from_config(stable.v3.e24.clean_loss_config(cfg['loss'])).cuda()
        fake=[i for i,r in enumerate(tr) if r.label]; auth=[i for i,r in enumerate(tr) if not r.label]; nf=min(len(fake),len(auth))
        base_train=FairDataset(tr,OUT/'jpeg_cache_train',True); base_val=FairDataset(va,OUT/'jpeg_cache_val',False)
        clean_val=RGBDCTDataset(base_val); q90_val=RGBDCTDataset(Fixed(base_val,90)); history=[]; started=time.time()
        for ep in range(1,EPOCHS+1):
            g=torch.Generator().manual_seed(SEED+ep); ids=[fake[i] for i in torch.randperm(len(fake),generator=g)[:nf]]+[auth[i] for i in torch.randperm(len(auth),generator=g)[:nf]]
            ids=[ids[i] for i in torch.randperm(len(ids),generator=g)]; plan=[(idx,SEED*1000000+ep*10000+j) for j,idx in enumerate(ids)]
            loader=DataLoader(CleanQ90(base_train),batch_size=4,sampler=plan,num_workers=2,pin_memory=True)
            set_mode(model); total=0.
            for st,b in enumerate(loader,1):
                b={k:(v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k,v in b.items()}; opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): out=stable.v3.fwd(model,b,cfg,False); loss=stable.v3.objective(out,b,seg,cfg)['total']
                loss.backward(); gn=torch.nn.utils.clip_grad_norm_(trainable,5.,error_if_nonfinite=True); opt.step(); total+=float(loss)
                if st==1 or st%100==0 or st==len(loader):
                    dump('progress.yaml',{'epoch':ep,'batch':st,'batches':len(loader),'loss':total/st,'grad_norm':float(gn),'elapsed_minutes':(time.time()-started)/60})
                    print('[E4]',ep,st,len(loader),total/st,flush=True)
            model.eval(); vals={}
            for name,ds in [('clean',clean_val),('q90',q90_val)]:
                z=fair.evaluate(model,ds,cfg,OUT,f'ep{ep:02d}_{name}'); m=z['metrics']['fake']
                vals[name]={'f1':float(m['oracle']['f1']),'precision':float(m['oracle']['precision']),'recall':float(m['oracle']['recall']),'auc':float(m['auc']),'threshold':float(m['oracle']['threshold'])}
            row={'epoch':ep,'train_loss':total/len(loader),**vals}; history.append(row)
            torch.save({'epoch':ep,'parent_epoch':76,'model':model.state_dict(),'optimizer':opt.state_dict(),'config':cfg,'history':history},OUT/'last.pt')
            best=max(history,key=lambda x:(x['q90']['f1'],x['clean']['f1']))
            if best['epoch']==ep: torch.save({'epoch':ep,'parent_epoch':76,'model':model.state_dict(),'optimizer':opt.state_dict(),'config':cfg,'history':history},OUT/'best.pt')
            summary.update(status='training',history=history,best_epoch=best['epoch'],best_clean=best['clean'],best_q90=best['q90']); dump('summary.yaml',summary)
        best=max(history,key=lambda x:(x['q90']['f1'],x['clean']['f1']))
        summary.update(status='completed',history=history,best_epoch=best['epoch'],best_clean=best['clean'],best_q90=best['q90'],decision='PROMOTE_joint_finetune' if best['q90']['f1']>.523 and best['clean']['f1']>=.92 else 'STOP')
        dump('summary.yaml',summary); print(json.dumps(summary,indent=2),flush=True)
    except BaseException:
        summary.update(status='failed',traceback=traceback.format_exc()); dump('summary.yaml',summary); raise
if __name__=='__main__': main()
