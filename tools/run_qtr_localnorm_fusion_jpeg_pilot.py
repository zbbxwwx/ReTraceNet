"""E1 QTR local-normalized reliability: one epoch CASIA2 JPEG pilot."""
import copy, hashlib, json, random, tempfile, time, traceback, types, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_imd_mit_full_v3
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from retrace.data import ManipulationDataset
from retracenet.final_modules import gray_quality_descriptor,norm2d

PARENT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
MANIFEST=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e2_qtr_localnorm_fusion_jpeg1_20260922')
SEED=3407; CONDITIONS=('clean',90,80,70)

def dump(name,obj):stable.dump(OUT/name,obj)
def local_norm(x,k=7,eps=1e-6):
    mu=F.avg_pool2d(x,k,1,k//2);var=(F.avg_pool2d(x.square(),k,1,k//2)-mu.square()).clamp_min(0)
    return (x-mu)*torch.rsqrt(var+eps)

def install_localnorm_qtr(model):
    qtr=model.qtr
    assert qtr is not None and qtr.use_reliability_gate and len(qtr.reliability_gates)==3
    for i,gate in enumerate(qtr.reliability_gates):
        old=gate[0];n=qtr.content_align[i][0].out_channels
        new=torch.nn.Conv2d(old.in_channels+n,old.out_channels,old.kernel_size,old.stride,old.padding,old.dilation,old.groups,bias=old.bias is not None,device=old.weight.device,dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_();new.weight[:,:old.in_channels].copy_(old.weight)
            if old.bias is not None:new.bias.copy_(old.bias)
        gate[0]=new
    def forward(self,image,content,detail):
        q=self.quality_mlp(gray_quality_descriptor(image));feats=[];maps=[];rels=[];qualified=[]
        for i in range(3):
            c=self.content_align[i](content[i])
            if c.shape[-2:]!=detail[i].shape[-2:]:c=F.interpolate(c,size=detail[i].shape[-2:],mode='bilinear',align_corners=False)
            qi=q[:,:,None,None].expand(-1,-1,detail[i].shape[-2],detail[i].shape[-1])
            x=torch.cat([detail[i],c,(norm2d(detail[i])-norm2d(c)).abs(),qi],1);f=self.qualifiers[i](x)
            feats.append(f);maps.append(f.abs().mean(1,keepdim=True))
            logits=self.reliability_gates[i](torch.cat([c,detail[i],qi,local_norm(f,7)],1));rel=torch.sigmoid(logits)
            rels.append(rel);qualified.append(detail[i]+rel*f if self.residual_refinement else detail[i]*(2*rel))
        return {'descriptor':q,'features':feats,'maps':maps,'reliability_maps':rels,'qualified_detail':qualified}
    qtr.forward=types.MethodType(forward,qtr);return qtr

def jpeg_view(sample,quality):
    vh,vw=map(int,sample['valid_hw'].tolist());fh,fw=sample['image'].shape[-2:]
    arr=(sample['image'][:,:vh,:vw].permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')
    import jpegio
    with tempfile.TemporaryDirectory(prefix='qtr_ln_') as folder:
        p=Path(folder)/'v.jpg';Image.fromarray(arr).save(p,'JPEG',quality=int(quality),subsampling=2)
        with Image.open(p) as im:pixels=np.asarray(im.convert('RGB'),dtype='float32')
        jpg=jpegio.read(str(p));coef=jpg.coef_arrays[0];qt=jpg.quant_tables[jpg.comp_info[0].quant_tbl_no].astype('float32')
    rgb=np.full((fh,fw,3),127.5,np.float32);rgb[:vh,:vw]=pixels
    cc=np.zeros((fh,fw),dtype=coef.dtype);h=min(fh,coef.shape[0]);w=min(fw,coef.shape[1]);cc[:h,:w]=coef[:h,:w]
    out=dict(sample);out.update(image=torch.from_numpy(rgb).permute(2,0,1)/255,dct=ManipulationDataset._dct_volume_from_coef(cc,fh,fw),qtable=torch.from_numpy(qt[None]),dct_available=torch.ones(1));return out

class Mixed(torch.utils.data.Dataset):
    def __init__(self,base):self.base=base;self.records=base.records
    def __len__(self):return len(self.base)
    def __getitem__(self,key):
        s=self.base[key];seed=key[1] if isinstance(key,tuple) else SEED+int(key);cond=CONDITIONS[int(np.random.default_rng(seed).integers(4))]
        return s if cond=='clean' else jpeg_view(s,cond)
class Fixed(torch.utils.data.Dataset):
    def __init__(self,base,q):self.base=base;self.records=base.records;self.q=q
    def __len__(self):return len(self.base)
    def __getitem__(self,key):return jpeg_view(self.base[key],self.q)

def main():
    OUT.mkdir(parents=True,exist_ok=True);summary={'status':'preparing','experiment':'E2_QTR_RelativeReliability_ForensicFusion','parent':str(PARENT),'epochs':1,'lr':1e-5,'train_conditions':list(CONDITIONS),'localnorm_window':7,'new_channel_init':'zero','trainable':'QTR + Forensic Fusion','gates':{'clean_f1_min':.936,'q70_auc_min':.60},'test_run':False,'external_read':False};dump('summary.yaml',summary)
    try:
        torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);ck=torch.load(PARENT,map_location='cpu');assert int(ck['epoch'])==76
        cfg=copy.deepcopy(ck['config']);cfg['training']['dct_sample_dropout']=0.;cfg['training']['learning_rate']=1e-5;cfg['training']['epochs']=1
        tr=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'train');va=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'val')
        model=fair.e.build_final_model(cfg['model']).cuda();model.load_state_dict(ck['model'],strict=True);model.eval()
        base_val=FairDataset(va,OUT/'jpeg_cache_val',False);batch=next(iter(DataLoader(base_val,batch_size=1,num_workers=0)));batch={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in batch.items()}
        with torch.no_grad():before=stable.v3.fwd(model,batch,cfg,False)['mask_logits'].float().cpu()
        qtr=install_localnorm_qtr(model)
        with torch.no_grad():after=stable.v3.fwd(model,batch,cfg,False)['mask_logits'].float().cpu()
        eq={'mean_abs_logit_diff':float((after-before).abs().mean()),'max_abs_logit_diff':float((after-before).abs().max())};summary['parent_equivalence']=eq;dump('summary.yaml',summary)
        if eq['max_abs_logit_diff']>1e-5:raise RuntimeError('parent equivalence failed '+str(eq))
        for p in model.parameters():p.requires_grad_(False)
        for p in qtr.parameters():p.requires_grad_(True)
        if model.balanced_forensic_fusions is None: raise RuntimeError('balanced_forensic_fusions missing')
        for p in model.balanced_forensic_fusions.parameters():p.requires_grad_(True)
        stable.train_mode(model);seg=stable.v3.e24.ReTraceLoss.from_config(stable.v3.e24.clean_loss_config(cfg['loss'])).cuda();opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=1e-5,momentum=.9,weight_decay=.0005)
        fake=[i for i,r in enumerate(tr) if r.label];auth=[i for i,r in enumerate(tr) if not r.label];nf=min(len(fake),len(auth));g=torch.Generator().manual_seed(SEED+1);ids=[fake[i] for i in torch.randperm(len(fake),generator=g)[:nf]]+[auth[i] for i in torch.randperm(len(auth),generator=g)[:nf]];ids=[ids[i] for i in torch.randperm(len(ids),generator=g)];plan=[(idx,SEED*1000000+j) for j,idx in enumerate(ids)]
        loader=DataLoader(Mixed(FairDataset(tr,OUT/'jpeg_cache_train',True)),batch_size=4,sampler=plan,num_workers=2,pin_memory=True);total=0.;started=time.time()
        for st,b in enumerate(loader,1):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=stable.v3.fwd(model,b,cfg,False);loss=stable.v3.objective(out,b,seg,cfg)['total']
            loss.backward();gn=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True);opt.step();total+=float(loss)
            if st==1 or st%25==0 or st==len(loader):dump('progress.yaml',{'batch':st,'batches':len(loader),'loss':total/st,'grad_norm':float(gn),'elapsed_minutes':(time.time()-started)/60});print('[qtr-ln]',st,len(loader),total/st,flush=True)
        checkpoint_path=OUT/'qtr_localnorm_epoch01.pt'
        torch.save({'epoch':1,'parent_epoch':76,'model':model.state_dict(),'optimizer':opt.state_dict(),'experiment':'E2_QTR_RelativeReliability_ForensicFusion','localnorm_window':7,'trainable':'QTR + Forensic Fusion'},checkpoint_path)
        if False and os.environ.get('RECOVER_SAVE_ONLY')=='1':
            previous=yaml.safe_load((OUT/'jpeg_pass_summary.yaml').read_text())
            previous.update(status='completed',checkpoint=str(checkpoint_path),checkpoint_recovered=True,parent_equivalence=eq,train_loss=total/len(loader))
            dump('summary.yaml',previous)
            print(json.dumps(previous,indent=2),flush=True)
            return
        model.eval();results={}
        for name,ds in [('clean',base_val),('q90',Fixed(base_val,90)),('q80',Fixed(base_val,80)),('q70',Fixed(base_val,70))]:
            z=fair.evaluate(model,ds,cfg,OUT,name);m=z['metrics']['fake'];results[name]={'f1':float(m['oracle']['f1']),'precision':float(m['oracle']['precision']),'recall':float(m['oracle']['recall']),'auc':float(m['auc']),'threshold':float(m['oracle']['threshold'])}
        passed=results['clean']['f1']>=.936 and results['q70']['auc']>=.60
        summary.update(status='completed',train_loss=total/len(loader),checkpoint=str(checkpoint_path),results=results,clean_gate_pass=results['clean']['f1']>=.936,q70_auc_gate_pass=results['q70']['auc']>=.60,decision='PASS_continue_IMD' if passed else 'REJECT_stop',test_run=False);dump('summary.yaml',summary);print(json.dumps(summary,indent=2),flush=True)
    except BaseException:
        summary.update(status='failed',stop_reason=traceback.format_exc());dump('summary.yaml',summary);raise
if __name__=='__main__':main()

