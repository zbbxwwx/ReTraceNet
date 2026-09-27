"""One-epoch CASIA2-only JPEG DG probe; no external datasets are read."""
import copy, io, json, math, random, sys, tempfile, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_imd_mit_full_v3  # patches model builder to MiT-V3
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from retrace.data import ManipulationDataset

PARENT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
MANIFEST=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/balanced_residual_v3_full30_20260912/clean_manifest.csv')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_mit_jpeg_only_dg_probe1_20260920')
QUALITIES=(70,80,90,100); SUBSAMPLINGS=(0,2); SEED=3407

def dump(name,obj): stable.dump(OUT/name,obj)

def jpeg_view(sample,quality,subsampling):
    vh,vw=map(int,sample['valid_hw'].tolist()); full_h,full_w=sample['image'].shape[-2:]
    arr=(sample['image'][:,:vh,:vw].permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')
    im=Image.fromarray(arr)
    import jpegio
    with tempfile.TemporaryDirectory(prefix='jpeg_dg_') as folder:
        path=Path(folder)/'view.jpg';im.save(path,format='JPEG',quality=int(quality),subsampling=int(subsampling))
        with Image.open(path) as dec:pixels=np.asarray(dec.convert('RGB'),dtype='float32')
        jpg=jpegio.read(str(path));coeff=jpg.coef_arrays[0];qtable=jpg.quant_tables[jpg.comp_info[0].quant_tbl_no].astype('float32')
    rgb=np.full((full_h,full_w,3),127.5,dtype='float32');rgb[:vh,:vw]=pixels
    coef=np.zeros((full_h,full_w),dtype=coeff.dtype);h=min(full_h,coeff.shape[0]);w=min(full_w,coeff.shape[1]);coef[:h,:w]=coeff[:h,:w]
    out=dict(sample);out.update(image=torch.from_numpy(rgb).permute(2,0,1)/255,
        dct=ManipulationDataset._dct_volume_from_coef(coef,full_h,full_w),qtable=torch.from_numpy(qtable[None]),dct_available=torch.ones(1))
    assert torch.equal(out['mask'],sample['mask'])
    return out

class TrainJPEG50(torch.utils.data.Dataset):
    def __init__(self,base):self.base=base;self.records=base.records
    def __len__(self):return len(self.base)
    def __getitem__(self,key):
        s=self.base[key];seed=key[1] if isinstance(key,tuple) else SEED+int(key)
        rng=np.random.default_rng(seed)
        if int(rng.integers(2))==0:return s
        return jpeg_view(s,QUALITIES[int(rng.integers(4))],SUBSAMPLINGS[int(rng.integers(2))])

class ShiftVal(torch.utils.data.Dataset):
    def __init__(self,base):self.base=base;self.records=base.records
    def __len__(self):return len(self.base)
    def __getitem__(self,key):
        s=self.base[key];raw=str(s['path']);seed=int(__import__('hashlib').sha256(raw.encode()).hexdigest()[:8],16)
        return jpeg_view(s,QUALITIES[seed%4],SUBSAMPLINGS[(seed//4)%2])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary={'status':'preparing','experiment':'CASIA2 epoch76 one-epoch JPEG-only DG probe','test_run':False,'external_datasets_read':False,
      'parent':str(PARENT),'lr':1e-5,'epochs':1,'train_mixture':'50% clean + 50% recompressed','qualities':list(QUALITIES),'subsampling':['444','420'],
      'dct_dropout':0.0,'consistency_weight':0.0,'clean_gate_f1':0.936}
    dump('summary.yaml',summary)
    try:
        torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
        ck=torch.load(PARENT,map_location='cpu');assert int(ck['epoch'])==76
        cfg=copy.deepcopy(ck['config']);cfg['training']['dct_sample_dropout']=0.0;cfg['training']['learning_rate']=1e-5;cfg['training']['epochs']=1
        train_records=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'train');val_records=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'val')
        fake=[i for i,r in enumerate(train_records) if r.label];auth=[i for i,r in enumerate(train_records) if not r.label];nf=min(len(fake),len(auth))
        base_train=FairDataset(train_records,OUT/'jpeg_cache_train',True);train_ds=TrainJPEG50(base_train)
        clean_val=FairDataset(val_records,OUT/'jpeg_cache_val',False);shift_val=ShiftVal(clean_val)
        model=fair.e.build_final_model(cfg['model']).cuda();model.load_state_dict(ck['model'],strict=True);stable.train_mode(model)
        seg=stable.v3.e24.ReTraceLoss.from_config(stable.v3.e24.clean_loss_config(cfg['loss'])).cuda()
        opt=torch.optim.SGD(model.parameters(),lr=1e-5,momentum=.9,weight_decay=.0005)
        gen=torch.Generator().manual_seed(SEED+1);ids=[fake[i] for i in torch.randperm(len(fake),generator=gen)[:nf]]+[auth[i] for i in torch.randperm(len(auth),generator=gen)[:nf]];ids=[ids[i] for i in torch.randperm(len(ids),generator=gen)]
        plan=[(idx,SEED*1000000+100000+j) for j,idx in enumerate(ids)]
        loader=DataLoader(train_ds,batch_size=4,sampler=plan,num_workers=2,pin_memory=True);total=0.;started=time.time()
        for st,b in enumerate(loader,1):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=stable.v3.fwd(model,b,cfg,False);terms=stable.v3.objective(out,b,seg,cfg);loss=terms['total']
            if not torch.isfinite(loss):raise FloatingPointError(st)
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();total+=float(loss)
            if st==1 or st%25==0 or st==len(loader):
                dump('progress.yaml',{'epoch':1,'batch':st,'batches':len(loader),'loss':total/st,'lr':1e-5,'grad_norm':float(gn),'elapsed_minutes':(time.time()-started)/60})
                print('[jpeg-probe]',st,len(loader),total/st,flush=True)
        clean=fair.evaluate(model,clean_val,cfg,OUT,'clean_val');shift=fair.evaluate(model,shift_val,cfg,OUT,'synthetic_shift_val')
        cq=clean['metrics']['fake']['oracle'];sq=shift['metrics']['fake']['oracle'];passed=float(cq['f1'])>=0.936
        state={'model':fair.e.copy_state(model),'optimizer':opt.state_dict(),'config':cfg,'epoch':1,'parent_epoch':76,'selection_threshold':cq['threshold'],'best_metric':cq['f1']}
        fair.e24.atomic_torch_save(state,OUT/'epoch01.pt')
        summary.update(status='completed',decision='keep_for_further_source-only_analysis' if passed else 'reject_clean_gate',clean_gate_pass=passed,
          train_loss=total/len(loader),clean_val={'f1':float(cq['f1']),'precision':float(cq['precision']),'recall':float(cq['recall']),'auc':float(clean['metrics']['fake']['auc']),'threshold':float(cq['threshold'])},
          synthetic_shift_val={'f1':float(sq['f1']),'precision':float(sq['precision']),'recall':float(sq['recall']),'auc':float(shift['metrics']['fake']['auc']),'threshold':float(sq['threshold'])})
        dump('summary.yaml',summary);print(json.dumps(summary,indent=2),flush=True)
    except BaseException:
        summary.update(status='failed',stop_reason=traceback.format_exc());dump('summary.yaml',summary);raise
if __name__=='__main__':main()
