"""Bounded in-memory fit of a fixed TRAIN batch; does not save or replace weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from retrace.data import read_manifest,pil_to_tensor
from retrace.utils import set_seed
from retracenet.final_modules import build_final_model
from train_e15_hr_corrected import make_optimizer,train_mode,dump
from train_e16_direct import objective
from e15_original_metrics import PixelStats
from train_v2_final_pipeline import resize_mask_for_supervision

HEADS={'D1':'e15_d1_logits','D2':'e15_d2_logits','D3':'e15_d3_logits','final':'mask_logits','teacher':'teacher_mask_logits'}


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def frozen_hash(model):
    trainable={n for n,p in model.named_parameters() if p.requires_grad}
    h=hashlib.sha256()
    for n,t in model.state_dict().items():
        if n not in trainable:
            h.update(n.encode()); h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def select_samples(records):
    # Stable selection by filename and GT area only; never inspect model predictions.
    groups={k:[] for k in ['S_tiny','S_large','D_small','D_large','authentic']}
    for r in sorted(records,key=lambda r:str(r.image)):
        if r.label==0:
            key='authentic'
        else:
            with Image.open(r.mask) as im: ratio=float((np.asarray(im.convert('L'))>127).mean())
            name=r.image.name
            if name.startswith('Tp_S') and 0<ratio<.01: key='S_tiny'
            elif name.startswith('Tp_S') and ratio>=.2: key='S_large'
            elif name.startswith('Tp_D') and .01<=ratio<.05: key='D_small'
            elif name.startswith('Tp_D') and ratio>=.2: key='D_large'
            else: continue
        if len(groups[key])<4: groups[key].append(r)
        if all(len(v)==4 for v in groups.values()): break
    assert all(len(v)==4 for v in groups.values()), {k:len(v) for k,v in groups.items()}
    def load(r,key):
        with Image.open(r.image) as im:
            rgb=im.convert('RGB'); size=rgb.size
            x=pil_to_tensor(rgb.resize((640,640),Image.Resampling.BILINEAR))[None]
        if r.mask:
            with Image.open(r.mask) as im: gt=im.convert('L').copy()
        else: gt=Image.new('L',size,0)
        assert gt.size==size
        y=(pil_to_tensor(gt.resize((640,640),Image.Resampling.NEAREST))>.5).float()[None]
        return {'path':str(r.image),'mask_path':str(r.mask) if r.mask else None,'group':key,'image':x,'mask':y,
                'gt_original':(np.asarray(gt)>127).astype(np.uint8),'label':r.label}
    fit=[load(v[0],k) for k,v in groups.items()]
    heldout=[load(r,k) for k,v in groups.items() for r in v[1:]]
    return fit,heldout


@torch.no_grad()
def score(model,samples,outdir,tag,save_predictions=False):
    model.eval()
    stats={k:PixelStats() for k in HEADS}
    fake={k:PixelStats() for k in HEADS}
    rows=[]
    for index,s in enumerate(samples):
        out=model(s['image'].cuda()); row={'path':s['path'],'group':s['group'],'heads':{}}
        stored={'image':s['image'][0].numpy(),'GT':s['mask'][0,0].numpy()}
        for name,key in HEADS.items():
            logits=out[key].float()
            native_gt=resize_mask_for_supervision(s['mask'].cuda(),logits.shape[-2:],hard=True)
            p=F.interpolate(logits.sigmoid(),size=s['gt_original'].shape,mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            m=PixelStats(); m.add(p,s['gt_original']); report=m.result()
            native_stats=PixelStats(); native_stats.add(logits.sigmoid().cpu().numpy(),native_gt.cpu().numpy())
            report['native_grid']=native_stats.result()
            report.update(native_shape=list(logits.shape),native_BCE=float(F.binary_cross_entropy_with_logits(logits,native_gt)),
                          probability_mean=float(p.mean()),probability_std=float(p.std()))
            row['heads'][name]=report
            stats[name].add(p,s['gt_original'])
            if s['label']: fake[name].add(p,s['gt_original'])
            stored[name]=F.interpolate(logits.sigmoid(),size=(640,640),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        rows.append(row)
        if save_predictions: np.savez_compressed(outdir/f'{tag}_{index:02d}.npz',**stored)
        del out
    result={'all':{k:v.result() for k,v in stats.items()},'fake':{k:v.result() for k,v in fake.items()},'per_image':rows}
    dump(outdir/(tag+'.yaml'),result)
    print(tag,json.dumps({k:{'fixed_F1':v['fixed']['f1'],'best_F1':v['oracle']['f1'],'auc':v['auc']} for k,v in result['fake'].items()}),flush=True)
    return result


def visualize(outdir,count):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(count,10,figsize=(25,2.6*count),squeeze=False)
    columns=['image','GT','teacher','D1 before','D2 before','D3 before','final before','D1 after','D3 after','final after']
    for i in range(count):
        before=np.load(outdir/f'fit_before_{i:02d}.npz'); after=np.load(outdir/f'fit_after_{i:02d}.npz')
        values=[before['image'].transpose(1,2,0),before['GT'],before['teacher'],before['D1'],before['D2'],before['D3'],before['final'],after['D1'],after['D3'],after['final']]
        for j,v in enumerate(values):
            axes[i,j].imshow(v,cmap='viridis' if j>=2 else 'gray',vmin=0,vmax=1)
            axes[i,j].axis('off')
            if i==0: axes[i,j].set_title(columns[j])
    fig.tight_layout(); fig.savefig(outdir/'fixed_batch_predictions.png',dpi=110); plt.close(fig)


def main(args):
    directory=Path(args.out); directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'result.yaml').exists(),'Audit result exists; refuse overwrite'
    start=time.time(); set_seed(3407)
    checkpoint_hash=file_hash(args.checkpoint)
    checkpoint=torch.load(args.checkpoint,map_location='cpu'); cfg=checkpoint['config']
    assert 'e16_direct' in cfg['model']['architecture']
    model=build_final_model(cfg['model']).cuda(); model.load_state_dict(checkpoint['model'],strict=True)
    epoch=int(checkpoint['epoch'])+1; del checkpoint
    optimizer=make_optimizer(model,1e-4)
    fit,heldout=select_samples(read_manifest(cfg['data']['manifest'],cfg['data']['root'],'train'))
    metadata={'checkpoint':args.checkpoint,'checkpoint_sha256':checkpoint_hash,'checkpoint_epoch':epoch,
              'split':'train ONLY','fit_samples':[{k:s[k] for k in ['path','mask_path','group']} for s in fit],
              'heldout_train_samples':[{k:s[k] for k in ['path','mask_path','group']} for s in heldout],
              'augmentation':'none; fixed 640 whole image resize','updates':200,'microbatch':1,'accumulation':len(fit),
              'lr':1e-4,'loss':'same E16 GT + aux + teacher KL .05; fixed for diagnostic',
              'trained':'same decoder + content proj; evidence stack frozen','weights_saved':False}
    dump(directory/'protocol.yaml',metadata)
    frozen_before=frozen_hash(model)
    before=score(model,fit,directory,'fit_before',True)
    heldout_before=score(model,heldout,directory,'heldout_train_before')
    trace=[]
    for step in range(1,201):
        train_mode(model); optimizer.zero_grad(set_to_none=True); totals={}
        for s in fit:
            out=model(s['image'].cuda()); terms=objective(out,s['mask'].cuda(),cfg,.05)
            assert torch.isfinite(terms['total'])
            (terms['total']/len(fit)).backward()
            for k,v in terms.items(): totals[k]=totals.get(k,0.)+float(v.detach())/len(fit)
            del out,terms
        norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5,error_if_nonfinite=True))
        optimizer.step()
        row={'update':step,'losses':totals,'gradient_norm_before_clip':norm,'elapsed_seconds':time.time()-start}; trace.append(row)
        if step==1 or step%10==0: print('[fit]',json.dumps(row),flush=True)
        if step in [1,10,25,50,100]: score(model,fit,directory,f'fit_step{step:03d}')
        dump(directory/'progress.yaml',row)
    after=score(model,fit,directory,'fit_after',True)
    heldout_after=score(model,heldout,directory,'heldout_train_after')
    assert frozen_hash(model)==frozen_before,'Frozen parameters or running buffers changed'
    assert file_hash(args.checkpoint)==checkpoint_hash,'Source checkpoint changed'
    for name in ['fake','all']:
        assert before[name]['teacher']==after[name]['teacher'],'Teacher predictions changed'
    result={'status':'PASS','protocol':metadata,'frozen_parameters_and_buffers_unchanged':True,'source_checkpoint_unchanged':True,
            'fit_before':before,'fit_after':after,'heldout_train_before':heldout_before,'heldout_train_after':heldout_after,
            'loss_trace':trace,'elapsed_seconds':time.time()-start,
            'limitation':'Fixed batch memorization is not validation generalization or proof frozen features suffice.'}
    dump(directory/'result.yaml',result)
    visualize(directory,len(fit))
    print('[AUDIT_COMPLETE]',str(directory/'result.yaml'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
