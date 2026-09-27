"""Matched E5 continuation pair: ONLY mechanism-chain trainability differs."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import yaml

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from retrace.data import read_manifest
from retrace.losses import ReTraceLoss
from retrace.utils import set_seed,atomic_torch_save
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import train_loader_for,clean_loss_config,weighted_deep_supervision_loss,tversky_loss,dher_v2_aux_loss
from train_e15_hr_corrected import dump,evaluate

ARMS=['A_decoder_only','B_decoder_mechanisms']
COMMON=['decoder.','mask_head.','log_var_head.','aux_heads.']
MECHANISM=['qtr.','hcv2.','dher.']


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def state_hash(model,frozen_only=False):
    trained={n for n,p in model.named_parameters() if p.requires_grad}
    h=hashlib.sha256()
    for n,t in model.state_dict().items():
        if frozen_only and n in trained: continue
        h.update(n.encode()); h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class ReplayDataset(Dataset):
    def __init__(self,dataset,indices,seeds):
        self.dataset,self.indices,self.seeds=dataset,indices,seeds
    def __len__(self): return len(self.indices)
    def __getitem__(self,position):
        index,seed=int(self.indices[position]),int(self.seeds[position])
        # Every item gets its own deterministic random stream, independent of workers/model.
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        sample=self.dataset[index]
        h=hashlib.sha256()
        h.update(sample['image'].numpy().tobytes());h.update(sample['mask'].numpy().tobytes())
        h.update(sample['label'].numpy().tobytes())
        sample.update(replay_index=index,replay_seed=seed,input_sha256=h.hexdigest())
        return sample


def optimizer_for(model,arm):
    allowed=COMMON+(MECHANISM if arm==ARMS[1] else [])
    for n,p in model.named_parameters(): p.requires_grad_(any(n.startswith(k) for k in allowed))
    # Same evaluation-mode layers in both arms: no BN-buffer/dropout confound.
    model.eval()
    groups=[]
    for prefix in COMMON+MECHANISM:
        params=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if params: groups.append({'params':params,'name':prefix,'lr':5e-6 if prefix in COMMON else 2e-6})
    assert not any(p.requires_grad for p in model.content_encoder.parameters())
    assert not any(p.requires_grad for p in model.forensic_stream.parameters())
    return torch.optim.AdamW(groups,weight_decay=.01)


def objective(out,batch,seg,cfg):
    y,lab=batch['mask'].cuda(),batch['label'].cuda()
    terms=seg(out,y,lab)
    aux=dher_v2_aux_loss(out,y,cfg)
    terms['total']=terms['total']+aux['dher_aux_total']
    ds=weighted_deep_supervision_loss(out,y,cfg)
    tv=tversky_loss(out['mask_logits'],y,float(cfg['loss'].get('tversky_alpha_fp',.3)),float(cfg['loss'].get('tversky_beta_fn',.7)))
    terms.update(weighted_deep_supervision=ds,tversky=tv)
    terms['total']=terms['total']+ds+float(cfg['loss']['weights'].get('tversky',0))*tv
    return terms


def audit_arm(model,opt,batch,seg,cfg,source_state):
    before=state_hash(model); frozen=state_hash(model,True)
    out=model(batch['image'].cuda()); terms=objective(out,batch,seg,cfg)
    initial_loss=float(terms['total'].detach())
    initial_prediction_hash=hashlib.sha256(out['mask_logits'].detach().cpu().numpy().tobytes()).hexdigest()
    terms['total'].backward()
    groups={}
    for group in opt.param_groups:
        prefix=group['name']; named=[(n,p) for n,p in model.named_parameters() if n.startswith(prefix)]
        grads=[p.grad for _,p in named if p.grad is not None]
        norm=sum(float(g.detach().square().sum()) for g in grads)**.5
        assert norm>0 and math.isfinite(norm),(prefix,norm)
        groups[prefix]={'parameters':sum(p.numel() for _,p in named),'with_grad':len(grads),'grad_norm':norm,'lr':group['lr']}
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5,error_if_nonfinite=True)
    opt.step()
    for prefix,g in groups.items():
        delta=sum(float((p.detach().cpu()-source_state[n]).abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix))
        assert delta>0,(prefix,delta)
        g['parameter_update_abs_sum']=delta
    assert frozen==state_hash(model,True)
    del out,terms
    model.load_state_dict(source_state,strict=True);opt.state.clear();opt.zero_grad(set_to_none=True)
    assert state_hash(model)==before
    return {'status':'PASS','groups':groups,'initial_loss':initial_loss,'initial_prediction_sha256':initial_prediction_hash,
            'frozen_state_unchanged':True,'audit_step_restored':True,'initial_state_sha256':before}


def main(args):
    directory=Path(args.out); directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'paired_summary.yaml').exists(),'Refuse duplicate run'
    summary={'status':'initializing','arms':{},'history':{},'test_run':False,'epochs_per_arm':5,
             'single_variable':'QTR/HCV/DHER requires_grad; decoder and localization heads train in both',
             'primary_comparison':'B minus A at matched final epoch, val original-grid fake pooled F1-best',
             'secondary_comparison':'best validation checkpoint within each identical 5-epoch budget',
             'no_per_arm_early_stop':True,'data_experiment_auto_start':False,'one_seed_limit':'Paired seed 3407; not statistical proof',
             'stop_reason':None}
    path=directory/'paired_summary.yaml'; dump(path,summary)
    try:
        set_seed(3407)
        checkpoint=torch.load(args.checkpoint,map_location='cpu'); source_state=checkpoint['model'];cfg=checkpoint['config']
        assert checkpoint.get('eval_variant','raw')=='raw','Explicit raw E5 required'
        assert cfg['model']['architecture']=='retracenet_b3_qtr_hcv2_dher_v21_mrfe'
        assert float(cfg['dher_v2_loss']['support'])==0 and float(cfg['dher_v2_loss']['refutation'])==0
        assert not cfg['loss']['weights'].get('weak_tiny',0)
        assert cfg['training'].get('consistency_probability',0)==0
        cfg['training']['batch_size']=1
        summary.update(source_checkpoint=args.checkpoint,source_checkpoint_sha256=sha_file(args.checkpoint),
                       source_checkpoint_epoch=checkpoint['epoch'],source_config=cfg,selection='val_original_grid_fake_pooled_F1_best',
                       optimizer={'type':'AdamW','decoder_and_heads_lr':5e-6,'mechanisms_lr':2e-6,'weight_decay':.01,'constant_lr':True},
                       layers_mode='eval for both arms; gradients enabled on selected parameters',amp=False,ema=False,
                       gradient_accumulation=4,gradient_clip=.5)
        dump(directory/'source_config.yaml',cfg)
        base=train_loader_for(cfg); dataset=base.dataset
        assert hasattr(base.sampler,'weights'),'Expected E5 weighted sampler'
        n=len(base.sampler); plan={}
        for ep in range(5):
            generator=torch.Generator().manual_seed(3407+ep)
            indices=torch.multinomial(base.sampler.weights,n,replacement=True,generator=generator).numpy()
            seeds=np.random.default_rng(103407+ep).integers(0,2**31-1,n,dtype=np.int64)
            plan[f'indices_{ep}']=indices;plan[f'seeds_{ep}']=seeds
        np.savez_compressed(directory/'shared_sample_and_augmentation_plan.npz',**plan)
        summary.update(training_microbatches_per_epoch=n,optimizer_steps_per_epoch=math.ceil(n/4),
                       optimizer_steps_per_arm=5*math.ceil(n/4),manifest_sha256=sha_file(cfg['data']['manifest']),
                       replay_plan_sha256=sha_file(directory/'shared_sample_and_augmentation_plan.npz'))
        replay=ReplayDataset(dataset,plan['indices_0'][:8],plan['seeds_0'][:8])
        audit_batch=next(b for b in DataLoader(replay,batch_size=1,num_workers=0) if b['mask'].sum()>0)
        seg=ReTraceLoss.from_config(clean_loss_config(cfg['loss'])).cuda()
        audits={}
        for arm in ARMS:
            model=build_final_model(cfg['model']).cuda(); model.load_state_dict(source_state,strict=True)
            opt=optimizer_for(model,arm)
            audits[arm]=audit_arm(model,opt,audit_batch,seg,cfg,source_state)
            print('[preflight]',arm,json.dumps(audits[arm]),flush=True)
            del model,opt;gc.collect();torch.cuda.empty_cache()
        assert audits[ARMS[0]]['initial_prediction_sha256']==audits[ARMS[1]]['initial_prediction_sha256']
        assert audits[ARMS[0]]['initial_loss']==audits[ARMS[1]]['initial_loss']
        summary['preflight']=audits;dump(directory/'preflight.yaml',audits)
        val=read_manifest(cfg['data']['manifest'],cfg['data']['root'],'val')
        model=build_final_model(cfg['model']).cuda();model.load_state_dict(source_state,strict=True)
        summary['status']='baseline_validation';dump(path,summary)
        baseline=evaluate(model,val,'E5_original_val',directory)
        summary['baseline']=baseline
        del model;gc.collect();torch.cuda.empty_cache()
        for arm in ARMS:
            set_seed(3407)
            adir=directory/arm;adir.mkdir()
            model=build_final_model(cfg['model']).cuda();model.load_state_dict(source_state,strict=True)
            opt=optimizer_for(model,arm); frozen_initial=state_hash(model,True)
            history=[]; best=-math.inf
            armcfg=json.loads(json.dumps(cfg));armcfg['training'].update(init_checkpoint=args.checkpoint,epochs=5,ema=False,
                selection_metric=summary['selection'],learning_rate=5e-6,gradient_accumulation_steps=4,
                constant_lr_epochs=5,min_lr_ratio=1.,early_stopping=None,
                parameter_groups=[{'patterns':[g['name']],'lr':g['lr'],'weight_decay':.01} for g in opt.param_groups])
            armcfg['paired_protocol']={k:summary[k] for k in ['single_variable','optimizer','gradient_accumulation','gradient_clip','replay_plan_sha256']}
            armcfg['paired_protocol']['arm']=arm;dump(adir/'config.yaml',armcfg)
            for ep in range(5):
                replay=ReplayDataset(dataset,plan[f'indices_{ep}'],plan[f'seeds_{ep}'])
                loader=DataLoader(replay,batch_size=1,shuffle=False,num_workers=4,pin_memory=True)
                model.eval();opt.zero_grad(set_to_none=True); totals={};start=time.time();stream=hashlib.sha256()
                summary.update(status='training',current_arm=arm,current_epoch=ep+1);dump(path,summary)
                print(f'[train] {arm} epoch={ep+1}/5 step=0/{n}',flush=True)
                with (adir/f'inputs_epoch_{ep+1:02d}.jsonl').open('w') as handle:
                    for bi,b in enumerate(loader):
                        identity={'position':bi,'index':int(b['replay_index']),'seed':int(b['replay_seed']),
                                  'path':b['path'][0],'input_sha256':b['input_sha256'][0]}
                        line=json.dumps(identity,sort_keys=True);stream.update(line.encode());handle.write(line+'\n')
                        out=model(b['image'].cuda()); terms=objective(out,b,seg,cfg)
                        if not torch.isfinite(terms['total']):raise FloatingPointError('Nonfinite loss')
                        accumulation=min(4,n-(bi//4)*4);(terms['total']/accumulation).backward()
                        if (bi+1)%4==0 or bi+1==n:
                            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5,error_if_nonfinite=True)
                            opt.step();opt.zero_grad(set_to_none=True)
                        for k,v in terms.items():totals[k]=totals.get(k,0.)+float(v.detach())
                        del out,terms
                        if bi==0 or (bi+1)%50==0 or bi+1==n:
                            elapsed=time.time()-start
                            progress={'arm':arm,'epoch':ep+1,'step':bi+1,'steps':n,'loss':totals['total']/(bi+1),
                                      'elapsed_minutes':elapsed/60,'eta_minutes':elapsed/(bi+1)*(n-bi-1)/60}
                            dump(directory/'progress.yaml',progress);print('[train]',json.dumps(progress),flush=True)
                stream_hash=stream.hexdigest()
                if arm==ARMS[1]: assert stream_hash==summary['history'][ARMS[0]][ep]['actual_input_stream_sha256'],'Paired data mismatch'
                assert state_hash(model,True)==frozen_initial,'Frozen params/buffers changed'
                summary['status']='validation';dump(path,summary)
                result=evaluate(model,val,f'epoch_{ep+1:02d}_original_val',adir)
                cur=result['metrics']['fake']['oracle']['f1'];assert math.isfinite(cur)
                row={'epoch':ep+1,'optimizer_steps':(ep+1)*math.ceil(n/4),'selection_value':cur,
                     'delta_vs_e5':cur-baseline['metrics']['fake']['oracle']['f1'],'validation':result,
                     'train_losses':{k:v/n for k,v in totals.items()},'actual_input_stream_sha256':stream_hash,
                     'frozen_state_unchanged':True,'elapsed_seconds':time.time()-start}
                if arm==ARMS[1]:row['delta_vs_matched_A']=cur-summary['history'][ARMS[0]][ep]['selection_value']
                isbest=cur>best;best=max(best,cur)
                state={'epoch':ep,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),
                       'config':armcfg,'eval_variant':'raw','phase':arm,'best_metric':best,'selection_threshold':result['metrics']['fake']['oracle']['threshold']}
                atomic_torch_save(state,adir/'last.pt')
                if isbest:
                    atomic_torch_save(state,adir/'best.pt')
                    summary['arms'][arm]={'best_F1':best,'best_epoch':ep+1,'best_checkpoint':str(adir/'best.pt')}
                del state
                history.append(row);summary['history'][arm]=history;dump(path,summary)
                print('[epoch_done]',arm,ep+1,'F1',cur,'delta_E5',row['delta_vs_e5'],flush=True)
            del model,opt;gc.collect();torch.cuda.empty_cache()
        summary.update(status='completed',stop_reason='both_arms_completed_equal_5_epochs',
                       final_epoch_delta_B_minus_A=summary['history'][ARMS[1]][-1]['selection_value']-summary['history'][ARMS[0]][-1]['selection_value'],
                       best_delta_B_minus_A=summary['arms'][ARMS[1]]['best_F1']-summary['arms'][ARMS[0]]['best_F1'],
                       identical_actual_inputs_verified=True)
        assert sha_file(args.checkpoint)==summary['source_checkpoint_sha256']
        dump(path,summary); print('[PAIR_COMPLETE]',summary['final_epoch_delta_B_minus_A'],flush=True)
    except BaseException:
        summary.update(status='failed',stop_reason=traceback.format_exc());dump(path,summary);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    main(p.parse_args())
