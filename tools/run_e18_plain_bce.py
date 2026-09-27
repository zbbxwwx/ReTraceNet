"""E18: matched E17-A replay, replacing only heteroscedastic localization with plain BCE."""
from __future__ import annotations
import argparse, gc, hashlib, json, math, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from retrace.data import read_manifest
from retrace.losses import ReTraceLoss
from retrace.utils import set_seed,atomic_torch_save
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import train_loader_for,clean_loss_config,weighted_deep_supervision_loss,tversky_loss,dher_v2_aux_loss
from train_e15_hr_corrected import dump,evaluate
from run_e17_e5_pair import ReplayDataset,sha_file,state_hash

TRAINABLE=['decoder.','mask_head.','aux_heads.']

def optimizer_for(model):
    for n,p in model.named_parameters():p.requires_grad_(any(n.startswith(x) for x in TRAINABLE))
    model.eval()
    groups=[]
    for prefix in TRAINABLE:
        ps=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        assert ps,prefix;groups.append({'params':ps,'name':prefix,'lr':5e-6})
    assert not any(p.requires_grad for p in model.log_var_head.parameters())
    return torch.optim.AdamW(groups,weight_decay=.01)

def objective(out,batch,seg,cfg):
    y,lab=batch['mask'].cuda(),batch['label'].cuda()
    terms=seg(out,y,lab)
    plain=F.binary_cross_entropy_with_logits(out['mask_logits'],y)
    loc_w=float(cfg['loss']['weights']['localization']);assert loc_w==1.0
    terms['total']=terms['total']-loc_w*terms['localization']+loc_w*plain
    terms['localization_heteroscedastic_diagnostic']=terms['localization']
    terms['localization']=plain
    aux=dher_v2_aux_loss(out,y,cfg);terms['total']=terms['total']+aux['dher_aux_total']
    ds=weighted_deep_supervision_loss(out,y,cfg)
    tv=tversky_loss(out['mask_logits'],y,float(cfg['loss'].get('tversky_alpha_fp',.3)),float(cfg['loss'].get('tversky_beta_fn',.7)))
    terms.update(weighted_deep_supervision=ds,tversky=tv)
    terms['total']=terms['total']+ds+float(cfg['loss']['weights'].get('tversky',0))*tv
    return terms

def audit(model,opt,batch,seg,cfg,source_state,e17):
    before=state_hash(model);frozen=state_hash(model,True)
    out=model(batch['image'].cuda());y=batch['mask'].cuda();terms=objective(out,batch,seg,cfg)
    exact=float((terms['localization']-F.binary_cross_entropy_with_logits(out['mask_logits'],y)).abs())
    assert exact==0.0
    pred=hashlib.sha256(out['mask_logits'].detach().cpu().numpy().tobytes()).hexdigest()
    assert pred==e17['preflight']['A_decoder_only']['initial_prediction_sha256']
    terms['total'].backward();groups={}
    for group in opt.param_groups:
        named=[(n,p) for n,p in model.named_parameters() if n.startswith(group['name'])]
        grads=[p.grad for _,p in named if p.grad is not None]
        norm=sum(float(g.detach().square().sum()) for g in grads)**.5
        assert math.isfinite(norm) and norm>0,(group['name'],norm)
        groups[group['name']]={'parameters':sum(p.numel() for _,p in named),'with_grad':len(grads),'grad_norm':norm,'lr':group['lr']}
    assert all(p.grad is None for p in model.log_var_head.parameters())
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5,error_if_nonfinite=True);opt.step()
    for prefix,row in groups.items():
        delta=sum(float((p.detach().cpu()-source_state[n]).abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix))
        assert delta>0;row['parameter_update_abs_sum']=delta
    assert frozen==state_hash(model,True)
    model.load_state_dict(source_state,strict=True);opt.state.clear();opt.zero_grad(set_to_none=True);assert state_hash(model)==before
    return {'status':'PASS','plain_bce_exact_abs_diff':exact,'initial_prediction_sha256':pred,'matches_E17_A_initial_prediction':True,
            'log_var_trainable_parameters':0,'log_var_gradients':0,'groups':groups,'frozen_state_unchanged':True,'audit_step_restored':True}

def main(args):
    outdir=Path(args.out);outdir.mkdir(parents=True,exist_ok=True);summary_path=outdir/'summary.yaml'
    assert not summary_path.exists(),'Refuse duplicate run'
    summary={'status':'initializing','experiment':'E18_plain_BCE','history':[],'test_run':False,'epochs':5,
             'single_change':'main localization term heteroscedastic BCE -> plain BCEWithLogits; log_var frozen/excluded',
             'stop_reason':None,'automatic_second_seed':False}
    dump(summary_path,summary)
    try:
        set_seed(3407)
        ck=torch.load(args.checkpoint,map_location='cpu');cfg=ck['config'];source_state=ck['model']
        assert ck.get('eval_variant','raw')=='raw';assert cfg['model']['architecture']=='retracenet_b3_qtr_hcv2_dher_v21_mrfe'
        cfg['training']['batch_size']=1
        e17=yaml.safe_load(Path(args.e17_summary).read_text());assert e17['status']=='completed'
        plan_path=Path(args.replay_plan);assert sha_file(plan_path)==e17['replay_plan_sha256']
        plan=np.load(plan_path);base=train_loader_for(cfg);dataset=base.dataset;n=len(base.sampler);assert n==e17['training_microbatches_per_epoch']
        summary.update(source_checkpoint=args.checkpoint,source_checkpoint_sha256=sha_file(args.checkpoint),source_checkpoint_epoch=ck['epoch'],
            source_config=cfg,replay_plan=str(plan_path),replay_plan_sha256=sha_file(plan_path),e17_A_reference={
                'best_F1':e17['arms']['A_decoder_only']['best_F1'],'best_epoch':e17['arms']['A_decoder_only']['best_epoch'],
                'final_F1':e17['history']['A_decoder_only'][-1]['selection_value'],'history':[x['selection_value'] for x in e17['history']['A_decoder_only']]},
            optimizer={'type':'AdamW','lr':5e-6,'weight_decay':.01,'constant_lr':True},gradient_accumulation=4,gradient_clip=.5,
            optimizer_steps_per_epoch=math.ceil(n/4),optimizer_steps_total=5*math.ceil(n/4),layers_mode='eval; selected parameters require gradients')
        seg=ReTraceLoss.from_config(clean_loss_config(cfg['loss'])).cuda()
        replay=ReplayDataset(dataset,plan['indices_0'][:8],plan['seeds_0'][:8]);batch=next(b for b in DataLoader(replay,batch_size=1,num_workers=0) if b['mask'].sum()>0)
        model=build_final_model(cfg['model']).cuda();model.load_state_dict(source_state,strict=True);opt=optimizer_for(model)
        summary['preflight']=audit(model,opt,batch,seg,cfg,source_state,e17);dump(outdir/'preflight.yaml',summary['preflight']);dump(summary_path,summary)
        print('[preflight]',json.dumps(summary['preflight']),flush=True)
        val=read_manifest(cfg['data']['manifest'],cfg['data']['root'],'val');frozen_initial=state_hash(model,True);best=-math.inf
        for ep in range(5):
            replay=ReplayDataset(dataset,plan[f'indices_{ep}'],plan[f'seeds_{ep}']);loader=DataLoader(replay,batch_size=1,shuffle=False,num_workers=4,pin_memory=True)
            model.eval();opt.zero_grad(set_to_none=True);totals={};stream=hashlib.sha256();start=time.time()
            summary.update(status='training',current_epoch=ep+1);dump(summary_path,summary);print(f'[train] epoch={ep+1}/5 step=0/{n}',flush=True)
            with (outdir/f'inputs_epoch_{ep+1:02d}.jsonl').open('w') as handle:
                for bi,b in enumerate(loader):
                    ident={'position':bi,'index':int(b['replay_index']),'seed':int(b['replay_seed']),'path':b['path'][0],'input_sha256':b['input_sha256'][0]}
                    line=json.dumps(ident,sort_keys=True);stream.update(line.encode());handle.write(line+'\n')
                    terms=objective(model(b['image'].cuda()),b,seg,cfg);assert torch.isfinite(terms['total'])
                    accumulation=min(4,n-(bi//4)*4);(terms['total']/accumulation).backward()
                    if (bi+1)%4==0 or bi+1==n:
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
                    for k,v in terms.items():totals[k]=totals.get(k,0.)+float(v.detach())
                    if bi==0 or (bi+1)%50==0 or bi+1==n:
                        elapsed=time.time()-start;progress={'epoch':ep+1,'step':bi+1,'steps':n,'loss':totals['total']/(bi+1),'elapsed_minutes':elapsed/60,'eta_minutes':elapsed/(bi+1)*(n-bi-1)/60}
                        dump(outdir/'progress.yaml',progress);print('[train]',json.dumps(progress),flush=True)
            stream_hash=stream.hexdigest();expected=e17['history']['A_decoder_only'][ep]['actual_input_stream_sha256'];assert stream_hash==expected
            assert state_hash(model,True)==frozen_initial
            summary['status']='validation';dump(summary_path,summary);result=evaluate(model,val,f'epoch_{ep+1:02d}_original_val',outdir)
            cur=result['metrics']['fake']['oracle']['f1'];row={'epoch':ep+1,'optimizer_steps':(ep+1)*math.ceil(n/4),'selection_value':cur,
                'delta_vs_E5':cur-e17['baseline']['metrics']['fake']['oracle']['f1'],'delta_vs_matched_E17_A':cur-e17['history']['A_decoder_only'][ep]['selection_value'],
                'train_losses':{k:v/n for k,v in totals.items()},'actual_input_stream_sha256':stream_hash,'matches_E17_A_inputs':True,
                'validation':result,'elapsed_seconds':time.time()-start}
            isbest=cur>best;best=max(best,cur);summary['history'].append(row)
            state={'epoch':ep,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':opt.state_dict(),'config':cfg,
                'eval_variant':'raw','phase':'E18_plain_BCE','best_metric':best,'selection_threshold':result['metrics']['fake']['oracle']['threshold']}
            atomic_torch_save(state,outdir/'last.pt')
            if isbest:atomic_torch_save(state,outdir/'best.pt');summary.update(best_F1=best,best_epoch=ep+1,best_checkpoint=str(outdir/'best.pt'))
            dump(summary_path,summary);print('[epoch_done]',ep+1,'F1',cur,'delta_A',row['delta_vs_matched_E17_A'],flush=True)
        final=summary['history'][-1]['selection_value'];tiny=summary['history'][summary['best_epoch']-1]['validation']['metrics']['Tp_S:lt_1']['fixed']['f1']
        full=summary['history'][summary['best_epoch']-1]['validation']['metrics']['all']['fixed']['f1'];e5=e17['baseline']['metrics']
        gates={'best_at_least_0p800':best>=.800,'final_delta_vs_A_at_least_0p003':final-e17['history']['A_decoder_only'][-1]['selection_value']>=.003,
               'all_fixed_drop_vs_E5_at_most_0p005':full>=e5['all']['fixed']['f1']-.005,'tiny_fixed_drop_vs_E5_at_most_0p005':tiny>=e5['Tp_S:lt_1']['fixed']['f1']-.005}
        summary.update(status='completed',stop_reason='fixed_5_epochs_completed',final_F1=final,final_delta_vs_E17_A=final-e17['history']['A_decoder_only'][-1]['selection_value'],gates=gates,qualifies_for_second_seed=all(gates.values()))
        assert sha_file(args.checkpoint)==summary['source_checkpoint_sha256'];dump(summary_path,summary);print('[COMPLETE]',json.dumps(gates),flush=True)
    except BaseException:
        summary.update(status='failed',stop_reason=traceback.format_exc());dump(summary_path,summary);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--e17-summary',required=True);p.add_argument('--replay-plan',required=True);main(p.parse_args())
