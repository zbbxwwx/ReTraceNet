"""Direct E15 decoder, frozen evidence stack, weak correct-pixel E5 teacher."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrace.data import read_manifest
from retrace.utils import load_config, set_seed, atomic_torch_save
from retracenet.final_modules import build_final_model
from train_e15_hr_corrected import dump, loader_for, train_mode, make_optimizer, evaluate, objective as gt_objective


def consistency(student, teacher, target, confidence=.9):
    teacher_prob = teacher.detach().float().sigmoid()
    eligible = ((teacher_prob >= confidence) & (target > .5)) | ((teacher_prob <= 1-confidence) & (target <= .5))
    # Bernoulli KL: subtract teacher entropy; teacher is always detached.
    p = teacher_prob.clamp(1e-6, 1-1e-6)
    entropy = -(p*p.log() + (1-p)*(1-p).log())
    kl = F.binary_cross_entropy_with_logits(student.float(), p, reduction='none') - entropy
    loss = (kl * eligible).sum() / eligible.sum().clamp_min(1)
    return loss, eligible.float().mean()


def objective(out, target, cfg, weight):
    terms = gt_objective(out, target, cfg)
    value, fraction = consistency(out['mask_logits'], out['teacher_mask_logits'], target)
    terms['teacher_kl'] = value
    terms['teacher_eligible_fraction'] = fraction
    terms['total'] = terms['total'] + weight * value
    return terms


def preflight(model, optimizer, batch, cfg, directory):
    train_mode(model)
    x, y = batch['image'].cuda(), batch['mask'].cuda()
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    prefixes = ['content_encoder.proj.', *[f'e15_decoder.stage{i}.' for i in range(1,5)],
                'e15_decoder.evidence_adapters.', 'e15_decoder.spatial_stem.',
                'e15_decoder.r1_', 'e15_decoder.r2_', 'e15_decoder.region_head.', 'e15_decoder.boundary_head.']
    grads = {key:0. for key in prefixes}
    # Make old logit differentiable for this audit only, proving no output dependency.
    hook = model.mask_head.register_forward_hook(lambda module, inputs, output: output.requires_grad_(True))
    out = model(x)
    direct_diff = float((out['mask_logits'] - out['direct_mask_logits']).detach().abs().max())
    assert direct_diff == 0
    dependency = torch.autograd.grad(out['mask_logits'].mean(), out['e15_old_logits'], allow_unused=True, retain_graph=True)[0]
    assert dependency is None, 'Final mask still depends on old mask logits'
    assert not out['teacher_mask_logits'].requires_grad
    old_before = out['teacher_mask_logits'].detach().clone()
    hook.remove()
    del out
    last = model.e15_decoder.region_head[-1]
    assert float(last.weight.detach().square().mean()) > 0, 'Direct head must not be zero initialized'
    # Explicit supervision-mask test: confident wrong teachers must be excluded.
    test_student = torch.zeros(1,1,1,4,device='cuda',requires_grad=True)
    test_teacher = torch.tensor([[[[10.,-10.,10.,-10.]]]],device='cuda')
    test_gt = torch.tensor([[[[1.,0.,0.,1.]]]],device='cuda')
    test_loss, test_fraction = consistency(test_student,test_teacher,test_gt)
    tg = torch.autograd.grad(test_loss,test_student)[0]
    assert float(test_fraction)==.5 and bool((tg[...,2:]==0).all()) and bool((tg[...,:2]!=0).all())
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        terms = objective(out,y,cfg,.05)
        assert torch.isfinite(terms['total'])
        terms['total'].backward()
        for prefix in prefixes:
            grads[prefix] = max(grads[prefix], sum(float(p.grad.detach().square().sum()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None)**.5)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5,error_if_nonfinite=True)
        optimizer.step()
        del out,terms
    updates = {key:sum(float((p.detach().cpu()-initial[n]).abs().sum()) for n,p in model.named_parameters() if n.startswith(key)) for key in prefixes}
    assert all(v>0 and math.isfinite(v) for v in grads.values()), grads
    assert all(v>0 for v in updates.values()), updates
    frozen_delta = max(float((p.detach().cpu()-initial[n]).abs().max()) for n,p in model.named_parameters() if not p.requires_grad)
    assert frozen_delta==0
    with torch.no_grad():
        out=model(x)
        teacher_drift=float((out['teacher_mask_logits']-old_before).abs().max())
        assert teacher_drift==0
        del out
    report={'status':'PASS','direct_output_max_diff':direct_diff,'final_gradient_to_old_logits':None,
            'teacher_detached':True,'teacher_wrong_pixel_exclusion_test':'PASS','teacher_drift':teacher_drift,
            'frozen_parameter_delta':frozen_delta,'gradient_norms':grads,'parameter_update_abs_sums':updates,
            'new_classifier_weight_rms':float(initial['e15_decoder.region_head.3.weight'].square().mean().sqrt()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'audit_steps_restored':True}
    dump(directory/'preflight.yaml',report)
    print(yaml.safe_dump(report,sort_keys=False),flush=True)
    model.load_state_dict(initial,strict=True)
    optimizer.state.clear(); optimizer.zero_grad(set_to_none=True)
    del initial,x,y,old_before
    gc.collect(); torch.cuda.empty_cache()
    return report


def main(config):
    cfg=load_config(config); directory=Path(cfg['output_dir']); directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'last.pt').exists(), 'Refuse to overwrite existing run'
    summary={'status':'initializing','phase':'direct_decoder_only','max_epochs':5,'first_gate_epoch':3,
             'gate_delta_vs_e15':.005,'forensic_frozen':True,'evidence_stack_frozen':True,'test_run':False,
             'selection':'val_original_grid_fake_pooled_F1_best','auc_used_for_gate':False,'history':[],
             'config':cfg,'stop_reason':None}
    summary_path=directory/'overnight_summary.yaml'; dump(summary_path,summary)
    try:
        set_seed(cfg['seed'])
        source=Path(cfg['training']['init_checkpoint'])
        checkpoint=torch.load(source,map_location='cpu')
        parent=yaml.safe_load(Path(cfg['reference_summary']).read_text())
        parent_epoch=int(checkpoint['epoch'])+1
        reference=next(r for r in parent['history'] if r['epoch']==parent_epoch)
        reference_f1=reference['selection_value']
        assert abs(reference_f1-checkpoint['best_metric'])<1e-8
        digest=hashlib.sha256()
        with source.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
        summary.update(initial_checkpoint=str(source),initial_checkpoint_sha256=digest.hexdigest(),
                       initial_epoch=parent_epoch,reference_e15=reference,reference_e15_f1=reference_f1,
                       e5_baseline=parent['e5_baseline'],catnet_validation=parent['catnet_validation'])
        model=build_final_model(cfg['model']).cuda()
        loaded=model.load_state_dict(checkpoint['model'],strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        del checkpoint
        model.e15_decoder.region_head[-1].reset_parameters()
        summary['initialization']={'all_reusable_keys_loaded_strictly':True,'reset_keys':['e15_decoder.region_head.3.weight','e15_decoder.region_head.3.bias']}
        optimizer=make_optimizer(model,float(cfg['training']['learning_rate']))
        loader=loader_for(cfg)
        summary['preflight']=preflight(model,optimizer,next(iter(loader)),cfg,directory)
        records=read_manifest(cfg['data']['manifest'],cfg['data']['root'],'val')
        best=-math.inf
        for ep in range(5):
            lr=1e-4 if ep<3 else 5e-5
            teacher_weight=[.05,.025,0.,0.,0.][ep]
            for group in optimizer.param_groups: group['lr']=lr
            train_mode(model); optimizer.zero_grad(set_to_none=True)
            start=time.time(); totals={}; sampled={}
            summary.update(status='training',current_epoch=ep+1,lr=lr,teacher_weight=teacher_weight)
            dump(summary_path,summary)
            print(f'[train] epoch={ep+1}/5 step=0/{len(loader)} lr={lr} teacher_weight={teacher_weight}',flush=True)
            for bi,b in enumerate(loader):
                x,y=b['image'].cuda(),b['mask'].cuda()
                out=model(x); terms=objective(out,y,cfg,teacher_weight)
                if not torch.isfinite(terms['total']): raise FloatingPointError('Nonfinite training loss')
                accumulation=min(4,len(loader)-(bi//4)*4)
                (terms['total']/accumulation).backward()
                if (bi+1)%4==0 or bi+1==len(loader):
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5,error_if_nonfinite=True)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                for k,v in terms.items(): totals[k]=totals.get(k,0.)+float(v.detach())
                for k in [b['geometry'][0],b['post_geometry_bucket'][0]]: sampled[k]=sampled.get(k,0)+1
                del out,terms,x,y
                if bi==0 or (bi+1)%25==0 or bi+1==len(loader):
                    elapsed=time.time()-start
                    progress={'epoch':ep+1,'step':bi+1,'steps':len(loader),'lr':lr,'teacher_weight':teacher_weight,
                              'loss':totals['total']/(bi+1),'elapsed_minutes':elapsed/60,
                              'eta_minutes':elapsed/(bi+1)*(len(loader)-bi-1)/60}
                    dump(directory/'progress.yaml',progress); print('[train] '+json.dumps(progress),flush=True)
            summary['status']='validation'; dump(summary_path,summary)
            result=evaluate(model,records,f'epoch_{ep+1:02d}_original_val',directory,include_old=True)
            current=result['metrics']['fake']['oracle']['f1']
            assert math.isfinite(current)
            assert abs(result['old_path']['fake']['oracle']['f1']-parent['e5_baseline']['metrics']['fake']['oracle']['f1'])<1e-8,'Frozen teacher changed'
            comparison={k:{'fixed_f1_delta':v['fixed']['f1']-reference['validation']['metrics'][k]['fixed']['f1'],
                           'oracle_f1_delta':v['oracle']['f1']-reference['validation']['metrics'][k]['oracle']['f1'],
                           'current_precision_fixed':v['fixed']['precision'],'current_recall_fixed':v['fixed']['recall']}
                        for k,v in result['metrics'].items() if k in reference['validation']['metrics']}
            row={'epoch':ep+1,'lr':lr,'teacher_weight':teacher_weight,'selection_value':current,
                 'delta_vs_e15':current-reference_f1,'train_losses':{k:v/len(loader) for k,v in totals.items()},
                 'sampled_geometry_and_buckets':sampled,'validation':result,'comparison_vs_e15':comparison,'elapsed_seconds':time.time()-start}
            is_best=current>best; best=max(best,current)
            state={'epoch':ep,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),
                   'config':cfg,'eval_variant':'raw','best_metric':best,'phase':'e16_direct',
                   'selection_threshold':result['metrics']['fake']['oracle']['threshold']}
            atomic_torch_save(state,directory/'last.pt')
            if is_best:
                atomic_torch_save(state,directory/'best.pt')
                summary.update(best_epoch=ep+1,best_F1=best,best_checkpoint=str(directory/'best.pt'),
                               selected_threshold=state['selection_threshold'])
            del state
            summary['history'].append(row); summary['completed_epochs']=ep+1
            dump(summary_path,summary)
            dump(directory/f'comparison_epoch_{ep+1:02d}.yaml',comparison)
            print(f'[epoch_done] epoch={ep+1} F1={current:.8f} delta_E15={current-reference_f1:.8f}',flush=True)
            if ep+1>=3 and best<reference_f1+.005:
                summary['stop_reason']='F1_gate_not_met_after_3_epochs: best < E15_reference + 0.005'
                break
        summary['status']='completed'; summary['stop_reason']=summary['stop_reason'] or 'maximum_5_epochs'
        dump(summary_path,summary); print('[complete] '+summary['stop_reason'],flush=True)
    except BaseException:
        summary.update(status='failed',stop_reason=traceback.format_exc()); dump(summary_path,summary)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True)
    main(p.parse_args().config)
