"""Reusable fixed-budget, resumable trainer for authorized G pilots/formal runs.

This file does not choose a route. A stage-specific command/decision must specify
the strategy and budget. Test samples are never loaded here.
"""
import argparse
import copy
import csv
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import run_fair_training_v1 as fair
from takeover_views import PairedDataset
from takeover_recalibrate import FAIR, atomic_json, sha


def training_step(model, batch, seg, cfg, strategy, cons_weight, rng, optimizer):
    optimizer.zero_grad(set_to_none=True)
    if strategy == 'paired':
        # A is the standard-view stop-gradient target. Backpropagate A first
        # and free its activations, then B; one optimizer update for the pair.
        a, b = batch['a'], batch['b']
        with torch.autocast('cuda', dtype=torch.bfloat16):
            oa = fair.fwd(model, a)
            la = fair.objective(oa, a, seg, cfg)['total']
            pa = oa['mask_logits'].detach().float().sigmoid()
        if not torch.isfinite(la):
            raise FloatingPointError('nonfinite standard view loss')
        la.backward()
        la_value = float(la.detach())
        del oa, la
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ob = fair.fwd(model, b)
            lb = fair.objective(ob, b, seg, cfg)['total']
        pb = ob['mask_logits'].float().sigmoid()
        valid = b['valid'].cuda().bool()
        consistency = F.mse_loss(pb[valid],pa[valid])
        loss = lb + cons_weight * consistency
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite degraded-view loss')
        loss.backward()
        value = la_value + float(loss.detach())
        cons_value = float(consistency.detach())
    else:
        if strategy == 'dct_dropout':
            keep = torch.rand((batch['image'].shape[0],1),generator=rng) >= .5
            batch = dict(batch)
            batch['dct_available'] = batch['dct_available'] * keep.to(batch['dct_available'].dtype)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = fair.objective(fair.fwd(model,batch),batch,seg,cfg)['total']
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite loss')
        loss.backward()
        value, cons_value = float(loss.detach()), 0.
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    optimizer.step()
    return value, cons_value, float(grad)


def rng_state(dropout_rng):
    return {'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all(),
            'numpy':np.random.get_state(), 'python':random.getstate(), 'dropout':dropout_rng.get_state()}


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock = (out/'execution.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path = out/'state.json'
    if state_path.exists() and json.loads(state_path.read_text()).get('status') == 'completed':
        print('already completed; preserving artifacts',flush=True)
        return
    torch.set_num_threads(4)
    fair.e24.set_seed(args.seed)
    source = torch.load(args.parent,map_location='cpu')
    cfg = copy.deepcopy(source['config'])
    cfg['data']['manifest'] = args.manifest
    cfg['training'].update(epochs=args.epochs,learning_rate=args.lr,batch_size=4,optimizer='SGD',ema=False)
    cfg['takeover'] = {'strategy':args.strategy,'lambda_cons':args.lambda_cons,'public_init':args.public_init,
                       'seed':args.seed,'optimizer_state_policy':'fresh_SGD_at_start;restore_on_resume',
                       'consistency':'MSE(p_B,stopgrad(p_A)) on valid pixels',
                       'paired_compute':'two views/two backward passes per optimizer update; sum of segmentation objectives'}
    train = fair.e.read_manifest(args.manifest,cfg['data']['root'],'train')
    val = fair.e.read_manifest(args.manifest,cfg['data']['root'],'val')
    cache = out/'jpeg_cache'
    ds = fair.FairDataset(train,cache,True)
    vd = fair.FairDataset(val,cache,False)
    fake = [i for i,r in enumerate(train) if r.label]
    auth = [i for i,r in enumerate(train) if not r.label]
    if not fake:
        raise ValueError('No fake training examples')
    plans = []
    for epoch in range(args.epochs):
        gen = torch.Generator().manual_seed(args.seed+epoch)
        if auth:
            # Select authentic with replacement only if too few to match fakes.
            ii = torch.randperm(len(auth),generator=gen).tolist() if len(auth)>=len(fake) else torch.randint(len(auth),(len(fake),),generator=gen).tolist()
            ids = fake + [auth[i] for i in ii[:len(fake)]]
        else:
            ids = fake.copy()
        ids = [ids[i] for i in torch.randperm(len(ids),generator=gen).tolist()]
        plans.append([(idx,args.seed*100000+epoch*100000+j) for j,idx in enumerate(ids)])
    plan_path = out/'sample_plan.pt'
    if plan_path.exists():
        assert torch.load(plan_path,weights_only=False) == plans
    else:
        torch.save(plans,plan_path)
    fair.e.dump(out/'config.yaml',cfg)
    atomic_json(out/'command.json',{'argv':sys.argv,'manifest_sha256':sha(args.manifest),
                'sample_plan_sha256':sha(plan_path),'code_sha256':sha(__file__),
                'views_sha256':sha(Path(__file__).with_name('takeover_views.py')),
                'parent_sha256':sha(args.parent),'train_count':len(train),'val_count':len(val),
                'test_run':False,'budget_epochs':args.epochs})
    model = fair.e.build_final_model(cfg['model']).cuda()
    if not args.public_init:
        model.load_state_dict(source['model'],strict=True)
    del source
    fair.mode_train(model)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=.9,weight_decay=.0005)
    seg = fair.e24.ReTraceLoss.from_config(fair.e24.clean_loss_config(cfg['loss'])).cuda()
    dropout_rng = torch.Generator().manual_seed(args.seed+990001)
    start, step_global, best, history = 1,0,-1.,[]
    resume = out/'last.pt'
    if resume.exists():
        ck = torch.load(resume,map_location='cpu')
        assert ck['config']['takeover']==cfg['takeover'], 'resume config mismatch'
        model.load_state_dict(ck['model'],strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        start,step_global,best,history=ck['epoch']+1,ck['global_step'],ck['best_so_far'],ck['history']
        rs=ck['rng'];torch.set_rng_state(rs['torch']);torch.cuda.set_rng_state_all(rs['cuda'])
        np.random.set_state(rs['numpy']);random.setstate(rs['python']);dropout_rng.set_state(rs['dropout'])
        del ck
    state={'status':'training','strategy':args.strategy,'epochs':args.epochs,'history':history,'best_F1':best,'test_run':False}
    atomic_json(state_path,state)
    try:
        total_steps=sum(math.ceil(len(p)/4) for p in plans)
        tds=PairedDataset(ds) if args.strategy=='paired' else ds
        for epoch in range(start,args.epochs+1):
            fair.mode_train(model)
            loader=DataLoader(tds,batch_size=4,sampler=plans[epoch-1],num_workers=args.workers,pin_memory=True)
            loss_sum=cons_sum=0.
            started=time.time()
            for index,batch in enumerate(loader,1):
                lr=args.lr*(1-step_global/total_steps)**.9
                for g in optimizer.param_groups:g['lr']=lr
                loss,cons,grad=training_step(model,batch,seg,cfg,args.strategy,args.lambda_cons,dropout_rng,optimizer)
                step_global+=1;loss_sum+=loss;cons_sum+=cons
                if index==1 or index%25==0 or index==len(loader):
                    progress={'epoch':epoch,'epochs':args.epochs,'batch':index,'batches':len(loader),
                              'lr':lr,'loss':loss_sum/index,'consistency':cons_sum/index,'grad_norm':grad,
                              'elapsed_seconds':time.time()-started}
                    atomic_json(out/'progress.json',progress)
                    print('[train]',json.dumps(progress),flush=True)
            state.update(status='validating',epoch=epoch)
            atomic_json(state_path,state)
            result=fair.evaluate(model,vd,cfg,out,f'epoch{epoch:02d}')
            score=result['metrics']['fake']['oracle']['f1']
            history.append({'epoch':epoch,'fake_F1_best':score,'loss':loss_sum/len(loader)})
            improved=score>best;best=max(best,score)
            ck={'model':fair.e.copy_state(model),'optimizer':optimizer.state_dict(),'config':cfg,
                'epoch':epoch,'global_step':step_global,'best_metric':score,'best_so_far':best,
                'selection_threshold':result['metrics']['fake']['oracle']['threshold'],
                'eval_variant':'raw','history':history,'rng':rng_state(dropout_rng)}
            fair.e24.atomic_torch_save(ck,out/'last.pt')
            if improved:
                fair.e24.atomic_torch_save(ck,out/'best.pt')
                state.update(best_epoch=epoch,best_checkpoint=str(out/'best.pt'))
            del ck
            state.update(status='training',best_F1=best,history=history)
            atomic_json(state_path,state);fair.e.dump(out/'summary.yaml',state)
        state.update(status='completed',stop_reason='fixed_budget_complete')
        atomic_json(state_path,state);fair.e.dump(out/'summary.yaml',state)
        atomic_json(out/'decision.json',{'status':'completed','route':'proxy_evaluation',
                    'evidence':{'clean_best':best},'reason':'Training complete; acceptance requires matched proxy results',
                    'rejected_routes':['select_on_external_test']})
    except BaseException:
        state.update(status='failed',error=traceback.format_exc())
        atomic_json(state_path,state)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True);p.add_argument('--parent',default=str(FAIR/'best.pt'))
    p.add_argument('--manifest',default=str(FAIR/'clean_manifest.csv'))
    p.add_argument('--strategy',choices=['control','dct_dropout','paired'],required=True)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--seed',type=int,default=8800);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--lambda-cons',type=float,default=.1);p.add_argument('--public-init',action='store_true')
    main(p.parse_args())
