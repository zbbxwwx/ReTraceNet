"""Matched E24 continuation vs native-grid DCT adaptation. No test, no new modules."""
import argparse, copy, hashlib, json, random, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import run_e24_dher_calibrated_writeback as e24
from native_dct_grid import NativeGridDataset, SeededSamples, attach_native_grid, forward_batch, pad_blocks

e=e24.e21
SOURCE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e24_dher_calibrated_writeback_20260910/best.pt')
BASELINE=.7982278425423237

def trainable(model):
    specs=[('dct_branch.output_projections.',1e-5),('dct_inject_adapters.',1e-5),
           ('mask_head.',1e-5),('decoder.stage1.fuse.',1e-5)]
    groups=[]
    for p in model.parameters():p.requires_grad_(False)
    for prefix,lr in specs:
        params=[];names=[]
        for n,p in model.named_parameters():
            if not n.startswith(prefix):continue
            if prefix=='decoder.stage1.fuse.' and not n.startswith(tuple(prefix+str(i)+'.' for i in [3,4,5])):continue
            p.requires_grad_(True);params.append(p);names.append(n)
        assert params,prefix
        groups.append({'params':params,'lr':lr,'weight_decay':0.0 if prefix=='mask_head.' else .01,'name':prefix,'names':names})
    model.eval() # Frozen BN statistics; trainable projections use LayerNorm/GELU.
    return groups

def frozen_hash(model):
    h=hashlib.sha256()
    for n,p in list(model.named_parameters())+list(model.named_buffers()):
        if isinstance(p,torch.nn.Parameter) and p.requires_grad:continue
        h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def batch(ds,index,seed):
    sample=SeededSamples(ds)[index,seed]
    return {k:v[None] if torch.is_tensor(v) else [v] for k,v in sample.items()}

def fingerprint(h,b):
    h.update(str(b['path']).encode())
    for k in ['image','mask','label','dct_available']:
        h.update(b[k].contiguous().numpy().tobytes())

def evaluate(model,ds,records,out,label):
    stats=e.ValidationStats();fmt={k:e.ValidationStats() for k in ['jpeg','nonjpeg']}
    with torch.inference_mode():
        for i,r in enumerate(records):
            b=batch(ds,i,0);o=forward_batch(model,b)
            with Image.open(r.image) as im:w,h=im.size
            y=np.zeros((h,w),dtype=np.int16) if r.mask is None else (np.asarray(Image.open(r.mask).convert('L'))>127).astype(np.int16)
            p=torch.nn.functional.interpolate(o['mask_logits'].float().sigmoid(),size=(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            stats.add(p,y,r.label,r.image)
            fmt['jpeg' if r.image.suffix.lower() in {'.jpg','.jpeg'} else 'nonjpeg'].add(p,y,r.label,r.image)
            if i==0 or (i+1)%200==0:print('[eval]',label,i+1,len(records),flush=True)
    result={'metrics':stats.result(),'formats':{k:v.result() for k,v in fmt.items()}}
    e.dump(out/(label+'.yaml'),result)
    return result

def audit(source,cfg,old_ds,new_ds,out):
    a=e.build_final_model(cfg['model']).cuda().eval();a.load_state_dict(source['model'],strict=True)
    b=e.build_final_model(cfg['model']).cuda().eval();b.load_state_dict(source['model'],strict=True);attach_native_grid(b)
    reports=[];selected=[]
    for ext in ['jpeg','nonjpeg']:
        found=next(i for i,r in enumerate(old_ds.records) if r.label and (r.image.suffix.lower() in {'.jpg','.jpeg'})==(ext=='jpeg'))
        selected.append(found)
    # Check block-phase-preserving padding with identifiable frequency positions.
    coef=np.arange(24*40).reshape(24,40);padded=pad_blocks(coef)
    assert np.array_equal(padded[:24,:40],coef)
    assert np.array_equal(padded[-8:,-8:],coef[-8:,-8:])
    for index in selected:
        for seed in [3,17,31,49]:
            x=batch(old_ds,index,seed);y=batch(new_ds,index,seed)
            for key in ['image','mask','label','dct_available']:assert torch.equal(x[key],y[key]),key
            assert y['dct'].shape[-1]%32==0 and y['dct'].shape[-2]%32==0
            with torch.no_grad():
                oa=forward_batch(a,x);ob=forward_batch(b,y)
            diff=float((oa['mask_logits']-ob['mask_logits']).abs().max())
            if not bool(y['dct_available'].item()):assert diff<1e-6,diff
            assert oa['mask_logits'].shape==ob['mask_logits'].shape
            reports.append({'path':x['path'][0],'seed':seed,'same_RGB_GT_availability':True,
                            'native_logit_difference_expected':diff,'shapes':b._native_shapes})
    grad_audit={}
    for name,model,ds in [('control',a,old_ds),('native_grid',b,new_ds)]:
        groups=trainable(model);opt=torch.optim.AdamW(groups)
        initial=e.copy_state(model);frozen=frozen_hash(model)
        x=batch(ds,selected[0],17)
        seg=e24.ReTraceLoss.from_config(e24.clean_loss_config(cfg['loss'])).cuda()
        loss=e24.objective(forward_batch(model,x),x,seg,cfg)['total']
        assert torch.isfinite(loss);loss.backward()
        g={group['name']:sum(float(p.grad.detach().square().sum()) for p in group['params'] if p.grad is not None)**.5 for group in groups}
        assert all(v>0 and np.isfinite(v) for v in g.values()),g
        opt.step()
        changes={group['name']:sum(float((dict(model.named_parameters())[n].detach().cpu()-initial[n]).abs().sum()) for n in group['names']) for group in groups}
        assert all(v>0 for v in changes.values()),changes
        assert frozen_hash(model)==frozen
        grad_audit[name]={'loss':float(loss),'grad_norm':g,'parameter_delta_L1':changes,'frozen_hash_unchanged':True}
    report={'status':'passed','strict_E24_load':True,'raw_block_padding_test':True,'samples':reports,'gradient_audit':grad_audit,
            'identity_note':'Native JPEG features intentionally differ at step0. NonJPEG predictions must remain identical.'}
    e.dump(out/'preflight.yaml',report)
    del a,b;torch.cuda.empty_cache()
    return report

def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.yaml').exists():raise RuntimeError('Refuse to overwrite an existing experiment')
    summary={'status':'initializing','test_run':False,'epochs_per_arm':args.epochs,'seed':3407,
             'source':str(SOURCE),'source_sha256':e.sha_file(SOURCE),'arms':{},
             'single_change':'encode original block-grid DCT then apply RGB geometry to encoded features',
             'context_note':'Native branch sees full original JPEG before feature cropping; crop context also changes as part of this pipeline.',
             'selection':'raw original-GT-grid fake pooled F1-best; fixed threshold .5 diagnostics',
             'acceptance':'native best>=.800 AND epoch3 advantage>=.003; all fixed drop<=.001 and tiny fixed drop<=.005 versus control epoch3',
             'inference_loader':'native_dct_grid.load_pipeline_checkpoint + NativeGridDataset + forward_batch'}
    def save():e.dump(out/'summary.yaml',summary)
    save()
    try:
        torch.set_num_threads(4);e24.set_seed(3407)
        source=torch.load(SOURCE,map_location='cpu');cfg=copy.deepcopy(source['config'])
        assert cfg['training']['batch_size']==1
        assert cfg['data'].get('dct_source_mode')=='original_jpeg'
        # Sampler itself is retained; materialize identical indices for both arms.
        base_loader=e24.train_loader_for(cfg);old_ds=base_loader.dataset
        new_ds=NativeGridDataset.from_dataset(old_ds)
        n=len(old_ds);plans=[]
        for epoch in range(args.epochs):
            gen=torch.Generator().manual_seed(3407+epoch)
            if isinstance(base_loader.sampler,WeightedRandomSampler):
                ids=torch.multinomial(base_loader.sampler.weights,base_loader.sampler.num_samples,base_loader.sampler.replacement,generator=gen).tolist()
            elif isinstance(base_loader.sampler,RandomSampler):ids=torch.randperm(n,generator=gen).tolist()
            else:ids=list(base_loader.sampler)
            plans.append([(idx,340700000+epoch*100000+j) for j,idx in enumerate(ids)])
        torch.save(plans,out/'sample_plan.pt')
        summary['sample_plan_sha256']=e.sha_file(out/'sample_plan.pt')
        summary['steps_per_epoch']=[len(p) for p in plans]
        e.dump(out/'config.yaml',cfg)
        summary['preflight']=audit(source,cfg,old_ds,new_ds,out);save()
        records=e.read_manifest(cfg['data']['manifest'],cfg['data']['root'],'val')
        val_old=e.ManipulationDataset(records,image_size=cfg['data']['image_size'],training=False,dct_forensic=True,dct_source_mode='original_jpeg')
        for arm,ds,vds in [('control',old_ds,val_old),('native_grid',new_ds,NativeGridDataset.from_dataset(val_old))]:
            e24.set_seed(3407)
            dest=out/arm;dest.mkdir(exist_ok=True)
            model=e.build_final_model(cfg['model']).cuda().eval();model.load_state_dict(source['model'],strict=True)
            if arm=='native_grid':attach_native_grid(model)
            groups=trainable(model);opt=torch.optim.AdamW(groups)
            frozen=frozen_hash(model)
            seg=e24.ReTraceLoss.from_config(e24.clean_loss_config(cfg['loss'])).cuda()
            entry={'status':'initial_validation','history':[],'best_F1':-1.,'trainable_groups':[{k:v for k,v in g.items() if k!='params'} for g in groups]}
            summary['arms'][arm]=entry;summary.update(status='running',current_arm=arm);save()
            initial=evaluate(model,vds,records,dest,'step0');entry['step0']=initial
            if arm=='control':assert abs(initial['metrics']['fake']['oracle']['f1']-BASELINE)<1e-8
            save()
            for epoch,plan in enumerate(plans,1):
                model.eval();opt.zero_grad(set_to_none=True);total=0.;inputs=hashlib.sha256();started=time.time()
                loader=DataLoader(SeededSamples(ds),batch_size=1,sampler=plan,num_workers=0)
                acc=int(cfg['training'].get('gradient_accumulation_steps',4))
                entry.update(status='training',epoch=epoch);save()
                for step,x in enumerate(loader,1):
                    fingerprint(inputs,x)
                    loss=e24.objective(forward_batch(model,x),x,seg,cfg)['total']
                    if not torch.isfinite(loss):raise FloatingPointError(f'{arm} {epoch} {step}')
                    window=min(acc,len(plan)-((step-1)//acc)*acc)
                    (loss/window).backward();total+=float(loss.detach())
                    if step%acc==0 or step==len(plan):
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.5,error_if_nonfinite=True)
                        opt.step();opt.zero_grad(set_to_none=True)
                    if step==1 or step%25==0 or step==len(plan):
                        elapsed=time.time()-started
                        progress={'arm':arm,'epoch':epoch,'epochs':args.epochs,'step':step,'steps':len(plan),'loss':total/step,
                                  'elapsed_minutes':elapsed/60,'eta_minutes':elapsed/step*(len(plan)-step)/60}
                        e.dump(out/'progress.yaml',progress);print('[train]',json.dumps(progress),flush=True)
                assert frozen_hash(model)==frozen,'Frozen weights or BN buffers changed'
                digest=inputs.hexdigest()
                if arm=='native_grid':assert digest==summary['arms']['control']['history'][epoch-1]['input_sha256'],'Sample/augmentation mismatch'
                entry['status']='validating';save()
                val=evaluate(model,vds,records,dest,f'epoch{epoch:02d}')
                score=val['metrics']['fake']['oracle']['f1']
                row={'epoch':epoch,'steps':len(plan),'optimizer_steps':(len(plan)+acc-1)//acc,'input_sha256':digest,'train_loss':total/len(plan),'validation':val,'selection_value':score,'frozen_unchanged':True}
                entry['history'].append(row)
                ck={'model':e.copy_state(model),'config':cfg,'optimizer':opt.state_dict(),'epoch':epoch,'best_metric':score,'eval_variant':'raw','dct_pipeline':arm,'inference_requires':'native_dct_grid.load_pipeline_checkpoint / forward_batch','selection_threshold':val['metrics']['fake']['oracle']['threshold']}
                e24.atomic_torch_save(ck,dest/'last.pt')
                if score>entry['best_F1']:
                    entry.update(best_F1=score,best_epoch=epoch,best_checkpoint=str(dest/'best.pt'));e24.atomic_torch_save(ck,dest/'best.pt')
                save();print('[epoch_done]',arm,epoch,score,flush=True)
            entry.update(status='completed',stop_reason='fixed_budget_completed');save()
            del model,opt,seg;torch.cuda.empty_cache()
        ctrl=summary['arms']['control']['history'][-1]['validation']['metrics'];new=summary['arms']['native_grid']['history'][-1]['validation']['metrics']
        delta=new['fake']['oracle']['f1']-ctrl['fake']['oracle']['f1']
        safe=(new['all']['fixed']['f1']>=ctrl['all']['fixed']['f1']-.001 and new['Tp_S:lt_1']['fixed']['f1']>=ctrl['Tp_S:lt_1']['fixed']['f1']-.005)
        summary.update(status='completed',stop_reason='both_fixed_budgets_completed_no_automatic_extension',epoch3_fake_advantage=delta,
                       acceptance_pass=bool(summary['arms']['native_grid']['best_F1']>=.800 and delta>=.003 and safe))
        save();print('PAIR_COMPLETE',summary['acceptance_pass'],flush=True)
    except BaseException:
        summary.update(status='failed',stop_reason=traceback.format_exc());save();raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--epochs',type=int,default=3)
    main(parser.parse_args())
