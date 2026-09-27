"""Balanced residual dual-stream QTR-HCV-DHER full 30-epoch training; no test."""
import argparse, copy, csv, hashlib, json, math, time, traceback
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import run_e24_dher_calibrated_writeback as e24
from fair_training_data import FairDataset,crop_outputs
from train_v2_final_pipeline import hcv_loss
from v3_unified_metrics import PixelCounts, AuthenticPixelFPR, ImageDetectionStats, PROTOCOL
e=e24.e21
BASE=Path('/group4/zouwenxuan/iml_generalization')
SOURCE=BASE/'outputs/retracenet_v2/e24_dher_calibrated_writeback_20260910/best.pt'
GROUPS=['content_encoder.','forensic_stream.','dct_branch.','balanced_forensic_fusions.','qtr.','hcv2.','qtr_aux_heads.','dher.','decoder.','mask_head.']

def objective(out,b,seg,cfg):
    bs,_,fh,fw=b['image'].shape;terms={}
    for i in range(bs):
        vh,vw=map(int,b['valid_hw'][i].tolist())
        oo=crop_outputs(out,i,bs,vh,vw,fh,fw)
        bb={'mask':b['mask'][i:i+1,:,:vh,:vw],'label':b['label'][i:i+1]}
        loss=e24.original_objective(oo,bb,seg,cfg)
        # E24's writer-adaptation entry point omits the existing HCV objective.
        # Full mechanism training must supervise all three configured HCV scales.
        with torch.autocast('cuda',enabled=False):
            if 'hcv2' in oo:
                hh={k:[t.float() for t in v] if isinstance(v,list) else v.float() if torch.is_tensor(v) else v for k,v in oo['hcv2'].items()}
                hterm=hcv_loss({'hcv2':hh},bb['mask'].float().cuda(),bb['label'].float().cuda(),cfg)['total']
            else:
                hterm=oo['mask_logits'].sum()*0.0
        loss['total']=loss['total']+hterm
        loss['hcv_existing_supervision']=hterm
        target=bb['mask'].float().cuda()
        def seg_aux(logit,tgt):
            z=F.interpolate(logit.float(),size=tgt.shape[-2:],mode='bilinear',align_corners=False)
            bce=F.binary_cross_entropy_with_logits(z,tgt)
            pr=z.sigmoid(); inter=(pr*tgt).sum((1,2,3)); den=pr.sum((1,2,3))+tgt.sum((1,2,3))
            return bce+(1-(2*inter+1)/(den+1)).mean()
        qlogits=oo.get('qtr_aux_logits',[])
        qloss=sum(seg_aux(z,target) for z in qlogits)/len(qlogits) if qlogits else oo['mask_logits'].sum()*0.0
        dher=oo.get('dher'); sl=[]
        if dher is not None:
            for ss,rr in zip(dher['support'],dher['refutation']):
                sp=ss.float()/(1+ss.float()); rp=rr.float()/(1+rr.float())
                tt=F.interpolate(target,size=sp.shape[-2:],mode='nearest')
                sp=sp.clamp(1e-5,1-1e-5);rp=rp.clamp(1e-5,1-1e-5)
                support_ce=-(tt*sp.log()+(1-tt)*(1-sp).log()).mean()
                refute_target=1-tt
                refute_ce=-(refute_target*rp.log()+(1-refute_target)*(1-rp).log()).mean()
                sl.append(support_ce+refute_ce)
        dloss=sum(sl)/len(sl) if sl else oo['mask_logits'].sum()*0.0
        loss['total']=loss['total']+0.15*qloss+0.05*dloss
        loss['qtr_direct_supervision']=qloss;loss['dher_direct_supervision']=dloss
        for k,v in loss.items():terms[k]=terms.get(k,0)+v/bs
    return terms

def fwd(model,b,cfg=None,train_dropout=False):
    x={k:b[k].cuda(non_blocking=True) for k in ['image','dct','qtable','dct_available']}
    if train_dropout and cfg is not None:
        p=float(cfg['training'].get('dct_sample_dropout',0.0))
        keep=(torch.rand_like(x['dct_available'].float())>=p).to(x['dct_available'].dtype)
        x['dct_available']=x['dct_available']*keep
    return model(**x)

def mode_train(model):
    model.train()
    for p in model.parameters():p.requires_grad_(True)

def audit(model,ds,seg,cfg,amp):
    mode_train(model);opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=0.,weight_decay=0.)
    norms={k:0. for k in GROUPS};changes={k:0. for k in GROUPS};record=[]
    ids=[i for i,r in enumerate(ds.records) if r.label][:4]
    b=next(iter(DataLoader(ds,batch_size=4,sampler=[(i,3407+i) for i in ids],num_workers=0)))
    origins=b['crop_origin'];assert bool((origins%8==0).all())
    assert bool((b['dct_available']==1).all())
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):out=fwd(model,b);terms=objective(out,b,seg,cfg)
        loss=terms['total'];assert torch.isfinite(loss)
        loss.backward()
        for k in GROUPS:
            g=sum(float(p.grad.float().square().sum()) for n,p in model.named_parameters() if n.startswith(k) and p.grad is not None)**.5
            norms[k]=max(norms[k],g)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        before={n:p.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        opt.step()
        for n,p in model.named_parameters():
            if n in before:
                for k in GROUPS:
                    if n.startswith(k):changes[k]+=float((p.detach()-before[n]).abs().sum())
        record.append(float(loss.detach()));del before
    assert all(v>0 and np.isfinite(v) for v in norms.values()),norms
    assert all(v>0 for v in changes.values()),changes
    # Loss uses only valid outputs: corrupt the cropped-away padding and recheck.
    model.eval()
    pad_id=next(i for i,r in enumerate(ds.records) if min(Image.open(r.image).size)<512)
    pb=next(iter(DataLoader(ds,batch_size=1,sampler=[(pad_id,9)])))
    with torch.no_grad():
        po=fwd(model,pb);l1=objective(po,pb,seg,cfg)['total']
        vh,vw=map(int,pb['valid_hw'][0].tolist());changed=dict(po)
        for key in ['mask_logits','log_var']:
            t=po[key].clone();t[...,vh:,:]=123;t[...,:,vw:]=-123;changed[key]=t
        l2=objective(changed,pb,seg,cfg)['total'];diff=float((l1-l2).abs());assert diff<1e-6,diff
    no_grad=[n for n,p in model.named_parameters() if p.grad is None]
    assert not any(n.startswith('hcv2.refs.') or n.startswith('hcv2.verifiers.') for n in no_grad),no_grad
    return {'status':'passed','steps':4,'grad_norm_max':norms,'parameter_delta_L1':changes,'losses':record,
            'padding_loss_invariance':diff,'last_probe_no_gradient_parameters':no_grad,
            'note':'Inactive auxiliary heads may lack gradients because original objective weights are zero; all core families must update.',
            'total_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'batch_size':4,'amp_bfloat16':amp,'audit_weight_decay':0.,'audit_momentum':0.,'max_memory_GB':torch.cuda.max_memory_allocated()/1024**3}

@torch.inference_mode()
def evaluate(model,ds,cfg,dest,label,locked_threshold=None,image_locked_threshold=None,evaluation_mode="validation_select"):
    model.eval();stats=e.ValidationStats();formats={k:e.ValidationStats() for k in ['jpeg','nonjpeg']}
    fake_locked=PixelCounts(locked_threshold) if locked_threshold is not None else None
    authentic=AuthenticPixelFPR(); image_detection=ImageDetectionStats()
    for i,r in enumerate(ds.records):
        b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)))
        o=fwd(model,b);vh,vw=map(int,b['valid_hw'][0].tolist())
        p=o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy();y=b['mask'][0,0,:vh,:vw].numpy().astype(np.int16)
        image_score=float(o['image_logit'].float().sigmoid().reshape(-1).mean().cpu())
        stats.add(p,y,r.label,r.image)
        formats['nonjpeg' if bool(b['generated'].item()) else 'jpeg'].add(p,y,r.label,r.image)
        image_detection.add(image_score,r.label)
        if r.label:
            if fake_locked is not None: fake_locked.add(p,y)
        else: authentic.add(p)
        if i==0 or (i+1)%200==0:print('[val]',label,i+1,len(ds),flush=True)
    metrics=stats.result(); selected=float(locked_threshold) if locked_threshold is not None else float(metrics['fake']['oracle']['threshold'])
    locked=(fake_locked.result() if fake_locked is not None else dict(metrics['fake']['oracle']))
    result={'protocol':dict(PROTOCOL),'evaluation_mode':evaluation_mode,'threshold_source':('dataset_val' if locked_threshold is not None else 'current_validation_fake_only'),'pixel_threshold':selected,'image_threshold_source':('dataset_val' if image_locked_threshold is not None else 'current_validation_all_images'),
            'pixel_localization_fake_only':{'test_f1_locked':locked,'auc':metrics['fake']['auc'],'test_f1_oracle_diagnostic':metrics['fake']['oracle']},
            'authentic_pixel_fpr':authentic.result(selected),'image_level_all':image_detection.result(image_locked_threshold),
            'preprocessing':{'input_size':'native original size padded right/bottom to multiple32','probability_interpolation':'bilinear before threshold if required','normalization':'RGB float [0,1]; model-owned encoder normalization','jpeg_handling':'JPEG original; non-JPEG converted once to q100 subsampling0','dct_block_alignment':'RGB/DCT/qtable identical JPEG; crop origins divisible by8','tta':'none','padding':'excluded from metrics'},'evaluator_version':'v3_unified_protocol_2','metrics':metrics,'formats':{k:v.result() for k,v in formats.items()}}
    e.dump(dest/(label+'.yaml'),result);return result

def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.yaml').exists():raise RuntimeError('Refuse duplicate run')
    summary={'status':'preparing','experiment':'ReTraceNet_Balanced_Residual_V3_Pilot','test_run':False,'epochs':30,
             'init':'public ResNet50 ImageNet + public DCT_djpeg; remaining layers seeded random, NO E24 localization weights',
             'scope':'CAT-like data/optimizer recipe, ReTraceNet architecture and original composite loss; not an isolated causal ablation',
             'selection':'val fake pooled F1-best; threshold chosen only on val; no polarity swapping',
             'test_policy':'not evaluated; existing test list left unchanged',
             'stop_reason':None}
    def save():e.dump(out/'summary.yaml',summary)
    save()
    try:
        torch.set_num_threads(4);e24.set_seed(3407)
        source=torch.load(SOURCE,map_location='cpu');cfg=copy.deepcopy(source['config']);del source
        audit_json=json.loads((BASE/'outputs/retracenet_v2/cat_fairness_audit_20260911/report.json').read_text())
        excluded={p for g in audit_json['content_cross_split_overlap']['pixels'] for p in g['train']}
        assert len(excluded)==46
        old=Path(cfg['data']['manifest']);rows=list(csv.DictReader(old.open()))
        newrows=[r for r in rows if not(r['split']=='train' and r['image'] in excluded)]
        assert len(rows)-len(newrows)==46
        manifest=out/'clean_manifest.csv'
        with manifest.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(newrows)
        (out/'excluded_train_duplicates.json').write_text(json.dumps(sorted(excluded),indent=2))
        cfg['data'].update(manifest=str(manifest),image_size=512,dct_source_mode='fair_unified_jpeg',area_stratified_sampling=False,
                           tiny_stratified_sampling=False,positive_crop_probability=0.,jpeg_probability=0.,blur_probability=0.,color_jitter=0.)
        cfg['model'].update(freeze_encoder_bn=False, freeze_forensic_bn=False,
                            enable_effective_mechanism_chain=False,
                            enable_strict_serial_chain=False,
                            enable_balanced_residual_chain=True,
                            enable_dher_visibility=False,
                            enable_dher_calibrated_writeback=False)
        cfg['training'].update(init_checkpoint=None,epochs=30,batch_size=4,gradient_accumulation_steps=1,learning_rate=.001,
                               optimizer='SGD',momentum=.9,weight_decay=.0005,amp=True,ema=False)
        cfg['training']['parameter_groups']=[]
        cfg['training']['dct_sample_dropout']=0.2
        cfg['training']['qtr_aux_weight']=0.15
        cfg['training']['dher_aux_weight']=0.05
        cfg['fair_training']={'seed':3407,'crop':'native 512, grid8 aligned, pad grey, valid-only loss',
            'input':'JPEG original; nonJPEG -> quality100 subsampling0; RGB and DCT from identical encoded file',
            'initialization':'public_pretrained_only','padding':'train512; val next32; no resize','loss':'original objective + HCV proposal + QTR auxiliary + DHER support/refutation supervision, valid regions',
            'architecture':'normal dual-stream; balanced residual+DCT fusion; residual QTR and residual HCV; decoder; residual DHER',
            'dct_dropout':'sample-level p=0.2 during training only',
            'lr':'SGD 0.001 poly(power=.9) across all updates; all parameters eligible; no staged freezing'}
        e.dump(out/'config.yaml',cfg)
        train=e.read_manifest(manifest,cfg['data']['root'],'train');val=e.read_manifest(manifest,cfg['data']['root'],'val')
        assert len(train)==8685 and len(val)==2494
        ta={str(r.image) for r in train};assert not (ta&excluded)
        ds=FairDataset(train,out/'jpeg_cache',True);vd=FairDataset(val,out/'jpeg_cache',False)
        fake=[i for i,r in enumerate(train) if r.label];auth=[i for i,r in enumerate(train) if not r.label]
        assert len(fake)==3487 and len(auth)==5198
        plans=[]
        for ep in range(30):
            gen=torch.Generator().manual_seed(3407+ep)
            ids=fake+[auth[i] for i in torch.randperm(len(auth),generator=gen)[:len(fake)].tolist()]
            ids=[ids[i] for i in torch.randperm(len(ids),generator=gen).tolist()]
            plans.append([(idx,340700000+ep*100000+j) for j,idx in enumerate(ids)])
        torch.save(plans,out/'sample_plan.pt')
        summary.update(train_size=len(train),val_size=len(val),train_fake=len(fake),train_authentic=len(auth),
                       samples_per_epoch=len(plans[0]),optimizer_steps_per_epoch=math.ceil(len(plans[0])/4),
                       manifest_sha256=e.sha_file(manifest),sample_plan_sha256=e.sha_file(out/'sample_plan.pt'),
                       excluded_train_duplicates=46,source_config_only=str(SOURCE))
        save()
        seg=e24.ReTraceLoss.from_config(e24.clean_loss_config(cfg['loss'])).cuda()
        model=e.build_final_model(cfg['model']).cuda()
        contract = {
            'status': 'passed',
            'two_stream_external_contract': ['content', 'unified_forensic'],
            'unified_forensic_definition': 'residual features plus DCT adapter writes before QTR',
            'dct_direct_decoder_path': False,
            'decoder_forensic_input': 'HCV residual-verified V1-V3 plus balanced N4',
            'raw_forensic_decoder_bypass': False,
            'qtr_residual_refinement': True,
            'hcv_semantic_residual': 'V=Q+0.1*verification_correction',
            'qtr_reliability_gate': bool(model.qtr.use_reliability_gate),
            'hcv_mandatory_q_fusion': bool(model.hcv2.mandatory_q_fusion),
            'balanced_forensic_fusion': model.balanced_forensic_fusions is not None,
            'qtr_residual_mode': bool(model.qtr.residual_refinement),
            'hcv_residual_mode': bool(model.hcv2.residual_verification),
            'dher_residual_identity_form': 'D1_refined = D1_pre + support_write + conflict_write',
        }
        assert model.dct_branch is not None and model.dct_inject_adapters is not None
        assert contract['qtr_reliability_gate'] and contract['balanced_forensic_fusion']
        assert contract['qtr_residual_mode'] and contract['hcv_residual_mode']
        assert not contract['hcv_mandatory_q_fusion']
        e.dump(out/'architecture_contract.yaml', contract)
        pristine=e.copy_state(model)
        result=audit(model,ds,seg,cfg,True)
        e.dump(out/'preflight.yaml',result);summary['preflight']=result;save()
        model.load_state_dict(pristine,strict=True);del pristine
        e24.set_seed(3407);mode_train(model)
        opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=.0005)
        summary.update(status='training',history=[],best_F1=-1.);save()
        total_steps=sum(math.ceil(len(p)/4) for p in plans);global_step=0
        for epoch,plan in enumerate(plans,1):
            mode_train(model);started=time.time();total=0.;seen=0
            loader=DataLoader(ds,batch_size=4,sampler=plan,num_workers=2,pin_memory=True)
            summary.update(status='training',epoch=epoch);save()
            for step,b in enumerate(loader,1):
                lr=.001*(1-global_step/total_steps)**.9
                for g in opt.param_groups:g['lr']=lr
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    terms=objective(fwd(model,b,cfg,True),b,seg,cfg);loss=terms['total']
                if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite epoch={epoch} step={step}')
                loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                opt.step();global_step+=1;seen+=len(b['path']);total+=float(loss.detach())
                if step==1 or step%25==0 or step==len(loader):
                    elapsed=time.time()-started
                    p={'epoch':epoch,'epochs':30,'batch':step,'batches':len(loader),'global_step':global_step,'lr':lr,
                       'loss':total/step,'grad_norm':float(gn),'elapsed_minutes':elapsed/60,'eta_epoch_minutes':elapsed/step*(len(loader)-step)/60}
                    e.dump(out/'progress.yaml',p);print('[train]',json.dumps(p),flush=True)
            assert seen==len(plan)
            summary['status']='validating';save()
            result=evaluate(model,vd,cfg,out,f'epoch{epoch:02d}')
            score=result['metrics']['fake']['oracle']['f1']
            row={'epoch':epoch,'train_loss':total/len(loader),'selection_value':score,'lr':lr,'global_step':global_step,'validation':result}
            summary['history'].append(row)
            ck={'model':e.copy_state(model),'optimizer':opt.state_dict(),'config':cfg,'epoch':epoch,'global_step':global_step,
                'eval_variant':'raw','best_metric':score,'selection_threshold':result['pixel_localization_fake_only']['test_f1_oracle_diagnostic']['threshold'],'image_selection_threshold':result['image_level_all']['test_f1_oracle_diagnostic']['threshold'],
                'input_pipeline':'fair_training_data.FairDataset, original-size validation; never legacy resize evaluator'}
            e24.atomic_torch_save(ck,out/'last.pt')
            if score>summary['best_F1']:
                summary.update(best_F1=score,best_epoch=epoch,best_checkpoint=str(out/'best.pt'));e24.atomic_torch_save(ck,out/'best.pt')
            save();print('[epoch_done]',epoch,score,'best',summary['best_F1'],flush=True)
        summary.update(status='completed',stop_reason='fixed_30_epoch_complete_no_external_selection');save()
    except BaseException:
        summary.update(status='failed',stop_reason=traceback.format_exc());save();raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);main(parser.parse_args())
