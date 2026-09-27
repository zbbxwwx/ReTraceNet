"""Re-select threshold on a dataset val split at 1024/768, then evaluate test."""
import argparse, copy, json, sys, traceback
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'tools'))
import run_cross_dataset_after_chain as cross
ORIGINAL_BUILD_FINAL_MODEL = cross.e.build_final_model
import eval_imd_mit_context_policies as ctx
from e15_original_metrics import ValidationStats
from v3_unified_metrics import AuthenticPixelFPR
from retrace.data import Record

TILE=1024; STRIDE=768; FULL_MAX=1024
def records(rows,split,dataset):
    return [Record(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') else None,
                   split=split,dataset=dataset,label=int(r['label'])) for r in rows]

@torch.inference_mode()
def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    summary={'status':'loading','dataset':a.dataset,'checkpoint':a.checkpoint,'split':a.split,
             'policy':{'full_max_side':FULL_MAX,'tile':TILE,'stride':STRIDE},'test_run':False}
    cross.save_yaml(out/'summary.yaml',summary)
    try:
        ck=torch.load(a.checkpoint,map_location='cpu'); cfg=copy.deepcopy(ck['config'])
        is_mit=any(str(k).startswith('content_encoder.net.') for k in ck['model'])
        builder=cross.e.build_final_model if is_mit else ORIGINAL_BUILD_FINAL_MODEL
        model=builder(cfg['model']).cuda().eval(); model.load_state_dict(ck['model'],strict=True)
        summary['content_backbone']='MiT' if is_mit else 'ResNet';cross.save_yaml(out/'summary.yaml',summary)
        split=json.loads(Path(a.split).read_text()); val=records(split['val'],'val',a.dataset)
        vr=ctx.evaluate_policy(model,val,out/'jpeg_cache',TILE,STRIDE,out,'tile1024_stride768',FULL_MAX)
        threshold=float(vr['primary']['threshold']); summary.update(status='evaluating_test',val=vr,locked_threshold=threshold,
          threshold_source=a.dataset+'_val_tile1024_stride768');cross.save_yaml(out/'summary.yaml',summary)
        recs=records(split['test'],'test',a.dataset);ds=cross.fair.FairDataset(recs,out/'jpeg_cache',False)
        stats=ValidationStats();locked=cross.LockedCounts(threshold);auth=AuthenticPixelFPR();sliding=0
        for i,r in enumerate(recs):
            with Image.open(r.image) as im:w,h=im.size
            if max(h,w)>FULL_MAX:scores,target=ctx.predict_large(model,r,out/'jpeg_cache',TILE,STRIDE);sliding+=1
            else:
                b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)));vh,vw=map(int,b['valid_hw'][0].tolist())
                o=cross.fair.fwd(model,b);scores=o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy();target=b['mask'][0,0,:vh,:vw].numpy().astype(np.int16)
            if scores.shape!=target.shape:scores=torch.nn.functional.interpolate(torch.from_numpy(scores)[None,None].float(),size=target.shape,mode='bilinear',align_corners=False)[0,0].numpy()
            stats.add(scores,target,r.label,r.image)
            if r.label:locked.add(scores,target)
            else:auth.add(scores)
            if i==0 or (i+1)%25==0:print('[1024-locked]',a.dataset,i+1,len(recs),flush=True)
        diag=stats.result(); result={'dataset':a.dataset,'checkpoint':a.checkpoint,'checkpoint_epoch':int(ck['epoch']),
          'threshold_source':summary['threshold_source'],'threshold':threshold,'policy':summary['policy'],'num_images':len(recs),'sliding_count':sliding,
          'fake_only_pooled':{'test_f1_locked':locked.result(),'auc':diag['fake']['auc'],'test_f1_oracle_diagnostic':diag['fake']['oracle']},
          'authentic_pixel_fpr':auth.result(threshold),'diagnostic_all_pixels':diag['all']}
        fake_locked=result['fake_only_pooled']['test_f1_locked']; ar=result['authentic_pixel_fpr']; tp,fn=int(fake_locked['tp']),int(fake_locked['fn']); fp=int(fake_locked['fp'])+int(ar['false_positive_pixels']); tn=int(fake_locked['tn'])+int(ar['authentic_pixels'])-int(ar['false_positive_pixels']); result['all_images_pooled']={'test_f1_locked':{'threshold':threshold,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn),'iou':tp/max(1,tp+fp+fn),'fpr':fp/max(1,fp+tn)},'auc':diag['all']['auc'],'population':'all valid pixels from fake and authentic test images'}
        cross.save_yaml(out/'result.yaml',result);summary.update(status='completed',test_run=True,test=result);cross.save_yaml(out/'summary.yaml',summary)
    except BaseException:
        summary.update(status='failed',stop_reason=traceback.format_exc());cross.save_yaml(out/'summary.yaml',summary);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--split',required=True);p.add_argument('--out',required=True);main(p.parse_args())

