"""E24 frozen validation: generated JPEG coefficients for non-JPEG inputs.

These coefficients describe a new quality100 encoding, NOT original history.
RGB and geometry remain identical; original JPEG inputs are untouched.
"""
import dataclasses, hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import run_e21_jpeg_history_branch as e

BASE=Path('/group4/zouwenxuan/iml_generalization')
CACHE=BASE/'outputs/retracenet_v2/oracle_overlap_20260911/paired'
OUT=BASE/'outputs/retracenet_v2/tiff_input_policy_audit_20260911'
OUT.mkdir(exist_ok=True)
(OUT/'generated_jpeg').mkdir(exist_ok=True)
summary={'status':'initializing','test_run':False,'training_run':False,
         'policy':'nonJPEG: original RGB + coefficients from newly generated JPEG quality100 subsampling0',
         'provenance':'generated coefficients; NOT original JPEG compression history',
         'checkpoint':str(BASE/'outputs/retracenet_v2/e24_dher_calibrated_writeback_20260910/best.pt')}
def save():e.dump(OUT/'summary.yaml',summary)
save()
try:
    torch.set_num_threads(4)
    state=torch.load(summary['checkpoint'],map_location='cpu');cfg=state['config']
    model=e.build_final_model(cfg['model']).cuda().eval()
    model.load_state_dict(state['model'],strict=True);model.disable_dct_evidence=False
    records=e.read_manifest(cfg['data']['manifest'],cfg['data']['root'],'val')
    assert len(records)==2494
    class Probe(e.ManipulationDataset):
        def _jpeg_info_from_original(self,r,size):
            if r.image.suffix.lower() in {'.jpg','.jpeg'}:
                return super()._jpeg_info_from_original(r,size)
            file=OUT/'generated_jpeg'/(hashlib.sha256(str(r.image).encode()).hexdigest()+'.jpg')
            if not file.exists():
                with Image.open(r.image) as im:im.convert('RGB').save(file,quality=100,subsampling=0)
            return super()._jpeg_info_from_original(dataclasses.replace(r,image=file),size)
    kwargs=dict(image_size=int(cfg['data'].get('image_size',640)),training=False,dct_forensic=True,dct_source_mode='original_jpeg')
    ds=Probe(records,**kwargs);original=e.ManipulationDataset(records,**kwargs)
    baseline=e.ValidationStats();probe=e.ValidationStats();start=time.time();checked=False
    summary['status']='evaluating';save()
    with torch.inference_mode(),open(OUT/'per_image.jsonl','w') as log:
        for i,r in enumerate(records):
            z=np.load(CACHE/f'{i:04d}.npz');y=z['gt'];old=z['on'];h,w=y.shape
            generated=r.image.suffix.lower() not in {'.jpg','.jpeg'}
            if generated:
                sample=ds[i]
                assert float(sample['dct_available'].item())==1
                if not checked:
                    orig=original[i]
                    assert torch.equal(sample['image'],orig['image'])
                    assert torch.equal(sample['mask'],orig['mask'])
                    b={k:orig[k][None] for k in ['image','dct','qtable','dct_available']}
                    out=e.forward(model,b)
                    replay=F.interpolate(out['mask_logits'].sigmoid(),size=(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
                    difference=float(np.max(np.abs(replay-old)))
                    assert difference<1e-5,difference
                    summary['sanity']={'original_RGB_identical':True,'GT_identical':True,'cached_baseline_max_diff':difference,'new_available':1,'old_available':float(orig['dct_available'].item())};checked=True;save()
                b={k:sample[k][None] for k in ['image','dct','qtable','dct_available']}
                out=e.forward(model,b)
                p=F.interpolate(out['mask_logits'].sigmoid(),size=(h,w),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            else:p=old
            baseline.add(old,y.astype(np.int16),r.label,r.image);probe.add(p,y.astype(np.int16),r.label,r.image)
            log.write(json.dumps({'path':str(r.image),'generated':generated,'mean_abs_probability_change':float(np.abs(p-old).mean())})+'\n')
            if i==0 or (i+1)%50==0:
                summary.update(completed=i+1,total=len(records),elapsed_minutes=(time.time()-start)/60);save()
                print('[progress]',i+1,len(records),round((time.time()-start)/60,2),flush=True)
    # Use the evaluator's existing public aggregation method.
    summary.update(status='completed',baseline=baseline.result(),probe=probe.result())
    save();print('COMPLETE',flush=True)
except BaseException:
    summary.update(status='failed',error=traceback.format_exc());save();raise
