"""Fixed-checkpoint CAT dependency interventions, validation only."""
import argparse, importlib.util, json, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import run_e21_jpeg_history_branch as e

BASE=Path('/group4/zouwenxuan/iml_generalization')
PAIRED=BASE/'outputs/retracenet_v2/oracle_overlap_20260911'
OUT=BASE/'outputs/retracenet_v2/cat_dependency_audit_20260911'
OUT.mkdir(exist_ok=True)
summary={'status':'initializing','test_run':False,'training_run':False,
 'interpretation':'Inference dependency interventions, not retrained single-stream ablations; zeroing may be out-of-distribution.',
 'modes':['baseline','dct_zero','rgb_zero','dct_spatial_mean','rgb_spatial_mean'],
 'threshold':0.5,'intervention_location':'stage4 RGB output / dc_stage4 DCT output, before stage5 fusion'}
def save(): e.dump(OUT/'summary.yaml',summary)
save()
try:
 torch.set_num_threads(4)
 rows=[json.loads(x) for x in (PAIRED/'B_per_image.jsonl').read_text().splitlines()]
 assert len(rows)==2494
 helper=BASE/'outputs/domain_casia2_eval/catnet_unified1247_best_20260831_015845/export_catnet_unified1247.py'
 spec=importlib.util.spec_from_file_location('cat_reference',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 ck=BASE/'outputs/catnet/splicing_dataset/CAT_full/best.pth.tar'
 m.update_config(m.config,argparse.Namespace(cfg=str(BASE/'repos/CAT-Net/experiments/CAT_full.yaml'),opts=['TEST.MODEL_FILE',str(ck),'TEST.FLIP_TEST','False']))
 class SharedGT(m.ManifestDataset):
  def get_tamp(self,index):
   z=np.load(PAIRED/'paired'/f'{index:04d}.npz')
   jpg=m.make_jpeg_for_catnet(self.rows[index]['image'],self.jpg_cache_dir)
   return self._create_tensor(jpg,z['gt'].astype(np.uint8))
 ds=SharedGT([{'image':r['path'],'mask':'','label':int(not r['bucket'].startswith('unknown'))} for r in rows],PAIRED/'cat_jpeg_cache')
 net=getattr(m.models,m.config.MODEL.NAME).get_seg_model(m.config)
 net.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=True)
 model=m.FullModel(net,m.CrossEntropy(ignore_label=m.config.TRAIN.IGNORE_LABEL,weight=ds.class_weights).cuda()).cuda().eval()
 control={'mode':'baseline','hits':{},'audit':{}}
 def hook_for(branch):
  def hook(module,inputs,outputs):
   control['hits'][branch]=control['hits'].get(branch,0)+1
   mode=control['mode']
   if mode.startswith(branch+'_'):
    altered=[torch.zeros_like(t) if mode.endswith('zero') else t.mean((-2,-1),keepdim=True).expand_as(t) for t in outputs]
    if mode.endswith('zero'): assert all(torch.count_nonzero(t)==0 for t in altered)
    else: assert all(torch.allclose(t.mean((-2,-1)),u.mean((-2,-1)),atol=1e-5) for t,u in zip(outputs,altered))
    if mode not in control['audit']:
     control['audit'][mode]={'shapes':[list(t.shape) for t in outputs],
      'before_rms':[float(t.square().mean().sqrt()) for t in outputs],
      'after_rms':[float(t.square().mean().sqrt()) for t in altered]}
    return altered
   return outputs
  return hook
 net.stage4.register_forward_hook(hook_for('rgb'));net.dc_stage4.register_forward_hook(hook_for('dct'))
 modes=summary['modes'];stats={mode:{fmt:e.ValidationStats() for fmt in ['all_formats','jpeg','nonjpeg']} for mode in modes}
 started=time.time();summary.update(status='running',checkpoint=str(ck),strict_load=True)
 save()
 with torch.inference_mode(),open(OUT/'per_image.jsonl','w') as log:
  for i,r in enumerate(rows):
   z=np.load(PAIRED/'paired'/f'{i:04d}.npz');y=z['gt'];h,w=y.shape
   im,lab,qt=ds[i];im=im[None].cuda();lab=lab[None].long().cuda();qt=qt[None].cuda() if torch.is_tensor(qt) else qt
   assert np.array_equal(lab[0,:h,:w].cpu().numpy()>0,y)
   fmt='jpeg' if Path(r['path']).suffix.lower() in {'.jpg','.jpeg'} else 'nonjpeg'
   row={'index':i,'path':r['path'],'bucket':r['bucket'],'format':fmt,'modes':{}}
   for mode in modes:
    control['mode']=mode;control['hits']={}
    _,prediction=model(im,lab,qt)
    assert control['hits']=={'rgb':1,'dct':1},control['hits']
    p=F.interpolate(prediction.float().softmax(1)[:,1:2],size=lab.shape[-2:],mode='bilinear',align_corners=False)[0,0].cpu().numpy()[:h,:w]
    if mode=='baseline':
     baseline=p.copy();c=p>=.5
     counts=[int((c&y).sum()),int((c&~y).sum()),int((~c&y).sum())]
     assert counts==r['cat_counts'],(i,counts,r['cat_counts'])
    row['modes'][mode]={'mean_abs_probability_diff':float(np.abs(p-baseline).mean()),'max_abs_probability_diff':float(np.abs(p-baseline).max())}
    label=int(not r['bucket'].startswith('unknown'))
    for group in ['all_formats',fmt]:stats[mode][group].add(p,y.astype(np.int16),label,Path(r['path']))
   log.write(json.dumps(row)+'\n');log.flush()
   if i==0 or (i+1)%25==0:
    elapsed=time.time()-started
    summary.update(completed=i+1,total=len(rows),elapsed_minutes=elapsed/60,eta_minutes=elapsed/(i+1)*(len(rows)-i-1)/60,hook_audit=control['audit'],baseline_counts_match=True)
    save();print('[progress]',i+1,len(rows),'minutes',round(elapsed/60,2),'ETA',round(summary['eta_minutes'],2),flush=True)
 summary.update(status='completed',completed=len(rows),metrics={mode:{fmt:s.result() for fmt,s in groups.items()} for mode,groups in stats.items()})
 save();print('COMPLETE',flush=True)
except BaseException:
 summary.update(status='failed',error=traceback.format_exc());save();raise
