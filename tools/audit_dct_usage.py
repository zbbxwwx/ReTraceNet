import sys,json,traceback
from pathlib import Path
import torch
import run_e21_jpeg_history_branch as e
OUT=Path(sys.argv[1]); OUT.mkdir(parents=True,exist_ok=True)
BASE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')
summary={'status':'running','test_run':False,'models':{}}
e.dump(OUT/'summary.yaml',summary)
try:
 for name,folder in [('E21','e21_jpeg_history_branch_20260910'),('E24','e24_dher_calibrated_writeback_20260910')]:
  ck=BASE/folder/'best.pt'; state=torch.load(ck,map_location='cpu'); cfg=state['config']
  model=e.build_final_model(cfg['model']).cuda().eval(); model.load_state_dict(state['model'],strict=True)
  records=e.read_manifest(cfg['data']['manifest'],cfg['data']['root'],'val')
  ds=e.ManipulationDataset(records,image_size=int(cfg['data'].get('image_size',640)),training=False,dct_forensic=True,dct_source_mode='original_jpeg')
  sanity=[]
  with torch.inference_mode():
   for i,r in enumerate(records):
    if r.mask is None: continue
    s=ds[i]
    if float(s['dct_available'].mean())<=0: continue
    b={k:s[k][None] for k in ['image','dct','qtable','dct_available']}
    model.disable_dct_evidence=False; a=e.forward(model,b)
    model.disable_dct_evidence=True; z=e.forward(model,b)
    assert float(a['dct_evidence']['disabled'])==0 and float(z['dct_evidence']['disabled'])==1
    d=(a['mask_logits'].float()-z['mask_logits'].float()).abs()
    sanity.append({'path':str(r.image),'availability':float(s['dct_available'].mean()),'mean_abs_logit_diff':float(d.mean()),'max_abs_logit_diff':float(d.max()),'disabled_on':0,'disabled_off':1})
    if len(sanity)==10: break
  assert len(sanity)==10
  dest=OUT/name; dest.mkdir(exist_ok=True); e.dump(dest/'sanity.yaml',sanity)
  print(name,'SANITY_PASS',json.dumps(sanity),flush=True)
  # Explicit argument is essential: evaluate resets the model flag on entry.
  full=e.evaluate(model,records,cfg,'with_dct',dest,ablate_dct=False)
  ab=e.evaluate(model,records,cfg,'without_dct',dest,ablate_dct=True)
  assert full['ablate_dct'] is False and ab['ablate_dct'] is True
  summary['models'][name]={'checkpoint':str(ck),'sanity':sanity,'with_dct':full,'without_dct':ab,'fake_oracle_delta':full['metrics']['fake']['oracle']['f1']-ab['metrics']['fake']['oracle']['f1']}
  e.dump(OUT/'summary.yaml',summary)
  print(name,'DONE',summary['models'][name]['fake_oracle_delta'],flush=True)
  del model,state; torch.cuda.empty_cache()
 summary['status']='completed'; e.dump(OUT/'summary.yaml',summary)
except BaseException:
 summary['status']='failed'; summary['error']=traceback.format_exc(); e.dump(OUT/'summary.yaml',summary); raise
