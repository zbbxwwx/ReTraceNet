"""Frozen IMD2020 val cue ablation. Never reads test."""
import json,sys,torch,yaml
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from retrace.data import Record
import run_balanced_residual_v3_full30 as v3
import run_v3_stable_bn_indomain as stable

BASE=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2')
RUN=BASE/'imd2020_stable_bn_v1_full50_20260913'
SPLIT=BASE/'v3_offline_master_20260912/indomain_IMD2020/split_721.json'
OUT=BASE/'imd2020_epoch49_cue_ablation_20260916';OUT.mkdir(parents=True,exist_ok=True)

def zero_tree(x):
 if torch.is_tensor(x):return torch.zeros_like(x)
 if isinstance(x,tuple):return tuple(zero_tree(v) for v in x)
 if isinstance(x,list):return [zero_tree(v) for v in x]
 if isinstance(x,dict):return {k:zero_tree(v) for k,v in x.items()}
 raise TypeError(type(x))

def main():
 split=json.loads(SPLIT.read_text());val=stable.records(split['val'],'val','IMD2020')
 ck=torch.load(RUN/'best.pt',map_location='cpu');cfg=ck['config']
 model=v3.e.build_final_model(cfg['model']).cuda().eval();model.load_state_dict(ck['model'],strict=True)
 results={}
 for name in ('full','dct_off','residual_off','content_off'):
  model.disable_dct_evidence=(name=='dct_off');handle=None
  if name=='residual_off':handle=model.forensic_stream.register_forward_hook(lambda m,a,o:zero_tree(o))
  if name=='content_off':handle=model.content_encoder.register_forward_hook(lambda m,a,o:zero_tree(o))
  try:
   r=stable.evaluate(model,val,OUT/f'jpeg_cache_{name}',None,None,'IMD2020',OUT/name,evaluation_mode='validation_ablation_oracle_diagnostic',threshold_source='IMD2020_val_oracle_diagnostic')
   q=r['pixel_localization_fake_only']['test_f1_oracle_diagnostic'];results[name]={'f1':float(q['f1']),'precision':float(q['precision']),'recall':float(q['recall']),'threshold':float(q['threshold']),'auc':float(r['pixel_localization_fake_only']['auc'])}
  finally:
   if handle is not None:handle.remove()
 model.disable_dct_evidence=False
 out={'status':'completed','checkpoint':str(RUN/'best.pt'),'checkpoint_epoch':int(ck['epoch']),'test_read':False,'protocol':'same IMD2020 val evaluator; fake-only pooled oracle diagnostic','results':results}
 (OUT/'summary.yaml').write_text(yaml.safe_dump(out,sort_keys=False));print(yaml.safe_dump(out,sort_keys=False))
if __name__=='__main__':main()
