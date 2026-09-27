"""Epoch76 baseline on the exact synthetic-shift CASIA2 val used by the probe."""
import copy,sys,yaml,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_imd_mit_full_v3
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
from run_casia2_mit_jpeg_only_probe import ShiftVal,PARENT,MANIFEST,OUT
def main():
 ck=torch.load(PARENT,map_location='cpu');cfg=copy.deepcopy(ck['config'])
 recs=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'val')
 ds=ShiftVal(FairDataset(recs,OUT/'jpeg_cache_val',False))
 model=fair.e.build_final_model(cfg['model']).cuda().eval();model.load_state_dict(ck['model'],strict=True)
 r=fair.evaluate(model,ds,cfg,OUT,'parent_epoch76_synthetic_shift_val');q=r['metrics']['fake']['oracle']
 o={'checkpoint_epoch':76,'synthetic_shift_val':{'f1':float(q['f1']),'precision':float(q['precision']),'recall':float(q['recall']),'auc':float(r['metrics']['fake']['auc']),'threshold':float(q['threshold'])}}
 stable.dump(OUT/'parent_shift_summary.yaml',o);print(yaml.safe_dump(o,sort_keys=False))
if __name__=='__main__':main()
