"""Evaluate the MiT-content full V3 checkpoint with a locked source-val threshold."""
import argparse, copy, sys
from pathlib import Path
import torch, yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_imd_mit_full_v3  # patches the final-model builder for MiT content
import run_cross_dataset_after_chain as cross
from takeover_recalibrate import BASE, atomic_json, sha

def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 ck=torch.load(a.checkpoint,map_location='cpu');cfg=copy.deepcopy(ck['config'])
 threshold=float(ck['selection_threshold']);ith=float(ck['image_selection_threshold'])
 model=cross.e.build_final_model(cfg['model']).cuda().eval();model.load_state_dict(ck['model'],strict=True)
 rows={}
 for name in a.datasets:
  rec=cross.parse_index(BASE/'paper_compare/mvss_five_indices'/(name+'.txt'),name)
  r=cross.evaluate_dataset(model,rec,out/'jpeg_cache'/name,threshold,ith,name,out,evaluation_mode='cross_domain_locked',threshold_source=a.threshold_source)
  r['image_threshold_source']=a.threshold_source
  cross.save_yaml(out/(name+'.yaml'),r)
  rows[name]={'pixel_localization_fake_only':r['pixel_localization_fake_only'],'authentic_pixel_fpr':r['authentic_pixel_fpr'],'image_level_all':r['image_level_all']}
 result={'status':'completed','checkpoint':a.checkpoint,'checkpoint_sha256':sha(a.checkpoint),'threshold':threshold,'threshold_source':a.threshold_source,'image_threshold':ith,'evaluation_mode':'cross_domain_locked','datasets':rows}
 cross.save_yaml(out/'final_summary.yaml',result);atomic_json(out/'summary.json',result)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--threshold-source',required=True);p.add_argument('--datasets',nargs='+',default=['CASIA1','Columbia','IMD2020']);main(p.parse_args())
