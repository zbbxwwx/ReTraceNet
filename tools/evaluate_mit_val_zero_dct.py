"""Evaluate a MiT-content full V3 checkpoint on the test part of a JSON split."""
import argparse, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_imd_mit_full_v3
import run_cross_dataset_after_chain as cross
from retrace.data import Record

def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 ck=torch.load(a.checkpoint,map_location='cpu');model=cross.e.build_final_model(ck['config']['model']).cuda().eval();model.load_state_dict(ck['model'],strict=True)
 model.disable_dct_evidence=True
 rows=json.loads(Path(a.split).read_text())['val'];records=[Record(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') else None,split='val',dataset=a.dataset,label=int(r['label'])) for r in rows]
 pt=float(ck['selection_threshold']);it=float(ck.get('image_selection_threshold',.5))
 result=cross.evaluate_dataset(model,records,out/'jpeg_cache',pt,it,a.dataset,out,evaluation_mode='in_domain_locked',threshold_source=a.dataset+'_val')
 result['image_threshold_source']=a.dataset+'_val';result['checkpoint']=a.checkpoint;result['checkpoint_epoch']=int(ck['epoch']);cross.save_yaml(out/'result.yaml',result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--split',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);main(p.parse_args())
