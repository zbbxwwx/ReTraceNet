"""Frozen external-four evaluator for a selected checkpoint and val threshold."""
import argparse, copy, json, traceback
from pathlib import Path
import torch
import run_cross_dataset_after_chain as cross
from takeover_recalibrate import BASE, atomic_json, sha

NAMES=['CASIA1','Columbia','NIST16','IMD2020']

def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'decision.json').exists():return
    state={'status':'running','checkpoint':a.checkpoint,'test_run':True};atomic_json(out/'state.json',state)
    try:
        ck=torch.load(a.checkpoint,map_location='cpu');cfg=copy.deepcopy(ck['config'])
        threshold=float(ck['selection_threshold']);image_threshold=float(ck['image_selection_threshold']);model=cross.e.build_final_model(cfg['model']).cuda().eval()
        model.load_state_dict(ck['model'],strict=True);rows={}
        for name in NAMES:
            dest=out/(name+'.yaml')
            if dest.exists():r=cross.load_yaml(dest)
            else:
                records=cross.parse_index(BASE/'paper_compare/mvss_five_indices'/(name+'.txt'),name)
                r=cross.evaluate_dataset(model,records,out/'jpeg_cache'/name,threshold,image_threshold,name,out)
            f=r['diagnostic_only']['fake'];rows[name]={'pixel_localization_fake_only':r['pixel_localization_fake_only'],'authentic_pixel_fpr':r['authentic_pixel_fpr'],'image_level_all':r['image_level_all'],'legacy_fake_diagnostic':{'fixed_0p5':f['fixed'],'auc':f['auc'],'oracle':f['oracle']}}
            atomic_json(out/'partial.json',rows)
        summary={'status':'completed','checkpoint':a.checkpoint,'checkpoint_sha256':sha(a.checkpoint),
                 'threshold':threshold,'threshold_source':'CASIA2_val','image_threshold':image_threshold,'image_threshold_source':'CASIA2_val','evaluation_mode':'cross_domain_locked','test_run':True,'datasets':rows}
        atomic_json(out/'decision.json',{'status':'completed','route':'GENERALIZATION_TRACK_DONE','evidence':rows,
                    'reason':'Frozen generalized checkpoint evaluated once','rejected_routes':['external_test_tuning']})
        atomic_json(out/'summary.json',summary);cross.save_yaml(out/'final_summary.yaml',summary)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());atomic_json(out/'state.json',state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);main(p.parse_args())
