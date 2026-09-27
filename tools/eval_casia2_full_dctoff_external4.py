"""CASIA2 epoch76: matched Full vs DCT-off on four external datasets."""
import copy, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import run_imd_mit_full_v3  # noqa
import run_cross_dataset_after_chain as cross
from takeover_recalibrate import BASE

CK=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_ep76_external4_full_dctoff_1024_20260921')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ck=torch.load(CK,map_location='cpu'); cfg=copy.deepcopy(ck['config'])
    thr=float(ck.get('selection_threshold',0.38)); ith=float(ck.get('image_selection_threshold',thr))
    model=cross.e.build_final_model(cfg['model']).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    summary={'status':'running','checkpoint':str(CK),'threshold':thr,'threshold_source':'CASIA2_val','inference':'1024/768','datasets':{}}
    cross.save_yaml(OUT/'summary.yaml',summary)
    for name in ['CASIA1','Columbia','IMD2020','NIST16']:
        rec=cross.parse_index(BASE/'paper_compare/mvss_five_indices'/(name+'.txt'),name)
        summary['datasets'][name]={}
        for mode,off in [('full',False),('dct_off',True)]:
            model.disable_dct_evidence=off
            r=cross.evaluate_dataset(model,rec,OUT/'jpeg_cache'/name,thr,ith,name,OUT,
                                     evaluation_mode='cross_domain_locked',threshold_source='CASIA2_val')
            cross.save_yaml(OUT/f'{name}__{mode}.yaml',r)
            summary['datasets'][name][mode]={'pixel_localization_fake_only':r['pixel_localization_fake_only'],'authentic_pixel_fpr':r['authentic_pixel_fpr'],'image_level_all':r['image_level_all']}
            cross.save_yaml(OUT/'summary.yaml',summary)
    model.disable_dct_evidence=False;summary['status']='completed';cross.save_yaml(OUT/'summary.yaml',summary)
if __name__=='__main__':main()
