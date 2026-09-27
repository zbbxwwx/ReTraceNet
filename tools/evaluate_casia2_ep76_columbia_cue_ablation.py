"""Frozen cue interventions for CASIA2 epoch76 on Columbia ALL 363."""
import copy, sys, traceback, yaml
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_imd_mit_full_v3 as mit
import eval_imd_mit_context_policies as ctx
import run_cross_dataset_after_chain as cross
from retrace.data import Record

CKPT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/CASIA2_mit_full100/best.pt')
FAKE_INDEX=Path('/group4/zouwenxuan/iml_generalization/paper_compare/mvss_five_indices/Columbia.txt')
AUTH_DIR=Path('/group4/zouwenxuan/iml_generalization/data/Columbia/AU')
FULL_RESULT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_ep76_columbia_all363_1024_768_20260920/result.yaml')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_ep76_columbia_all363_cue_ablation_20260920')
THRESHOLD=0.38

@torch.inference_mode()
def predict_large_1024_768(model, record, cache):
    scores,target=ctx.predict_large(model,record,cache,1024,768)
    return scores,target,0.5
mit.stable.cross.predict_large_record=predict_large_1024_768

def zero_tree(x):
    if torch.is_tensor(x): return torch.zeros_like(x)
    if isinstance(x,tuple): return tuple(zero_tree(v) for v in x)
    if isinstance(x,list): return [zero_tree(v) for v in x]
    if isinstance(x,dict): return {k:zero_tree(v) for k,v in x.items()}
    raise TypeError(type(x))

def compact(result):
    f=result['pixel_localization_fake_only']['test_f1_locked']
    return {'precision':float(f['precision']),'recall':float(f['recall']),'f1':float(f['f1']),
            'pixel_auc':float(result['pixel_localization_fake_only']['auc']),
            'authentic_pixel_fpr':float(result['authentic_pixel_fpr']['fpr']),
            'oracle_f1_diagnostic':float(result['pixel_localization_fake_only']['test_f1_oracle_diagnostic']['f1']),
            'oracle_threshold_diagnostic':float(result['pixel_localization_fake_only']['test_f1_oracle_diagnostic']['threshold'])}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    summary={'status':'loading','checkpoint_epoch':76,'threshold':THRESHOLD,'threshold_source':'CASIA2_val','inference':'1024/768','counts':{'all':363,'fake':180,'authentic':183},'conditions':{}}
    cross.save_yaml(OUT/'summary.yaml',summary)
    try:
        fake=cross.parse_index(FAKE_INDEX,'Columbia')
        auth=[Record(image=p,mask=None,split='test',dataset='Columbia',label=0) for p in sorted(AUTH_DIR.iterdir()) if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}]
        recs=fake+auth
        assert (len(fake),len(auth),len(recs))==(180,183,363)
        full=yaml.safe_load(FULL_RESULT.read_text())
        summary['conditions']['full']=compact(full)
        ck=torch.load(CKPT,map_location='cpu'); assert int(ck['epoch'])==76
        model=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval(); model.load_state_dict(ck['model'],strict=True)
        for name in ('dct_off','residual_off'):
            model.disable_dct_evidence=(name=='dct_off'); handle=None
            if name=='residual_off':
                handle=model.forensic_stream.register_forward_hook(lambda m,a,o:zero_tree(o))
            try:
                result=cross.evaluate_dataset(model,recs,OUT/f'jpeg_cache_{name}',THRESHOLD,float(ck.get('image_selection_threshold',0.0)),f'Columbia_all363_{name}',OUT/name,evaluation_mode='cross_domain_locked_cue_intervention',threshold_source='CASIA2_val')
                result.update(condition=name,checkpoint_epoch=76,intervention_not_retrained=True,actual_large_image_inference={'full_max_side':1024,'tile':1024,'stride':768})
                cross.save_yaml(OUT/f'{name}.yaml',result)
                summary['conditions'][name]=compact(result)
                cross.save_yaml(OUT/'summary.yaml',summary)
            finally:
                if handle is not None: handle.remove()
                model.disable_dct_evidence=False
        summary['status']='completed'; cross.save_yaml(OUT/'summary.yaml',summary)
        print(yaml.safe_dump(summary,sort_keys=False),flush=True)
    except BaseException:
        summary.update(status='failed',stop_reason=traceback.format_exc()); cross.save_yaml(OUT/'summary.yaml',summary); raise
if __name__=='__main__': main()
