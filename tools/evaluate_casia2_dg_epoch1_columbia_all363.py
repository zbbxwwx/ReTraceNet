"""Evaluate CASIA2 DG epoch1 on the complete Columbia 363-image set."""
import copy, sys, traceback
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_imd_mit_full_v3 as mit
import eval_imd_mit_context_policies as ctx
import run_cross_dataset_after_chain as cross
from retrace.data import Record

CKPT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_mit_dg_paired_ep76_ft10_v2_20260920/best.pt')
FAKE_INDEX=Path('/group4/zouwenxuan/iml_generalization/paper_compare/mvss_five_indices/Columbia.txt')
AUTH_DIR=Path('/group4/zouwenxuan/iml_generalization/data/Columbia/AU')
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/casia2_mit_dg_epoch1_columbia_all363_1024_768_20260920')
THRESHOLD=0.09

@torch.inference_mode()
def predict_large_1024_768(model, record, cache):
    scores,target=ctx.predict_large(model,record,cache,1024,768)
    return scores,target,0.5
mit.stable.cross.predict_large_record=predict_large_1024_768

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    status={'status':'loading','checkpoint':str(CKPT),'threshold_source':'CASIA2_val','threshold':THRESHOLD,'evaluation_mode':'cross_domain_locked','inference_policy':{'full_max_side':1024,'tile':1024,'stride':768}}
    cross.save_yaml(OUT/'status.yaml',status)
    try:
        fake=cross.parse_index(FAKE_INDEX,'Columbia')
        auth=[Record(image=p,mask=None,split='test',dataset='Columbia',label=0) for p in sorted(AUTH_DIR.iterdir()) if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}]
        recs=fake+auth
        assert (len(fake),len(auth),len(recs))==(180,183,363)
        assert len({str(r.image.resolve()) for r in recs})==363,'duplicate image paths'
        for r in recs:
            assert r.image.exists(),r.image
            if r.label==1: assert r.mask is not None and r.mask.exists(),r.mask
        ck=torch.load(CKPT,map_location='cpu')
        assert int(ck['epoch'])==1,ck['epoch']
        model=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval()
        model.load_state_dict(ck['model'],strict=True)
        result=cross.evaluate_dataset(model,recs,OUT/'jpeg_cache',THRESHOLD,float(ck.get('image_selection_threshold',0.0)),'Columbia_all363',OUT,evaluation_mode='cross_domain_locked',threshold_source='CASIA2_val')
        result.update(checkpoint=str(CKPT),checkpoint_epoch=1,dataset_counts={'all':363,'fake':180,'authentic':183},actual_large_image_inference={'full_max_side':1024,'tile':1024,'stride':768})
        cross.save_yaml(OUT/'result.yaml',result)
        f=result['pixel_localization_fake_only']['test_f1_locked']
        compact={'status':'completed','checkpoint_epoch':1,'threshold_source':'CASIA2_val','threshold':THRESHOLD,'inference':'1024/768','counts':{'all':363,'fake':180,'authentic':183},'fake_only_pooled':{'precision':f['precision'],'recall':f['recall'],'f1':f['f1'],'tp':f['tp'],'fp':f['fp'],'fn':f['fn'],'tn':f['tn']},'pixel_auc':result['pixel_localization_fake_only']['auc'],'authentic_pixel_fpr':result['authentic_pixel_fpr'],'target_oracle_f1_diagnostic_only':result['pixel_localization_fake_only']['test_f1_oracle_diagnostic']}
        cross.save_yaml(OUT/'summary.yaml',compact)
        print(compact,flush=True)
    except BaseException:
        status.update(status='failed',stop_reason=traceback.format_exc())
        cross.save_yaml(OUT/'status.yaml',status)
        raise
if __name__=='__main__': main()

