"""IMD coverage ep29 cross-domain evaluation with IMD-val locked threshold."""
import argparse, copy, json, sys, traceback
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_imd_mit_full_v3 as mit
import eval_imd_mit_context_policies as ctx
import run_cross_dataset_after_chain as cross
from retrace.data import Record

@torch.inference_mode()
def predict_large_1024_768(model,record,cache):
    scores,target=ctx.predict_large(model,record,cache,1024,768)
    return scores,target,0.5
mit.stable.cross.predict_large_record=predict_large_1024_768

SPLITS={
 'CASIA1':'/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_CASIA1/split_721.json',
 'Columbia':'/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json',
 'NIST16':'/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_NIST16/split_721.json',
 'CASIA2':'/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/casia2_split.json',
 'FantasticReality':'/group4/zouwenxuan/iml_generalization/datasets/FantasticReality_v1_localization/split_721_random_stratified.json',
}

def add_all_pooled(result):
    fake=result['pixel_localization_fake_only']['test_f1_locked'];auth=result['authentic_pixel_fpr']
    tp,fn=int(fake['tp']),int(fake['fn']);fp=int(fake['fp'])+int(auth['false_positive_pixels']);tn=int(fake['tn'])+int(auth['authentic_pixels'])-int(auth['false_positive_pixels'])
    p=tp/max(1,tp+fp);r=tp/max(1,tp+fn);f=2*p*r/max(1e-12,p+r);diag=result.get('diagnostic_only',{}).get('all',{})
    result['pixel_localization_all_images_pooled']={'test_f1_locked':{'threshold':result['pixel_threshold'],'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':p,'recall':r,'f1':f,'iou':tp/max(1,tp+fp+fn),'fpr':fp/max(1,fp+tn)},'auc':diag.get('auc'),'test_f1_oracle_diagnostic':diag.get('oracle'),'population':'all valid pixels from fake and authentic target test images','threshold_source':'IMD2020_val'}

def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.append and (out/'summary.yaml').exists():
        import yaml
        summary=yaml.safe_load((out/'summary.yaml').read_text()) or {}
        summary.setdefault('datasets',{});summary['status']='loading_append'
    else:
        summary={'status':'loading','checkpoint':a.checkpoint,'threshold_source':'IMD2020_val','evaluation_mode':'cross_domain_locked','inference_policy':{'full_max_side':1024,'tile':1024,'stride':768},'datasets':{}}
    cross.save_yaml(out/'summary.yaml',summary)
    try:
        ck=torch.load(a.checkpoint,map_location='cpu');model=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval();model.load_state_dict(ck['model'],strict=True)
        thr=float(ck['selection_threshold']);ith=float(ck.get('image_selection_threshold',0.0));summary.update(checkpoint_epoch=int(ck['epoch']),pixel_threshold=thr,image_threshold=ith,status='running');cross.save_yaml(out/'summary.yaml',summary)
        for name in a.datasets:
            rows=json.loads(Path(SPLITS[name]).read_text())['test'];recs=[Record(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') else None,split='test',dataset=name,label=int(r['label'])) for r in rows]
            result=cross.evaluate_dataset(model,recs,out/'jpeg_cache'/name,thr,ith,name,out,evaluation_mode='cross_domain_locked',threshold_source='IMD2020_val')
            result['checkpoint']=a.checkpoint;result['checkpoint_epoch']=int(ck['epoch']);result['actual_large_image_inference']={'full_max_side':1024,'tile':1024,'stride':768};add_all_pooled(result)
            cross.save_yaml(out/f'{name}.yaml',result);summary['datasets'][name]=result;cross.save_yaml(out/'summary.yaml',summary)
        summary['status']='completed';cross.save_yaml(out/'summary.yaml',summary)
    except BaseException:
        summary.update(status='failed',stop_reason=traceback.format_exc());cross.save_yaml(out/'summary.yaml',summary);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--datasets',nargs='+',default=list(SPLITS));p.add_argument('--append',action='store_true');main(p.parse_args())
