"""Evaluate one Stable-BN in-domain checkpoint once with its val-locked thresholds."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from retrace.data import Record
import run_balanced_residual_v3_full30 as v3
import run_cross_dataset_after_chain as cross

def recs(rows, split, dataset):
    return [Record(image=Path(r['image']), mask=Path(r['mask']) if r.get('mask') else None,
                   split=split, dataset=dataset, label=int(r['label'])) for r in rows]

def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.checkpoint,map_location='cpu')
    model=v3.e.build_final_model(ck['config']['model']).cuda()
    model.load_state_dict(ck['model'],strict=True);model.eval()
    split=json.loads(Path(a.split).read_text());test=recs(split['test'],'test',a.dataset)
    pixel_t=float(ck['selection_threshold']);image_t=float(ck.get('image_selection_threshold',.5))
    result=cross.evaluate_dataset(model,test,out/'jpeg_cache',pixel_t,image_t,a.dataset,out,
        evaluation_mode='in_domain_locked_epoch22_probe',threshold_source=a.dataset+'_val_epoch22')
    result['checkpoint']=str(Path(a.checkpoint));result['checkpoint_epoch']=int(ck['epoch'])
    result['pixel_threshold_locked']=pixel_t;result['image_threshold_locked']=image_t
    v3.e.dump(out/'result.yaml',result)
    print(json.dumps({'checkpoint_epoch':ck['epoch'],'pixel_threshold':pixel_t,
      'fake_locked':result['pixel_localization_fake_only']['test_f1_locked'],
      'fake_auc':result['pixel_localization_fake_only']['auc'],
      'fake_oracle':result['pixel_localization_fake_only']['test_f1_oracle_diagnostic'],
      'authentic_fpr':result['authentic_pixel_fpr'],'image_level':result['image_level_all']},ensure_ascii=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--split',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);main(p.parse_args())
