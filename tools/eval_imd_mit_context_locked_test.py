"""Evaluate the adopted 1024/768 policy on IMD test at its val-locked threshold."""
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import run_imd_mit_full_v3
import run_cross_dataset_after_chain as cross
import eval_imd_mit_context_policies as ctx
from e15_original_metrics import ValidationStats
from v3_unified_metrics import AuthenticPixelFPR

@torch.inference_mode()
def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    summary=cross.load_yaml(Path(a.val_summary)); threshold=float(summary['policies']['tile1024_stride768']['primary']['threshold'])
    ck=torch.load(a.checkpoint,map_location='cpu');model=cross.e.build_final_model(copy.deepcopy(ck['config']['model'])).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    rows=json.loads(Path(a.split).read_text())['test']; recs=ctx.records(rows,'test','IMD2020')
    ds=cross.fair.FairDataset(recs,out/'jpeg_cache',False);stats=ValidationStats();locked=cross.LockedCounts(threshold);auth=AuthenticPixelFPR();sliding=0
    for i,r in enumerate(recs):
        with Image.open(r.image) as im:w,h=im.size
        if max(h,w)>1024:scores,target=ctx.predict_large(model,r,out/'jpeg_cache',1024,768);sliding+=1
        else:
            b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)));vh,vw=map(int,b['valid_hw'][0].tolist());o=cross.fair.fwd(model,b);scores=o['mask_logits'].float().sigmoid()[0,0,:vh,:vw].cpu().numpy();target=b['mask'][0,0,:vh,:vw].numpy().astype(np.int16)
        if scores.shape!=target.shape:scores=torch.nn.functional.interpolate(torch.from_numpy(scores)[None,None].float(),size=target.shape,mode='bilinear',align_corners=False)[0,0].numpy()
        stats.add(scores,target,r.label,r.image)
        if r.label:locked.add(scores,target)
        else:auth.add(scores)
        if i==0 or (i+1)%25==0:print('[locked-test]',i+1,len(recs),flush=True)
    diag=stats.result();result={'dataset':'IMD2020_test','checkpoint':a.checkpoint,'checkpoint_epoch':int(ck['epoch']),'threshold_source':'IMD2020_val_tile1024_stride768','threshold':threshold,'policy':{'full_max_side':1024,'tile':1024,'stride':768},'num_images':len(recs),'sliding_count':sliding,'fake_only_pooled':{'test_f1_locked':locked.result(),'auc':diag['fake']['auc'],'test_f1_oracle_diagnostic':diag['fake']['oracle']},'authentic_pixel_fpr':auth.result(threshold)}
    cross.save_yaml(out/'result.yaml',result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',required=True);p.add_argument('--val-summary',required=True);p.add_argument('--out',required=True);main(p.parse_args())
