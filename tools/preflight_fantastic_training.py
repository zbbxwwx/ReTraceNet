"""One-batch data/model/loss/backward preflight for FantasticReality."""
import argparse, copy, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run_imd_mit_full_v3
import run_v3_stable_bn_indomain as stable

def main(a):
 split=json.loads(Path(a.split).read_text());records=stable.records(split['train'],'train','FantasticReality')
 fake=[i for i,r in enumerate(records) if r.label][:2];real=[i for i,r in enumerate(records) if not r.label][:2]
 ds=stable.FairDataset(records,Path(a.out).parent/'preflight_jpeg_cache',True,positive_crop_prob=0.,positive_crop_retries=20)
 batch=next(iter(DataLoader(ds,batch_size=4,sampler=[(i,340700000+j) for j,i in enumerate(fake+real)],num_workers=0)))
 ck=torch.load(a.template,map_location='cpu');cfg=copy.deepcopy(ck['config']);cfg['training']['dct_sample_dropout']=.2
 model=stable.v3.e.build_final_model(cfg['model']).cuda();stable.train_mode(model)
 seg=stable.v3.e24.ReTraceLoss.from_config(stable.v3.e24.clean_loss_config(cfg['loss'])).cuda()
 with torch.autocast('cuda',dtype=torch.bfloat16):
  out=stable.v3.fwd(model,batch,cfg,True);terms=stable.v3.objective(out,batch,seg,cfg);loss=terms['total']
 if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
 loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True))
 labels=batch['label'].tolist();fg=[float(batch['mask'][i].float().mean()) for i in range(4)]
 report={'status':'passed','labels':labels,'batch_image_shape':list(batch['image'].shape),'batch_mask_shape':list(batch['mask'].shape),'dct_shape':list(batch['dct'].shape),'dct_available':batch['dct_available'].reshape(-1).tolist(),'mask_foreground_ratio':fg,'loss':float(loss.detach()),'grad_norm':gn,'mask_logits_shape':list(out['mask_logits'].shape)}
 Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--template',required=True);p.add_argument('--out',required=True);main(p.parse_args())
