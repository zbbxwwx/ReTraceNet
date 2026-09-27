"""CASIA2-only OOD proxy: threshold frozen on clean validation, no external test."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import run_fair_training_v1 as fair
from run_cross_dataset_after_chain import LockedCounts
from takeover_views import PROXY_CONDITIONS, ProxyDataset
from takeover_recalibrate import FAIR, atomic_json, sha


@torch.inference_mode()
def evaluate(model, ds, threshold):
    stats = fair.e.ValidationStats()
    locked = LockedCounts(threshold)
    authentic = LockedCounts(threshold)
    per_image=[]
    for i,r in enumerate(ds.records):
        b=next(iter(DataLoader(ds,batch_size=1,sampler=[i],num_workers=0)))
        o=fair.fwd(model,b)
        h,w=map(int,b['valid_hw'][0].tolist())
        p=o['mask_logits'].float().sigmoid()[0,0,:h,:w].cpu().numpy()
        y=b['mask'][0,0,:h,:w].numpy()
        stats.add(p,y,r.label,r.image)
        if r.label: locked.add(p,y)
        else: authentic.add(p,y)
        single=LockedCounts(threshold);single.add(p,y)
        per_image.append({'path':str(r.image),'label':r.label,'locked':single.result()})
        if i==0 or (i+1)%200==0:print('[proxy]',i+1,len(ds),flush=True)
    return {'metrics':stats.result(),'fake_locked':locked.result(),'authentic_locked':authentic.result(),
            'per_image':per_image}


def main(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    ck=torch.load(args.checkpoint,map_location='cpu');cfg=ck['config']
    model=fair.e.build_final_model(cfg['model']).cuda().eval()
    model.load_state_dict(ck['model'],strict=True)
    threshold=float(ck['selection_threshold'])
    records=fair.e.read_manifest(args.manifest,cfg['data']['root'],'val')
    base=fair.FairDataset(records,out/'jpeg_cache',False)
    protocol={'checkpoint':args.checkpoint,'checkpoint_sha256':sha(args.checkpoint),
              'threshold':threshold,'conditions':PROXY_CONDITIONS,'test_run':False,
              'manifest_sha256':sha(args.manifest),'macro':'equal-condition mean of fake locked F1, clean excluded',
              'views_sha256':sha(Path(__file__).with_name('takeover_views.py'))}
    if (out/'protocol.json').exists():
        assert json.loads((out/'protocol.json').read_text())==protocol,'proxy protocol changed on resume'
    else:atomic_json(out/'protocol.json',protocol)
    table={}
    for name in PROXY_CONDITIONS+['dct_off']:
        path=out/(name+'.json')
        if path.exists(): result=json.loads(path.read_text())
        else:
            model.disable_dct_evidence=(name=='dct_off')
            ds=base if name=='dct_off' else ProxyDataset(base,name)
            print('[condition]',name,flush=True)
            result=evaluate(model,ds,threshold)
            atomic_json(path,result)
        table[name]={'fake_locked_F1':result['fake_locked']['f1'],
                     'fake_F1_best':result['metrics']['fake']['oracle']['f1'],
                     'auc':result['metrics']['fake']['auc'],
                     'authentic_FPR':result['authentic_locked']['fpr']}
        atomic_json(out/'progress.json',{'completed_conditions':list(table),'total_conditions':16})
    macro=sum(table[n]['fake_locked_F1'] for n in PROXY_CONDITIONS if n!='clean')/(len(PROXY_CONDITIONS)-1)
    result={'status':'completed','protocol':protocol,'conditions':table,'robustness_macro_F1':macro,
            'clean_fake_F1_best':table['clean']['fake_F1_best'],
            'dct_off_fake_F1_best':table['dct_off']['fake_F1_best']}
    atomic_json(out/'summary.json',result);fair.e.dump(out/'summary.yaml',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--manifest',default=str(FAIR/'clean_manifest.csv'));main(p.parse_args())
