"""Val-select a cross-seed probability ensemble, then evaluate deduplicated test once."""
import itertools, json, sys
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import eval_columbia_topk_ensemble as ev
import run_cross_dataset_after_chain as cross
from e15_original_metrics import ValidationStats
from v3_unified_metrics import AuthenticPixelFPR

RUN=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/columbia_mit_dedup_topk_20260921')
SPLIT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/v3_offline_master_20260912/indomain_Columbia/split_721.json')
SEEDS=(3407,2026,42)

def seed_dirs(seed,part):
    root=RUN/f'seed_{seed}'/'topk_ensemble_1024_768'
    summary=yaml.safe_load((root/'summary.yaml').read_text())
    return root,[root/f"pred_{part}_ep{int(x['epoch']):03d}" for x in summary['selected']]

def seed_prob(root,dirs,part,index):
    return np.mean([np.load(d/f'{index:04d}.npy').astype(np.float32) for d in dirs],axis=0)

def aggregate(records,roots_dirs,part,weights,threshold=None):
    stats=ValidationStats();locked=cross.LockedCounts(threshold) if threshold is not None else None;auth=AuthenticPixelFPR()
    target_root=roots_dirs[0][0]
    for i,r in enumerate(records):
        probs=[seed_prob(root,dirs,part,i) for root,dirs in roots_dirs]
        p=sum(float(w)*x for w,x in zip(weights,probs))
        t=np.load(target_root/f'target_{part}_{i:04d}.npy').astype(np.int16)
        stats.add(p,t,r.label,r.image)
        if threshold is not None:
            if r.label:locked.add(p,t)
            else:auth.add(p)
    diag=stats.result()
    if threshold is None:return diag
    f=locked.result();ar=auth.result(threshold)
    tp,fn=int(f['tp']),int(f['fn']);fp=int(f['fp'])+int(ar['false_positive_pixels']);tn=int(f['tn'])+int(ar['authentic_pixels'])-int(ar['false_positive_pixels'])
    return {'fake_only_pooled':{'test_f1_locked':f,'auc':diag['fake']['auc'],'test_f1_oracle_diagnostic':diag['fake']['oracle']},'authentic_pixel_fpr':ar,'all_images_pooled':{'test_f1_locked':{'threshold':threshold,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':2*tp/max(1,2*tp+fp+fn)},'auc':diag['all']['auc']}}

def weight_grid():
    # Coarse predeclared grid. Require at least two seeds to contribute.
    vals=(0.,.25,.5,.75,1.)
    for w in itertools.product(vals,repeat=3):
        if abs(sum(w)-1.)<1e-9 and sum(x>0 for x in w)>=2:yield w

def main():
    split=json.loads(SPLIT.read_text());val=ev.records(ev.dedupe(split['val']),'val');test=ev.records(ev.dedupe(split['test']),'test')
    val_rd=[seed_dirs(s,'val') for s in SEEDS];test_rd=[seed_dirs(s,'test') for s in SEEDS]
    trials=[]
    for w in weight_grid():
        d=aggregate(val,val_rd,'val',w);q=d['fake']['oracle'];trials.append({'weights':list(w),'f1':float(q['f1']),'auc':float(d['fake']['auc']),'threshold':float(q['threshold'])})
    best=max(trials,key=lambda x:(x['f1'],x['auc']))
    # This is the sole cross-seed test computation; all selection above used val only.
    result=aggregate(test,test_rd,'test',best['weights'],best['threshold'])
    result.update(dataset='Columbia',seeds=list(SEEDS),weights=best['weights'],threshold=float(best['threshold']),threshold_source='Columbia_dedup_val_cross_seed_probability_ensemble',val_selection=best,policy={'tile':1024,'stride':768,'full_max_side':1024},test_evaluations=1)
    (RUN/'cross_seed_ensemble_result.yaml').write_text(yaml.safe_dump(result,sort_keys=False))
    (RUN/'cross_seed_ensemble_val_trials.yaml').write_text(yaml.safe_dump({'trials':trials,'selected':best},sort_keys=False))
    print(yaml.safe_dump(result,sort_keys=False))

if __name__=='__main__':main()
