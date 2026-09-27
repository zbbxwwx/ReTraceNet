"""Deterministic duplicate-grouped stratified 7:2:1 in-domain V3 training."""
import argparse,copy,hashlib,json,math,os,random,time,traceback,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from torch.utils.data import DataLoader
from retrace.data import Record
import run_balanced_residual_v3_full30 as v3
import run_cross_dataset_after_chain as cross
from fair_training_data import FairDataset

def dump(p,x): v3.e.dump(Path(p),x)
def split_rows(rows,seed):
 # Conservative source isolation: join exact RGB duplicates, exact perceptual-hash matches,
 # and all rows sharing a dataset-provided donor/source token.
 parent=list(range(len(rows)))
 def find(x):
  while parent[x]!=x: parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b: parent[b]=a
 seen={}
 for i,r in enumerate(rows):
  keys=['rgb:'+r['rgb_sha256'],'dhash:'+r['dhash']]+['source:'+str(t) for t in r.get('source_tokens_candidate',[])]
  for key in keys:
   if key in seen: union(i,seen[key])
   else: seen[key]=i
 groups={}
 for i,r in enumerate(rows): groups.setdefault(find(i),[]).append(r)
 strata={}
 for gid,items in groups.items():
  fake=[r for r in items if int(r['label'])]
  rep=fake[0] if fake else items[0]
  key=(int(bool(fake)),rep.get('area','unknown'),rep.get('manipulation_type','unknown'))
  signature='|'.join(sorted(r['rgb_sha256'] for r in items))
  strata.setdefault(key,[]).append((signature,items))
 out={k:[] for k in ['train','val','test']}
 for key,gs in strata.items():
  gs.sort(key=lambda z:hashlib.sha256(f'{seed}:{z[0]}'.encode()).hexdigest())
  n=len(gs);nt=int(round(.7*n));nv=int(round(.2*n))
  if n>=3:nt=max(1,min(nt,n-2));nv=max(1,min(nv,n-nt-1))
  elif n==2:nt,nv=1,1
  else:nt,nv=1,0
  for j,(_,items) in enumerate(gs):out['train' if j<nt else 'val' if j<nt+nv else 'test'].extend(items)
 return out

def records(rows,split,name):
 return [Record(image=Path(r['image']),mask=Path(r['mask']) if r.get('mask') else None,split=split,dataset=name,label=int(r['label'])) for r in rows]
def main(a):
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);summary={'status':'preparing','dataset':a.dataset,'split':'duplicate-grouped stratified 7:2:1','test_run':False}
 dump(out/'summary.yaml',summary)
 try:
  rows=[json.loads(x) for x in Path(a.inventory).read_text().splitlines() if x.strip()];sp=split_rows(rows,a.seed)
  keys=lambda r:set(['rgb:'+r['rgb_sha256'],'dhash:'+r['dhash']]+['source:'+str(t) for t in r.get('source_tokens_candidate',[])])
  isolation={k:set().union(*(keys(r) for r in v)) if v else set() for k,v in sp.items()}
  assert not(isolation['train']&isolation['val'] or isolation['train']&isolation['test'] or isolation['val']&isolation['test'])
  isolation_audit={'status':'passed','grouping':['exact_rgb_sha256','exact_dhash','source_tokens_candidate'],'counts':{k:len(v) for k,v in sp.items()},'note':'source tokens are inventory-derived candidates; provenance unavailable is recorded rather than claimed as verified'};dump(out/'source_isolation_audit.yaml',isolation_audit)
  split_doc={k:[{'image':r['image'],'mask':r.get('mask'),'label':r['label'],'rgb_sha256':r['rgb_sha256']} for r in v] for k,v in sp.items()};(out/'split_721.json').write_text(json.dumps(split_doc,indent=2))
  ck=torch.load(a.template,map_location='cpu');cfg=copy.deepcopy(ck['config']);cfg['training']['epochs']=a.epochs;cfg['training']['dct_sample_dropout']=.2
  tr,va,te=(records(sp[k],k,a.dataset) for k in ['train','val','test']);ds=FairDataset(tr,out/'jpeg_cache',True);vd=FairDataset(va,out/'jpeg_cache',False);td=FairDataset(te,out/'jpeg_cache',False)
  fake=[i for i,r in enumerate(tr) if r.label];auth=[i for i,r in enumerate(tr) if not r.label]; assert fake and auth and va and te
  seg=v3.e24.ReTraceLoss.from_config(v3.e24.clean_loss_config(cfg['loss'])).cuda();model=v3.e.build_final_model(cfg['model']).cuda();v3.mode_train(model)
  opt=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=.0005);total_steps=a.epochs*math.ceil(2*min(len(fake),len(auth))/4);gs=0;history=[];best=-1
  summary.update(status='training',counts={k:len(v) for k,v in sp.items()},label_counts={k:{str(z):sum(int(r['label'])==z for r in v) for z in [0,1]} for k,v in sp.items()});dump(out/'summary.yaml',summary)
  for ep in range(1,a.epochs+1):
   gen=torch.Generator().manual_seed(a.seed+ep);nf=min(len(fake),len(auth));ids=[fake[i] for i in torch.randperm(len(fake),generator=gen)[:nf]]+[auth[i] for i in torch.randperm(len(auth),generator=gen)[:nf]];ids=[ids[i] for i in torch.randperm(len(ids),generator=gen)];plan=[(idx,a.seed*1000000+ep*100000+j) for j,idx in enumerate(ids)]
   loader=DataLoader(ds,batch_size=4,sampler=plan,num_workers=2,pin_memory=True);v3.mode_train(model);tot=0
   for st,b in enumerate(loader,1):
    lr=.001*(1-gs/max(total_steps,1))**.9
    for g in opt.param_groups:g['lr']=lr
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16): terms=v3.objective(v3.fwd(model,b,cfg,True),b,seg,cfg);loss=terms['total']
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();gs+=1;tot+=float(loss)
    if st==1 or st%25==0: dump(out/'progress.yaml',{'dataset':a.dataset,'epoch':ep,'epochs':a.epochs,'batch':st,'batches':len(loader),'loss':tot/st});print('[train]',a.dataset,ep,st,len(loader),flush=True)
   vr=cross.evaluate_dataset(model,va,out/'jpeg_cache_val',None,None,a.dataset,out/f'val_epoch{ep:02d}',evaluation_mode='validation_select',threshold_source=a.dataset+'_val_current');score=vr['pixel_localization_fake_only']['test_f1_oracle_diagnostic']['f1'];row={'epoch':ep,'val_fake_F1':score,'threshold':vr['pixel_localization_fake_only']['test_f1_oracle_diagnostic']['threshold'],'image_threshold':vr['image_level_all']['test_f1_oracle_diagnostic']['threshold']};history.append(row)
   state={'model':v3.e.copy_state(model),'optimizer':opt.state_dict(),'config':cfg,'epoch':ep,'best_metric':score,'selection_threshold':row['threshold'],'image_selection_threshold':row['image_threshold']};v3.e24.atomic_torch_save(state,out/'last.pt')
   if score>best:best=score;best_ep=ep;v3.e24.atomic_torch_save(state,out/'best.pt')
   summary.update(history=history,best_F1=best,best_epoch=best_ep,best_checkpoint=str(out/'best.pt'));dump(out/'summary.yaml',summary)
  bestck=torch.load(out/'best.pt',map_location='cpu');model.load_state_dict(bestck['model'],strict=True);test=cross.evaluate_dataset(model,te,out/'jpeg_cache_test',float(bestck['selection_threshold']),float(bestck['image_selection_threshold']),a.dataset,out/'test_once',evaluation_mode='in_domain_locked',threshold_source=a.dataset+'_val');test['evaluation_mode']='in_domain_locked';test['threshold_source']=a.dataset+'_val';test['image_threshold_source']=a.dataset+'_val';dump(out/'test_once'/f'{a.dataset}.yaml',test)
  summary.update(status='completed',test_run=True,test_checkpoint_epoch=best_ep,test_locked_threshold=float(bestck['selection_threshold']),test=test,stop_reason='fixed_budget_complete_test_once');dump(out/'summary.yaml',summary)
 except BaseException:
  summary.update(status='failed',stop_reason=traceback.format_exc());dump(out/'summary.yaml',summary);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--inventory',required=True);p.add_argument('--template',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=30);p.add_argument('--seed',type=int,default=3407);main(p.parse_args())
