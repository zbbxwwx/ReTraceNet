import copy,sys,traceback
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import run_v3_stable_bn_indomain as stable
import run_fair_training_v1 as fair
from fair_training_data import FairDataset
import run_e4_new_dct_encoder_q90 as e4
PARENT=e4.PARENT; MANIFEST=e4.MANIFEST; OUT=e4.OUT/'epoch02_ablation'; CK=e4.OUT/'epoch02_frozen_for_ablation.pt'
def dump(x):stable.dump(OUT/'summary.yaml',x)
def norm(ps):
 s=sum(float(p.grad.detach().float().square().sum()) for p in ps if p.grad is not None);return s**.5
try:
 OUT.mkdir(parents=True,exist_ok=True); summary={'status':'preparing','checkpoint':str(CK),'test_read':False};dump(summary)
 parent=torch.load(PARENT,map_location='cpu');cfg=copy.deepcopy(parent['config']); ck=torch.load(CK,map_location='cpu')
 model=fair.e.build_final_model(cfg['model']).cuda();model.load_state_dict(parent['model'],strict=True)
 channels=tuple(cfg['model'].get('forensic_channels',(64,128,256,512)));model.dct_branch=e4.RGBDCTEncoder(channels).cuda();model.load_state_dict(ck['model'],strict=True)
 va=fair.e.read_manifest(MANIFEST,cfg['data']['root'],'val');base=e4.FairDataset(va,OUT/'cache',False);clean=e4.RGBDCTDataset(base);q90=e4.RGBDCTDataset(e4.Fixed(base,90))
 results={}
 for mode,off in [('full',False),('dct_off',True)]:
  model.disable_dct_evidence=off;model.eval();results[mode]={}
  for name,ds in [('clean',clean),('q90',q90)]:
   z=fair.evaluate(model,ds,cfg,OUT,f'{mode}_{name}');m=z['metrics']['fake'];results[mode][name]={'f1':float(m['oracle']['f1']),'precision':float(m['oracle']['precision']),'recall':float(m['oracle']['recall']),'auc':float(m['auc']),'threshold':float(m['oracle']['threshold'])}
 # Gradient and mode audit on the same validation source; no optimizer step.
 model.disable_dct_evidence=False
 for p in model.parameters():p.requires_grad_(False)
 for p in model.dct_branch.parameters():p.requires_grad_(True)
 for p in model.balanced_forensic_fusions.parameters():p.requires_grad_(True)
 e4.set_mode(model)
 batch=next(iter(DataLoader(e4.CleanQ90(e4.FairDataset(fair.e.read_manifest(MANIFEST,cfg['data']['root'],'train'),OUT/'train_cache',True)),batch_size=4,shuffle=False,num_workers=0)))
 batch={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in batch.items()};seg=stable.v3.e24.ReTraceLoss.from_config(stable.v3.e24.clean_loss_config(cfg['loss'])).cuda()
 with torch.autocast('cuda',dtype=torch.bfloat16):out=stable.v3.fwd(model,batch,cfg,False);loss=stable.v3.objective(out,batch,seg,cfg)['total']
 loss.backward()
 dp=list(model.dct_branch.parameters());fp=list(model.balanced_forensic_fusions.parameters());frozen=[p for n,p in model.named_parameters() if not (n.startswith('dct_branch.') or n.startswith('balanced_forensic_fusions.'))]
 frozen_bn=[(n,m.training) for n,m in model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and not (n.startswith('dct_branch.') or n.startswith('balanced_forensic_fusions.'))]
 audit={'loss':float(loss),'dct_grad_norm':norm(dp),'fusion_grad_norm':norm(fp),'frozen_grad_tensor_count':sum(p.grad is not None for p in frozen),'frozen_bn_count':len(frozen_bn),'frozen_bn_in_train_mode':sum(x[1] for x in frozen_bn),'all_frozen_bn_eval':not any(x[1] for x in frozen_bn),'trainable_parameter_count':sum(p.numel() for p in dp+fp),'frozen_parameter_count':sum(p.numel() for p in frozen)}
 summary.update(status='completed',results=results,audit=audit,interpretation={'q90_dct_off_delta_f1':results['dct_off']['q90']['f1']-results['full']['q90']['f1'],'clean_dct_off_delta_f1':results['dct_off']['clean']['f1']-results['full']['clean']['f1']});dump(summary);print(summary,flush=True)
except BaseException:
 summary.update(status='failed',traceback=traceback.format_exc());dump(summary);raise
