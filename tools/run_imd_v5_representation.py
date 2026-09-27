"""V5 representation reboot: matched R0 and R1/R2/R3/R4 on fixed IMD protocol."""
import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
TRU=Path('/group4/zouwenxuan/iml_generalization/repos/TruFor/TruFor_train_test')
sys.path.insert(0,str(TRU))
import torch
from torch import nn
import torch.nn.functional as F
import run_v3_stable_bn_indomain as stable
from retracenet.final_modules import QualityTraceQualification,HCV2
from retracenet.decoder import ProgressiveDecoder
from retrace.blocks import LayerNorm2d

MIT_WEIGHTS=TRU/'pretrained_models/segformers/mit_b2.pth'
NP_WEIGHTS=TRU/'pretrained_models/noiseprint++/noiseprint++.th'

class MiTB2RGB(nn.Module):
 def __init__(self):
  super().__init__()
  from lib.models.cmx.encoders.dual_segformer import mit_b2
  self.net=mit_b2();self.net.init_weights_onepath(str(MIT_WEIGHTS))
 def forward(self,x):
  n=self.net;b=x.shape[0];out=[]
  for pe,blocks,norm in [(n.patch_embed1,n.block1,n.norm1),(n.patch_embed2,n.block2,n.norm2),(n.patch_embed3,n.block3,n.norm3),(n.patch_embed4,n.block4,n.norm4)]:
   x,h,w=pe(x)
   for blk in blocks:x=blk(x,h,w)
   x=norm(x).reshape(b,h,w,-1).permute(0,3,1,2).contiguous();out.append(x)
  return out

class NoiseprintPyramid(nn.Module):
 def __init__(self,channels=(64,128,256,512)):
  super().__init__()
  from lib.models.DnCNN import make_net
  self.dncnn=make_net(3,kernels=[3]*17,features=[64]*16+[1],bns=[False]+[True]*15+[False],acts=['relu']*16+['linear'],dilats=[1]*17,bn_momentum=.1,padding=1)
  state=torch.load(NP_WEIGHTS,map_location='cpu')['network'];self.dncnn.load_state_dict(state)
  for p in self.dncnn.parameters():p.requires_grad_(False)
  mods=[];inc=1
  for i,c in enumerate(channels):
   stride=4 if i==0 else 2
   mods.append(nn.Sequential(nn.Conv2d(inc,c,3,stride=stride,padding=1,bias=False),LayerNorm2d(c),nn.GELU(),nn.Conv2d(c,c,3,padding=1,bias=False),LayerNorm2d(c),nn.GELU()));inc=c
  self.stages=nn.ModuleList(mods)
 def train(self,mode=True):
  super().train(mode);self.dncnn.eval();return self
 def forward(self,x):
  with torch.no_grad():z=self.dncnn(x)
  out=[]
  for s in self.stages:z=s(z);out.append(z)
  return out

_build=stable.v3.e.build_final_model
def build_v5(cfg):
 v=os.environ['V5_VARIANT'];m=_build(cfg)
 cc=(256,512,1024,2048);dc=(64,128,256,512)
 if v in ('R2','R3','R4'):
  m.content_encoder=MiTB2RGB();cc=(64,128,320,512)
 if v in ('R1','R3','R4'):
  m.forensic_stream=NoiseprintPyramid(dc);m.dct_branch=None;m.dct_inject_adapters=None;m.balanced_forensic_fusions=None
 # Rebuild the entire downstream contract for the selected representation.
 m.decoder=ProgressiveDecoder(content_channels=cc,detail_channels=dc,decoder_channels=128)
 m.mask_head=nn.Conv2d(128,1,1);m.log_var_head=nn.Conv2d(128,1,1);m.aux_heads=nn.ModuleList(nn.Conv2d(128,1,1) for _ in range(3));m.image_head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,1))
 m.dher=None;m.dher_visibility=None;m.dher_calibrated_writeback=None;m.splicing_region_affinity=None;m.correspondence=None;m.p2_preserve=None;m.mask_refine=None;m.region_refine=None;m.content_context=None
 if v=='R4':m.qtr=None;m.hcv2=None;m.qtr_aux_heads=nn.ModuleList();m.enable_balanced_residual_chain=False;m.enable_strict_serial_chain=False
 else:
  m.qtr=QualityTraceQualification(content_channels=cc[:3],detail_channels=dc[:3],use_reliability_gate=True,residual_refinement=True)
  m.hcv2=HCV2(content_channels=cc[:3],detail_channels=dc[:3],residual_verification=True)
  m.qtr_aux_heads=nn.ModuleList(nn.Conv2d(c,1,1) for c in dc[:3]);m.enable_balanced_residual_chain=True;m.enable_strict_serial_chain=False
 return m

stable.v3.e.build_final_model=build_v5
def freeze_bn_v5(model):
 n=0
 for branch in (getattr(model,'content_encoder',None),getattr(model,'dct_branch',None)):
  if branch is None:continue
  for module in branch.modules():
   if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval();n+=1
 # Frozen Noiseprint++ must always use its learned statistics.
 fs=getattr(model,'forensic_stream',None)
 if isinstance(fs,NoiseprintPyramid):fs.dncnn.eval()
 return n
stable.freeze_bn=freeze_bn_v5

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=('R0','R1','R2','R3','R4'),required=True);p.add_argument('--dataset',required=True);p.add_argument('--split',required=True);p.add_argument('--template',required=True);p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--seed',type=int,default=3407);p.add_argument('--resume',action='store_true');p.add_argument('--skip-test',action='store_true')
 a=p.parse_args();os.environ['V5_VARIANT']=a.variant
 # Fill the stable runner's unchanged options; final-mask-only is mandatory.
 # Representation-screening gate: these runs locate promising families rather
 # than squeeze the last fraction of a point from a formal training recipe.
 for k,v in dict(positive_crop_prob=0.,positive_crop_retries=20,hcv_aux_scale=1.,qtr_aux_scale=.15,dher_aux_scale=.05,forensic_mode='full',hcv_semantic_bypass=False,verified_residual_decoder=False,decoder_ds_targeted=False,final_mask_only=True,splicing_region_affinity=False,region_affinity_weight=.1,early_gate_epoch=25,early_gate_min_f1=.48).items():setattr(a,k,v)
 stable.main(a)
