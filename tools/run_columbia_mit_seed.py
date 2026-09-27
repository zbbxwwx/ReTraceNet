"""IMD full V3 control: replace only ResNet content features by MiT-B2.

MiT stages are projected to the original ResNet channel contract, leaving the
Residual+DCT fusion, QTR, HCV, DHER, decoder, heads, and composite objective
unchanged.
"""
import argparse, copy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
import run_v3_columbia_seed_ensemble as stable

TRU = Path('/group4/zouwenxuan/iml_generalization/repos/TruFor/TruFor_train_test')
sys.path.insert(0, str(TRU))
MIT_WEIGHTS = TRU / 'pretrained_models/segformers/mit_b2.pth'


class MiTB2ContentCompat(nn.Module):
    """MiT-B2 RGB encoder with projections to the original V3 contract."""
    def __init__(self):
        super().__init__()
        from lib.models.cmx.encoders.dual_segformer import mit_b2
        self.net = mit_b2()
        self.net.init_weights_onepath(str(MIT_WEIGHTS))
        self.proj = nn.ModuleList([
            nn.Conv2d(64, 256, 1, bias=False),
            nn.Conv2d(128, 512, 1, bias=False),
            nn.Conv2d(320, 1024, 1, bias=False),
            nn.Conv2d(512, 2048, 1, bias=False),
        ])

    def forward(self, x):
        n, b, feats = self.net, x.shape[0], []
        stages = ((n.patch_embed1,n.block1,n.norm1),(n.patch_embed2,n.block2,n.norm2),
                  (n.patch_embed3,n.block3,n.norm3),(n.patch_embed4,n.block4,n.norm4))
        for i, (pe, blocks, norm) in enumerate(stages):
            x, h, w = pe(x)
            for blk in blocks:
                x = blk(x, h, w)
            x = norm(x).reshape(b,h,w,-1).permute(0,3,1,2).contiguous()
            feats.append(self.proj[i](x))
        return feats


original_build = stable.v3.e.build_final_model
def build_mit_full(cfg):
    model = original_build(cfg)
    model.content_encoder = MiTB2ContentCompat()
    return model

stable.v3.e.build_final_model = build_mit_full

def freeze_bn(model):
    count = 0
    for branch in (getattr(model,'content_encoder',None), getattr(model,'dct_branch',None)):
        if branch is None:
            continue
        for module in branch.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval(); count += 1
    return count
stable.freeze_bn = freeze_bn

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True);p.add_argument('--split',required=True)
    p.add_argument('--template',required=True);p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--learning-rate',type=float,default=.001);p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--resume',action='store_true');p.add_argument('--skip-test',action='store_true')
    a=p.parse_args()
    for k,v in dict(positive_crop_prob=0.,positive_crop_retries=20,hcv_aux_scale=1.,
        qtr_aux_scale=.15,dher_aux_scale=.05,forensic_mode='full',hcv_semantic_bypass=False,
        verified_residual_decoder=False,decoder_ds_targeted=False,final_mask_only=False,
        splicing_region_affinity=False,region_affinity_weight=.1,early_gate_epoch=0,
        early_gate_min_f1=.5).items(): setattr(a,k,v)
    stable.main(a)
