"""MiT-full V3 control with ImageNet normalization on Content input only."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
import run_imd_mit_full_v3 as mit
import eval_imd_mit_context_policies as context_eval


class ImageNetNormalizedMiT(mit.MiTB2ContentCompat):
    """Exact MiT content encoder with the pretrained model's input contract."""
    def __init__(self):
        super().__init__()
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return super().forward((x - self.imagenet_mean) / self.imagenet_std)


original_build = mit.original_build
def build_normalized_mit_full(cfg):
    model = original_build(cfg)
    model.content_encoder = ImageNetNormalizedMiT()
    return model


mit.stable.v3.e.build_final_model = build_normalized_mit_full


@torch.inference_mode()
def predict_large_1024_768(model, record, cache):
    scores, target = context_eval.predict_large(model, record, cache, 1024, 768)
    # Image classification is not trained in this experiment; retain a fixed
    # scalar solely to satisfy the shared evaluator interface.
    return scores, target, 0.5


# Checkpoint selection and final locked test must use the adopted IMD policy.
mit.stable.cross.predict_large_record = predict_large_1024_768


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--template", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    a = p.parse_args()
    for key, value in dict(
        positive_crop_prob=0., positive_crop_retries=20, hcv_aux_scale=1.,
        qtr_aux_scale=.15, dher_aux_scale=.05, forensic_mode="full",
        hcv_semantic_bypass=False, verified_residual_decoder=False,
        decoder_ds_targeted=False, final_mask_only=False,
        splicing_region_affinity=False, region_affinity_weight=.1,
        early_gate_epoch=0, early_gate_min_f1=.5,
    ).items():
        setattr(a, key, value)
    mit.stable.main(a)
