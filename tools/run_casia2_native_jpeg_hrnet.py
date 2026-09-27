"""Train MiT-B2 + Native-JPEG Forensic HRNet-W32 + QTR/HCV/DHER."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

import run_imd_mit_full_v3 as mit
from native_jpeg_fair_data import NativeJPEGFairDataset


stable = mit.stable
FORENSIC_PRETRAIN = None
_mit_builder = stable.v3.e.build_final_model


def build_native_jpeg_hrnet(config: dict):
    cfg = copy.deepcopy(config)
    cfg["forensic_stream_type"] = "native_jpeg_hrnet_w32"
    cfg["enable_dct_parallel"] = False
    cfg["dct_pretrained_path"] = None
    cfg["enable_dct_gate"] = False
    cfg["enable_dct_evidence_head"] = False
    cfg["native_jpeg_layout"] = "quantized_y_dct_64_block_grid"
    config.clear()
    config.update(copy.deepcopy(cfg))
    model = _mit_builder(cfg)
    if FORENSIC_PRETRAIN:
        checkpoint = torch.load(FORENSIC_PRETRAIN, map_location="cpu")
        state = checkpoint.get("encoder", checkpoint)
        result = model.forensic_stream.load_state_dict(state, strict=True)
        print({"forensic_pretrain": str(FORENSIC_PRETRAIN), "keys": len(state), "missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}, flush=True)
    if getattr(model, "dct_branch", None) is not None:
        raise RuntimeError("legacy CAT-Net DCT branch was unexpectedly constructed")
    return model


stable.v3.e.build_final_model = build_native_jpeg_hrnet
stable.FairDataset = NativeJPEGFairDataset
stable.cross.fair.FairDataset = NativeJPEGFairDataset
if hasattr(stable.v3, "fair"):
    stable.v3.fair.FairDataset = NativeJPEGFairDataset


def smoke_test(template: Path) -> None:
    checkpoint = torch.load(template, map_location="cpu")
    cfg = copy.deepcopy(checkpoint["config"])
    model = build_native_jpeg_hrnet(cfg["model"]).cuda().train()
    image = torch.rand(2, 3, 256, 256, device="cuda")
    native = torch.randint(-20, 21, (2, 64, 32, 32), device="cuda").float()
    qtable = torch.randint(1, 100, (2, 1, 8, 8), device="cuda").float()
    available = torch.ones(2, 1, device="cuda")
    output = model(image, dct=native, qtable=qtable, dct_available=available)
    loss = output["mask_logits"].float().square().mean()
    loss.backward()

    forensic_grad = sum(
        float(parameter.grad.detach().float().norm())
        for name, parameter in model.named_parameters()
        if "forensic_stream" in name and parameter.grad is not None
    )
    if not torch.isfinite(loss) or forensic_grad <= 0.0:
        raise RuntimeError("Native-JPEG HRNet smoke backward failed")
    forbidden = [name for name, _ in model.named_parameters() if "dct_branch" in name]
    if forbidden:
        raise RuntimeError(f"legacy DCT parameters found: {forbidden[:5]}")
    print(
        {
            "status": "smoke_passed",
            "mask_shape": tuple(output["mask_logits"].shape),
            "loss": float(loss.detach()),
            "forensic_grad_norm_sum": forensic_grad,
            "legacy_dct_parameter_count": 0,
            "forensic_stream": type(model.forensic_stream).__name__,
        },
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CASIA2")
    parser.add_argument("--split", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--forensic-pretrain", default="")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    global FORENSIC_PRETRAIN
    args = parse_args()
    FORENSIC_PRETRAIN = args.forensic_pretrain or None
    if args.smoke_only:
        smoke_test(Path(args.template))
        return
    for key, value in {
        "positive_crop_prob": 0.0,
        "positive_crop_retries": 20,
        "hcv_aux_scale": 1.0,
        "qtr_aux_scale": 0.15,
        "dher_aux_scale": 0.05,
        "forensic_mode": "full",
        "hcv_semantic_bypass": False,
        "verified_residual_decoder": False,
        "decoder_ds_targeted": False,
        "final_mask_only": False,
        "splicing_region_affinity": False,
        "region_affinity_weight": 0.1,
        "early_gate_epoch": 0,
        "early_gate_min_f1": 0.5,
    }.items():
        setattr(args, key, value)
    stable.main(args)


if __name__ == "__main__":
    main()

