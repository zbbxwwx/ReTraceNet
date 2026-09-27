from pathlib import Path
import torch
import yaml

from retrace.data import build_loader
from retrace.model import build_model
from retrace.utils import load_checkpoint, load_config

ROOT = Path("/group4/zouwenxuan/iml_generalization")
CFG = ROOT / "repos/ReTraceNet/configs/casia2_retrace_resnet50_poscrop70_stage3_dhel_residual_lr5e5.yaml"
CKPT = ROOT / "outputs/retracenet/resnet50_poscrop70_stage3_dhel_residual_lr5e5/last.pt"

device = torch.device("cuda:0")
cfg = load_config(str(CFG))
loader = build_loader({**cfg["data"], "batch_size": 1}, "val", shuffle=False)

model = build_model(cfg["model"]).to(device).eval()
ckpt = load_checkpoint(str(CKPT), torch.device("cpu"))
state = ckpt["model"]
ms = model.state_dict()
state = {k: v for k, v in state.items() if k in ms and tuple(v.shape) == tuple(ms[k].shape)}
model.load_state_dict(state, strict=False)

gain = torch.sigmoid(model.dhel_residual_gain.detach()).item()
raw_gain = model.dhel_residual_gain.detach().item()
print("epoch", ckpt.get("epoch"), "best_metric", ckpt.get("best_metric"))
print("dhel_residual_gain_raw", raw_gain)
print("dhel_residual_gain_sigmoid", gain)

stats = {
    "logits_sum": 0.0,
    "logits_abs_sum": 0.0,
    "delta_abs_sum": 0.0,
    "n": 0,
    "logits_max": -1e9,
    "logits_min": 1e9,
    "pos_sum": 0.0,
    "pos_n": 0,
    "neg_sum": 0.0,
    "neg_n": 0,
}

with torch.no_grad():
    for i, batch in enumerate(loader):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device).float()

        out = model(image)
        r = out["dhel_residual_logits"].float()
        delta = gain * r

        stats["logits_sum"] += r.sum().item()
        stats["logits_abs_sum"] += r.abs().sum().item()
        stats["delta_abs_sum"] += delta.abs().sum().item()
        stats["n"] += r.numel()
        stats["logits_max"] = max(stats["logits_max"], r.max().item())
        stats["logits_min"] = min(stats["logits_min"], r.min().item())

        pos = mask > 0.5
        neg = ~pos
        if pos.any():
            stats["pos_sum"] += r[pos].sum().item()
            stats["pos_n"] += pos.sum().item()
        if neg.any():
            stats["neg_sum"] += r[neg].sum().item()
            stats["neg_n"] += neg.sum().item()

        if i >= 255:
            break

print("dhel_residual_logits_mean", stats["logits_sum"] / stats["n"])
print("dhel_residual_logits_abs_mean", stats["logits_abs_sum"] / stats["n"])
print("dhel_residual_logits_min", stats["logits_min"])
print("dhel_residual_logits_max", stats["logits_max"])
print("final_mean_abs_delta", stats["delta_abs_sum"] / stats["n"])
print("residual_mean_on_gt_pos", stats["pos_sum"] / max(1, stats["pos_n"]))
print("residual_mean_on_gt_neg", stats["neg_sum"] / max(1, stats["neg_n"]))