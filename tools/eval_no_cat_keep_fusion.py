import copy
import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import run_imd_mit_full_v3
import run_cross_dataset_after_chain as cross

from retrace.data import Record
from retracenet.final_modules import BalancedForensicFusion

BASE = ROOT.parents[1]
OLD = BASE / (
    "outputs/retracenet_v2/"
    "casia1_mit_v3_indomain100_20260921/best.pt"
)
OUT = BASE / (
    "outputs/retracenet_v2/"
    "mit_v3_no_cat_ablation/val_no_cat_fusion"
)
OUT.mkdir(parents=True, exist_ok=True)


def inject_without_cat(self, detail, dct, qtable, dct_available):
    fused = []

    for base, fusion in zip(
        detail, self.balanced_forensic_fusions
    ):
        zeros = torch.zeros_like(base)
        available = base.new_zeros((base.shape[0], 1))

        result, _, _ = fusion(base, zeros, available)
        fused.append(result)

    assert len(fused) == len(detail)
    return fused, None, None


# 1. 读取旧权重，关闭 CAT 构建
ck = torch.load(OLD, map_location="cpu", weights_only=False)
cfg = copy.deepcopy(ck["config"])

cfg["model"]["enable_dct_parallel"] = False
cfg["model"]["dct_pretrained_path"] = None
cfg["model"]["enable_balanced_residual_chain"] = True

model = cross.e.build_final_model(cfg["model"])

assert model.dct_branch is None

# 2. 恢复旧模型的四层融合模块
channels = cfg["model"]["forensic_channels"]

model.balanced_forensic_fusions = nn.ModuleList([
    BalancedForensicFusion(int(c)) for c in channels
])

# 3. 严格检查全部保留权重
old_state = ck["model"]
target = model.state_dict()

missing = [k for k in target if k not in old_state]
mismatch = [
    k for k in target
    if k in old_state
    and target[k].shape != old_state[k].shape
]

removed = [k for k in old_state if k not in target]

allowed = (
    "dct_branch.",
    "dct_inject_adapters.",
    "dct_evidence_head.",
)

unexpected = [
    k for k in removed
    if not k.startswith(allowed)
]

print("Missing:", len(missing))
print("Mismatch:", len(mismatch))
print("Unexpected removals:", unexpected[:20])

if missing or mismatch or unexpected:
    raise RuntimeError("Weight verification failed")

clean = {k: old_state[k] for k in target}
model.load_state_dict(clean, strict=True)

# 4. 保存无 CAT 权重
compact = OUT / "no_cat_fusion.pt"

torch.save({
    "model": clean,
    "config": cfg,
    "epoch": ck["epoch"],
    "selection_threshold": ck["selection_threshold"],
    "image_selection_threshold":
        ck.get("image_selection_threshold", 0.5),
    "phase": "no_cat_zero_dct_fusion",
    "source_checkpoint": str(OLD),
}, compact)

# 5. 替换融合逻辑，不执行 CAT
model._inject_dct_evidence = types.MethodType(
    inject_without_cat, model
)

model = model.cuda().eval()

print("CAT encoder:", model.dct_branch)
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)

# 6. 使用 CASIA1 原验证集
splits = list(BASE.glob(
    "**/indomain_CASIA1/split_721.json"
))

if not splits:
    raise FileNotFoundError("CASIA1 split not found")

# 使用上次评测的同一份划分
split_path = splits[0]
print("Split:", split_path)

rows = json.loads(
    split_path.read_text(encoding="utf-8")
)["val"]

records = [
    Record(
        image=Path(r["image"]),
        mask=Path(r["mask"]) if r.get("mask") else None,
        split="val",
        dataset="CASIA1",
        label=int(r["label"]),
    )
    for r in rows
]

assert len(records) == 361, "Check CASIA1 split"

result = cross.evaluate_dataset(
    model,
    records,
    OUT / "jpeg_cache",
    float(ck["selection_threshold"]),
    float(ck.get("image_selection_threshold", 0.5)),
    "CASIA1",
    OUT,
    evaluation_mode="in_domain_locked",
    threshold_source="CASIA1_val",
)

cross.save_yaml(OUT / "result.yaml", result)

pixel = result["pixel_localization_fake_only"]
locked = pixel["test_f1_locked"]

print("========== RESULT ==========")
print("F1:", locked["f1"])
print("Recall:", locked["recall"])
print("AUC:", pixel["auc"])
print("Saved:", compact)
