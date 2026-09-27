import copy
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_imd_mit_full_v3
import run_cross_dataset_after_chain as cross

BASE = Path("/group4/zouwenxuan/iml_generalization")

SOURCE = (
    BASE / "outputs/retracenet_v2"
    / "casia1_mit_v3_indomain100_20260921/best.pt"
)

OUTPUT = (
    BASE / "outputs/retracenet_v2"
    / "mit_v3_no_cat_ablation/init.pt"
)

# 1. 读取旧模型
ckpt = torch.load(
    SOURCE, map_location="cpu", weights_only=False
)

old_state = ckpt["model"]
cfg = copy.deepcopy(ckpt["config"])

# 2. 关闭 CAT 并行分支
cfg["model"]["enable_dct_parallel"] = False
cfg["model"]["dct_pretrained_path"] = None
cfg["model"]["enable_balanced_residual_chain"] = True
cfg["model"]["enable_dct_evidence_head"] = False

# 3. 构建不含 CAT 分支的新模型
model = cross.e.build_final_model(cfg["model"])
target = model.state_dict()

# 4. 严格检查权重兼容性
missing = [k for k in target if k not in old_state]

mismatch = [
    k for k in target
    if k in old_state
    and target[k].shape != old_state[k].shape
]

removed = [k for k in old_state if k not in target]

cat_prefixes = (
    "dct_branch.",
    "dct_inject_adapters.",
    "balanced_forensic_fusions.",
    "dct_evidence_head.",
)

unexpected = [
    k for k in removed
    if not k.startswith(cat_prefixes)
]

print("========== CHECK ==========")
print("Missing:", missing[:20])
print("Shape mismatch:", mismatch[:20])
print("Unexpected removals:", unexpected[:20])
print("Removed CAT-related keys:", len(removed))

if missing or mismatch or unexpected:
    raise RuntimeError("Weight compatibility check failed")

# 5. 严格加载所有保留的权重
clean_state = {
    k: old_state[k] for k in target
}

model.load_state_dict(clean_state, strict=True)

assert model.dct_branch is None
assert model.dct_inject_adapters is None
assert model.balanced_forensic_fusions is None

print("All retained weights loaded successfully")
print(
    "Model parameters:",
    sum(p.numel() for p in model.parameters())
)

# 6. 独立保存，不覆盖旧模型
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model": clean_state,
        "config": cfg,
        "epoch": ckpt.get("epoch"),
        "phase": "no_cat_structural_ablation",
        "source_checkpoint": str(SOURCE),
        "source_metric_with_cat": ckpt.get("best_metric"),
    },
    OUTPUT
)

print("Saved:", OUTPUT)
