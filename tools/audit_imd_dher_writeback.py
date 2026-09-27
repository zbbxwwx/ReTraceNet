"""Read-only DHER write-back magnitude audit on IMD2020 validation fakes."""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from retrace.data import Record
from fair_training_data import FairDataset
import run_balanced_residual_v3_full30 as v3


def records(rows):
    return [Record(image=Path(r["image"]), mask=Path(r["mask"]) if r.get("mask") else None,
                   split="val", dataset="IMD2020", label=int(r["label"])) for r in rows]


def scalar(x):
    return float(x.detach().float().cpu()) if torch.is_tensor(x) else float(x)


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(a.split).read_text())
    rows = records(split["val"])
    ck = torch.load(a.checkpoint, map_location="cpu")
    cfg = ck["config"]
    model = v3.e.build_final_model(cfg["model"]).cuda().eval()
    model.load_state_dict(ck["model"], strict=True)
    if model.dher is None:
        raise RuntimeError("checkpoint has no DHER")

    captured = {}
    h = model.dher.register_forward_pre_hook(lambda _m, args: captured.__setitem__("pre", args[0].detach()))
    ds = FairDataset(rows, out / "jpeg_cache", False)
    per = []
    for i, r in enumerate(rows):
        if not r.label:
            continue
        with Image.open(r.image) as im:
            w, hh = im.size
        if max(hh, w) > a.max_dimension:
            continue
        batch = next(iter(DataLoader(ds, batch_size=1, sampler=[i], num_workers=0)))
        with torch.no_grad():
            output = v3.fwd(model, batch, cfg, False)
        diag = output.get("dher", {})
        pre = captured["pre"].float()
        sw = diag.get("support_write")
        cw = diag.get("conflict_write")
        if sw is None or cw is None:
            raise RuntimeError("DHER diagnostics do not expose support_write/conflict_write")
        sw, cw = sw.detach().float(), cw.detach().float()
        total = sw + cw
        pre_norm = pre.norm().clamp_min(1e-12)
        def cos(write):
            wn = write.norm()
            return float((pre.flatten() @ write.flatten() / (pre_norm * wn).clamp_min(1e-12)).cpu())
        row = {
            "path": str(r.image),
            "foreground_ratio": float(batch["mask"].float().mean()),
            "support_rms_over_d1_rms": float((sw.square().mean().sqrt() / pre.square().mean().sqrt().clamp_min(1e-12)).cpu()),
            "conflict_rms_over_d1_rms": float((cw.square().mean().sqrt() / pre.square().mean().sqrt().clamp_min(1e-12)).cpu()),
            "total_write_rms_over_d1_rms": float((total.square().mean().sqrt() / pre.square().mean().sqrt().clamp_min(1e-12)).cpu()),
            "support_cos_with_d1": cos(sw),
            "conflict_cos_with_d1": cos(cw),
            "mean_abs_pre_post_difference": float(total.abs().mean().cpu()),
            "relative_l2_pre_post_difference": float((total.norm() / pre_norm).cpu()),
        }
        for key in ("support_gate", "conflict_gate"):
            value = diag.get(key)
            if value is not None:
                value = value.detach().float()
                row[key + "_mean"] = float(value.mean().cpu())
                row[key + "_std"] = float(value.std(unbiased=False).cpu())
                row[key + "_saturation_low"] = float((value < .05).float().mean().cpu())
                row[key + "_saturation_high"] = float((value > .95).float().mean().cpu())
        for key in ("alpha_support", "alpha_conflict"):
            if diag.get(key) is not None:
                row[key] = scalar(diag[key])
        per.append(row)
        if len(per) >= a.max_samples:
            break
    h.remove()
    metric_keys = [k for k in per[0] if k not in ("path", "foreground_ratio")]
    aggregate = {}
    for k in metric_keys:
        vals = np.asarray([r[k] for r in per], dtype=np.float64)
        aggregate[k] = {"mean": float(vals.mean()), "median": float(np.median(vals)), "max": float(vals.max()), "min": float(vals.min())}
    result = {
        "status": "completed", "checkpoint": a.checkpoint, "split": a.split,
        "selection": f"first {a.max_samples} fake val images with max dimension <= {a.max_dimension}",
        "samples": len(per), "aggregate": aggregate, "per_image": per,
        "interpretation_note": "Direct magnitude audit. Ratios near zero establish effective identity write-back; cosine equality alone does not."
    }
    (out / "summary.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    print(yaml.safe_dump({k: v for k, v in result.items() if k != "per_image"}, sort_keys=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--split", required=True)
    p.add_argument("--out", required=True); p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--max-dimension", type=int, default=1024)
    main(p.parse_args())
