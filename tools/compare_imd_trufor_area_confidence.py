"""Same-image IMD2020 error decomposition: Stable-BN V3 versus TruFor."""
import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from retrace.data import Record
from fair_training_data import FairDataset
import run_balanced_residual_v3_full30 as v3
import run_cross_dataset_after_chain as cross


def records(rows):
    return [Record(image=Path(r["image"]), mask=Path(r["mask"]) if r.get("mask") else None,
                   split="val", dataset="IMD2020", label=int(r["label"])) for r in rows]


def counts(p, y, t):
    z = p >= t; y = y.astype(bool)
    tp = int((z & y).sum()); fp = int((z & ~y).sum()); fn = int((~z & y).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return tp, fp, fn, pr, rc, 2 * tp / max(2 * tp + fp + fn, 1)


def binned_auc(p, y, bins=1000):
    y = y.astype(bool); idx = np.minimum((np.asarray(p, np.float32) * bins).astype(np.int32), bins)
    hp = np.bincount(idx[y].reshape(-1), minlength=bins + 1).astype(np.float64)
    hn = np.bincount(idx[~y].reshape(-1), minlength=bins + 1).astype(np.float64)
    if hp.sum() == 0 or hn.sum() == 0: return None
    tp = np.cumsum(hp[::-1]); fp = np.cumsum(hn[::-1])
    return float(np.trapz(tp / hp.sum(), fp / hn.sum()))


def pred_ours(model, record, dataset, index, cache):
    with Image.open(record.image) as im: w, h = im.size
    if max(h, w) > 1024:
        return cross.predict_large_record(model, record, cache)[:2]
    batch = next(iter(DataLoader(dataset, batch_size=1, sampler=[index], num_workers=0)))
    vh, vw = map(int, batch["valid_hw"][0])
    p, _ = cross.predict_probability(model, batch)
    return p[:vh, :vw].numpy(), batch["mask"][0, 0, :vh, :vw].numpy().astype(np.uint8)


def distribution(p, y):
    fg = p[y]; bg = p[~y]
    return {
        "fg_mean": float(fg.mean()), "bg_mean": float(bg.mean()),
        "fg_p10": float(np.quantile(fg, .10)), "fg_p25": float(np.quantile(fg, .25)),
        "fg_p50": float(np.quantile(fg, .50)), "fg_p75": float(np.quantile(fg, .75)),
        "bg_top1_mean": float(bg[bg >= np.quantile(bg, .99)].mean()),
        "bg_top5_mean": float(bg[bg >= np.quantile(bg, .95)].mean()),
    }


def pooled(rows, prefix):
    tp = sum(int(r[prefix + "_tp"]) for r in rows); fp = sum(int(r[prefix + "_fp"]) for r in rows); fn = sum(int(r[prefix + "_fn"]) for r in rows)
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": pr, "recall": rc, "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def group(rows, predicate):
    part = [r for r in rows if predicate(r)]
    score_keys = ("fg_mean", "fg_p10", "fg_p25", "fg_p50", "fg_p75", "bg_mean", "bg_top1_mean", "bg_top5_mean")
    return {"images": len(part), "ours": pooled(part, "ours") if part else None,
            "trufor": pooled(part, "trufor") if part else None,
            "ours_score_distribution": {k: float(np.mean([r["ours_" + k] for r in part])) for k in score_keys} if part else None,
            "trufor_score_distribution": {k: float(np.mean([r["trufor_" + k] for r in part])) for k in score_keys} if part else None,
            "mean_delta_per_image_f1": float(np.mean([r["delta_f1"] for r in part])) if part else None}


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(a.split).read_text()); rec = records(split["val"])
    ck = torch.load(a.checkpoint, map_location="cpu"); cfg = ck["config"]
    model = v3.e.build_final_model(cfg["model"]).cuda().eval(); model.load_state_dict(ck["model"], strict=True)
    ours_t = float(ck["selection_threshold"]); trufor_t = float(a.trufor_threshold)
    dataset = FairDataset(rec, out / "jpeg_cache", False); rows = []
    for i, r in enumerate(rec):
        if not r.label: continue
        po, y = pred_ours(model, r, dataset, i, out / "jpeg_cache")
        y = np.asarray(y).astype(bool, copy=False)
        z = np.load(Path(a.trufor_pred) / (r.image.stem + ".npz")); pt = np.asarray(z["map"], np.float32)
        if po.shape != y.shape or pt.shape != y.shape: raise RuntimeError((r.image, po.shape, pt.shape, y.shape))
        co = counts(po, y, ours_t); ct = counts(pt, y, trufor_t); do = distribution(po, y); dt = distribution(pt, y)
        with Image.open(r.image) as im:
            fmt = str(im.format or "unknown").lower(); w, h = im.size
        gt_ratio = float(y.mean())
        area_group = "lt1" if gt_ratio < .01 else "1_5" if gt_ratio < .05 else "5_20" if gt_ratio < .20 else "gte20"
        row = {"path": str(r.image), "format": fmt, "height": h, "width": w, "sliding": int(max(h, w) > 1024),
               "gt_foreground_ratio": gt_ratio, "area_group": area_group,
               "ours_threshold": ours_t, "trufor_threshold": trufor_t,
               "ours_pred_ratio": float((po >= ours_t).mean()), "trufor_pred_ratio": float((pt >= trufor_t).mean()),
               "ours_auc": binned_auc(po, y), "trufor_auc": binned_auc(pt, y)}
        for prefix, c, dist in (("ours", co, do), ("trufor", ct, dt)):
            row.update({prefix + "_tp": c[0], prefix + "_fp": c[1], prefix + "_fn": c[2], prefix + "_precision": c[3], prefix + "_recall": c[4], prefix + "_f1": c[5]})
            row.update({prefix + "_" + k: v for k, v in dist.items()})
        row["delta_f1"] = row["trufor_f1"] - row["ours_f1"]
        rows.append(row)
        if len(rows) == 1 or len(rows) % 50 == 0: print("[compare]", len(rows), flush=True)
    rows.sort(key=lambda x: x["delta_f1"], reverse=True)
    with (out / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {"status": "completed", "checkpoint": a.checkpoint, "trufor_predictions": a.trufor_pred,
               "thresholds": {"ours": ours_t, "trufor": trufor_t}, "fake_images": len(rows),
               "overall": group(rows, lambda _r: True),
               "by_area": {g: group(rows, lambda r, g=g: r["area_group"] == g) for g in ("lt1", "1_5", "5_20", "gte20")},
               "by_sliding": {"false": group(rows, lambda r: not r["sliding"]), "true": group(rows, lambda r: bool(r["sliding"]))},
               "by_format": {g: group(rows, lambda r, g=g: r["format"] == g) for g in sorted(set(r["format"] for r in rows))},
               "mean_scores": {k: float(np.mean([r[k] for r in rows])) for k in ("ours_fg_mean", "trufor_fg_mean", "ours_bg_mean", "trufor_bg_mean", "ours_bg_top1_mean", "trufor_bg_top1_mean", "ours_bg_top5_mean", "trufor_bg_top5_mean")},
               "top_trufor_leads": [{k: r[k] for k in ("path", "format", "sliding", "gt_foreground_ratio", "ours_f1", "trufor_f1", "delta_f1", "ours_pred_ratio", "trufor_pred_ratio")} for r in rows[:100]],
               "protocol_note": "Each model uses its own globally selected IMD val threshold. Per-image F1 ranks errors; pooled fake-pixel metrics remain primary."}
    (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    print(yaml.safe_dump({k: v for k, v in summary.items() if k != "top_trufor_leads"}, sort_keys=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--split", required=True)
    p.add_argument("--trufor-pred", required=True); p.add_argument("--trufor-threshold", type=float, required=True); p.add_argument("--out", required=True)
    main(p.parse_args())
