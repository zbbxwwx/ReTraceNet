"""Compare large-image tile policies for the IMD MiT-full checkpoint.

Both policies use the same direct-full cutoff (1024 pixels), dataset records,
checkpoint, JPEG/DCT construction, probability stitching, and threshold sweep.
Only tile size and stride differ.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_imd_mit_full_v3  # patches the V3 builder to construct MiT content
import run_cross_dataset_after_chain as cross
from retrace.data import ManipulationDataset, Record
from e15_original_metrics import ValidationStats


def records(rows, split="val", dataset="IMD2020"):
    return [Record(image=Path(r["image"]),
                   mask=Path(r["mask"]) if r.get("mask") else None,
                   split=split, dataset=dataset, label=int(r["label"]))
            for r in rows]


def tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    final = length - tile
    if starts[-1] != final:
        starts.append(final)
    if not all(v % 8 == 0 for v in starts):
        raise RuntimeError(f"non-aligned tile origin: {length=} {tile=} {stride=} {starts[-1]=}")
    return starts


@torch.inference_mode()
def predict_large(model, record, cache, tile, stride):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    jpeg_path = record.image
    if jpeg_path.suffix.lower() not in {".jpg", ".jpeg"}:
        jpeg_path = cache / (hashlib.sha256(str(record.image).encode()).hexdigest() + ".jpg")
        if not jpeg_path.exists():
            with Image.open(record.image) as image:
                image.convert("RGB").save(jpeg_path, quality=100, subsampling=0)
    import jpegio
    jpeg = jpegio.read(str(jpeg_path))
    coefficients = jpeg.coef_arrays[0]
    qtable = jpeg.quant_tables[jpeg.comp_info[0].quant_tbl_no].astype("float32")
    with Image.open(jpeg_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    if record.mask is None:
        target = np.zeros((height, width), dtype=np.int16)
    else:
        with Image.open(record.mask) as image:
            target = (np.asarray(image.convert("L")) > 127).astype(np.int16)
    if target.shape != (height, width):
        raise ValueError(f"image/mask mismatch: {record.image}: {rgb.shape} versus {target.shape}")

    # Multiple-of-32 padding keeps every final origin divisible by 8.
    padded_h = max(tile, ((height + 31) // 32) * 32)
    padded_w = max(tile, ((width + 31) // 32) * 32)
    canvas = np.full((padded_h, padded_w, 3), 127.5, dtype=np.float32)
    canvas[:height, :width] = rgb
    coef_canvas = np.zeros((padded_h, padded_w), dtype=coefficients.dtype)
    coef_canvas[:coefficients.shape[0], :coefficients.shape[1]] = coefficients
    accum = torch.zeros((padded_h, padded_w), dtype=torch.float32)
    weight = torch.zeros((padded_h, padded_w), dtype=torch.float32)
    for top in tile_starts(padded_h, tile, stride):
        for left in tile_starts(padded_w, tile, stride):
            rgb_tile = torch.from_numpy(canvas[top:top+tile, left:left+tile].copy()).permute(2, 0, 1)[None] / 255
            coef_tile = coef_canvas[top:top+tile, left:left+tile]
            batch = {
                "image": rgb_tile,
                "dct": ManipulationDataset._dct_volume_from_coef(coef_tile, tile, tile)[None],
                "qtable": torch.from_numpy(qtable[None, None]),
                "dct_available": torch.ones((1, 1)),
            }
            output = cross.fair.fwd(model, batch)
            prob = output["mask_logits"].float().sigmoid()[0, 0].cpu()
            accum[top:top+tile, left:left+tile] += prob
            weight[top:top+tile, left:left+tile] += 1
    if not bool((weight > 0).all()):
        raise RuntimeError("sliding-window coverage gap")
    return (accum / weight)[:height, :width].numpy(), target


@torch.inference_mode()
def evaluate_policy(model, recs, cache, tile, stride, out, name, full_max=1024):
    ds = cross.fair.FairDataset(recs, cache, False)
    stats = ValidationStats()
    direct_count = sliding_count = 0
    for index, record in enumerate(recs):
        with Image.open(record.image) as image:
            width, height = image.size
        if max(height, width) > full_max:
            scores, target = predict_large(model, record, cache, tile, stride)
            sliding_count += 1
        else:
            batch = next(iter(DataLoader(ds, batch_size=1, sampler=[index], num_workers=0)))
            valid_h, valid_w = map(int, batch["valid_hw"][0].tolist())
            output = cross.fair.fwd(model, batch)
            scores = output["mask_logits"].float().sigmoid()[0, 0, :valid_h, :valid_w].cpu().numpy()
            target = batch["mask"][0, 0, :valid_h, :valid_w].numpy().astype(np.int16)
            direct_count += 1
        if scores.shape != target.shape:
            scores = torch.nn.functional.interpolate(torch.from_numpy(scores)[None, None].float(),
                size=target.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
        stats.add(scores, target, record.label, record.image)
        if index == 0 or (index + 1) % 25 == 0:
            print(f"[{name}] {index+1}/{len(recs)} direct={direct_count} sliding={sliding_count}", flush=True)
    result = stats.result()
    payload = {
        "policy": name,
        "dataset": "IMD2020_val",
        "num_images": len(recs),
        "direct_count": direct_count,
        "sliding_count": sliding_count,
        "full_max_side": full_max,
        "tile": tile,
        "stride": stride,
        "pixel_stitching": "uniform overlap probability average",
        "fake_only_pooled": result["fake"],
        "primary": result["fake"]["oracle"],
        "threshold_source": "IMD2020_val_current_policy",
        "oracle_role": "validation selection (not a test oracle)",
    }
    cross.save_yaml(Path(out) / f"{name}.yaml", payload)
    return payload


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"status": "loading", "checkpoint": a.checkpoint, "split": a.split,
               "controlled_variable": "large-image tile and stride only",
               "fixed_full_max_side": 1024, "policies": {}}
    cross.save_yaml(out / "summary.yaml", summary)
    try:
        checkpoint = torch.load(a.checkpoint, map_location="cpu")
        model = cross.e.build_final_model(copy.deepcopy(checkpoint["config"]["model"])).cuda().eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        split = json.loads(Path(a.split).read_text())
        recs = records(split["val"])
        for name, tile, stride in (("tile512_stride384", 512, 384),
                                   ("tile1024_stride768", 1024, 768)):
            summary["status"] = f"evaluating_{name}"
            cross.save_yaml(out / "summary.yaml", summary)
            result = evaluate_policy(model, recs, out / "jpeg_cache", tile, stride, out, name)
            summary["policies"][name] = result
            cross.save_yaml(out / "summary.yaml", summary)
        base = float(summary["policies"]["tile512_stride384"]["primary"]["f1"])
        alt = float(summary["policies"]["tile1024_stride768"]["primary"]["f1"])
        delta = alt - base
        if delta >= .01:
            decision = "adopt_1024_768_and_run_locked_test"
        elif delta < .005:
            decision = "discard_larger_tile_policy"
        else:
            decision = "borderline_0.5_to_1pp_no_automatic_adoption"
        summary.update(status="completed", base_f1=base, alternative_f1=alt,
                       delta_f1=delta, delta_pp=100 * delta, decision=decision,
                       checkpoint_epoch=int(checkpoint.get("epoch", -1)))
        cross.save_yaml(out / "summary.yaml", summary)
        print(json.dumps({"base_f1":base,"alternative_f1":alt,"delta_pp":100*delta,
                          "decision":decision}, indent=2), flush=True)
    except BaseException:
        summary.update(status="failed", stop_reason=traceback.format_exc())
        cross.save_yaml(out / "summary.yaml", summary)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
