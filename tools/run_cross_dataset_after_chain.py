"""Wait for the complete training chain, then evaluate four external datasets.

The checkpoint and threshold are selected only on CASIA2 validation. External
dataset oracle values are reported as diagnostics and never used for selection.
"""
import argparse
import copy
import hashlib
import json
import time
import traceback
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import run_fair_training_v1 as fair
from retrace.data import ManipulationDataset, Record
from e15_original_metrics import ValidationStats
from v3_unified_metrics import PixelCounts, AuthenticPixelFPR, ImageDetectionStats, PROTOCOL

e = fair.e


def load_yaml(path):
    import yaml
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def save_yaml(path, value):
    e.dump(Path(path), value)


def wait_completed(path, poll_seconds, label):
    while True:
        state = load_yaml(path)
        status = state.get("status")
        if status == "completed":
            return state
        if status == "failed":
            raise RuntimeError(f"{label} failed: {state.get('stop_reason')}")
        print("[wait]", label, status, state.get("epoch"), flush=True)
        time.sleep(poll_seconds)


def parse_index(path, dataset):
    records = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_number}: expected image mask label")
            image, mask, label = Path(fields[0]), Path(fields[1]), int(fields[2])
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f"{path}:{line_number}: {image} or {mask}")
            records.append(Record(image=image, mask=mask, split="external", dataset=dataset, label=label))
    if not records:
        raise ValueError(f"empty index: {path}")
    return records


class LockedCounts:
    def __init__(self, threshold):
        self.threshold = float(threshold)
        self.tp = self.fp = self.fn = self.tn = 0

    def add(self, scores, target):
        pred = np.asarray(scores) >= self.threshold
        truth = np.asarray(target) > 0
        self.tp += int(np.logical_and(pred, truth).sum())
        self.fp += int(np.logical_and(pred, ~truth).sum())
        self.fn += int(np.logical_and(~pred, truth).sum())
        self.tn += int(np.logical_and(~pred, ~truth).sum())

    def result(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        return {"threshold": self.threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "iou": tp / max(tp + fp + fn, 1), "fpr": fp / max(fp + tn, 1)}


def tile_starts(length, tile=512, stride=384):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    assert all(value % 8 == 0 for value in starts)
    return starts


@torch.inference_mode()
def predict_probability(model, batch):
    _, _, height, width = batch["image"].shape
    if max(height, width) <= 1024:
        output=fair.fwd(model,batch)
        return output["mask_logits"].float().sigmoid()[0,0].cpu(), float(output["image_logit"].float().sigmoid().reshape(-1).mean().cpu())
    tile,stride=512,384; accum=torch.zeros((height,width)); weight=torch.zeros((height,width)); image_scores=[]
    for top in tile_starts(height,tile,stride):
        for left in tile_starts(width,tile,stride):
            tiled={"image":batch["image"][...,top:top+tile,left:left+tile],"dct":batch["dct"][...,top:top+tile,left:left+tile],"qtable":batch["qtable"],"dct_available":batch["dct_available"]}
            output=fair.fwd(model,tiled); probability=output["mask_logits"].float().sigmoid()[0,0].cpu()
            image_scores.append(float(output["image_logit"].float().sigmoid().reshape(-1).mean().cpu()))
            accum[top:top+tile,left:left+tile]+=probability; weight[top:top+tile,left:left+tile]+=1
    if not bool((weight>0).all()): raise RuntimeError("sliding-window coverage gap")
    return accum/weight,float(np.mean(sorted(image_scores,reverse=True)[:max(1,int(np.ceil(.2*len(image_scores))))]))


@torch.inference_mode()
def predict_large_record(model, record, cache):
    """Tile RGB and coefficients before expanding the 21-channel DCT volume."""
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
        raise ValueError(f"image/mask mismatch: {record.image} {rgb.shape} {target.shape}")
    padded_h, padded_w = max(512, ((height + 31) // 32) * 32), max(512, ((width + 31) // 32) * 32)
    canvas = np.full((padded_h, padded_w, 3), 127.5, dtype=np.float32)
    canvas[:height, :width] = rgb
    coef_canvas = np.zeros((padded_h, padded_w), dtype=coefficients.dtype)
    coef_canvas[:coefficients.shape[0], :coefficients.shape[1]] = coefficients
    accum = torch.zeros((padded_h, padded_w), dtype=torch.float32)
    weight = torch.zeros((padded_h, padded_w), dtype=torch.float32)
    image_scores = []
    for top in tile_starts(padded_h):
        for left in tile_starts(padded_w):
            rgb_tile = torch.from_numpy(canvas[top:top + 512, left:left + 512].copy()).permute(2, 0, 1)[None] / 255
            coef_tile = coef_canvas[top:top + 512, left:left + 512]
            batch = {"image": rgb_tile,
                     "dct": ManipulationDataset._dct_volume_from_coef(coef_tile, 512, 512)[None],
                     "qtable": torch.from_numpy(qtable[None, None]),
                     "dct_available": torch.ones((1, 1))}
            # Optional context for models that fuse a whole-image thumbnail
            # with this coordinate-aligned local tile. Baseline models ignore it.
            if model.content_encoder.__class__.__name__ == "GlobalLocalContentEncoder":
                whole = np.asarray(
                    Image.fromarray(rgb.astype(np.uint8)).resize((512, 512), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
                batch["global_image"] = torch.from_numpy(whole.copy()).permute(2, 0, 1)[None] / 255
                batch["global_roi"] = torch.tensor(
                    [[left / width, top / height,
                      min(left + 512, width) / width, min(top + 512, height) / height]],
                    dtype=torch.float32,
                )
            output = fair.fwd(model, batch)
            probability = output["mask_logits"].float().sigmoid()[0, 0].cpu()
            image_scores.append(float(output["image_logit"].float().sigmoid().reshape(-1).mean().cpu()))
            accum[top:top + 512, left:left + 512] += probability
            weight[top:top + 512, left:left + 512] += 1
    if not bool((weight > 0).all()):
        raise RuntimeError("large-record sliding-window coverage gap")
    return (accum / weight)[:height, :width].numpy(), target, float(np.mean(sorted(image_scores,reverse=True)[:max(1,int(np.ceil(.2*len(image_scores))))]))


@torch.inference_mode()
def evaluate_dataset(model, records, cache, threshold, image_threshold, dataset_name, out, evaluation_mode="cross_domain_locked", threshold_source=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    ds=fair.FairDataset(records,cache,False); stats=ValidationStats(); fake_locked=PixelCounts(threshold) if threshold is not None else None
    authentic=AuthenticPixelFPR(); image_detection=ImageDetectionStats(); label_counts={0:0,1:0}
    for index,record in enumerate(records):
        with Image.open(record.image) as image: original_width,original_height=image.size
        if max(original_height,original_width)>1024:
            scores,target,image_score=predict_large_record(model,record,cache)
        else:
            batch=next(iter(DataLoader(ds,batch_size=1,sampler=[index],num_workers=0))); vh,vw=map(int,batch["valid_hw"][0].tolist())
            probability,image_score=predict_probability(model,batch); scores=probability[:vh,:vw].numpy(); target=batch["mask"][0,0,:vh,:vw].numpy().astype(np.int16)
        if scores.shape != target.shape:
            scores=torch.nn.functional.interpolate(torch.from_numpy(np.asarray(scores))[None,None].float(),size=target.shape,mode="bilinear",align_corners=False)[0,0].numpy()
        assert scores.shape==target.shape
        stats.add(scores,target,record.label,record.image); image_detection.add(image_score,record.label)
        if record.label:
            if fake_locked is not None: fake_locked.add(scores,target)
        else: authentic.add(scores)
        label_counts[int(record.label)]+=1
        if index==0 or (index+1)%100==0: print("[external]",dataset_name,index+1,len(records),flush=True)
    diagnostics=stats.result()
    selected=float(threshold) if threshold is not None else float(diagnostics["fake"]["oracle"]["threshold"])
    image_result=image_detection.result(image_threshold)
    selected_image=float(image_threshold) if image_threshold is not None else float(image_result["test_f1_oracle_diagnostic"]["threshold"])
    result={"dataset":dataset_name,"num_images":len(records),"label_counts":label_counts,"protocol":dict(PROTOCOL),"evaluation_mode":evaluation_mode,"threshold_source":(threshold_source or ("CASIA2_val" if threshold is not None else f"{dataset_name}_val_current")),"pixel_threshold":selected,"image_threshold_source":("CASIA2_val" if image_threshold is not None else f"{dataset_name}_val_current"),"image_threshold":selected_image,
      "input_policy":"JPEG original; nonJPEG -> JPEG q100 subsampling0; RGB and DCT from identical file",
      "large_image_inference":"512x512 grid8-aligned tiles, stride384; overlap pixel probabilities averaged; one image score is fixed top-20%-tile mean",
      "locked_from":"selected checkpoint validation fake-only pooled F1-best",
      "pixel_localization_fake_only":{"test_f1_locked":(fake_locked.result() if fake_locked is not None else dict(diagnostics["fake"]["oracle"])),"auc":diagnostics["fake"]["auc"],"test_f1_oracle_diagnostic":diagnostics["fake"]["oracle"]},
      "authentic_pixel_fpr":authentic.result(selected),"image_level_all":image_result,
      "diagnostic_only":diagnostics,
      "preprocessing":{"input_size":"native original size; 512 tiles only when max side >1024","sliding_window_size":512,"stride":384,"pixel_stitching":"overlap probability sum / weight sum","probability_interpolation":"bilinear before threshold if output/GT shapes differ","normalization":"RGB float [0,1]; model-owned encoder normalization","jpeg_handling":"JPEG original; non-JPEG converted once to q100 subsampling0","dct_block_alignment":"tile x/y starts divisible by 8; RGB/DCT/qtable from identical JPEG; tiles never re-JPEG-encoded","tta":"none","padding":"right/bottom only and cropped before metrics"},
      "evaluator_version":"v3_unified_protocol_2","git_commit":(subprocess.run(["git","-C",str(Path(__file__).resolve().parents[1]),"rev-parse","HEAD"],text=True,capture_output=True).stdout.strip() or "unavailable"),
      "note":"External oracle is diagnostic only; primary pixel localization is fake-only at the validation-locked threshold."}
    print("[protocol] evaluation_mode=%s threshold_source=%s pixel_threshold=%.6f image_threshold=%.6f" % (evaluation_mode,result["threshold_source"],selected,selected_image),flush=True)
    save_yaml(out/f"{dataset_name}.yaml",result); return result


def main(args):
    parent_dir, post_dir, out = Path(args.parent), Path(args.post), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"status": "waiting", "test_run": True, "datasets": {},
               "selection_policy": "checkpoint and threshold selected only on CASIA2 validation"}
    save_yaml(out / "summary.yaml", summary)
    try:
        parent = wait_completed(parent_dir / "summary.yaml", args.poll_seconds, "fair_training")
        post = wait_completed(post_dir / "summary.yaml", args.poll_seconds, "post_training")
        candidates = [("fair_parent", float(parent["best_F1"]), Path(parent["best_checkpoint"]))]
        control_summary_path = post_dir / "control_continue" / "summary.yaml"
        if control_summary_path.exists():
            control = load_yaml(control_summary_path)
            if control.get("status") == "completed":
                candidates.append(("control_continue", float(control["best_F1"]), Path(control["best_checkpoint"])))
        dropout_summary_path = post_dir / "dct_dropout_p50" / "summary.yaml"
        if bool(post.get("accepted")) and dropout_summary_path.exists():
            dropout = load_yaml(dropout_summary_path)
            candidates.append(("accepted_dct_dropout", float(dropout["best_F1"]), Path(dropout["best_checkpoint"])))
        source, score, checkpoint_path = max(candidates, key=lambda item: item[1])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = copy.deepcopy(checkpoint["config"])
        model = e.build_final_model(cfg["model"]).cuda().eval()
        model.load_state_dict(checkpoint["model"], strict=True)
        threshold = float(checkpoint["selection_threshold"]); image_threshold=float(checkpoint["image_selection_threshold"])
        summary.update(status="evaluating", selected_source=source, selected_checkpoint=str(checkpoint_path),
                       casia2_selection_F1=score, locked_threshold=threshold, candidates=[
                           {"source": name, "F1": value, "checkpoint": str(path)} for name, value, path in candidates])
        save_yaml(out / "summary.yaml", summary)
        index_root = Path(args.index_root)
        mapping = {"CASIA1": index_root / "CASIA1.txt", "Columbia": index_root / "Columbia.txt",
                   "NIST16": index_root / "NIST16.txt", "IMD2020": index_root / "IMD2020.txt"}
        for name, index_path in mapping.items():
            records = parse_index(index_path, name)
            result = evaluate_dataset(model, records, out / "jpeg_cache" / name, threshold, image_threshold, name, out)
            summary["datasets"][name] = {"num_images": result["num_images"], "pixel_localization_fake_only": result["pixel_localization_fake_only"],
                                         "authentic_pixel_fpr": result["authentic_pixel_fpr"],
                                         "image_level_all": result["image_level_all"]}
            save_yaml(out / "summary.yaml", summary)
        summary.update(status="completed", stop_reason="four_external_datasets_complete")
        save_yaml(out / "summary.yaml", summary)
    except BaseException:
        summary.update(status="failed", stop_reason=traceback.format_exc())
        save_yaml(out / "summary.yaml", summary)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    main(parser.parse_args())
