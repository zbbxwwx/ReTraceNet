import argparse, inspect, json, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from IMDLBenCo.datasets import JsonDataset
from IMDLBenCo.registry import MODELS, POSTFUNCS


def counts(tp, fp, fn, tn=None):
    precision = float(tp / max(1, tp + fp)); recall = float(tp / max(1, tp + fn))
    result = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": precision,
              "recall": recall, "f1": float(2 * tp / max(1, 2 * tp + fp + fn))}
    if tn is not None: result["tn"] = int(tn)
    return result


def auc(pos, neg):
    below = np.cumsum(neg) - neg
    return float(((pos * below).sum() + .5 * (pos * neg).sum()) / max(1, pos.sum() * neg.sum()))


def scan(pos, neg):
    tp = np.cumsum(pos[::-1])[::-1]; fp = np.cumsum(neg[::-1])[::-1]; fn = pos.sum() - tp
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn); k = int(np.argmax(f1))
    result = counts(tp[k], fp[k], fn[k]); result["threshold"] = k / 1000
    return result


def run(model, model_name, manifest, size, threshold=None, image_threshold=None):
    rows = json.loads(Path(manifest).read_text())
    post_name = f"{model_name}_post_func".lower()
    post = POSTFUNCS.get(post_name) if POSTFUNCS.has(post_name) else None
    dataset = JsonDataset(manifest, is_resizing=True, output_size=(size, size), post_funcs=post)
    fake_pos = np.zeros(1001, np.int64); fake_neg = np.zeros(1001, np.int64)
    auth_hist = np.zeros(1001, np.int64); locked = [0, 0, 0, 0]
    auth_fp = auth_pixels = 0; image_scores = []; image_labels = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            sample = torch.utils.data.default_collate([dataset[index]])
            sample = {key: value.cuda() if torch.is_tensor(value) else value for key, value in sample.items()}
            # MVSS-Net computes its auxiliary loss inside forward even during
            # evaluation.  JsonDataset does not emit an edge target unless an
            # edge post-function is explicitly configured, so provide a
            # shape-compatible placeholder.  It cannot affect pred_mask.
            if model_name == "MVSSNet" and "edge_mask" not in sample:
                sample["edge_mask"] = torch.zeros_like(sample["mask"])
            output = model(**sample)["pred_mask"].float()
            mask_path = row[1] if isinstance(row, list) else row.get("mask")
            image_path = row[0] if isinstance(row, list) else row["image"]
            if mask_path and str(mask_path).lower() != "negative":
                with Image.open(mask_path) as handle: target = np.asarray(handle.convert("L")) > 127
            else:
                with Image.open(image_path) as handle: target = np.zeros((handle.height, handle.width), bool)
            label = int(target.any())
            probability = F.interpolate(output, size=target.shape, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            bins = np.minimum((probability * 1000).astype(np.int16), 1000)
            if label:
                fake_pos += np.bincount(bins[target], minlength=1001)
                fake_neg += np.bincount(bins[~target], minlength=1001)
            else: auth_hist += np.bincount(bins.ravel(), minlength=1001)
            top = max(1, probability.size // 100)
            image_scores.append(float(np.partition(probability.ravel(), -top)[-top:].mean())); image_labels.append(label)
            if threshold is not None:
                pred = probability >= threshold
                if label:
                    locked[0] += int((pred & target).sum()); locked[1] += int((pred & ~target).sum())
                    locked[2] += int((~pred & target).sum()); locked[3] += int((~pred & ~target).sum())
                else: auth_fp += int(pred.sum()); auth_pixels += pred.size
            if index == 0 or (index + 1) % 100 == 0: print("[eval]", index + 1, len(rows), flush=True)
    image_pos = np.zeros(1001, np.int64); image_neg = np.zeros(1001, np.int64)
    for score, label in zip(image_scores, image_labels):
        (image_pos if label else image_neg)[min(1000, int(score * 1000))] += 1
    return dict(rows=len(rows), fake=sum(image_labels), authentic=len(rows)-sum(image_labels),
                fake_pos=fake_pos, fake_neg=fake_neg, auth_hist=auth_hist, locked=locked,
                auth_fp=auth_fp, auth_pixels=auth_pixels, image_pos=image_pos, image_neg=image_neg)


def build_model(name, checkpoint):
    constructor = MODELS.get(name); saved_args = vars(checkpoint["args"])
    parameters = inspect.signature(constructor if isinstance(constructor, (types.FunctionType, types.MethodType)) else constructor.__init__).parameters
    kwargs = {key: value for key, value in saved_args.items() if key in parameters}
    if name == "PSCC_Net": kwargs["pretrain_path"] = "/group4/zouwenxuan/iml_generalization/repos/PSCC-Net/models/hrnet_w18_small_v2.pth"
    if name == "Cat_Net": kwargs["cfg_file"] = "/group4/zouwenxuan/iml_generalization/envs/mesorch/lib/python3.10/site-packages/IMDLBenCo/statics/model_zoo/configs/CAT_full.yaml"
    model = constructor(**kwargs).cuda(); model.load_state_dict(checkpoint["model"], strict=True)
    if name == "MVSSNet": model.train()
    else: model.eval()
    return model


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(args.model, checkpoint)
    val = run(model, args.model, args.val, args.size); pixel_best = scan(val["fake_pos"], val["fake_neg"])
    image_best = scan(val["image_pos"], val["image_neg"])
    test = run(model, args.model, args.test, args.size, pixel_best["threshold"], image_best["threshold"])
    tp, fp, fn, tn = test["locked"]
    fake = counts(tp, fp, fn, tn); all_images = counts(tp, fp + test["auth_fp"], fn, tn + test["auth_pixels"] - test["auth_fp"])
    ik = int(image_best["threshold"] * 1000); itp = int(np.cumsum(test["image_pos"][::-1])[::-1][ik]); ifp = int(np.cumsum(test["image_neg"][::-1])[::-1][ik])
    image_result = counts(itp, ifp, int(test["image_pos"].sum()) - itp, int(test["image_neg"].sum()) - ifp)
    result = {"status": "completed", "model": args.model, "dataset": "FantasticReality",
              "threshold_source": "FantasticReality_val", "pixel_threshold": pixel_best["threshold"],
              "image_threshold": image_best["threshold"], "counts": {k: test[k] for k in ("rows", "fake", "authentic")},
              "fake_only_pooled": {**fake, "auc": auc(test["fake_pos"], test["fake_neg"]), "oracle_diagnostic": scan(test["fake_pos"], test["fake_neg"])},
              "authentic_pixel_fpr": float(test["auth_fp"] / max(1, test["auth_pixels"])),
              "all_images_pooled": {**all_images, "auc": auc(test["fake_pos"], test["fake_neg"] + test["auth_hist"])},
              "image_level_all": {**image_result, "auc": auc(test["image_pos"], test["image_neg"])},
              "validation_selection": {"pixel": pixel_best, "image": image_best}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True); Path(args.out).write_text(yaml.safe_dump(result, sort_keys=False)); print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val", required=True); parser.add_argument("--test", required=True); parser.add_argument("--out", required=True); parser.add_argument("--size", type=int, required=True)
    main(parser.parse_args())
