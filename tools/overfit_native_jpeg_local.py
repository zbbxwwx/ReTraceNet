from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
from pathlib import Path

import jpegio
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from retracenet.native_jpeg_hrnet import NativeJPEGForensicHRNet


def encode(image):
    output = io.BytesIO()
    image.save(output, "JPEG", quality=95, subsampling=0, optimize=False)
    return output.getvalue()


def decode_q50(image):
    output = io.BytesIO()
    image.save(output, "JPEG", quality=50, subsampling=0, optimize=False)
    return Image.open(io.BytesIO(output.getvalue())).convert("RGB")


def native(blob):
    fd, name = tempfile.mkstemp(suffix=".jpg")
    try:
        with os.fdopen(fd, "wb") as handle: handle.write(blob)
        jpeg = jpegio.read(name)
        coefficients = np.asarray(jpeg.coef_arrays[0], np.float32)
        qtable = np.asarray(jpeg.quant_tables[0], np.float32)
    finally:
        try: os.unlink(name)
        except FileNotFoundError: pass
    height = coefficients.shape[0] // 8 * 8; width = coefficients.shape[1] // 8 * 8
    blocks = coefficients[:height, :width].reshape(height // 8, 8, width // 8, 8)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(height // 8, width // 8, 64)
    return torch.from_numpy(blocks.transpose(2, 0, 1).copy()), torch.from_numpy(qtable.copy())[None]


class FixedLocalDataset(Dataset):
    def __init__(self, paths, size=256):
        self.samples = []
        for index, path in enumerate(paths):
            image = Image.open(path).convert("RGB")
            width, height = image.size
            scale = max(size / width, size / height, 1.0)
            if scale > 1: image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
            width, height = image.size
            left = (width - size) // 2; top = (height - size) // 2
            clean = image.crop((left, top, left + size, top + size))
            donor = decode_q50(clean)
            # Fixed, large, 8x8-aligned region; positions vary deterministically across images.
            region = 128
            x = 32 + (index % 5) * 16; y = 32 + ((index // 5) % 5) * 16
            composed = clean.copy(); composed.paste(donor.crop((x, y, x + region, y + region)), (x, y))
            dct, qtable = native(encode(composed))
            mask = torch.zeros(1, size // 4, size // 4)
            mask[:, y // 4:(y + region) // 4, x // 4:(x + region) // 4] = 1
            self.samples.append((dct, qtable, mask, str(path), (x, y, region, region)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, index): return self.samples[index]


class Localizer(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = NativeJPEGForensicHRNet(); self.head = nn.Conv2d(64, 1, 1)
        for parameter in self.encoder.residual_bank.parameters(): parameter.requires_grad = False
        for parameter in self.encoder.residual_stem.parameters(): parameter.requires_grad = False
    def forward(self, dct, qtable):
        batch = dct.shape[0]
        image = torch.zeros(batch, 3, dct.shape[-2] * 8, dct.shape[-1] * 8, device=dct.device)
        return self.head(self.encoder(image, dct, qtable, torch.ones(batch, 1, device=dct.device))[0])


def metrics(target, probability):
    y = target.numpy().ravel(); p = probability.numpy().ravel()
    precision, recall, threshold = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / (precision + recall + 1e-12); best = int(np.argmax(f1))
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1[best]),
            "precision": float(precision[best]), "recall": float(recall[best]),
            "threshold": float(threshold[min(best, len(threshold) - 1)])}


def save_map(output, target, probability, epoch):
    directory = output / "maps"; directory.mkdir(exist_ok=True)
    Image.fromarray((target[0, 0].numpy() * 255).astype(np.uint8)).save(directory / "target.png")
    Image.fromarray((probability[0, 0].numpy() * 255).astype(np.uint8)).save(directory / f"prob_epoch{epoch:03d}.png")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--images", required=True)
    parser.add_argument("--global-pretrain", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=32); parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args(); torch.manual_seed(3407); np.random.seed(3407)
    split = json.loads((Path(args.global_pretrain).parent / "split.json").read_text())
    paths = [Path(path) for path in split["train"][:args.count]]
    dataset = FixedLocalDataset(paths); loader = DataLoader(dataset, batch_size=8, shuffle=False)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps({"images": list(map(str, paths)), "q1": 50, "q2": 95,
        "subsampling": "444", "crop": 256, "region": 128, "feature_mask_grid": "64x64",
        "all_coordinates_divisible_by_8": all(all(value % 8 == 0 for value in sample[4]) for sample in dataset.samples)}, indent=2))
    model = Localizer().cuda(); checkpoint = torch.load(args.global_pretrain, map_location="cpu")
    result = model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
    print({"loaded": len(checkpoint["encoder"]), "missing": result.missing_keys, "unexpected": result.unexpected_keys}, flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device="cuda")); history = []
    fixed_batch = next(iter(loader)); fixed_target = fixed_batch[2]
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.; encoder_grad = head_grad = 0.
        for dct, qtable, target, _, _ in loader:
            dct, qtable, target = dct.cuda(), qtable.cuda(), target.cuda(); logits = model(dct, qtable)
            probability = logits.sigmoid(); intersection = (probability * target).sum((1,2,3))
            dice = 1 - ((2 * intersection + 1) / (probability.sum((1,2,3)) + target.sum((1,2,3)) + 1)).mean()
            loss = bce(logits, target) + .5 * dice; optimizer.zero_grad(set_to_none=True); loss.backward()
            encoder_grad = sum(float(p.grad.norm()) for p in model.encoder.parameters() if p.grad is not None)
            head_grad = sum(float(p.grad.norm()) for p in model.head.parameters() if p.grad is not None)
            optimizer.step(); total_loss += float(loss) * len(dct)
        model.eval(); probabilities = []; targets = []
        with torch.no_grad():
            for dct, qtable, target, _, _ in loader:
                probabilities.append(model(dct.cuda(), qtable.cuda()).sigmoid().cpu()); targets.append(target)
        probability = torch.cat(probabilities); target = torch.cat(targets); report = metrics(target, probability)
        row = {"epoch": epoch, "loss": total_loss / len(dataset), "encoder_grad_norm_sum": encoder_grad,
               "head_grad_norm_sum": head_grad, **report}; history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2)); print(row, flush=True)
        if epoch in (1, 5, 10, 25, 50, 100): save_map(output, target, probability, epoch)
        if report["auc"] >= .95 and report["f1"] >= .90:
            torch.save({"encoder": model.encoder.state_dict(), "head": model.head.state_dict(), "epoch": epoch, "metrics": report}, output / "passed.pt")
            (output / "decision.json").write_text(json.dumps({"passed": True, "epoch": epoch, **report}, indent=2)); return
    (output / "decision.json").write_text(json.dumps({"passed": False, "best_auc": max(x["auc"] for x in history), "best_f1": max(x["f1"] for x in history)}, indent=2))


if __name__ == "__main__": main()
