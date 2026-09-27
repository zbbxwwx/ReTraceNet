"""E26: realistic donor-host compression history branch pretraining.

Fixes the E25 failure mode by using cross-image donor->host splicing instead of
same-image JPEG patches, and by never updating the original forensic stream.
Only the compression evidence branch is pretrained, then DCT adapters + QTR +
DHER visibility write-back are adapted while HCV/decoder/backbones stay frozen.
No test evaluation is run.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch
from torch import nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import run_e21_jpeg_history_branch as e21
from retrace.data import pil_to_tensor, read_manifest
from retrace.losses import ReTraceLoss
from retrace.utils import atomic_torch_save, set_seed
from retracenet.final_modules import build_final_model
from train_v2_final_pipeline import clean_loss_config, dher_v2_aux_loss, tversky_loss, train_loader_for, weighted_deep_supervision_loss

EXPERIMENT = "E26_donor_host_compression_branch_pretrain"
SOURCE_BEST_F1 = 0.7982278425423237


def dump(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=int(quality), subsampling=0)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def jitter(image: Image.Image, strength: float) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(random.uniform(1 - strength, 1 + strength))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(1 - strength, 1 + strength))
    image = ImageEnhance.Color(image).enhance(random.uniform(1 - strength, 1 + strength))
    return image


def dct_volume_from_coef(coef: np.ndarray, height: int, width: int) -> torch.Tensor:
    coef = coef[:height, :width].astype("float32", copy=False)
    dct = np.zeros((21, height, width), dtype=np.float32)
    dct[0] = (coef == 0)
    abs_coef = np.abs(coef)
    for value in range(1, 20):
        dct[value] = (abs_coef == value)
    dct[20] = (abs_coef >= 20)
    return torch.from_numpy(dct)


def jpeg_info(image: Image.Image, quality: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import jpegio
    rgb = image.convert("RGB")
    width, height = rgb.size
    pad_w = ((width + 7) // 8) * 8
    pad_h = ((height + 7) // 8) * 8
    if (pad_w, pad_h) != (width, height):
        padded = Image.new("RGB", (pad_w, pad_h), color=(128, 128, 128))
        padded.paste(rgb, (0, 0))
        rgb = padded
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            tmp_name = handle.name
        rgb.save(tmp_name, format="JPEG", quality=int(quality), subsampling=0)
        jpg = jpegio.read(tmp_name)
        coef = jpg.coef_arrays[0].astype("float32")
        ci = jpg.comp_info[0]
        qtable = jpg.quant_tables[ci.quant_tbl_no].astype("float32")
    finally:
        if tmp_name is not None:
            try: os.unlink(tmp_name)
            except OSError: pass
    return dct_volume_from_coef(coef, height, width), torch.from_numpy(qtable[None, :, :]), torch.tensor([1.0], dtype=torch.float32)


class DonorHostSpliceGenerator:
    def __init__(self, records, image_size: int = 640) -> None:
        self.records = [r for r in records if r.image.exists()]
        if len(self.records) < 2:
            raise RuntimeError("need at least two training images")
        self.image_size = int(image_size)

    def _load_square(self, index: int | None = None) -> Image.Image:
        record = self.records[random.randrange(len(self.records))] if index is None else self.records[index % len(self.records)]
        with Image.open(record.image) as handle:
            image = handle.convert("RGB")
        short = max(1, min(image.size))
        ratio = self.image_size * random.uniform(0.85, 1.45) / short
        new_size = (max(self.image_size, round(image.width * ratio)), max(self.image_size, round(image.height * ratio)))
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        left = random.randint(0, image.width - self.image_size) if image.width > self.image_size else 0
        top = random.randint(0, image.height - self.image_size) if image.height > self.image_size else 0
        return image.crop((left, top, left + self.image_size, top + self.image_size)).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

    def _region(self) -> tuple[Image.Image, tuple[int, int, int, int], str]:
        size = self.image_size
        bucket = random.choices(["lt_1", "1_5", "5_20", "gte_20"], weights=[0.08, 0.24, 0.34, 0.34], k=1)[0]
        if bucket == "lt_1": area = random.uniform(0.005, 0.01)
        elif bucket == "1_5": area = random.uniform(0.012, 0.05)
        elif bucket == "5_20": area = random.uniform(0.055, 0.20)
        else: area = random.uniform(0.20, 0.50)
        aspect = math.exp(random.uniform(math.log(0.4), math.log(2.5)))
        h = max(16, min(size - 4, int(round(size * math.sqrt(area / aspect)))))
        w = max(16, min(size - 4, int(round(size * math.sqrt(area * aspect)))))
        left = random.randint(0, size - w)
        top = random.randint(0, size - h)
        box = (left, top, left + w, top + h)
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        mode = random.random()
        if mode < 0.45:
            draw.ellipse(box, fill=255)
        elif mode < 0.85:
            try: draw.rounded_rectangle(box, radius=random.randint(0, max(1, min(w, h)//4)), fill=255)
            except Exception: draw.rectangle(box, fill=255)
        else:
            pts=[]
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            for k in range(random.randint(5, 9)):
                ang = 2*math.pi*k/random.randint(6, 10) + random.uniform(-0.25, 0.25)
                rr = random.uniform(0.35, 0.55)
                pts.append((cx + math.cos(ang)*w*rr, cy + math.sin(ang)*h*rr))
            draw.polygon(pts, fill=255)
        if random.random() < 0.75:
            mask = mask.filter(ImageFilter.GaussianBlur(random.uniform(0.25, 1.6))).point(lambda p: 255 if p > 80 else 0)
        return mask, box, bucket

    def make(self) -> dict[str, Any]:
        host_q = random.choice([86, 90, 94, 98])
        donor_q = random.choice([38, 45, 55, 65, 75])
        final_q = host_q if random.random() < 0.8 else random.choice([78, 86, 94])
        host = jpeg_roundtrip(self._load_square(), host_q)
        donor = jpeg_roundtrip(self._load_square(), donor_q)
        donor = jitter(donor, random.uniform(0.04, 0.14))
        if random.random() < 0.4:
            donor = donor.filter(ImageFilter.GaussianBlur(random.uniform(0.15, 0.8)))
        mask, box, bucket = self._region()
        crop = donor.crop(box)
        alpha = mask.crop(box).filter(ImageFilter.GaussianBlur(random.uniform(0.2, 1.3)))
        mixed = host.copy()
        mixed.paste(crop, box[:2], alpha)
        mixed = jpeg_roundtrip(mixed, final_q)
        dct, qtable, available = jpeg_info(mixed, final_q)
        diff = abs(host_q - donor_q)
        q_class = 0 if diff < 10 else (1 if diff < 20 else (2 if diff < 35 else 3))
        return {
            "image": pil_to_tensor(mixed).float(),
            "mask": (pil_to_tensor(mask) > 0.5).float(),
            "dct": dct.float(),
            "qtable": qtable.float(),
            "dct_available": available.float(),
            "q_class": torch.tensor(q_class, dtype=torch.long),
            "meta": {"host_q": host_q, "donor_q": donor_q, "final_q": final_q, "bucket": bucket},
        }


def stack(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([s["image"] for s in samples]).cuda(non_blocking=True),
        "mask": torch.stack([s["mask"] for s in samples]).cuda(non_blocking=True),
        "dct": torch.stack([s["dct"] for s in samples]).cuda(non_blocking=True),
        "qtable": torch.stack([s["qtable"] for s in samples]).cuda(non_blocking=True),
        "dct_available": torch.stack([s["dct_available"] for s in samples]).cuda(non_blocking=True),
        "q_class": torch.stack([s["q_class"] for s in samples]).cuda(non_blocking=True),
        "meta": [s["meta"] for s in samples],
    }


class CompressionBranchHead(nn.Module):
    def __init__(self, c1: int = 64, c4: int = 512, hidden: int = 96, classes: int = 4) -> None:
        super().__init__()
        self.mask_head = nn.Sequential(nn.Conv2d(c1, hidden, 3, padding=1, bias=False), nn.GroupNorm(8, hidden), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1, bias=False), nn.GroupNorm(8, hidden), nn.GELU(), nn.Conv2d(hidden, 1, 1))
        self.cls_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c4, hidden), nn.GELU(), nn.Linear(hidden, classes))
    def forward(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mask_head(feats[0]), self.cls_head(feats[3])


def dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p=logits.sigmoid().flatten(1); y=target.flatten(1)
    inter=(p*y).sum(1); den=p.sum(1)+y.sum(1)
    return (1-(2*inter+1)/(den+1)).mean()


def set_pretrainable(model, head) -> list[str]:
    for p in model.parameters(): p.requires_grad_(False)
    names=[]
    for n,p in model.named_parameters():
        trainable = n.startswith("dct_branch.")
        p.requires_grad_(trainable)
        if trainable: names.append(n)
    for p in head.parameters(): p.requires_grad_(True)
    model.eval(); model.dct_branch.train(); model.forensic_stream.eval(); head.train()
    return names


def pretrain(model, cfg, outdir: Path, summary: dict, epochs: int, steps: int, batch_size: int, lr: float) -> None:
    records = read_manifest(cfg["data"]["manifest"], cfg["data"]["root"], "train")
    gen = DonorHostSpliceGenerator(records, int(cfg["data"].get("image_size", 640)))
    head = CompressionBranchHead().cuda()
    names = set_pretrainable(model, head)
    opt = torch.optim.AdamW([
        {"params":[p for n,p in model.named_parameters() if n.startswith("dct_branch.") and p.requires_grad], "lr": lr, "weight_decay":0.01, "name":"dct_branch."},
        {"params":list(head.parameters()), "lr": lr, "weight_decay":0.01, "name":"compression_branch_head."},
    ])
    history=[]
    summary["pretrain"]={"task":"cross_image_donor_host_compression_discrepancy", "epochs":epochs, "steps_per_epoch":steps, "batch_size":batch_size, "lr":lr, "trainable_count":sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable_examples":names[:80], "history":history}
    for epoch in range(1, epochs+1):
        started=time.time(); totals={}
        for step in range(1, steps+1):
            batch=stack([gen.make() for _ in range(batch_size)])
            feats=model.dct_branch(batch["dct"], batch["qtable"])
            logits_low, q_logits=head(feats)
            target_low=F.interpolate(batch["mask"], size=logits_low.shape[-2:], mode="nearest")
            bce=F.binary_cross_entropy_with_logits(logits_low, target_low)
            dsc=dice(logits_low, target_low)
            ce=F.cross_entropy(q_logits, batch["q_class"])
            loss=bce+dsc+0.2*ce
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],1.0,error_if_nonfinite=True); opt.step()
            vals={"loss":loss,"bce":bce,"dice":dsc,"quality_ce":ce}
            for k,v in vals.items(): totals[k]=totals.get(k,0.0)+float(v.detach())
            if step==1 or step%50==0 or step==steps:
                elapsed=time.time()-started
                progress={"stage":"donor_host_pretrain","epoch":epoch,"epochs":epochs,"step":step,"steps":steps,"loss":totals["loss"]/step,"bce":totals["bce"]/step,"dice":totals["dice"]/step,"quality_ce":totals["quality_ce"]/step,"elapsed_minutes":elapsed/60,"eta_minutes":elapsed/step*(steps-step)/60}
                dump(outdir/"progress.yaml",progress); print("[pretrain]",json.dumps(progress),flush=True)
        row={"epoch":epoch,"losses":{k:v/steps for k,v in totals.items()},"elapsed_seconds":time.time()-started}
        history.append(row); atomic_torch_save({"epoch":epoch,"model":e21.copy_state(model),"head":head.state_dict(),"config":cfg,"phase":"E26_donor_host_pretrain","pretrain_losses":row["losses"]}, outdir/"pretrained_compression_branch.pt"); dump(outdir/"summary.yaml", summary)


def real_forward(model,batch):
    return model(batch["image"].cuda(non_blocking=True), dct=batch["dct"].cuda(non_blocking=True), qtable=batch["qtable"].cuda(non_blocking=True), dct_available=batch["dct_available"].cuda(non_blocking=True))


def objective(out,batch,seg,cfg):
    target=batch["mask"].cuda(non_blocking=True); label=batch["label"].cuda(non_blocking=True)
    terms=seg(out,target,label); aux=dher_v2_aux_loss(out,target,cfg); deep=weighted_deep_supervision_loss(out,target,cfg); tv=tversky_loss(out["mask_logits"],target,float(cfg["loss"].get("tversky_alpha_fp",0.3)),float(cfg["loss"].get("tversky_beta_fn",0.7)))
    terms["total"]=terms["total"]+aux["dher_aux_total"]+deep+float(cfg["loss"]["weights"].get("tversky",0.0))*tv
    terms.update(weighted_deep_supervision=deep,tversky=tv)
    return terms


def set_adaptable(model):
    prefixes=("dct_branch.","dct_inject_adapters.","qtr.","dher_visibility.","dher_calibrated_writeback.")
    names=[]
    for n,p in model.named_parameters():
        trainable=n.startswith(prefixes)
        p.requires_grad_(trainable)
        if trainable: names.append(n)
    model.eval()
    for module_name in ("dct_branch","dct_inject_adapters","qtr","dher_visibility","dher_calibrated_writeback"):
        m=getattr(model,module_name,None)
        if m is not None: m.train()
    # Explicitly keep original forensic stream, HCV, DHER core, decoder, and heads frozen.
    assert not any(p.requires_grad for p in model.forensic_stream.parameters())
    assert not any(p.requires_grad for p in model.hcv2.parameters())
    assert not any(p.requires_grad for p in model.dher.parameters())
    assert not any(p.requires_grad for p in model.decoder.parameters())
    return names


def opt_adapt(model):
    specs=[("dct_branch.",3e-6,0.01),("dct_inject_adapters.",5e-6,0.01),("qtr.",3e-6,0.01),("dher_visibility.",1e-5,0.01),("dher_calibrated_writeback.",1e-5,0.01)]
    groups=[]
    for prefix,lr,wd in specs:
        params=[p for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if params: groups.append({"params":params,"lr":lr,"weight_decay":wd,"name":prefix})
    return torch.optim.AdamW(groups)


def compact(result):
    m=result["metrics"]
    out={"fake_oracle_f1":m["fake"]["oracle"]["f1"],"fake_oracle_threshold":m["fake"]["oracle"].get("threshold"),"fake_fixed_f1":m["fake"]["fixed"]["f1"],"fake_fixed_precision":m["fake"]["fixed"]["precision"],"fake_fixed_recall":m["fake"]["fixed"]["recall"],"all_oracle_f1":m["all"]["oracle"]["f1"]}
    for key in ["Tp_S:lt_1","Tp_S:1_5","Tp_S:5_20","Tp_S:gte_20","Tp_D:gte_20"]:
        if key in m: out[key]={"oracle_f1":m[key]["oracle"]["f1"],"fixed_f1":m[key]["fixed"]["f1"],"fixed_precision":m[key]["fixed"].get("precision"),"fixed_recall":m[key]["fixed"].get("recall")}
    return out


def finetune(model,cfg,outdir:Path,summary:dict,epochs:int):
    names=set_adaptable(model); opt=opt_adapt(model); seg=ReTraceLoss.from_config(clean_loss_config(cfg["loss"])).cuda(); loader=train_loader_for(cfg); val_records=e21.read_manifest(cfg["data"]["manifest"],cfg["data"]["root"],"val"); acc=int(cfg["training"].get("gradient_accumulation_steps",4)); best={"value":-math.inf,"epoch":None}
    summary["finetune"]={"policy":"train compression branch + QTR + DHER visibility; freeze forensic/HCV/DHER core/decoder", "epochs":epochs,"trainable_count":sum(p.numel() for p in model.parameters() if p.requires_grad),"trainable_examples":names[:100],"optimizer_groups":[{"name":g.get("name"),"lr":g.get("lr"),"params":sum(p.numel() for p in g["params"])} for g in opt.param_groups],"history":[]}
    for epoch in range(1,epochs+1):
        started=time.time(); totals={}; opt.zero_grad(set_to_none=True); summary.update(status="finetuning",current_epoch=epoch); dump(outdir/"summary.yaml",summary)
        for step,batch in enumerate(loader,1):
            out=real_forward(model,batch); terms=objective(out,batch,seg,cfg)
            if not torch.isfinite(terms["total"]): raise FloatingPointError(f"nonfinite epoch={epoch} step={step}")
            (terms["total"]/acc).backward()
            if step%acc==0 or step==len(loader): torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],0.5,error_if_nonfinite=True); opt.step(); opt.zero_grad(set_to_none=True)
            for k,v in terms.items(): totals[k]=totals.get(k,0.0)+float(v.detach())
            if step==1 or step%50==0 or step==len(loader):
                elapsed=time.time()-started; progress={"stage":"selective_adapt","epoch":epoch,"epochs":epochs,"step":step,"steps":len(loader),"loss":totals["total"]/step,"elapsed_minutes":elapsed/60,"eta_minutes":elapsed/step*(len(loader)-step)/60}; dump(outdir/"progress.yaml",progress); print("[finetune]",json.dumps(progress),flush=True)
        result=e21.evaluate(model,val_records,cfg,f"epoch_{epoch:02d}_original_val",outdir); score=result["metrics"]["fake"]["oracle"]["f1"]; row={"epoch":epoch,"selection_value":score,"delta_vs_source":score-SOURCE_BEST_F1,"validation_compact":compact(result),"validation":result,"train_losses":{k:v/len(loader) for k,v in totals.items()},"elapsed_seconds":time.time()-started}
        if score>best["value"]:
            best={"value":score,"epoch":epoch}; atomic_torch_save({"epoch":epoch,"model":e21.copy_state(model),"optimizer":opt.state_dict(),"config":cfg,"eval_variant":"raw","phase":"E26_selective_adapt","best_metric":score,"selection_threshold":result["metrics"]["fake"]["oracle"].get("threshold")},outdir/"best.pt"); row["is_best"]=True; summary.update(best_F1=score,best_epoch=epoch,best_checkpoint=str(outdir/"best.pt"))
        else: row["is_best"]=False
        summary["finetune"]["history"].append(row); dump(outdir/"summary.yaml",summary); print("[epoch_done]",json.dumps({"epoch":epoch,"fake_F1":score,"delta_vs_source":score-SOURCE_BEST_F1,"best_F1":best["value"],"TpS_gte20":row["validation_compact"].get("Tp_S:gte_20")},ensure_ascii=False),flush=True)
        if epoch==1 and score < SOURCE_BEST_F1 - 0.004:
            summary.update(status="stopped",stop_reason="epoch1_drop_exceeded_0p004_in_E26_selective_adapt"); dump(outdir/"summary.yaml",summary); print("[early_stop] epoch1_drop_exceeded_0p004",flush=True); return best
    return best


def main(args):
    outdir=Path(args.out); outdir.mkdir(parents=True,exist_ok=True); summary_path=outdir/"summary.yaml"
    if summary_path.exists(): raise RuntimeError(f"refuse duplicate run: {summary_path}")
    summary={"status":"initializing","experiment":EXPERIMENT,"source":"E24 best","test_run":False,"selection":"val_original_grid_fake_pooled_F1_best","stop_reason":None}
    dump(summary_path,summary)
    try:
        set_seed(args.seed); source=torch.load(args.checkpoint,map_location="cpu"); cfg=copy.deepcopy(source["config"]); cfg["experiment"]=EXPERIMENT; cfg["training"].update(ema=False,amp=False,batch_size=1,gradient_accumulation_steps=4,grad_clip=0.5,selection_metric="val_original_grid_fake_pooled_F1_best"); dump(outdir/"config.yaml",cfg)
        model=build_final_model(cfg["model"]).cuda(); model.load_state_dict(source["model"],strict=True); summary.update(source_checkpoint=args.checkpoint,source_best_F1=float(source.get("best_metric",SOURCE_BEST_F1)),protocol="original_GT_grid; probability_bilinear; valid_GT>=0; raw; thresholds=101; no_TTA")
        dump(summary_path,summary)
        pretrain(model,cfg,outdir,summary,args.pretrain_epochs,args.pretrain_steps,args.pretrain_batch_size,args.pretrain_lr)
        summary.update(status="pretrain_completed",pretrained_compression_branch=str(outdir/"pretrained_compression_branch.pt")); dump(summary_path,summary)
        best=finetune(model,cfg,outdir,summary,args.finetune_epochs)
        if (outdir/"best.pt").exists():
            val_records=e21.read_manifest(cfg["data"]["manifest"],cfg["data"]["root"],"val"); best_state=torch.load(outdir/"best.pt",map_location="cpu"); model.load_state_dict(best_state["model"],strict=True); full=e21.evaluate(model,val_records,cfg,"best_original_val",outdir); model.disable_dct_evidence=True; ablate=e21.evaluate(model,val_records,cfg,"best_dct_ablation_original_val",outdir); model.disable_dct_evidence=False; summary.update(final_full_metrics=full,final_compact=compact(full),final_dct_ablation_metrics=ablate,dct_contribution_fake_F1=full["metrics"]["fake"]["oracle"]["f1"]-ablate["metrics"]["fake"]["oracle"]["f1"],acceptance_pass=bool(full["metrics"]["fake"]["oracle"]["f1"]>=0.800))
        if summary.get("status")!="stopped": summary.update(status="completed",stop_reason=f"fixed_pretrain_{args.pretrain_epochs}_adapt_{args.finetune_epochs}_budget_completed")
        summary.update(best_F1=best["value"],best_epoch=best["epoch"]); dump(summary_path,summary); print("[E26_COMPLETE]",json.dumps({"status":summary["status"],"best_F1":summary.get("best_F1"),"delta_vs_source":(summary.get("best_F1") or 0.0)-SOURCE_BEST_F1,"acceptance_pass":summary.get("acceptance_pass"),"stop_reason":summary.get("stop_reason")},ensure_ascii=False),flush=True)
    except BaseException:
        summary.update(status="failed",stop_reason=traceback.format_exc()); dump(summary_path,summary); raise


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--out",required=True); p.add_argument("--pretrain-epochs",type=int,default=4); p.add_argument("--pretrain-steps",type=int,default=2200); p.add_argument("--pretrain-batch-size",type=int,default=2); p.add_argument("--pretrain-lr",type=float,default=5e-6); p.add_argument("--finetune-epochs",type=int,default=3); p.add_argument("--seed",type=int,default=3407); main(p.parse_args())
