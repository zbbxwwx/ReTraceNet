import argparse, json
from pathlib import Path
import numpy as np
import yaml
from PIL import Image


def load_manifest(directory):
    return json.loads((Path(directory) / "manifest.json").read_text())


def collect(directory):
    positive = np.zeros(1001, np.int64); negative = np.zeros(1001, np.int64); authentic = np.zeros(1001, np.int64)
    entries = []; image_pos = np.zeros(1001, np.int64); image_neg = np.zeros(1001, np.int64)
    for row in load_manifest(directory):
        data = np.load(row["prediction"]); probability = np.asarray(data["map"], np.float32); label = int(row["label"])
        if label:
            with Image.open(row["mask"]) as handle: target = np.asarray(handle.convert("L")) > 127
        else: target = np.zeros(probability.shape, bool)
        if probability.shape != target.shape:
            import torch, torch.nn.functional as F
            probability = F.interpolate(torch.from_numpy(probability)[None,None], size=target.shape, mode="bilinear", align_corners=False)[0,0].numpy()
        bins = np.minimum((probability * 1000).astype(np.int16), 1000)
        if label:
            positive += np.bincount(bins[target], minlength=1001); negative += np.bincount(bins[~target], minlength=1001)
        else: authentic += np.bincount(bins.ravel(), minlength=1001)
        score = float(data["score"]); (image_pos if label else image_neg)[min(1000, int(score * 1000))] += 1
        entries.append((probability, target, label))
    return positive, negative, authentic, image_pos, image_neg, entries


def counts(tp, fp, fn):
    return {"precision": float(tp/max(1,tp+fp)), "recall": float(tp/max(1,tp+fn)), "f1": float(2*tp/max(1,2*tp+fp+fn))}
def scan(p,n):
    tp=np.cumsum(p[::-1])[::-1];fp=np.cumsum(n[::-1])[::-1];fn=p.sum()-tp;f=2*tp/np.maximum(1,2*tp+fp+fn);k=int(np.argmax(f));return {**counts(tp[k],fp[k],fn[k]),"threshold":k/1000}
def auc(p,n):
    below=np.cumsum(n)-n;return float(((p*below).sum()+.5*(p*n).sum())/max(1,p.sum()*n.sum()))
def main(a):
    vp,vn,va,vip,vin,_=collect(a.val_pred);pb=scan(vp,vn);ib=scan(vip,vin)
    tp,tn,ta,tip,tin,entries=collect(a.test_pred);threshold=pb["threshold"];c=[0,0,0];af=ap=0
    for prob,target,label in entries:
        pred=prob>=threshold
        if label:c[0]+=int((pred&target).sum());c[1]+=int((pred&~target).sum());c[2]+=int((~pred&target).sum())
        else:af+=int(pred.sum());ap+=pred.size
    fake=counts(*c);allm=counts(c[0],c[1]+af,c[2]);ik=int(ib['threshold']*1000);itp=np.cumsum(tip[::-1])[::-1][ik];ifp=np.cumsum(tin[::-1])[::-1][ik]
    result={"status":"completed","model":"TruFor","threshold_source":"FantasticReality_val","pixel_threshold":threshold,
            "fake_only_pooled":{**fake,"auc":auc(tp,tn),"oracle_diagnostic":scan(tp,tn)},"authentic_pixel_fpr":float(af/max(1,ap)),
            "all_images_pooled":{**allm,"auc":auc(tp,tn+ta)},"image_level_all":{**counts(itp,ifp,tip.sum()-itp),"auc":auc(tip,tin)},
            "validation_selection":{"pixel":pb,"image":ib}}
    Path(a.out).write_text(yaml.safe_dump(result,sort_keys=False));print(yaml.safe_dump(result,sort_keys=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--val-pred',required=True);p.add_argument('--test-pred',required=True);p.add_argument('--out',required=True);main(p.parse_args())
