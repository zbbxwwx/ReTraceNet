from __future__ import annotations
import argparse, io, os, random, tempfile, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve
import yaml, jpegio

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from retracenet.native_jpeg_hrnet import NativeJPEGForensicHRNet

def enc(im,q,sub=0):
    b=io.BytesIO(); im.save(b,'JPEG',quality=int(q),subsampling=int(sub),optimize=False); return b.getvalue()

def native(blob):
    fd,p=tempfile.mkstemp(suffix='.jpg')
    try:
        with os.fdopen(fd,'wb') as f:f.write(blob)
        j=jpegio.read(p); c=np.asarray(j.coef_arrays[0],np.float32); q=np.asarray(j.quant_tables[0],np.float32)
    finally:
        try:os.unlink(p)
        except FileNotFoundError:pass
    h=c.shape[0]//8*8; w=c.shape[1]//8*8
    d=c[:h,:w].reshape(h//8,8,w//8,8).transpose(0,2,1,3).reshape(h//8,w//8,64)
    return torch.from_numpy(d.transpose(2,0,1).copy()),torch.from_numpy(q.copy()).unsqueeze(0)

def crop(im,size,rng):
    im=im.convert('RGB'); w,h=im.size; s=max(size/w,size/h,1)
    if s>1: im=im.resize((round(w*s),round(h*s)),Image.Resampling.BICUBIC)
    w,h=im.size; x=rng.randrange(w-size+1); y=rng.randrange(h-size+1)
    return im.crop((x,y,x+size,y+size))

class LocalHistory(Dataset):
    def __init__(self,paths,size=256,seed=3407,train=True,repeats=20):
        self.paths=list(map(str,paths)); self.size=size; self.seed=seed; self.train=train; self.repeats=repeats; self.epoch=0
    def set_epoch(self,e):self.epoch=e
    def __len__(self):return len(self.paths)*self.repeats*2
    def __getitem__(self,i):
        pair=i//2; base=pair%len(self.paths); rep=pair//len(self.paths); salt=self.epoch if self.train else 99991
        r=random.Random(self.seed+base*7919+rep*104729+salt*1000003)
        im=crop(Image.open(self.paths[base]),self.size,r)
        qfinal=r.choice((75,80,85,90,95)); sub=r.choice((0,2)); positive=(i%2)==1
        # Both classes undergo the same decode/paste/final-encode operations. Only history differs.
        direction=rep%2; low=r.choice((50,55,60,65,70)); high=r.choice((85,90,95,100)); qbg=low if direction==0 else high; subfirst=r.choice((0,2)); bg=Image.open(io.BytesIO(enc(im,qbg,subfirst))).convert('RGB')
        donor_q=((high if direction==0 else low) if positive else qbg)
        donor=Image.open(io.BytesIO(enc(im,donor_q,subfirst))).convert('RGB')
        bw=r.randrange(6,22)*8; bh=r.randrange(6,22)*8
        x=r.randrange(0,(self.size-bw)//8+1)*8; y=r.randrange(0,(self.size-bh)//8+1)*8
        composed=bg.copy(); composed.paste(donor.crop((x,y,x+bw,y+bh)),(x,y))
        d,q=native(enc(composed,qfinal,sub))
        mask=torch.zeros(1,self.size//4,self.size//4)
        if positive: mask[:,y//4:(y+bh)//4,x//4:(x+bw)//4]=1
        return d,q,mask,torch.tensor(float(positive))

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=NativeJPEGForensicHRNet(); self.head=nn.Conv2d(64,1,1)
        for p in self.encoder.residual_bank.parameters():p.requires_grad=False
        for p in self.encoder.residual_stem.parameters():p.requires_grad=False
    def forward(self,d,q):
        b=d.shape[0]; z=torch.zeros(b,3,d.shape[-2]*8,d.shape[-1]*8,device=d.device)
        return self.head(self.encoder(z,d,q,torch.ones(b,1,device=d.device))[0])

def metrics(model,loader):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for d,q,y,_ in loader:
            p=torch.sigmoid(model(d.cuda(),q.cuda())).cpu(); ys.append(y.flatten().numpy()); ps.append(p.flatten().numpy())
    y=np.concatenate(ys); p=np.concatenate(ps); auc=roc_auc_score(y,p); pr,rc,th=precision_recall_curve(y,p)
    f=2*pr*rc/(pr+rc+1e-12); k=int(np.nanargmax(f)); return {'auc':float(auc),'f1':float(f[k]),'precision':float(pr[k]),'recall':float(rc[k]),'threshold':float(th[min(k,len(th)-1)])}

def main():
    a=argparse.ArgumentParser(); a.add_argument('--images',required=True);a.add_argument('--out',required=True);a.add_argument('--global-pretrain',required=True)
    a.add_argument('--epochs',type=int,default=20);a.add_argument('--batch-size',type=int,default=8);a.add_argument('--workers',type=int,default=4);a.add_argument('--smoke',action='store_true');a=a.parse_args()
    random.seed(3407);np.random.seed(3407);torch.manual_seed(3407)
    import json
    split=json.loads((Path(a.global_pretrain).parent/'split.json').read_text()); paths=[]
    va=split['val']; tr=split['train']; reps=20
    assert not set(tr)&set(va)
    if a.smoke:tr=tr[:4];va=va[:2];reps=2;a.epochs=1;a.workers=0;a.batch_size=2
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'split.json').write_text(json.dumps({'train':list(map(str,tr)),'val':list(map(str,va))},indent=2))
    td=LocalHistory(tr,train=True,repeats=reps);vd=LocalHistory(va,train=False,repeats=4)
    tl=DataLoader(td,a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True);vl=DataLoader(vd,a.batch_size,num_workers=a.workers)
    m=Model().cuda(); ck=torch.load(a.global_pretrain,map_location='cpu'); res=m.encoder.load_state_dict(ck['encoder'],strict=True);print({'loaded':a.global_pretrain,'missing':res.missing_keys,'unexpected':res.unexpected_keys},flush=True)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],3e-4,weight_decay=1e-4);lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0],device='cuda'));best=-1;hist=[]
    for ep in range(1,a.epochs+1):
        td.set_epoch(ep);m.train();ls=0;n=0;t=time.time()
        for d,q,y,_ in tl:
            d,q,y=d.cuda(non_blocking=True),q.cuda(non_blocking=True),y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);z=m(d,q);bce=lossfn(z,y);p=torch.sigmoid(z);inter=(p*y).sum((1,2,3));dice=1-((2*inter+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)).mean();loss=bce+0.5*dice;loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();ls+=loss.item()*len(d);n+=len(d)
        met=metrics(m,vl);row={'epoch':ep,'train_loss':ls/n,'val':met,'seconds':time.time()-t};hist.append(row);print(row,flush=True)
        torch.save({'encoder':m.encoder.state_dict(),'head':m.head.state_dict(),'epoch':ep,'val':met},out/'last.pt')
        if met['f1']>best:best=met['f1'];torch.save({'encoder':m.encoder.state_dict(),'head':m.head.state_dict(),'epoch':ep,'val':met,'task':'local_compression_history_mask'},out/'best.pt')
        (out/'summary.yaml').write_text(yaml.safe_dump({'status':'training','best_f1':best,'history':hist},sort_keys=False))
    (out/'complete.json').write_text(json.dumps({'best_f1':best,'epochs':a.epochs}))
    print('LOCAL_PRETRAIN_COMPLETE',best,flush=True)
if __name__=='__main__':main()
