from __future__ import annotations
import argparse, io, json, os, random, tempfile, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import yaml
import jpegio

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from retracenet.native_jpeg_hrnet import NativeJPEGForensicHRNet

QUALITY=list(range(50,101))
SUBS=(0,2)  # Pillow: 0=4:4:4, 2=4:2:0

def encode(im,quality,subsampling):
    b=io.BytesIO(); im.save(b,format='JPEG',quality=int(quality),subsampling=int(subsampling),optimize=False)
    return b.getvalue()

def native_from_bytes(blob):
    fd,path=tempfile.mkstemp(suffix='.jpg')
    try:
        with os.fdopen(fd,'wb') as f: f.write(blob)
        j=jpegio.read(path)
        coef=np.asarray(j.coef_arrays[0],dtype=np.float32)
        qt=np.asarray(j.quant_tables[0],dtype=np.float32)
    finally:
        try: os.unlink(path)
        except FileNotFoundError: pass
    h=(coef.shape[0]//8)*8; w=(coef.shape[1]//8)*8; coef=coef[:h,:w]
    blocks=coef.reshape(h//8,8,w//8,8).transpose(0,2,1,3).reshape(h//8,w//8,64)
    return torch.from_numpy(blocks.transpose(2,0,1).copy()),torch.from_numpy(qt.copy()).unsqueeze(0)

def crop_image(im,size,rng):
    im=im.convert('RGB')
    w,h=im.size
    scale=max(size/w,size/h,1.0)
    if scale>1: im=im.resize((int(round(w*scale)),int(round(h*scale))),Image.Resampling.BICUBIC)
    w,h=im.size
    x=rng.randrange(max(1,w-size+1));y=rng.randrange(max(1,h-size+1))
    return im.crop((x,y,x+size,y+size))

class PairedDJPEG(Dataset):
    def __init__(self,paths,size=256,seed=3407,training=True):
        self.paths=[str(x) for x in paths];self.size=size;self.seed=seed;self.training=training;self.epoch=0
    def set_epoch(self,e):self.epoch=int(e)
    def __len__(self):return len(self.paths)*2
    def __getitem__(self,index):
        base=index//2; label=index%2
        # Pair labels share crop, Q2, and final subsampling exactly.
        salt=0
        rng=random.Random(self.seed+salt*10000019+base*97)
        im=crop_image(Image.open(self.paths[base]),self.size,rng)
        q2=95; sub2=0
        if label==0:
            final=encode(im,q2,sub2)
        else:
            q1=60
            sub1=0
            first=encode(im,q1,sub1)
            final=encode(Image.open(io.BytesIO(first)).convert('RGB'),q2,sub2)
        dct,qt=native_from_bytes(final)
        return dct,qt,torch.tensor(label,dtype=torch.long),torch.tensor([q2,sub2],dtype=torch.long)

class DJPEGClassifier(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=NativeJPEGForensicHRNet();self.head=nn.Linear(512,2)
        for p in self.encoder.residual_bank.parameters():p.requires_grad=False
        for p in self.encoder.residual_stem.parameters():p.requires_grad=False
    def forward(self,dct,qt):
        b=dct.shape[0];h=dct.shape[-2]*8;w=dct.shape[-1]*8
        # Zero RGB makes the residual anchor exactly zero; Stage A learns JPEG only.
        image=torch.zeros(b,3,h,w,device=dct.device,dtype=dct.dtype)
        feats=self.encoder(image,dct,qt,torch.ones(b,1,device=dct.device))
        return self.head(feats[-1].mean((-2,-1)))

def evaluate(model,loader,device):
    model.eval(); ys=[];ps=[];correct=total=0;loss_sum=0.;ce=nn.CrossEntropyLoss()
    with torch.no_grad():
        for d,q,y,_ in loader:
            d,q,y=d.to(device),q.to(device),y.to(device);z=model(d,q);loss=ce(z,y)
            prob=z.softmax(1)[:,1];ys.extend(y.cpu().tolist());ps.extend(prob.cpu().tolist())
            correct+=(z.argmax(1)==y).sum().item();total+=len(y);loss_sum+=loss.item()*len(y)
    return {'loss':loss_sum/max(total,1),'accuracy':correct/max(total,1),'auc':roc_auc_score(ys,ps)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--images',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--crop-size',type=int,default=256);ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--seed',type=int,default=3407)
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    paths=sorted([p for p in Path(a.images).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'}])
    rng=random.Random(a.seed);rng.shuffle(paths);paths=paths[:128];nval=16;val=paths[:nval];train=paths[nval:]
    if a.smoke: train=train[:4];val=val[:2];a.epochs=1;a.workers=0;a.batch_size=2
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'split.json').write_text(json.dumps({'train':[str(x) for x in train],'val':[str(x) for x in val]},indent=2))
    tr=PairedDJPEG(train,a.crop_size,a.seed,True);va=PairedDJPEG(val,a.crop_size,a.seed,False)
    tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True,persistent_workers=False)
    vl=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=False)
    dev=torch.device('cuda');model=DJPEGClassifier().to(dev);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=a.lr*.05);ce=nn.CrossEntropyLoss();best=-1;hist=[]
    print({'train_originals':len(train),'val_originals':len(val),'samples_per_epoch':len(tr),'epochs':a.epochs,'exposures':len(tr)*a.epochs},flush=True)
    for ep in range(1,a.epochs+1):
        tr.set_epoch(ep);model.train();tot=correct=0;ls=0.;t=time.time()
        for step,(d,q,y,meta) in enumerate(tl,1):
            d,q,y=d.to(dev,non_blocking=True),q.to(dev,non_blocking=True),y.to(dev,non_blocking=True)
            opt.zero_grad(set_to_none=True);z=model(d,q);loss=ce(z,y);loss.backward();
            if step==1:
                gn=sum(float(p.grad.norm()) for p in model.encoder.parameters() if p.grad is not None)
                hg=sum(float(p.grad.norm()) for p in model.head.parameters() if p.grad is not None)
                print({'epoch':ep,'encoder_grad_norm_sum':gn,'head_grad_norm_sum':hg,'logit_mean':z.detach().mean(0).cpu().tolist(),'logit_std':z.detach().std(0).cpu().tolist()},flush=True)
            nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            ls+=loss.item()*len(y);correct+=(z.argmax(1)==y).sum().item();tot+=len(y)
            if step==1 or step%250==0:print(f'[djpeg] ep={ep} step={step}/{len(tl)} loss={ls/tot:.5f} acc={correct/tot:.4f}',flush=True)
        metrics=evaluate(model,vl,dev);sched.step();row={'epoch':ep,'train_loss':ls/tot,'train_accuracy':correct/tot,'val':metrics,'lr':opt.param_groups[0]['lr'],'seconds':time.time()-t};hist.append(row)
        payload={'status':'training','best_auc':max(best,metrics['auc']),'best_epoch':max(hist,key=lambda x:x['val']['auc'])['epoch'],'history':hist,'train_originals':len(train),'val_originals':len(val),'samples_per_epoch':len(tr)}
        (out/'summary.yaml').write_text(yaml.safe_dump(payload,sort_keys=False));torch.save({'encoder':model.encoder.state_dict(),'head':model.head.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'history':hist},out/'last.pt')
        if metrics['auc']>best:
            best=metrics['auc'];torch.save({'encoder':model.encoder.state_dict(),'head':model.head.state_dict(),'epoch':ep,'val':metrics,'input_layout':'quantized_y_dct_64_block_grid','task':'single_vs_double_jpeg'},out/'best.pt')
        print(row,flush=True)
    print('DJPEG_PRETRAIN_COMPLETE',best,flush=True)
if __name__=='__main__':main()
