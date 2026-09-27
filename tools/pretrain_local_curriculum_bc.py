from __future__ import annotations

import argparse, io, json, os, tempfile
from pathlib import Path
import jpegio, numpy as np, torch
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from retracenet.native_jpeg_hrnet import NativeJPEGForensicHRNet

def encode(im,q,sub):
    b=io.BytesIO();im.save(b,'JPEG',quality=q,subsampling=sub,optimize=False);return b.getvalue()
def native(blob):
    fd,name=tempfile.mkstemp(suffix='.jpg')
    try:
        with os.fdopen(fd,'wb') as f:f.write(blob)
        j=jpegio.read(name);c=np.asarray(j.coef_arrays[0],np.float32);q=np.asarray(j.quant_tables[0],np.float32)
    finally:
        try:os.unlink(name)
        except FileNotFoundError:pass
    h=c.shape[0]//8*8;w=c.shape[1]//8*8;b=c[:h,:w].reshape(h//8,8,w//8,8).transpose(0,2,1,3).reshape(h//8,w//8,64)
    return torch.from_numpy(b.transpose(2,0,1).copy()),torch.from_numpy(q.copy())[None]
def center_crop(path,size=256):
    im=Image.open(path).convert('RGB');w,h=im.size;s=max(size/w,size/h,1)
    if s>1:im=im.resize((round(w*s),round(h*s)),Image.Resampling.BICUBIC)
    w,h=im.size;return im.crop(((w-size)//2,(h-size)//2,(w-size)//2+size,(h-size)//2+size))

B_TRAIN=[(50,90),(60,95),(70,90),(90,70),(95,80),(80,95)]
B_VAL=[(55,95),(65,90),(85,70),(95,75)]
C_TRAIN=[(50,95,0,128),(60,90,2,96),(70,95,2,64),(90,70,0,48),(95,80,2,32),(80,95,0,128)]
C_VAL=[(55,90,2,40),(65,95,0,72),(85,70,2,56),(95,75,0,104)]

class Paired(Dataset):
    def __init__(self,paths,mode,train):
        self.samples=[];specs=(B_TRAIN if train else B_VAL) if mode=='B' else (C_TRAIN if train else C_VAL)
        for image_index,path in enumerate(paths):
            clean=center_crop(path)
            for spec_index,spec in enumerate(specs):
                if mode=='B':q1,q2=spec;sub=0;region=128
                else:q1,q2,sub,region=spec
                x=((24+((image_index+spec_index*3)%9)*16)//8)*8;y=((24+((image_index*3+spec_index)%9)*16)//8)*8
                x=min(x,256-region);y=min(y,256-region);x=x//8*8;y=y//8*8
                donor=Image.open(io.BytesIO(encode(clean,q1,sub))).convert('RGB')
                for positive in (0,1):
                    composed=clean.copy();source=donor if positive else clean
                    composed.paste(source.crop((x,y,x+region,y+region)),(x,y));dct,qt=native(encode(composed,q2,sub))
                    mask=torch.zeros(1,64,64);roi=torch.zeros_like(mask);roi[:,y//4:(y+region)//4,x//4:(x+region)//4]=1
                    if positive:mask.copy_(roi)
                    key=f'q1={q1}_q2={q2}_sub={sub}_size={region}'
                    self.samples.append((dct,qt,mask,roi,positive,key))
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):return self.samples[i]
class Model(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=NativeJPEGForensicHRNet();self.head=nn.Conv2d(64,1,1)
        for p in self.encoder.residual_bank.parameters():p.requires_grad=False
        for p in self.encoder.residual_stem.parameters():p.requires_grad=False
    def forward(self,d,q):
        b=d.shape[0];z=torch.zeros(b,3,d.shape[-2]*8,d.shape[-1]*8,device=d.device)
        return self.head(self.encoder(z,d,q,torch.ones(b,1,device=d.device))[0])
def score(y,p):
    y=y.numpy().ravel();p=p.numpy().ravel();pr,rc,th=precision_recall_curve(y,p);f=2*pr*rc/(pr+rc+1e-12);k=int(f.argmax())
    return dict(auc=float(roc_auc_score(y,p)),f1=float(f[k]),precision=float(pr[k]),recall=float(rc[k]),threshold=float(th[min(k,len(th)-1)]))
def evaluate(model,loader):
    model.eval();ys=[];ps=[];rs=[];positives=[];keys=[]
    with torch.no_grad():
        for d,q,y,r,pos,key in loader:
            ps.append(model(d.cuda(),q.cuda()).sigmoid().cpu());ys.append(y);rs.append(r);positives.append(pos);keys.extend(key)
    y=torch.cat(ys);p=torch.cat(ps);r=torch.cat(rs);pos=torch.cat(positives).bool();overall=score(y,p);groups={}
    for key in sorted(set(keys)):
        idx=torch.tensor([x==key for x in keys]);groups[key]=score(y[idx],p[idx])
        negative=idx&~pos;groups[key]['negative_roi_fpr']=float((p[negative][r[negative].bool()]>=overall['threshold']).float().mean())
    return overall,groups
def main():
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['B','C'],required=True);a.add_argument('--init',required=True);a.add_argument('--global-split',required=True);a.add_argument('--out',required=True);a.add_argument('--epochs',type=int,default=15);a=a.parse_args()
    torch.manual_seed(3407);np.random.seed(3407);split=json.loads(Path(a.global_split).read_text());train=Paired(split['train'],a.mode,True);val=Paired(split['val'],a.mode,False)
    tl=DataLoader(train,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(3407));vl=DataLoader(val,batch_size=8)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model=Model().cuda();ck=torch.load(a.init,map_location='cpu');model.encoder.load_state_dict(ck['encoder'],strict=True);model.head.load_state_dict(ck['head'],strict=True)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.],device='cuda'));hist=[];best=-1
    print({'mode':a.mode,'train_samples':len(train),'val_samples':len(val),'init_epoch':ck.get('epoch')},flush=True)
    for ep in range(1,a.epochs+1):
        model.train();total=0
        for d,q,y,_,_,_ in tl:
            d,q,y=d.cuda(),q.cuda(),y.cuda();z=model(d,q);p=z.sigmoid();inter=(p*y).sum((1,2,3));dice=1-((2*inter+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)).mean();loss=lossfn(z,y)+.5*dice;opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=float(loss)*len(d)
        overall,groups=evaluate(model,vl);row={'epoch':ep,'loss':total/len(train),'overall':overall,'groups':groups};hist.append(row);print(row,flush=True)
        (out/'summary.json').write_text(json.dumps({'status':'training','history':hist},indent=2))
        if overall['f1']>best:best=overall['f1'];torch.save({'encoder':model.encoder.state_dict(),'head':model.head.state_dict(),'epoch':ep,'val':overall,'groups':groups,'mode':a.mode},out/'best.pt')
    hardest_auc=min(v['auc'] for v in hist[-1]['groups'].values());max_neg=max(v['negative_roi_fpr'] for v in hist[-1]['groups'].values());passed=max(x['overall']['auc'] for x in hist)>=.90 and best>=.70 and hardest_auc>=.75 and max_neg<=.25
    (out/'decision.json').write_text(json.dumps({'passed':passed,'best_f1':best,'best_auc':max(x['overall']['auc'] for x in hist),'final_hardest_group_auc':hardest_auc,'final_max_negative_roi_fpr':max_neg},indent=2))
if __name__=='__main__':main()
