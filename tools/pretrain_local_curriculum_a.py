from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from overfit_native_jpeg_local import FixedLocalDataset, Localizer, metrics


def evaluate(model, loader):
    model.eval(); probabilities=[]; targets=[]
    with torch.no_grad():
        for dct,qtable,target,_,_ in loader:
            probabilities.append(model(dct.cuda(),qtable.cuda()).sigmoid().cpu());targets.append(target)
    return metrics(torch.cat(targets),torch.cat(probabilities))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--global-pretrain',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--epochs',type=int,default=15);args=parser.parse_args()
    torch.manual_seed(3407);np.random.seed(3407);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    split=json.loads((Path(args.global_pretrain).parent/'split.json').read_text())
    train=FixedLocalDataset([Path(x) for x in split['train']]);val=FixedLocalDataset([Path(x) for x in split['val']])
    train_loader=DataLoader(train,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(3407));val_loader=DataLoader(val,batch_size=8)
    model=Localizer().cuda();checkpoint=torch.load(args.global_pretrain,map_location='cpu');load=model.encoder.load_state_dict(checkpoint['encoder'],strict=True)
    print({'train_originals':len(train),'val_originals':len(val),'missing':load.missing_keys,'unexpected':load.unexpected_keys},flush=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0],device='cuda'));history=[];best=-1
    for epoch in range(1,args.epochs+1):
        model.train();total=0
        for dct,qtable,target,_,_ in train_loader:
            dct,qtable,target=dct.cuda(),qtable.cuda(),target.cuda();logits=model(dct,qtable);probability=logits.sigmoid()
            intersection=(probability*target).sum((1,2,3));dice=1-((2*intersection+1)/(probability.sum((1,2,3))+target.sum((1,2,3))+1)).mean()
            loss=criterion(logits,target)+.5*dice;optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();total+=float(loss)*len(dct)
        report=evaluate(model,val_loader);row={'epoch':epoch,'train_loss':total/len(train),**report};history.append(row);print(row,flush=True)
        (out/'summary.json').write_text(json.dumps({'status':'training','best_f1':max(x['f1'] for x in history),'history':history},indent=2))
        if report['f1']>best:best=report['f1'];torch.save({'encoder':model.encoder.state_dict(),'head':model.head.state_dict(),'epoch':epoch,'val':report},out/'best.pt')
    passed=max(x['auc'] for x in history)>=.90 and max(x['f1'] for x in history)>=.70
    (out/'decision.json').write_text(json.dumps({'passed':passed,'best_auc':max(x['auc'] for x in history),'best_f1':max(x['f1'] for x in history)},indent=2))
if __name__=='__main__':main()
