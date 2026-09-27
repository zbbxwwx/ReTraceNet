"""Train TruFor on IMD 7:2:1 and select checkpoints by original-grid pooled fake F1."""
import argparse,json,logging,os,random,sys,time
from pathlib import Path
import numpy as np,torch,yaml
import torch.nn.functional as F
from PIL import Image
from tensorboardX import SummaryWriter

def starts(n,tile,stride):
 if n<=tile:return [0]
 out=list(range(0,n-tile+1,stride))
 if out[-1]!=n-tile:out.append(n-tile)
 return out
def tile_prob(model,arr,device):
 h,w=arr.shape[:2];x=torch.from_numpy(arr.transpose(2,0,1)).float().unsqueeze(0).to(device)/256.0;ph=(-h)%8;pw=(-w)%8
 if ph or pw:x=F.pad(x,(0,pw,0,ph),mode='reflect')
 pred,_,_,_=model(x,save_np=False)
 if pred.shape[-2:]!=x.shape[-2:]:pred=F.interpolate(pred,size=x.shape[-2:],mode='bilinear',align_corners=False)
 return F.softmax(pred,dim=1)[0,1,:h,:w].float().cpu().numpy()
def probability(model,path,device,full_max=1536,tile=1024,stride=768):
 a=np.asarray(Image.open(path).convert('RGB'));h,w=a.shape[:2]
 if max(h,w)<=full_max:return tile_prob(model,a,device)
 acc=np.zeros((h,w),np.float32);cnt=np.zeros((h,w),np.float32)
 for y in starts(h,tile,stride):
  for x in starts(w,tile,stride):
   p=tile_prob(model,a[y:min(y+tile,h),x:min(x+tile,w)],device);hh,ww=p.shape;acc[y:y+hh,x:x+ww]+=p;cnt[y:y+hh,x:x+ww]+=1
 return acc/np.maximum(cnt,1)
def pooled_validate(model,rows,device):
 hp=np.zeros(101,np.int64);hn=np.zeros(101,np.int64);used=0
 model.eval()
 with torch.no_grad():
  for j,r in enumerate(rows,1):
   if not int(r['label']):continue
   p=probability(model,r['image'],device);y=np.asarray(Image.open(r['mask']).convert('L'))>127
   if p.shape!=y.shape:raise RuntimeError((r['image'],p.shape,y.shape))
   b=np.minimum((p*100).astype(np.int16),100);hp+=np.bincount(b[y],minlength=101);hn+=np.bincount(b[~y],minlength=101);used+=1
   if used==1 or used%50==0:print('[locked-val]',used,flush=True)
 tp=np.cumsum(hp[::-1])[::-1];fp=np.cumsum(hn[::-1])[::-1];pos=int(hp.sum());fn=pos-tp;f1=2*tp/np.maximum(2*tp+fp+fn,1);k=int(np.argmax(f1));pr=tp[k]/max(tp[k]+fp[k],1);rc=tp[k]/max(pos,1)
 return {'fake_images':used,'threshold':k/100.0,'f1':float(f1[k]),'precision':float(pr),'recall':float(rc),'tp':int(tp[k]),'fp':int(fp[k]),'fn':int(fn[k]),'protocol':'original_GT_grid; probability_bilinear; fake-only pooled; thresholds=0:0.01:1; sliding overlap average'}
def main(a):
 os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);repo=Path(a.repo);sys.path.insert(0,str(repo));os.chdir(repo)
 from lib.config import config,update_config
 from lib.utils import get_model,get_optimizer,FullModel,adjust_learning_rate
 from lib.core.function import train
 from dataset.data_core import myDataset
 import albumentations
 class Args:pass
 x=Args();x.experiment=a.experiment;x.gpu=[0];x.opts=[];update_config(config,x)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 torch.backends.cudnn.benchmark=bool(config.CUDNN.BENCHMARK);torch.backends.cudnn.deterministic=bool(config.CUDNN.DETERMINISTIC);torch.backends.cudnn.enabled=bool(config.CUDNN.ENABLED)
 os.environ['PYTHONHASHSEED']=str(a.seed);device=torch.device('cuda:0');out=repo/'weights'/a.experiment;out.mkdir(parents=True,exist_ok=True);log=out/'locked_train.log'
 logging.basicConfig(filename=str(log),level=logging.INFO);writer=SummaryWriter(str(repo/'log'/config.MODEL.NAME/(a.experiment+'_locked')));wd={'writer':writer,'train_global_steps':0,'valid_global_steps':0}
 aug=albumentations.load(config.TRAIN.AUG,data_format='yaml') if config.TRAIN.AUG else None;crop=(config.TRAIN.IMAGE_SIZE[1],config.TRAIN.IMAGE_SIZE[0]);ds=myDataset(config,crop,False,'train',aug=aug);loader=torch.utils.data.DataLoader(ds,batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,shuffle=config.TRAIN.SHUFFLE,num_workers=config.WORKERS)
 split=json.loads(Path(a.split).read_text());val=split['val'];base=get_model(config);base=torch.nn.DataParallel(base,device_ids=[0]).cuda();model=FullModel(base,config);opt=get_optimizer(model,config);epoch_iters=np.int32(len(ds)/config.TRAIN.BATCH_SIZE_PER_GPU);num_iters=config.TRAIN.END_EPOCH*epoch_iters;history=[];best=-1.;best_ep=0;start_ep=0
 if a.resume and Path(a.resume).is_file():
  ck=torch.load(a.resume,map_location='cpu');model.model.module.load_state_dict(ck['state_dict'],strict=True);opt.load_state_dict(ck['optimizer']);history=list(ck.get('history',[]));best=float(ck.get('best_value',-1.));best_ep=int(max(history,key=lambda z:z.get('f1',-1)).get('epoch',0)) if history else int(ck.get('epoch',0));start_ep=int(ck.get('epoch',0));print('[resume]',a.resume,'start_epoch',start_ep,'best',best,flush=True)
 for ep in range(start_ep,config.TRAIN.END_EPOCH):
  ds.shuffle();train(ep,config.TRAIN.END_EPOCH,epoch_iters,config.TRAIN.LR,num_iters,loader,opt,model,wd,adjust_learning_rate=adjust_learning_rate);torch.cuda.empty_cache()
  metric=pooled_validate(model.model.module,val,device);metric['epoch']=ep+1;history.append(metric);print('[epoch]',metric,flush=True)
  state={'epoch':ep+1,'best_value':max(best,metric['f1']),'best_key':'pooled_fake_F1_best_original_grid','state_dict':model.model.module.state_dict(),'optimizer':opt.state_dict(),'selection_threshold':metric['threshold'],'history':history,'split':a.split,'seed':a.seed}
  torch.save(state,out/'checkpoint.pth.tar')
  if metric['f1']>best:best=metric['f1'];best_ep=ep+1;torch.save(state,out/'best.pth.tar')
  (out/'summary.yaml').write_text(yaml.safe_dump({'status':'training','best_F1':best,'best_epoch':best_ep,'history':history,'selection_protocol':metric['protocol'],'split':a.split,'seed':a.seed},sort_keys=False))
 writer.close();(out/'summary.yaml').write_text(yaml.safe_dump({'status':'completed','best_F1':best,'best_epoch':best_ep,'history':history,'selection_protocol':history[-1]['protocol'],'split':a.split,'seed':a.seed},sort_keys=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--experiment',default='trufor_imd721');p.add_argument('--split',required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--seed',type=int,default=1337);p.add_argument('--resume',default='');main(p.parse_args())
