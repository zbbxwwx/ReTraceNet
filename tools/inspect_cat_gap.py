"""Read-only validation failure inspection; never train or use test."""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

BASE = Path('/group4/zouwenxuan/iml_generalization')
ROOT = BASE/'outputs/retracenet_v2/oracle_overlap_20260911'
OUT = ROOT/'failure_inspection'
OUT.mkdir(exist_ok=True)
rows = [json.loads(x) for x in (ROOT/'B_per_image.jsonl').read_text().splitlines()]
assert len(rows) == 2494
def score(c):
    t,p,n = c
    return float(2*t/max(1,2*t+p+n))
groups = {}
for i,r in enumerate(rows):
    if r['bucket'].startswith('unknown'): continue
    z = np.load(ROOT/'paired'/f'{i:04d}.npz')
    h,w = z['gt'].shape
    for group in [r['bucket'],r['bucket']+'|'+Path(r['path']).suffix.lower(),
                  'fake|'+Path(r['path']).suffix.lower(),
                  'fake|'+('maxdim>640' if max(h,w)>640 else 'maxdim<=640')]:
        g=groups.setdefault(group,{'n':0,'cat':np.zeros(3,dtype=np.int64),'e24':np.zeros(3,dtype=np.int64)})
        g['n']+=1;g['cat']+=r['cat_counts'];g['e24']+=r['e24_on_counts']
summary={'status':'running','test_run':False,'threshold':0.5,'groups':{
    k:{'n':v['n'],'cat_f1':score(v['cat']),'e24_f1':score(v['e24']),
       'cat_counts':v['cat'].tolist(),'e24_counts':v['e24'].tolist()} for k,v in groups.items()},
    'notes':['Selected worst failures are not representative performance estimates.',
             'CAT non-JPEG quality100 conversion differs from E24 availability masking.',
             'Existing HCV ReferenceNet already combines proposal-weighted local and global references.']}
def save(): (OUT/'report.json').write_text(json.dumps(summary,indent=2))
save()
selected=sorted([i for i,r in enumerate(rows) if r['bucket']=='Tp_S:gte_20'],
                key=lambda i:rows[i]['cat_correct_e24_on_wrong_foreground'],reverse=True)[:20]
helper=BASE/'outputs/domain_casia2_eval/catnet_unified1247_best_20260831_015845/export_catnet_unified1247.py'
spec=importlib.util.spec_from_file_location('cat_reference',helper)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ck=BASE/'outputs/catnet/splicing_dataset/CAT_full/best.pth.tar'
m.update_config(m.config,argparse.Namespace(cfg=str(BASE/'repos/CAT-Net/experiments/CAT_full.yaml'),opts=['TEST.MODEL_FILE',str(ck),'TEST.FLIP_TEST','False']))
class SharedGT(m.ManifestDataset):
    def get_tamp(self,index):
        z=np.load(ROOT/'paired'/f'{index:04d}.npz')
        jpg=m.make_jpeg_for_catnet(self.rows[index]['image'],self.jpg_cache_dir)
        return self._create_tensor(jpg,z['gt'].astype(np.uint8))
ds=SharedGT([{'image':r['path'],'mask':'','label':int(not r['bucket'].startswith('unknown'))} for r in rows],ROOT/'cat_jpeg_cache')
torch.set_num_threads(4)
net=getattr(m.models,m.config.MODEL.NAME).get_seg_model(m.config)
net.load_state_dict(torch.load(ck,map_location='cpu')['state_dict'],strict=True)
model=m.FullModel(net,m.CrossEntropy(ignore_label=m.config.TRAIN.IGNORE_LABEL,weight=ds.class_weights).cuda()).cuda().eval()
def gray(x): return Image.fromarray(np.uint8(np.clip(x,0,1)*255)).convert('RGB')
cards=[];summary['selected']=[]
with torch.inference_mode():
    for rank,i in enumerate(selected,1):
        r=rows[i];z=np.load(ROOT/'paired'/f'{i:04d}.npz');y=z['gt'].astype(bool);h,w=y.shape
        image,label,qt=ds[i]
        _,pred=model(image[None].cuda(),label[None].long().cuda(),qt[None].cuda() if torch.is_tensor(qt) else qt)
        prob=F.interpolate(pred.float().softmax(1)[:,1:2],size=label.shape[-2:],mode='bilinear',align_corners=False)[0,0].cpu().numpy()[:h,:w]
        c=prob>=.5;a=z['on']>=.5
        counts=[int((c&y).sum()),int((c&~y).sum()),int((~c&y).sum())]
        assert counts==r['cat_counts'],(i,counts,r['cat_counts'])
        diff=z['on']-z['off'];heat=np.zeros((h,w,3),dtype=np.uint8)
        heat[:,:,0]=np.uint8(np.clip(diff/.1,0,1)*255);heat[:,:,2]=np.uint8(np.clip(-diff/.1,0,1)*255)
        panels=[('Original',Image.open(r['path']).convert('RGB')),('GT',gray(y)),
                ('E24 probability',gray(z['on'])),('CAT probability',gray(prob)),
                ('E24 mask',gray(a)),('CAT mask',gray(c)),
                ('E24 missed foreground',gray(y&~a)),('DCT difference (+red/-blue; .1)',Image.fromarray(heat))]
        card=Image.new('RGB',(1280,580),'white');draw=ImageDraw.Draw(card)
        draw.text((8,4),f'{rank:02d} {Path(r["path"]).name} | E24 {score(r["e24_on_counts"]):.4f} CAT {score(counts):.4f} | {w}x{h}',fill='black')
        for j,(title,im) in enumerate(panels):
            x=(j%4)*320;yy=30+(j//4)*275
            draw.text((x+5,yy),title,fill='black');im.thumbnail((310,248))
            card.paste(im,(x+(320-im.width)//2,yy+20))
        file=f'{rank:02d}_{i:04d}.png';card.save(OUT/file);cards.append(card)
        summary['selected'].append({'index':i,'file':file,'path':r['path'],'e24_f1':score(r['e24_on_counts']),'cat_f1':score(counts),'recovered_foreground':r['cat_correct_e24_on_wrong_foreground']})
        save();print(f'[inspect] {rank}/20 {file}',flush=True)
contact=Image.new('RGB',(1280,290*len(cards)),'white')
for j,card in enumerate(cards):contact.paste(card.resize((640,290)),((j%2)*640,(j//2)*290))
contact=contact.crop((0,0,1280,290*((len(cards)+1)//2)));contact.save(OUT/'contact_sheet.jpg')
summary['status']='completed';save();print('COMPLETE',flush=True)
