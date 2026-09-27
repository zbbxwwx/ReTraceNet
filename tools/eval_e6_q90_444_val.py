import copy,yaml,torch
from pathlib import Path
import run_e6_multi_quality_chroma as e6
import run_e5_paired_chroma_adaptation as e5
OUT=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/imd_small_window_e6_q90_444_20260924')
parent=torch.load(e6.PARENT,map_location='cpu')
ep=torch.load('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/e6_q90_q80_joint_adaptation_20260923/epoch02.pt',map_location='cpu')
cfg=copy.deepcopy(parent['config']);cfg['training']['dct_sample_dropout']=0.0
model=e6.fair.e.build_final_model(cfg['model']).cuda().eval();model.load_state_dict(ep['model'],strict=True)
records=e6.fair.e.read_manifest(e6.MANIFEST,cfg['data']['root'],'val')
base=e6.FairDataset(records,OUT/'e6_q90_444_cache',False)
threshold=float(yaml.safe_load(open(e6.OUT/'summary.yaml'))['history'][-1]['clean']['oracle']['threshold'])
q=e5.eval_ds(model,e6.FixedQuality(base,90,0),cfg,'e6_ep02_q90_444',threshold)
res={'status':'completed','checkpoint':str(e6.OUT/'epoch02.pt'),'population':'CASIA2 val','jpeg_quality':90,'subsampling':'4:4:4','clean_locked_threshold':threshold,'metrics':q,'test_read':False}
(OUT/'e6_ep02_q90_444_val.yaml').write_text(yaml.safe_dump(res,sort_keys=False));print(yaml.safe_dump(res,sort_keys=False))
