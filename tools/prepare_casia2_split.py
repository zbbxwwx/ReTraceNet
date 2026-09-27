import csv, json
from pathlib import Path

manifest=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/fair_training_v1_full_20260911/clean_manifest.csv')
out=Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/mit_full_offline_20260918/casia2_split.json')
rows=list(csv.DictReader(manifest.open()))
data={k:[] for k in ('train','val','test')}
for r in rows:
    split=r['split']
    if split not in data: continue
    data[split].append({'image':r['image'],'mask':r.get('mask') or None,'label':int(r['label'])})
# The established fair protocol has train/val but no separately consumed test.
# Cross-domain evaluation only needs a CASIA2-val-selected checkpoint/threshold.
if not data['test']: data['test']=list(data['val'])
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(data,indent=2))
print({k:len(v) for k,v in data.items()},out)
