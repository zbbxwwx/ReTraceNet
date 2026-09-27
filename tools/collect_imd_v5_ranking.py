from pathlib import Path
import yaml

ROOT = Path('/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2/imd_v5_representation_reboot_20260918')
TARGETS = (3000, 6000, 9000, 12000)

def read(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)

def metrics_for(variant, row):
    if row.get('val_fake_precision') is not None:
        return row.get('val_fake_precision'), row.get('val_fake_recall')
    p = ROOT / variant / f"val_epoch{int(row['epoch']):03d}" / 'IMD2020.yaml'
    if not p.exists():
        return None, None
    q = read(p)['pixel_localization_fake_only']['test_f1_oracle_diagnostic']
    return q.get('precision'), q.get('recall')

rows = {}
for variant in ('R0','R1','R2','R3','R4'):
    p = ROOT / variant / 'summary.yaml'
    if not p.exists():
        continue
    s = read(p) or {}
    hist = s.get('history') or []
    if not hist:
        continue
    best = max(hist, key=lambda x: float(x['val_fake_f1']))
    precision, recall = metrics_for(variant, best)
    milestones = {}
    for target in TARGETS:
        r = min(hist, key=lambda x: abs(int(x['global_step']) - target))
        milestones[f'f1_at_{target}_steps'] = {
            'f1': float(r['val_fake_f1']), 'actual_step': int(r['global_step']),
            'epoch': int(r['epoch'])}
    f1 = float(best['val_fake_f1'])
    band = 'failure' if f1 < .53 else ('effective' if f1 < .58 else ('breakthrough_058' if f1 < .62 else ('breakthrough_062' if f1 < .64 else 'breakthrough_064')))
    rows[variant] = {
        'best_f1': f1, 'best_precision': precision, 'best_recall': recall,
        'best_auc': best.get('val_fake_auc'), 'best_threshold': best.get('threshold'),
        'best_epoch': int(best['epoch']), 'best_global_step': int(best['global_step']),
        **milestones, 'final_epoch_f1': float(hist[-1]['val_fake_f1']),
        'final_epoch': int(hist[-1]['epoch']), 'performance_band': band,
        'checkpoint': str(ROOT / variant / 'best.pt')}

comparisons = {}
def diff(a,b,name):
    if a in rows and b in rows:
        comparisons[name] = rows[a]['best_f1'] - rows[b]['best_f1']
diff('R1','R0','forensic_representation_R1_minus_R0')
diff('R2','R0','content_backbone_R2_minus_R0')
diff('R3','R2','noiseprint_under_mit_R3_minus_R2')
diff('R3','R1','mit_under_noiseprint_R3_minus_R1')
diff('R3','R4','qtr_hcv_value_R3_minus_R4')
out = {'protocol': {'dataset':'IMD2020','budget':'50 epochs x 279 steps = 13950 optimizer updates','selection':'val fake-only pooled oracle during validation','objective':'final BCE + Dice','DHER':'disabled','seed':3407}, 'variants': rows, 'comparisons': comparisons, 'ranking': sorted(rows, key=lambda k: rows[k]['best_f1'], reverse=True)}
with open(ROOT/'ranking_enhanced.yaml','w',encoding='utf-8') as f:
    yaml.safe_dump(out,f,sort_keys=False,allow_unicode=True)
print(yaml.safe_dump(out,sort_keys=False,allow_unicode=True))
