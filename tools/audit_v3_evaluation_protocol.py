"""Fail-fast static and synthetic audit for every V3 reporting path."""
from pathlib import Path
import json,sys,numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from v3_unified_metrics import PixelCounts,AuthenticPixelFPR,ImageDetectionStats,PROTOCOL
files={n:Path(__file__).with_name(n).read_text() for n in ['run_balanced_residual_v3_full30.py','run_cross_dataset_after_chain.py','evaluate_v3_external_four.py','run_v3_indomain_721.py']}
checks={
 'formal_no_test_loading':"read_manifest(manifest,cfg['data']['root'],'test')" not in files['run_balanced_residual_v3_full30.py'],
 'cross_pixel_threshold_CASIA2':"threshold_source':'CASIA2_val'" in files['evaluate_v3_external_four.py'],
 'domain_own_val_threshold':"threshold_source=a.dataset+'_val'" in files['run_v3_indomain_721.py'],
 'domain_separate_image_threshold':'image_selection_threshold' in files['run_v3_indomain_721.py'],
 'fake_only_pixel_counts':'if record.label:' in files['run_cross_dataset_after_chain.py'],
 'authentic_separate':'authentic.add(scores)' in files['run_cross_dataset_after_chain.py'],
 'probability_before_threshold':'mode="bilinear"' in files['run_cross_dataset_after_chain.py'],
 'overlap_average':'accum / weight' in files['run_cross_dataset_after_chain.py'],
 'tile_grid8':'assert all(value % 8 == 0 for value in starts)' in files['run_cross_dataset_after_chain.py'],
 'one_image_topk_score':'np.ceil(.2*len(image_scores))' in files['run_cross_dataset_after_chain.py'],
 'source_isolation':'source_isolation_audit.yaml' in files['run_v3_indomain_721.py'],
 'explicit_diagnostic_names':'test_f1_oracle_diagnostic' in files['run_cross_dataset_after_chain.py'],
}
f=PixelCounts(.5);f.add([.9,.1,.8,.2],[1,1,0,0]);checks['pooled_counts_synthetic']=f.result()['f1']==.5
a=AuthenticPixelFPR();a.add([.9,.1]);checks['auth_fpr_denominator']=abs(a.result(.5)['fpr']-.5)<1e-9
i=ImageDetectionStats();[i.add(s,y) for s,y in [(.9,1),(.1,0)]];checks['image_one_scalar_per_image']=i.result(.5)['count']==2
report={'status':'passed' if all(checks.values()) else 'failed','checks':checks,'protocol':PROTOCOL}
print(json.dumps(report,indent=2));
if not all(checks.values()):raise SystemExit(1)
