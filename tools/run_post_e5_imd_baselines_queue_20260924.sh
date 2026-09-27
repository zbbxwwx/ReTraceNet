#!/usr/bin/env bash
set -uo pipefail
R=/group4/zouwenxuan/iml_generalization
REPO=$R/repos/ReTraceNet
PY=$R/envs/mesorch/bin/python
OUT=$R/outputs/retracenet_v2/post_e5_imd_baselines_queue_20260924
mkdir -p "$OUT"
LOG=$OUT/queue.log
note(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }
export CUDA_VISIBLE_DEVICES=0
note "WAIT E5 locked CASIA2 test pid=1255841"
while kill -0 1255841 2>/dev/null; do sleep 30; done
note "E5 test process ended"
cd "$REPO"
note "START E5 matched-budget no-distill control"
"$PY" -u tools/run_e5_paired_chroma_no_distill_control.py >> "$OUT/e5_no_distill.log" 2>&1
note "END E5 no-distill rc=$?"
note "START IMD coverage epoch29 error profile"
"$PY" -u tools/profile_imd2020_coverage_ep29_errors.py >> "$OUT/imd_error_profile.log" 2>&1
note "END IMD profile rc=$?"
note "START repaired SAFIRE/MVSS continuation"
bash tools/rerun_safire_mvss_after_main.sh >> "$OUT/baseline_repair.log" 2>&1
note "END SAFIRE/MVSS repair rc=$?"
note "START PSCC/CAT continuation"
bash tools/queue_other_imd_models.sh >> "$OUT/baseline_other.log" 2>&1
note "END PSCC/CAT queue rc=$?"
note "ALL REQUESTED STAGES FINISHED"
