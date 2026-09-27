#!/bin/bash
set -u
BASE=/group4/zouwenxuan/iml_generalization
REPO=$BASE/repos/ReTraceNet
PY=$BASE/envs/mesorch/bin/python
ROOT=$BASE/outputs/retracenet_v2/v3_offline_master_20260912
PILOT=$BASE/outputs/retracenet_v2/balanced_residual_v3_pilot_20260912
FORMAL=$BASE/outputs/retracenet_v2/balanced_residual_v3_full30_20260912
INV=$BASE/outputs/retracenet_v2/final_experiment_summary_20260912/data_inventory
IDX=$BASE/paper_compare/mvss_five_indices
mkdir -p "$ROOT"
exec 9>"$ROOT/execution.lock"; flock -n 9 || exit 9
log(){ printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$ROOT/master.log"; }
state(){ printf 'status: running\nstage: %s\nupdated_at: %s\n' "$1" "$(date -Is)" > "$ROOT/state.yaml.tmp"; mv "$ROOT/state.yaml.tmp" "$ROOT/state.yaml"; }
run_stage(){ name=$1; shift; state "$name"; log "START $name: $*"; "$@" > "$ROOT/${name}.log" 2>&1; rc=$?; echo "$rc" > "$ROOT/${name}.exit_code"; if [ "$rc" -eq 0 ]; then log "DONE $name"; else log "FAILED $name rc=$rc; continuing"; fi; return 0; }
state WAIT_PILOT
log "waiting current five-epoch pilot pid 248850"
while kill -0 248850 2>/dev/null; do sleep 30; done
log "pilot exited; formal training starts regardless of pilot metrics"
run_stage FORMAL_30 env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "$REPO/tools/run_balanced_residual_v3_full30.py" --out "$FORMAL"
CKPT=$FORMAL/best.pt
if [ -f "$CKPT" ]; then
 run_stage CROSS_FOUR env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "$REPO/tools/evaluate_v3_external_four.py" --checkpoint "$CKPT" --out "$ROOT/cross_four"
else
 log "SKIP CROSS_FOUR: formal best.pt missing"
fi
for DS in CASIA1 Columbia NIST16 IMD2020; do
 if [ -f "$CKPT" ] && [ -f "$INV/$DS.jsonl" ]; then
  run_stage "INDOMAIN_$DS" env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u "$REPO/tools/run_v3_indomain_721.py" --dataset "$DS" --inventory "$INV/$DS.jsonl" --template "$CKPT" --out "$ROOT/indomain_$DS" --epochs 30 --seed 3407
 else
  log "SKIP INDOMAIN_$DS: checkpoint or inventory missing"
 fi
done
printf 'status: completed\nstage: DONE\nupdated_at: %s\nformal_checkpoint: %s\n' "$(date -Is)" "$CKPT" > "$ROOT/state.yaml"
log "ALL OFFLINE STAGES FINISHED"
