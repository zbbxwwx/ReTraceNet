#!/usr/bin/env bash
set -u
REPO=/group4/zouwenxuan/iml_generalization/repos/ReTraceNet
PY=/group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python
BASE=/group4/zouwenxuan/iml_generalization
CHECK=$BASE/outputs/retracenet_v2/imd_mit_context_check_20260919
OLD=$BASE/outputs/retracenet_v2/imd_mitb2_full_v3_100ep_20260918
OUT=$BASE/outputs/retracenet_v2/imd_mitb2_full_v3_imagenet_norm_20260919
SPLIT=$BASE/outputs/retracenet_v2/v3_offline_master_20260912/indomain_IMD2020/split_721.json
TEMPLATE=$BASE/outputs/retracenet_v2/imd2020_stable_bn_full100_20260916/best.pt
mkdir -p "$OUT"
log(){ echo "[$(date -Iseconds)] $*" | tee -a "$OUT/queue.log"; }

log "waiting for controlled tile-policy evaluation on GPU0"
while true; do
  status=$($PY -c "import yaml,pathlib; p=pathlib.Path('$CHECK/summary.yaml'); print((yaml.safe_load(p.read_text()) or {}).get('status','missing') if p.exists() else 'missing')" 2>/dev/null || echo missing)
  [ "$status" = completed ] && break
  [ "$status" = failed ] && { log "context check failed"; exit 20; }
  sleep 60
done

decision=$($PY -c "import yaml; print((yaml.safe_load(open('$CHECK/summary.yaml')) or {}).get('decision',''))")
if [ "$decision" = adopt_1024_768_and_run_locked_test ]; then
  log "larger tile passed +1pp gate; run IMD test at its val-locked threshold"
  CUDA_VISIBLE_DEVICES=0 "$PY" tools/eval_imd_mit_context_locked_test.py \
    --checkpoint "$OLD/best.pt" --split "$SPLIT" --val-summary "$CHECK/summary.yaml" \
    --out "$CHECK/test_tile1024_stride768_locked" > "$CHECK/test_tile1024_stride768.log" 2>&1 || exit 21
fi

log "context check complete; launch ImageNet-normalization-only control"
cd "$REPO" || exit 2
CUDA_VISIBLE_DEVICES=0 "$PY" tools/run_imd_mit_full_v3_imagenet_norm.py \
  --dataset IMD2020 --split "$SPLIT" --template "$TEMPLATE" --out "$OUT" \
  --epochs 100 --batch-size 4 --learning-rate .001 --skip-test \
  > "$OUT/train.log" 2>&1 &
train_pid=$!
echo "$train_pid" > "$OUT/train_pid"
log "training pid=$train_pid"

gate30_done=0
while kill -0 "$train_pid" 2>/dev/null; do
  read epoch new_f1 old_f1 delta <<< "$($PY - <<PY
import pathlib,yaml
def load(p):
    p=pathlib.Path(p)
    return yaml.safe_load(p.read_text()) if p.exists() else {}
n=load('$OUT/summary.yaml') or {}; o=load('$OLD/summary.yaml') or {}
nh=n.get('history') or []; oh=o.get('history') or []
if not nh: print('0 nan nan nan')
else:
    r=nh[-1]; ep=int(r['epoch']); nf=float(r['val_fake_f1'])
    candidates=[x for x in oh if int(x.get('global_step',-1))<=int(r.get('global_step',-1))]
    old=float(candidates[-1]['val_fake_f1']) if candidates else float('nan')
    print(ep,nf,old,nf-old)
PY
)"
  if [ "$epoch" -ge 30 ] && [ "$gate30_done" -eq 0 ]; then
    gate30_done=1
    log "gate30 epoch=$epoch normalized=$new_f1 matched_original=$old_f1 delta=$delta"
  fi
  if [ "$epoch" -ge 50 ]; then
    keep=$($PY -c "import math; print(int(float('$delta')>=.02))")
    if [ "$keep" -eq 0 ]; then
      log "STOP at >=7k updates: matched improvement below +2pp (delta=$delta)"
      kill "$train_pid" 2>/dev/null || true
      wait "$train_pid" 2>/dev/null || true
      $PY - <<PY
import pathlib,yaml
p=pathlib.Path('$OUT/gate_decision.yaml')
p.write_text(yaml.safe_dump({'decision':'stopped_at_7k_updates','epoch':int('$epoch'),'normalized_f1':float('$new_f1'),'matched_original_f1':float('$old_f1'),'delta_f1':float('$delta')},sort_keys=False))
PY
      exit 0
    fi
  fi
  sleep 120
done
wait "$train_pid"
rc=$?
log "normalization training exited rc=$rc"
exit "$rc"
