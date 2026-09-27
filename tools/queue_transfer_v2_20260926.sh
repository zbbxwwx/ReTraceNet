#!/bin/bash
set -euo pipefail
R=/group4/zouwenxuan/iml_generalization
P=$R/repos/ReTraceNet
PY=$R/envs/mesorch/bin/python
O=$R/outputs/retracenet_v2/local_paired_transfer_v2_20260926
G=$R/outputs/retracenet_v2/native_jpeg_hrnet_djpeg_stage2_global_div2k_20260925/best.pt
S=$R/outputs/retracenet_v2/v3_target_adapted_stable_20260913/CASIA1/split_721.json
export CUDA_VISIBLE_DEVICES=1
mkdir -p "$O"
exec >>"$O/queue.log" 2>&1
trap 'echo "FAILED line=$LINENO"' ERR
"$PY" "$P/tools/pretrain_native_jpeg_local_paired_v2.py" --images unused --out "$O/pretrain" --global-pretrain "$G" --epochs 20
"$PY" "$P/tools/probe_native_transfer_v2.py" --split "$S" --global-ck "$G" --local-ck "$O/pretrain/best.pt" --out "$O/probe"
if "$PY" -c "import json;exit(0 if json.load(open('$O/probe/decision.json'))['pass_gate'] else 1)"; then
 echo "TRANSFER_GATE_PASS CASIA1 full training"
 "$PY" "$P/tools/run_casia2_native_jpeg_hrnet.py" --dataset CASIA1 --split "$S" --template "$R/outputs/retracenet_v2/casia2_mit_ep76_manipdiv_posttrain1_20260921/best.pt" --out "$O/CASIA1" --epochs 100 --batch-size 4 --learning-rate 1e-3 --seed 3407 --forensic-pretrain "$O/pretrain/best.pt"
else
 echo "TRANSFER_GATE_FAIL no full training"
fi
