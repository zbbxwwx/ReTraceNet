kill -CONT 795151 795156 2>/dev/null || true
kill -TERM 795151 795156 2>/dev/null || true
sleep 8
pkill -TERM -f 'run_imd_mit_full_v3.py --dataset Columbia' 2>/dev/null || true
sleep 5
cd /group4/zouwenxuan/iml_generalization/repos/ReTraceNet
ROOT=/group4/zouwenxuan/iml_generalization/outputs/retracenet_v2
COL=$ROOT/columbia_mit_v3_indomain200_20260921
OUT=$COL/test_best_now_1024_768
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 /group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python -u tools/eval_split_1024_locked_all.py --dataset Columbia --split "$ROOT/v3_offline_master_20260912/indomain_Columbia/split_721.json" --checkpoint "$COL/best.pt" --out "$OUT" > "$OUT/run.log" 2>&1
echo DONE
