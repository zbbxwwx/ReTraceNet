#!/usr/bin/env bash
set -euo pipefail
cd /group4/zouwenxuan/iml_generalization/repos/ReTraceNet
exec /group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python run_dher_v21_ab.py --gpu-a 0 --gpu-b 1
