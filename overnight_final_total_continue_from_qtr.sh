#!/usr/bin/env bash
set -euo pipefail
cd /group4/zouwenxuan/iml_generalization/repos/ReTraceNet
exec /group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python overnight_final_total_continue_from_qtr.py --gpu 1
