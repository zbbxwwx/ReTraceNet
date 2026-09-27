#!/usr/bin/env bash
set -euo pipefail
cd /group4/zouwenxuan/iml_generalization/repos/ReTraceNet
exec /group4/zouwenxuan/iml_generalization/envs/mesorch/bin/python \
  /group4/zouwenxuan/iml_generalization/repos/ReTraceNet/overnight_hmsi_runner.py "$@"
