#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 run_continual_benchmark.py \
    --task-suites walker2d_dynamics \
    --condition-index 1 4 --composition-spaces parameter policy \
    --test-adapt-steps 0 --frozen-eval-policy pool \
    "$@"
