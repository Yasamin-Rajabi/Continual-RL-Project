#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Assumes you already activated a virtual environment and installed requirements.
python3 - <<'PY'
import platform, sys
import torch, gymnasium, mujoco, stable_baselines3
print("python:", sys.version.split()[0], platform.machine())
print("torch:", torch.__version__)
print("gymnasium:", gymnasium.__version__)
print("mujoco:", mujoco.__version__)
print("stable-baselines3:", stable_baselines3.__version__)
print("mps available:", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
PY

python3 sanity_check_pool.py
python3 tasks.py --check
python3 walker2d_smoke.py

SMOKE_ROOT="${SMOKE_ROOT:-$PWD/_walker2d_smoke_output}"
rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"/{agents,runs,plots,analysis}

# Tiny CPU run that intentionally forces a merge on the third task. It is NOT
# a performance test; 4k steps/task is only enough to exercise SAC, checkpoint
# inheritance, arithmetic merging, KL pair selection, and distillation end-to-end.
python3 run_continual_benchmark.py \
  --task-suites walker2d_dynamics \
  --seeds 11 \
  --task-sequence 0 1 2 \
  --total-timesteps 4000 \
  --learning-starts 500 \
  --random-actions-end 750 \
  --batch-size 128 \
  --pool-size 2 \
  --eval-every 2000 \
  --num-evals 1 \
  --retention-eval-episodes 1 \
  --test-adapt-steps 0 \
  --distill-extra-steps 300 \
  --max-distill-buffer 1000 \
  --similarity-samples 128 \
  --distill-max-samples 500 \
  --distill-epochs 2 \
  --distill-batch-size 128 \
  --analysis-log-every 1000 \
  --condition-index 1 4 \
  --train-shared \
  --skip-retention \
  --skip-survey-metrics \
  --save-root "$SMOKE_ROOT/agents" \
  --runs-root "$SMOKE_ROOT/runs" \
  --plots-root "$SMOKE_ROOT/plots" \
  --analysis-root "$SMOKE_ROOT/analysis" \
  --cpu

echo
echo "SMOKE TEST PASSED"
echo "Output: $SMOKE_ROOT"
echo "Do not interpret the 4k-step returns as learning quality."
