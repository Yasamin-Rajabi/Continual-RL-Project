#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Structural + live MuJoCo checks.
python3 sanity_check_pool.py
python3 tasks.py --check

# Main development run. Start with the moderate dynamics suite; it is designed
# to converge much sooner than Ant while still producing distinct gait experts.
python3 run_continual_benchmark.py \
  --task-suites walker2d_dynamics \
  --seeds 1 2 3 \
  --total-timesteps 200000 \
  --pool-size 5 \
  --batch-size 256 \
  --policy-lr 3e-4 \
  --q-lr 3e-4 \
  --learning-starts 5000 \
  --random-actions-end 10000 \
  --eval-every 10000 \
  --num-evals 5 \
  --distill-extra-steps 10000 \
  --similarity-samples 2048 \
  --max-distill-buffer 50000 \
  --distill-max-samples 20000 \
  --distill-epochs 16 \
  --distill-lr 5e-4 \
  --distill-batch-size 256 \
  --distill-test-frac 0.2 \
  --train-shared

# Only after the moderate suite is verified, try --task-suites
# walker2d_mixed_dynamics if you want a stronger consolidation stress test.
# If individual specialists are still visibly improving at 200k, raise the
# training budget to 300k/task before interpreting merge differences.
