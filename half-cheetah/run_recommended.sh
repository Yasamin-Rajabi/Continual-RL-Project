#!/usr/bin/env bash
set -euo pipefail

# Fast structural checks (no MuJoCo required).
python3 sanity_check_pool.py

# Main development run: both suites, four conditions, 3 seeds.
python3 run_continual_benchmark.py \
  --task-suites halfcheetah_vel halfcheetah_wind_vel \
  --seeds 1 2 3 \
  --total-timesteps 300000 \
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
  --distill-epochs 8 \
  --distill-lr 3e-4 \
  --distill-batch-size 256 \
  --distill-test-frac 0.2

# For a publication run, use --seeds 1 2 3 4 5.
# If HalfCheetahWindVel is still learning at 300k/task, rerun from scratch
# with --total-timesteps 500000 rather than tuning the merge only on one mode.
