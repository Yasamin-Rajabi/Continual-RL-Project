# Validation: replay-trained policy student variant

Date: 2026-09-09

## Implemented

- Added `--policy-student-replay` to `run_sac.py` and `run_continual_benchmark.py` in:
  - half-cheetah
  - Walker2D
  - Hopper
  - metaworld
  - meta-world compatibility mirror
- Added condition label `combined_policy_student`.
- Added standalone-novel SAC objective and detached-expert routing objective.
- Added alternating update order: novel expert first, alpha/gate second.
- Added direct standalone-novel storage for this variant; ordinary `combined_policy` still projects.
- Added TensorBoard scalars and plots for novel actor loss and routing loss.
- Added SLURM main/scratch jobs for HalfCheetah, Walker2D, and canonical `metaworld/`.

## Checks performed

1. `python -m compileall` succeeds for the project.
2. `sanity_check_pool.py` passes in all five mirrored runtime directories.
3. New sanity test verifies:
   - novel SAC loss gives gradients to current-expert parameters but not alpha/alpha-mass;
   - routing loss gives gradients to alpha/alpha-mass but not current-expert parameters;
   - `combined_policy_student` stores the standalone current expert directly.
4. `bash -n` succeeds for all six delivered SLURM scripts.
5. Mocked SLURM submission confirms each main script submits 8 jobs:
   - 4 methods x deterministic/stochastic,
   - with the correct deterministic/stochastic scratch `afterok` dependency group.

## Not validated here

The execution environment used for this code review does not contain the full Gymnasium/MuJoCo/TensorBoard/MetaWorld runtime, so no real simulator learning run was performed. The first cluster run should therefore still be treated as an integration test before launching a large seed sweep.
