# CKA-RL: HalfCheetah continual-learning experiments

This repository is a modified implementation of **Continual Knowledge Adaptation for Reinforcement Learning (CKA-RL)**. The continual benchmark now uses two MuJoCo task suites:

- **HalfCheetahVel** (`halfcheetah_vel`): the desired forward velocity changes between tasks.
- **HalfCheetahWindVel** (`halfcheetah_wind_vel`): the desired velocity changes and each task also has a fixed, hidden external wind force.

The original CKA-RL pool merge selected the most cosine-similar knowledge vectors in **weight space** and averaged them. This version keeps the bounded knowledge pool but selects the merge pair in **policy-output space**: it evaluates every pool entry on a balanced subset of states from each candidate pair's two lineage buffers and merges the pair with the **lowest symmetric KL divergence between their diagonal Gaussian policies (mean + log-standard-deviation heads)**.

For distillation modes, arithmetic averaging is only the student initialization. The selected pair is then jointly distilled by minimizing teacher-to-student Gaussian KL on the two parent buffers.

## Four experimental cases

All four cases use exactly the same behavioral-KL pair selector so that the comparison is controlled.

| Case | Knowledge representation | Alpha mass | Merge after pair selection |
|---|---|---:|---|
| `baseline` | `classic_cka` task vector | No | Arithmetic average |
| `distil_only` | `classic_cka` task vector | No | Joint policy KL distillation |
| `weight_only` | `weight_delta` | Yes | Arithmetic average |
| `combined` | `weight_delta` | Yes | Joint policy KL distillation |

The mean and log-std heads are now **aligned pool slots**. A single alpha vector weights a whole policy knowledge item, and one merge decision is applied to both heads. This avoids the old bug where the two heads could merge different task pairs.

## Environment definitions

### HalfCheetahVel

For target velocity `v*`, each step uses

```text
reward = -abs(x_velocity - v*) - 0.05 * sum(action^2)
```

The target velocity is not appended to the observation. For diagnostics, `success` is the fraction of steps with `abs(x_velocity - target_velocity) <= 0.2`; it does not change the reward. The eight default targets are:

```text
0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 1.25, 2.25 m/s
```

### HalfCheetahWindVel

The reward is the same target-velocity objective, but a task-specific external force is applied through MuJoCo's `xfrc_applied` every simulation frame. The wind is hidden from the observation. The default `(wind_x, wind_z)` values are:

```text
(-2.5, 0.0), (2.5, 0.0), (0.0, -5.0), (0.0, 5.0),
(-1.25, -2.5), (1.25, 2.5), (-2.5, 5.0), (2.5, -5.0)
```

The default continual sequence is the eight tasks followed by the same eight tasks again:

```text
0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
```

This gives relearning/retention information and forces multiple pool merges with `pool_size=5`.

## Installation

```bash
pip install -r requirements.txt
```

Then run the CPU-only pool tests before spending MuJoCo time:

```bash
python3 sanity_check_pool.py
```

The sanity test checks all four modes, aligned head pools, shared alpha, frozen later-task encoder, behavioral-KL pair selection, KL distillation, and compact policy snapshots.

## Quick smoke test

This is intentionally small and only verifies that the full pipeline reaches a merge and produces plots:

```bash
python3 run_continual_benchmark.py --quick-test
```

## Recommended experiment

A good starting point for the actual comparison is:

```bash
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
  --distill-extra-steps 10000 \
  --similarity-samples 2048 \
  --max-distill-buffer 50000 \
  --distill-max-samples 20000 \
  --distill-epochs 8 \
  --distill-lr 3e-4 \
  --distill-batch-size 256
```

For a final paper-quality run, use **5 seeds**. If the WindVel tasks are visibly under-trained at 300k steps, increase `--total-timesteps` to **500k** before changing the merge hyperparameters.

### Recommended defaults and why

| Parameter | Value | Rationale |
|---|---:|---|
| SAC actor / critic LR | `3e-4` | Standard stable starting point for MuJoCo SAC |
| Batch size | `256` | Good compute/variance trade-off |
| Replay buffer | `1,000,000` | Enough coverage across one task |
| `gamma` | `0.99` | Standard locomotion horizon |
| target smoothing `tau` | `0.005` | Stable SAC target updates |
| random exploration | `10k` steps | Fills replay before policy dominates |
| learning starts | `5k` steps | Avoids critic updates on a tiny buffer |
| pool size | `5` | Forces compression while retaining several distinct skills |
| stored post-task states | `10k` | Provides a useful behavioral reference distribution |
| maximum buffer / slot | `50k` | Bounds memory after repeated merges |
| KL comparison states | `2048` | Usually enough for stable pair ranking without expensive all-buffer evaluation |
| distillation rows | `20k` max | Balanced between the two selected parents |
| distillation epochs | `8` | Enough refinement from the averaged initialization without making merging dominant in runtime |
| distillation LR | `3e-4` | Conservative for a student already initialized near the two teachers |
| held-out distillation fraction | `0.2` | Lets you detect memorization / poor merge generalization |

## What gets plotted

`run_continual_benchmark.py` writes plots under `plots_halfcheetah_continual/<suite>/`.

It includes:

- continual training episodic return;
- periodic evaluation return, velocity error, and success fraction;
- SAC actor and critic losses and entropy coefficient;
- actor parameter drift, historical-alpha entropy, and alpha-mass (weight modes);
- selected merge-pair symmetric KL, pool-wide KL statistics, pool length, merge time, and merge-lineage composition heatmaps;
- distillation train/held-out KL plus mean/log-std MSE diagnostics;
- zero-shot performance **before** training each new task;
- checkpoint-by-task retention heatmaps after every continual task;
- average performance on all tasks seen so far;
- average forgetting (`best previous return - current return`);
- final checkpoint per-task return, velocity error, and success;
- final return-vs-forgetting and return-vs-tracking-error trade-off plots;
- task training, merge-buffer collection, and finalize/merge timings;
- `summary_metrics.csv` aggregated per seed/condition.

Retention evaluation is cached, so rerunning the plotting script does not repeat all MuJoCo evaluation unless the cache is removed.

## Single-task / manual chain runs

`run_sac.py` can still be called directly. For the first task:

```bash
python3 run_sac.py \
  --task-suite halfcheetah_vel \
  --task-id 0 \
  --fusion-mode classic_cka \
  --no-distillation \
  --save-dir agents/manual/root
```

For a later task, pass the root checkpoint and latest checkpoint through `--prev-units`. The benchmark runner does this automatically and is less error-prone.

## Important implementation fixes in this version

- Mean/log-std pools can no longer choose different merge pairs.
- Both heads share one alpha vector, matching one coefficient per knowledge item.
- Merge selection compares **full policy distributions**, not flattened weights.
- Distillation jointly optimizes both Gaussian heads with KL instead of unrelated head-wise MSE targets.
- Distillation teachers are reconstructed from the actual pool entries, avoiding the old mismatch between a stored task delta and targets from the whole historical mixture.
- The shared encoder is frozen after the root task, so historical policy behavior does not silently drift when later tasks train.
- Periodic evaluation uses a separate environment instead of resetting/stepping the training environment and corrupting replay transitions.
- Gymnasium `final_observation` / autoreset layouts are handled across old and new API variants.
- Logging no longer reads undefined actor/temperature losses on non-policy-update steps.
- Every condition stores raw rollout states because behavioral KL is required even when distillation is disabled.
- The old MetaWorld-specific benchmark/result scripts have been replaced or deprecated in favor of the new HalfCheetah runner.

## Main files

```text
cka_rl.py                    aligned policy pool, behavioral KL, joint distillation
knowledge_pools.py           one-head tensor storage for aligned policy slots
policy_utils.py              bounded SAC log-std + numerically stable Gaussian KL
tasks.py                     HalfCheetahVel / HalfCheetahWindVel task sequences
halfcheetah_envs.py          target-velocity reward + hidden MuJoCo wind
run_sac.py                   SAC training for one continual task
run_continual_benchmark.py   all four modes, multiple seeds, evaluation and plots
analysis_logging.py          TensorBoard and boundary snapshots
sanity_check_pool.py         fast CPU-only structural tests
run_recommended.sh           recommended 3-seed benchmark command
```
