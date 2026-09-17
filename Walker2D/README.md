> **Updated implementation:** read `../IMPLEMENTATION_NOTES.md` and
> `../EVALUATION_GUIDE.md` first. The material below describes the uploaded
> project's older presets; its old defaults and performance claims do not
> validate this revised task-blind/policy-space implementation. Use
> `run_comparison.sh` for the new defaults and train fresh checkpoints.

# CKA-RL Walker2D continual-dynamics benchmark

This directory is a Walker2D sibling of the project's revised `half-cheetah`
directory. The CKA-RL / SAC / knowledge-pool / distillation logic was carried
over from that mature directory. The environment/task definitions, defaults,
Kaggle entry points, diagnostics, and smoke tests were adapted for Walker2D.

## What is being evaluated

The default suite is `walker2d_dynamics`. Every task has the **same objective**:
stay healthy and track a forward velocity of 1.5 m/s. Only robot/contact
dynamics change. This avoids creating a fake merge gap from contradictory
rewards while still producing specialists that may need different gaits.

Every step uses

    reward = healthy_reward - abs(x_velocity - 1.5) - control_cost

and preserves Walker2d-v5's normal unhealthy termination. Direct construction
is wrapped in a 1000-step `TimeLimit`.

The policy and critic both receive a six-value task vector appended to the raw
Walker observation:

    [target_velocity,
     right_mass_scale,
     left_mass_scale,
     foot_friction_scale,
     joint_damping_scale,
     actuator_strength_scale]

Walker2d-v5 contributes 17 raw observation values and 6 actions, so this
benchmark uses observation shape `(23,)` and action shape `(6,)`.

### Default moderate suite

| ID | Name | Right mass | Left mass | Foot friction | Joint damping | Motor strength |
|---:|---|---:|---:|---:|---:|---:|
| 0 | nominal | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1 | right-heavy | 1.25 | 0.90 | 1.00 | 1.00 | 1.00 |
| 2 | left-heavy | 0.90 | 1.25 | 1.00 | 1.00 | 1.00 |
| 3 | slippery-feet | 1.00 | 1.00 | 0.70 | 1.00 | 1.00 |
| 4 | high-damping | 1.00 | 1.00 | 1.00 | 1.50 | 1.00 |
| 5 | weak-motors | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 |

The default continual sequence is `0 1 2 3 4 5 0 1 2 3 4 5`. With the normal
pool size of 5 this forces consolidation and then tests a second pass over the
same tasks.

An optional `walker2d_mixed_dynamics` suite is also included. It combines
moderate perturbations and is intended only after the default suite has been
validated.

## Important implementation details

- Right/left leg body mass and body inertia are scaled together.
- Foot friction is found by the geoms attached to `foot` and `foot_left`; it
  does not depend on fragile foot-geom XML names.
- Damping is changed on the six Walker2D actuated joints.
- Motor strength is changed through MuJoCo actuator gear scaling.
- `mujoco.mj_setConst` and `mujoco.mj_forward` are called after the model edits.
- Actual unhealthy Walker falls are stored as terminal transitions; 1000-step
  time-limit truncations continue to bootstrap. This inherits the corrected
  replay handling in the revised HalfCheetah code.
- Evaluation uses a separate environment and logs Walker-specific
  `test_episode_length` and `test_fall_rate` in addition to return, velocity
  error, and success.
- Run manifests/fingerprints include `walker2d_envs.py`, so changing environment
  code invalidates stale checkpoints instead of silently reusing them.

## macOS local test

Python 3.11 is the safest choice for a clean test environment. Python 3.12 is
also reasonable if all wheels install normally.

From Terminal:

```bash
cd /path/to/Walker2D
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
bash local_smoke_test.sh
```

On Apple Silicon, the smoke benchmark deliberately runs the tiny training test
on CPU. You do not need CUDA. MuJoCo's Python package is installed through
`requirements.txt`.

If `python3.11` is not installed, use a compatible `python3` you already have,
or install Python 3.11 first and repeat the commands above.

### What the local test checks

`local_smoke_test.sh` performs four layers of validation:

1. imports and prints Python / Torch / Gymnasium / MuJoCo / SB3 versions;
2. runs `sanity_check_pool.py`, exercising CKA pool insertion, pair selection,
   arithmetic merging, behavioral-KL selection, distillation, alpha-mass,
   shared-encoder continuation, and checkpoint/model-selection guards;
3. runs `tasks.py --check` and `walker2d_smoke.py`, which instantiate every
   Walker task, verify `(23,)` observations and `(6,)` actions, numerically
   confirm each dynamics scale, and execute random MuJoCo rollouts looking for
   NaNs or reset/termination problems;
4. runs a deliberately tiny three-task CKA-RL benchmark with pool size 2, so the
   third task **must trigger a merge**. It runs Baseline and Combined end to
   end, including the distillation path.

The final line should be:

    SMOKE TEST PASSED

The 4k-step training returns from this test are **not** a learning-quality test.
The important things are that no assertion/exception occurs, the dynamics
checks say `[ok]`, training reaches the merge/distillation path, and checkpoint
files are written.

If macOS rendering/OpenGL creates trouble, do not enable rendering for this
smoke test; all included tests are headless.

## Kaggle

You can attach either the unzipped `Walker2D` directory as a Kaggle Dataset or
the supplied `Walker2D_CKA-RL.zip`. `kaggle_script.ipynb` detects either form,
copies/extracts it into `/kaggle/working/walker2d`, and runs from there.

Recommended notebook order:

```bash
bash run_kaggle.sh setup
bash run_kaggle.sh sanity
bash run_kaggle.sh pilot
```

The pilot uses four tasks, Baseline + Combined, 20k training steps/task and an
actual pool merge. It is for integration validation, not final conclusions.

After the pilot is healthy, the full moderate benchmark can be launched with:

```bash
TOTAL_TIMESTEPS=200000 \
SEEDS="101 102 103" \
SCRATCH_SEEDS="201 202" \
bash run_kaggle.sh all
```

The packaged `run_kaggle.sh` default is 150k steps/task if
`TOTAL_TIMESTEPS` is not supplied. For the first real learning curves, 150k is
a useful checkpoint. If specialist curves are still clearly improving there,
use 200k (and, only if necessary, 300k) before interpreting merge quality.
Do not deepen the perturbations merely to make Baseline fail until individual
specialists themselves are learning reliably.

`bash run_kaggle.sh all` runs:

- the structural/task sanity checks;
- from-scratch `plain` and `distill_skip` single-task baselines needed for
  forward-transfer/survey metrics;
- continual Baseline and Combined runs on the full default sequence.

`run_recommended.sh` is an alternative full-development command and defaults to
200k steps/task and three continual seeds.

## What healthy learning should look like

The custom reward has a healthy bonus of 1 per step. A perfectly tracking,
healthy 1000-step episode with negligible control cost therefore approaches a
return around 1000. This is only a scale reference, not an expected threshold.

For a properly learning specialist, the more diagnostic trends are:

- episode length should rise toward the 1000-step limit;
- fall rate should decrease;
- absolute velocity error should decrease;
- success rate should increase;
- episodic return should increase and eventually stabilize.

A low return accompanied by very short episodes / high fall rate indicates a
balance-learning problem. Long episodes but poor velocity error indicate that
the agent learned survival without the target gait. Those two failure modes
should be diagnosed separately.

## Results to send back for review

Please send the following after the macOS smoke test and/or Kaggle pilot:

- full stdout from `tasks.py --check` and `walker2d_smoke.py`;
- the end of the `sanity_check_pool.py` output;
- any exception traceback, if present;
- for the pilot/real run: learning curves or TensorBoard values for
  `charts/test_episodic_return`, `charts/test_velocity_error`,
  `charts/test_success`, `charts/test_episode_length`, and
  `charts/test_fall_rate`;
- the Baseline-vs-Combined plots/CSV from the pilot or main run;
- merge/distillation logs, especially selected pair, symmetric KL, held-out KL,
  and selected distillation epoch.

Those are enough to distinguish an environment bug, an SAC convergence issue,
a too-strong task perturbation, and a genuine consolidation difference.

## Files most relevant to the port

- `walker2d_envs.py` — Walker2D environment, reward, task conditioning, dynamics
  perturbations.
- `tasks.py` — moderate and optional mixed task suites.
- `walker2d_smoke.py` — live MuJoCo/dynamics assertions.
- `local_smoke_test.sh` — macOS integration test.
- `run_kaggle.sh` — Kaggle setup/pilot/full entry point.
- `kaggle_script.ipynb` — ready-to-run Kaggle notebook.
- `run_continual_benchmark.py` — four-way orchestration, inherited from the
  revised HalfCheetah version with Walker defaults.
- `run_sac.py` — inherited SAC/CKA-RL training plus Walker fall/episode metrics.
- `cka_rl.py`, `knowledge_pools.py`, `shared_arch.py`, `policy_utils.py` — mature
  consolidation logic retained from the revised HalfCheetah directory.
