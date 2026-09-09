# Running and evaluating the revised code

Use fresh checkpoints, scratch baselines, pretrained encoders and metric caches.
Old results do not establish performance after removing context or changing
composition. Run commands from the relevant environment directory, not the
repository root; the original project uses local unqualified module imports.

## 1. Setup and smoke checks

Use Python 3.10 or newer. Each environment directory has requirements.txt;
requirements-dev.txt adds pytest. MetaWorld retains its pinned installation
procedure in `run_kaggle.sh setup`; its requirements file alone deliberately
does not install MetaWorld/MuJoCo. Review that script for the target environment.
Installation and real simulator compatibility were not validated here.

From the repository root, CPU structural tests do not require simulators:

```bash
python -m pytest -q tests/test_policy_space.py
CRL_FAMILY=Walker2D python -m pytest -q tests/test_policy_space.py
CRL_FAMILY=Hopper python -m pytest -q tests/test_policy_space.py
CRL_FAMILY=metaworld python -m pytest -q tests/test_policy_space.py
```

Then from an environment directory, with its real dependencies installed:

```bash
python sanity_check_pool.py
python tasks.py --check
python run_continual_benchmark.py --quick-test --skip-forward-transfer
```

The quick test exercises insertion/merging at a reduced budget; it is not a
statistical performance experiment. For Walker2D, `python walker2d_smoke.py`
also checks that configured physical perturbations were applied. The synthetic
harness in `tests/synthetic_smoke.py` deliberately substitutes environment/replay
adapters and must not be described as a MuJoCo or MetaWorld test.

## 2. Main comparison

Every command below runs baseline and combined in **both** composition spaces.
The examples use frozen, post-consolidation pool evaluation without test updates.
Choose a per-family Delta before running, and use the same Delta and B for all
conditions and their scratch baselines. The three locomotion examples below use
300000; MetaWorld retains 150000 as a pilot default, not a paper reproduction.

```bash
# From half-cheetah/
python run_continual_benchmark.py \
  --task-suites halfcheetah_vel halfcheetah_wind_vel \
  --condition-index 1 4 --composition-spaces parameter policy \
  --seeds 1 2 3 4 5 --total-timesteps 300000 --distill-buffer-steps 10000 \
  --retention-eval-episodes 20 --test-adapt-steps 0 --skip-forward-transfer

# From Walker2D/
python run_continual_benchmark.py \
  --task-suites walker2d_dynamics \
  --seeds 1 2 3 4 5 --total-timesteps 300000 --distill-buffer-steps 10000 \
  --retention-eval-episodes 20 --skip-forward-transfer

# From Hopper/ (new benchmark extension)
python run_continual_benchmark.py \
  --task-suites hopper_dynamics \
  --seeds 1 2 3 4 5 --total-timesteps 300000 --distill-buffer-steps 10000 \
  --retention-eval-episodes 20 --skip-forward-transfer

# From metaworld/ (paper TASK LIST, not exact reproduction)
python run_continual_benchmark.py \
  --task-suites mw_paper10 \
  --seeds 1 2 3 4 5 --total-timesteps 150000 --distill-buffer-steps 10000 \
  --retention-eval-episodes 20 --skip-forward-transfer
```

The five seeds and twenty evaluation episodes are suggested comparison settings,
not a claim that this sample size resolves every performance difference. Pilot
learning curves should be checked before committing to the full budget. Hopper
and the harder MetaWorld tasks especially need real-environment validation.

Each directory also has `run_comparison.sh`. The no-argument Python benchmark
uses three seeds and three retention episodes; those are convenient defaults,
not a publication recommendation. `--pool-size` defaults to 5 in locomotion and
4 in MetaWorld. Select a value smaller than the sequence length so consolidation
is actually exercised.

The main comparisons answer different questions:

| Comparison | What it tests |
|---|---|
| baseline vs combined | Complete proposed parameter-space method vs classic implementation |
| baseline_policy vs combined_policy | Consolidation/storage methods under policy-mixture adaptation |
| combined vs combined_policy | Parameter composition vs policy composition plus required insertion projection |
| baseline vs baseline_policy | Classic parameter method vs its new mixture/projection analogue |

Do not call the last two pure inference-only ablations: policy-space training,
SAC density, adaptation semantics and bounded-storage projection differ too.
For a component ablation later use `--condition-index 0 --composition-spaces
parameter` and hold all additional knobs constant.

## 3. Forward transfer requires fresh matched scratch curves

The default benchmark computes retention even when scratch checkpoints are
missing, and reports FT as unavailable rather than silently reusing mismatched
baselines. `--skip-forward-transfer` avoids the scratch lookup entirely.

For example, from half-cheetah/:

```bash
python scratch_baselines.py \
  --task-suites halfcheetah_vel halfcheetah_wind_vel \
  --variants plain --seeds 101 102 103 \
  --total-timesteps 300000 --distill-buffer-steps 10000

python run_continual_benchmark.py \
  --task-suites halfcheetah_vel halfcheetah_wind_vel \
  --seeds 1 2 3 4 5 --scratch-seeds 101 102 103 \
  --total-timesteps 300000 --distill-buffer-steps 10000 \
  --retention-eval-episodes 20 --skip-training
```

Use the equivalent suite/budget in the other directories. `--skip-training`
requires complete matching checkpoints from the main comparison. Match replay
collection schedule, SAC settings, encoder/skip architecture, evaluation action
mode and B, not just Delta. Scratch uses the same frozen tail. One plain scratch
architecture suffices for both composition spaces because an empty history is
a single Gaussian policy. Opting into the observation skip needs matching
`distill_skip` scratch curves for the distillation conditions.

New run identities reject incompatible reused checkpoints. Save alternative
experiments under different roots rather than relying on automatic retraining,
which removes stale partial outputs at the same paths.

## 4. Exploration/warmup experiment

Keep everything else, including the seed list and pool capacity, fixed:

```bash
# Policy controls the environment during approximately steps 5k-10k.
--learning-starts 5000 --random-actions-end 5000 --alpha-warmup-steps 5000

# Legacy collection: random actions through the same warmup interval.
--learning-starts 5000 --random-actions-end 10000 --alpha-warmup-steps 5000
```

These fragments are flags to append to a benchmark command. Use separate
`--save-root`, `--runs-root`, `--plots-root`, `--analysis-root` and
`--scratch-save-root` values for the two experiments. Retrain matched scratch
baselines with each collection schedule when computing FT. Compare early
learning AUC, final A_N, FG/BWT and mixing-weight histories. The first setting
still uses off-policy replay; old random transitions are not filtered out.

## 5. Metrics already implemented

Let `S[t,j]` denote checkpoint t evaluated on unique task j, under one fixed
checkpoint policy/action mode. Let `d[t] = S[t, task_at_position_t]` and
`f[j] = S[last,j]`. All success values are in [0,1].

### Final performance, forgetting and backward transfer

**A_N** is the mean `f[j]` over unique task IDs. **FG** is the mean of
`max(d[t] - f[task_t], 0)` over all sequence positions except the last. **BWT** is
the mean of `f[task_t] - d[t]` over the same positions, without clipping.
Higher A_N/BWT and lower FG are better. If every old task only loses success,
BWT equals minus FG; reporting both is not two independent pieces of evidence.

The repeated-task convention matters: A_N counts each unique task once;
FG/BWT compare every non-final training occurrence. A repeatedly encountered
task can therefore contribute multiple terms. This is not the same as a
unique-task-only forgetting definition.

The retention plot/summary also computes **peak-to-final forgetting**: for a
unique task, its best recorded success after it was first learned minus final
success. This is a separate diagnostic, not necessarily identical to FG.

### Forward transfer

Normalized learning-curve AUC comes from monitored `charts/test_success` or
`charts/test_episodic_return`, including the initial and final checkpoints.
FT is averaged only over first encounters of previously unseen tasks after the
root task. Repeated-task relearning is not counted as forward transfer.

    FT_success[j] = (AUC_success[j] - AUC_success_scratch[j])
                    / (1 - AUC_success_scratch[j])

    FT_return[j] = (AUC_return[j] - AUC_return_scratch[j])
                   / (U - AUC_return_scratch[j])

`U=0` for these HalfCheetah negative tracking rewards. `U=1000` for Walker2D
and Hopper, whose maximum healthy reward is 1 per step over 1000 steps, before
error/control penalties. MetaWorld retains its shaped-return scale `U=2000`
(10 times its 200-step horizon). Zero/no-positive headroom is not divided by.
The old `1 - AUC/AUC_scratch` shortcut is valid only for U=0; it was not carried
over incorrectly to survival-reward environments. FT_success is usually easier
to compare across task families because reward scales differ.

FT monitoring evaluates the active training policy, while default retention
evaluates the finalized pool. They answer acquisition and consolidated-retention
questions respectively; do not treat them as the same checkpoint object.

### Other outputs

The code records return, success, per-task and sequence retention matrices,
zero-shot initial performance, velocity error for locomotion and task/object
error for MetaWorld where provided. Locomotion success is the episode's fraction
of steps satisfying its threshold (health is also required in Walker/Hopper),
then averaged across episodes. Early termination shortens this denominator;
return and survival diagnostics should accompany success. MetaWorld success is
whether success occurred at any point during the episode, not the fraction of
steps after success was first latched.

Pool diagnostics include pair similarities, merge lineage, distillation losses,
pool size, parameter norms, mixing logits/mass/entropy and wall-clock timing.
Policy-space runs additionally save projection validation/training component-KL.
Those fit diagnostics are not substitutes for measured returns.

## 6. Evaluation protocols must be labeled separately

`--test-adapt-steps 0 --frozen-eval-policy pool` is the main no-test-update setting:
uniform composition of the finalized pool, no task-ID routing, zero residual
and no novel expert. This actually includes the final consolidation event.

`--frozen-eval-policy snapshot --test-adapt-steps 0` evaluates the exact active
policy saved before insertion/final merging. It answers a different question
and can hide final-consolidation loss; present it separately.

`--test-adapt-steps K --frozen-eval-policy pool` spends K reward interactions on
each evaluated checkpoint/task before scoring. This retains the code's
immediate-reward REINFORCE heuristic for logits (and a previously learnable
scale), now with the correct mixture likelihood. It is not a task-inference
module, full SAC retraining, or return-to-go policy-gradient method. Historical
heads remain frozen; mass stays one because evaluation has no new expert.
Report K explicitly and never put these numbers in a "no interaction" table.

`--eval-action-mode deterministic` is the default. Policy mixtures use weighted
squashed component means; this can produce a compromise action that no component
would choose. `--eval-action-mode stochastic` evaluates the actual mixture.
A useful secondary result is stochastic frozen-pool success/return across enough
episodes. Match action mode in scratch and continual learning curves for FT.
Changing it also changes the run identity for monitored curves, so do not reuse
mismatched FT logs.

## 7. Reporting

Report mean and standard deviation across training seeds, individual seed
points where possible, per-task final success/return, the retention heatmap,
A_N, FG/BWT and FT_success with its matched scratch protocol. Include Delta,
B, pool capacity, number of tasks/occurrences, policy composition, evaluation
object, action mode and any test-adaptation budget. Also report runtime and
projection cost: equal environment interactions do not imply equal compute.

A meaningful claim of better consolidation needs more than low KL: check
post-consolidation retention under the same evaluation rule. Do not reuse the
draft's earlier performance tables as evidence for this revised implementation.

## Replay-trained policy student variant

The updated code adds `--policy-student-replay` for the combined policy-space condition. The execution Gaussian mixture controls the environment, while the standalone current expert is optimized from replay with SAC; alpha and alpha-mass are then optimized in a separate routing step. The new condition is labeled `combined_policy_student`, stores the standalone novel expert directly, and leaves ordinary `combined_policy` unchanged. See `POLICY_STUDENT_REPLAY.md` for the precise update order and equations.
