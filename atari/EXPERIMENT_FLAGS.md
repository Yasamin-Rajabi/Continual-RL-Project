# Experiment / ablation flags

The default code favors a stable final implementation, but every small optimization
that can change learning is exposed so it can be disabled in ablations.

## Shared encoder

| Flags | Meaning |
|---|---|
| `--no-train-shared --no-freeze-root-encoder` | Scratch root encoder learns on task 0, then is frozen/reloaded for later tasks. Default without pretraining. |
| `--pretrained-encoder PATH --no-train-shared` | Optional legacy/pretraining ablation: freeze the supplied encoder from task 0 onward. |
| `--train-shared` | Friend-workflow encoder rule: learn/fine-tune the encoder continually; later tasks start from the latest encoder, never reset to the root/pretrained file. No pretraining is required. |
| `--freeze-root-encoder` | Random-frozen encoder ablation. Mutually exclusive with `--train-shared`. |
| `--encoder-linear-out` | Encoder ends in a linear layer. Retained for optional pretraining/architecture ablations; serialized modules are validated at load time. |
| `--no-encoder-linear-out` | Original encoder with trailing ReLU. |

The present code is **not an exact reproduction of paper CKA-RL**, because the
knowledge vectors cover the policy heads while `fc` is a separate shared encoder.
A structurally coherent paper-like baseline is therefore:

```bash
--condition-index 1 \
--no-train-shared --no-freeze-root-encoder --encoder-from-base \
--no-encoder-linear-out --no-use-alpha-scale \
--no-collect-cosine-buffers
```

This learns the root representation once, fixes it thereafter, and keeps classic
CKA/cosine/arithmetic merging free of the new stabilizers.

## Four method conditions

1. `baseline`: `classic_cka`, cosine pair selection, arithmetic merge.
2. `distil_only`: `classic_cka`, symmetric-KL pair selection, policy distillation.
3. `weight_only`: `weight_delta`, cosine pair selection, arithmetic merge.
4. `combined`: `weight_delta`, symmetric-KL pair selection, policy distillation.

`--weight-use-alpha-mass/--no-weight-use-alpha-mass` controls whether conditions
3/4 also use learned alpha mass. Disable it to isolate the representation change.


## Friend-method controls merged from latest master

The latest master workflow keeps pretraining support only as an optional legacy/ablation path;
the supplied Kaggle notebook runs from scratch with `--train-shared`.  The following controls
make the new method changes individually switchable:

| Default in condition runner | Ablation | Effect |
|---|---|---|
| `--condition-alpha-scale` | `--no-condition-alpha-scale` | Classic modes learn alpha scale from 1; weight-delta modes use fixed scale 5. Disabling this lets `--use-alpha-scale/--fix-alpha-scale` choose one global rule. |
| `--distill-observation-skip` | `--no-distill-observation-skip` | Distillation modes concatenate raw observation to shared features before the policy heads. |
| `--alpha-lr 5e-3` | change value | Separate learning rate for knowledge-mixture parameters. |
| `--alpha-warmup-steps 5000` | `0` | Early weight-delta phase learns historical mixing before the new residual/mass moves. It activates only when at least two historical slots exist. |
| `--alpha-mass-reg 0.05` | `0` | Regularizes effective mass after warmup. |
| `--drift-reg 1.0` | `0` | When `--train-shared` and distillation are both active, penalizes encoder drift on historical observations. |
| `--distill-encoder-lr-mult 0.1` | `1.0` | Slows the shared encoder on later distillation tasks. |
| `--alpha-entropy-reg 0.01` | `0` | Entropy bonus on the **actual scaled** historical mixture during effective weight-delta warmup. |
| `--test-adapt-steps 5000` | `0` | Test-time adaptation of alpha-only mixture parameters for retention/final-row metrics. |

`--test-adapt-steps 0` restores frozen-checkpoint evaluation. FG/BWT now re-evaluate both the
diagonal checkpoint and final checkpoint with the same adaptation protocol and common episode
seeds. Forward transfer remains defined only on first encounters of previously unseen tasks;
repeated encounters are relearning/savings.

## Small optimizations / legacy switches

| Default | Disable for ablation | Effect |
|---|---|---|
| `--constrain-alpha-mass` | `--no-constrain-alpha-mass` | Positive normalized-softplus mass instead of a scalar that can hit zero/negative. |
| `--distill-select-best-val` | `--no-distill-select-best-val` | Restore best held-out-KL student epoch instead of always keeping the last epoch. |
| `--no-collect-cosine-buffers` | `--collect-cosine-buffers` | Cosine-only modes skip unused post-training rollout states; enable to equalize post-training interaction counts. |
| `--no-use-alpha-scale` | `--use-alpha-scale` | Optional learned global scaling of historical alpha logits. |
| `--autotune --no-autotune-init-from-alpha` | `--autotune-init-from-alpha` | Legacy entropy autotuning starts at alpha=1; optional flag starts at `--alpha`. |

If `--no-autotune` is used, `--alpha` is the fixed SAC entropy coefficient.

## Reproducibility and resuming

Every task checkpoint now contains `run_manifest.json`. A checkpoint is resumable
only if its training configuration, training-source fingerprint, runtime package
versions, pretrained encoder contents, and parent checkpoint identities still
match. Old pre-manifest checkpoints are intentionally considered stale.

Scratch baselines for Forward Transfer must use the same encoder treatment, SAC settings,
evaluation cadence, training budget, and **actor-head architecture** as the continual run.
`scratch_baselines.py` therefore caches two variants by default: `plain` for Baseline/Weight-Only
and `distill_skip` for Distill-Only/Combined when observation skip is enabled. The default
`run_kaggle.sh` follows the current no-pretraining `--train-shared` workflow; TD-JEPA remains
available only as an explicit ablation.

## Diagnostics retained for upcoming TODOs

- `source_ids` identify sequence occurrences separately from semantic `task_ids`.
- merge snapshots/logs preserve task-level and source-occurrence lineage.
- KL logs include mean, p95, and max tails for selected merge pairs and distillation.
- parent-balanced replay/distillation is intentionally **not** lineage-balanced yet;
  that behavior is the control for the upcoming exponential-decay investigation.
- critic reset/persistence is intentionally unchanged pending the critic TODO.




////////////////

I inspected the plots carefully. My first impression is: **I do not think this is primarily a PPO hyperparameter problem.** There are signs that the PPO optimizer is behaving normally; the problem seems to be in **the continual composition / alpha mechanism / knowledge transfer schedule**. The good news is that the diagnostics are actually very informative.

I will go through each observation and then suggest a tuning plan.

---

## 1. The main failure: catastrophic behavior after task transitions

The evaluation curve is the most important one:

![Image](https://images.openai.com/static-rsc-4/_ey6CBmGhw33Cy8V4CMpd4qAFEm0fpj0eUJiK1Oa_I65k83bDRYbl0uAt2R_YEXKgiwGMELbr425xLR_NFIS-Br4phdRKwM1jJHm_qSBDYj0s0dqYa0HbMY9oyxM0FKVjyTtFfBGxnzetbWnsE0D6zahUBQHO9qUzqdTVJE56YbQ2XHSpeckaltwtcmSkidd?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/4hG8BoDEB3qc3z4attPvLMbpA5Gslm2kJCVIDn5bzeyKxG2RbWCHVMrSzsCSLxXt6GCYomadgWYazPGpKBTqD3kJhLAom9OhwCmZ4MSH_4N9Yw4Fp3RZiFcDiR-Qb1HOKTOPudm0EFfr9bVHPLmHowWlOT9pB_gzKhpovwzUQVFSQVccmf1BogcCwITcPAyv?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/d-7ufRs-4CwGVLJ-NBNd57r6Zy_L9ZmNSATpCA_IkW1l5fMH6Uscea0JyImYGQK22lM30AQvX1DuL6m3Fkk7cbDZKXfm8Qm4kc1iFo4GIqwxbdyjB8yxiDmFsIAHCJVf7gU1JuilyO_asAyxjA4kPs4lQ3vL7GEOO47pLZGviY93lz5XkcVMPzZthrPqeMBz?purpose=fullsize)

Your reward:

* Task 0 starts around **22**
* After transition to task 1 → drops to ~5
* Task 2 → recovers partially
* Task 4 → recovers to ~22
* Task 5/6 → collapses again
* Final tasks remain weak

The heatmap confirms this:

* Task 0 retention: excellent
* Task 4 retention: excellent
* Task 1/5/6/7: weak

This is **not what I expect from bad PPO learning**.

If PPO was the issue, we would see:

* high policy loss instability,
* exploding entropy,
* bad learning curves everywhere.

Instead:

Your PPO loss:

* stabilizes quickly
* becomes almost zero after learning phases

So PPO is converging.

The issue is:

> the merged policy is choosing a bad mixture after some tasks.

---

# 2. Alpha mass is almost frozen

This plot is the biggest clue:

### Historical-mixture alpha mass

Your alpha mass stays:

```
0.94 - 0.96
```

almost everywhere.

Meaning:

The mixture gate is saying:

> "Use almost only previous historical policy."

This is happening even when the current task needs adaptation.

For continual RL, this is dangerous.

The current task contribution is too weak.

Your method:

```
new policy =
alpha_old * old knowledge
+
alpha_new * current task
```

but your alpha:

```
alpha_old ≈ 0.96
alpha_new ≈ 0.04
```

So the new task is barely entering.

---

## Change #1: increase alpha learning rate

Currently:

```bash
--alpha-learning-rate 2.5e-4
```

I would try:

```
--alpha-learning-rate 1e-3
```

or even:

```
--alpha-learning-rate 2.5e-3
```

for Atari.

Why?

Your PPO LR is:

```
2.5e-4
```

but alpha is a tiny vector. It needs faster adaptation.

---

# 3. Alpha entropy regularization is too strong

Currently:

```python
--alpha-entropy-reg 0.01
```

This encourages uniform mixtures.

But your result shows the opposite problem:

The model keeps old mixture.

Try:

```
--alpha-entropy-reg 0.001
```

or:

```
0
```

I would start:

```bash
--alpha-entropy-reg 0.001
```

---

# 4. Alpha mass regularization is probably hurting you

Current:

```bash
--alpha-mass-reg 0.05
```

Your alpha mass curve tells me the constraint is too strong.

This regularizer:

```text
keep alpha mass stable
```

is fighting:

```text
adapt to new task
```

For Atari I would test:

```
0.005
```

instead of:

```
0.05
```

Ten times smaller.

---

# 5. Alpha initialization is probably wrong

You use:

```bash
--alpha-init Randn
```

From your alpha entropy curve:

The first tasks start with low entropy and then suddenly jump.

This suggests the initial mixture is not well behaved.

I would change:

```
--alpha-init Uniform
```

Why?

For a new task:

You do not know whether:

* task 0
* task 1
* task 2

is useful.

Uniform gives:

```
1/N
```

starting point.

Then learning decides.

---

# 6. Pool size

Current:

```bash
--pool-size 5
```

For Freeway:

you have:

```
11 sequential tasks
```

with repeats.

A pool of 5 means:

many merges happen.

Your lineage plot shows:

```
2 -> 3 source lineages
```

after some merges.

I would try:

```
pool-size 8
```

or:

```
pool-size 10
```

for the final experiment.

Your contribution is bounded memory, but 5 might simply be too aggressive.

---

# 7. Distillation seems too weak, not too strong

Look at:

```
Distillation held-out KL
```

Values:

```
1e-4 ~ 7e-4
```

This is very small.

Meaning:

student ≈ teacher.

So distillation is not destroying behavior.

The issue is:

the teacher itself is wrong because alpha selection is wrong.

I would NOT tune:

```
distill-lr
distill-epochs
```

yet.

---

# 8. PPO changes I would make (secondary)

Only after alpha fixes.

Current:

```
num_steps=128
num_envs=8
batch=1024
```

For Atari this is small.

I would use:

```
num_envs=16
num_steps=256
```

giving:

```
4096 rollout batch
```

Then:

```
num_minibatches=8
```

This usually gives more stable Atari PPO.

So:

```bash
--num-envs 16
--num-steps 256
--num-minibatches 8
```

---

# My first tuning experiment

I would NOT change everything.

Run this:

```bash
python run_continual_benchmark.py \
 --task-suites freeway \
 --condition-index 4 \
 --seeds 1 \
 --total-timesteps 1000000 \
 --pool-size 8 \
 --alpha-init Uniform \
 --alpha-learning-rate 1e-3 \
 --alpha-entropy-reg 0.001 \
 --alpha-mass-reg 0.005 \
 --distill-lr 3e-4 \
 --distill-epochs 8
```

Keep PPO unchanged first.

---

# Second experiment if still bad

Then:

```bash
--num-envs 16
--num-steps 256
--num-minibatches 8
```

---

# One more thing I noticed

Your orchestrator default:

```python
--alpha-factor=1e-3
```

combined with:

```python
alpha-scale
```

may be important.

For weight-delta mode you have:

```python
fix_alpha_scale=True
```

and:

```
alpha scale = 5
```

I want to verify this because your alpha mass curve suggests the fixed scale may be dominating.

Could you send me:

1. the `training_meta.json` from this run
2. the exact command used to launch this seed
3. the values of:

   * alpha_scale
   * alpha_factor
   * alpha_warmup_steps
   * composition_space

Then I can tell whether the failure is from:

* alpha optimization,
* weight-delta scaling,
* or the merge-selection mechanism.

Right now my strongest hypothesis is:

**alpha adaptation is too conservative; the method is preserving old knowledge but not allowing enough new-task plasticity.**
