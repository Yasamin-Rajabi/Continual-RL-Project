# Replay-trained standalone policy student

## Flag

Use `--policy-student-replay` with the combined policy-space method only:

```bash
python run_continual_benchmark.py \
  --condition-index 4 \
  --composition-spaces policy \
  --policy-student-replay ...
```

The benchmark labels this condition `combined_policy_student`, so it never reuses ordinary `combined_policy` checkpoints.

## Intended semantics

Historical policies remain frozen.  Let

- `pi_i` be frozen historical Gaussian policy heads,
- `alpha_i = softmax(logits)_i`,
- `m = sigmoid(alpha_mass)` be historical mass,
- `pi_new` be the standalone trainable current expert.

After pool-search warmup, the behavior/execution policy is

`pi_exec = m * sum_i alpha_i pi_i + (1-m) * pi_new`.

The execution mixture is used for environment actions, replay collection, critic targets, entropy-temperature tuning, and evaluation.

## Alternating actor update

At every SAC actor-update cycle after warmup:

1. **Novel expert step.** Sample replay states and optimize only `pi_new` with the standard SAC actor loss

   `E_s,a~pi_new [ alpha_SAC log pi_new(a|s) - Q(s,a) ]`.

   The replay states were collected by `pi_exec`. Historical heads and alpha/gate parameters are not in this loss.

2. **Routing step.** Recompute the execution-mixture objective after the novel step and optimize only `alpha` and `alpha_mass`. Component Gaussian means/log-stds are detached in this step, while mixture weights and exact mixture log-density remain differentiable.

Thus the current expert updates first, then the routing adapts to the improved current expert.

During the initial pool-search warmup, `pi_new` is excluded and frozen. Only historical alpha coefficients are optimized, matching the existing historical-only search phase.

## Why alpha-mass can go to zero

`m` is historical mass. Therefore:

- `m -> 1`: execute almost entirely from historical experts;
- `m -> 0`: execute almost entirely from `pi_new`.

If the standalone current expert becomes better than the historical mixture, the routing SAC objective can push `m` toward zero. The existing double-well regularizer keeps 0 and 1 as preferred extremes but does not decide which extreme is better; the RL objective decides that.

Effective mass is already logged as `analysis/mean/alpha_mass` and is plotted by the benchmark.

## Storage behavior

This variant intentionally does **not** project the final execution mixture into a new Gaussian head. It inserts `pi_new` itself as the new pool entry.

That is essential to the hypothesis being tested: if a projected mixture were stored, residual dependence on historical policies could be hidden by projection. Direct storage makes failure visible when alpha-mass remains high or retention degrades after finalization.

The ordinary `combined_policy` method is unchanged and still projects its final execution mixture before pool insertion.

## What is not added

There is no behavior-cloning or teacher-KL term in this variant. The novel expert learns off-policy through SAC from replay states/critic values. This keeps the comparison focused on decoupling behavior-policy execution from standalone-expert optimization.

## Diagnostics

New TensorBoard losses:

- `losses/novel_actor_loss`
- `losses/mixture_weight_actor_loss`

Existing useful diagnostics:

- `analysis/mean/alpha_mass`
- historical alpha weights/entropy
- final success/return
- retention matrices and A_N / FG / BWT / FT
