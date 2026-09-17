# Implementation notes: task-blind observations and optional policy composition

Updated 8 September 2026 from the uploaded `Continual-RL-Project.zip`.
Read `VALIDATION_REPORT.md` before starting expensive runs. Passing synthetic
checks does not establish MuJoCo/MetaWorld compatibility or learning quality.

## 1. Scope and defaults

Updated runners: `half-cheetah/`, `Walker2D/`, `metaworld/`, and the compatible
legacy `meta-world/` directory. Added `Hopper/`, which did not exist in the
uploaded archive. Ant and MiniGrid algorithms were not changed. Shell scripts
were normalized to LF, including otherwise-untouched folders.

Each benchmark defaults to condition indices **1 and 4**, crossed with
`--composition-spaces parameter policy`:

| Output label | Stored representation / adaptation | Overflow pair consolidation | Inference composition |
|---|---|---|---|
| `baseline` | Classic CKA residual-vector implementation | Existing cosine selection + arithmetic averaging | Parameters |
| `combined` | Full heads, task residual and sigmoid history mass | Existing behavioral selector + KL distillation | Parameters |
| `baseline_policy` | Complete base-plus-residual experts with a shared trainable task residual | Existing cosine selection + arithmetic averaging | Policy distribution mixture |
| `combined_policy` | Complete historical experts and a gated novel expert | Existing behavioral selector + KL distillation | Policy distribution mixture |

**Inference composition is independent of overflow pair consolidation.** Selecting
`policy` does not change which pair the baseline selects or silently turn its
pair merger into the combined method. It does require a separate projection at
insertion, described below. Thus `baseline_policy` is a new experimental analogue,
not the published CKA-RL baseline, and the policy/parameter comparison includes
the storage projection needed by the bounded-pool design.

Disable the new option with `--composition-spaces parameter`. Individual SAC
runs accept `--composition-space parameter` or `--composition-space policy`.
The old ablations remain available: 2 = `distil_only`, 3 = `weight_only`, 0 = all
four conditions. `--condition-index 0 --composition-spaces parameter policy`
therefore runs eight configurations.

## 2. Explicit task context removed

In HalfCheetah and Walker2D, `get_task()` no longer applies
`TaskConditionedObservationWrapper`; the call is commented out. Encoder
pretraining no longer installs it either. The legacy wrapper and
`make_task_specific_observation()` are retained as **identity operations**, so
calling them cannot accidentally reintroduce context.

The actor and critic now receive native 17-dimensional observations in both
families. Velocity targets, winds and dynamics parameters still configure the
environment and can appear in diagnostic `info` fields. They are not concatenated
to network inputs. Replay `task_ids` and `source_ids` remain lineage metadata,
not features. Hopper uses its native 11-dimensional observation and 3-dimensional
action space, without extra context.

MetaWorld's native observation was not given an extra integer ID or parameter
vector by the modern runner. Its ordinary goal-observable 39-dimensional input
is retained. **Native goal/object coordinates can convey task context**; this
patch does not claim that MetaWorld is goal-hidden or that its tasks cannot be
inferred from observations. Removing native goals would define a different
benchmark, not merely undo the extra concatenation.

Target velocities/rewards themselves were not changed in HalfCheetah. With the
target no longer observed, HalfCheetah-Vel has a latent goal. A single frozen,
memoryless policy may consequently struggle to satisfy conflicting target
velocities. No task-inference module was added; optional test adaptation is
reported separately.

## 3. Buffer budget is now inside Delta

`--total-timesteps` means **Delta total learning-environment interactions**.
`--distill-buffer-steps B` is the clearer alias of the retained
`--distill-extra-steps B`; despite the old name, these are not extra steps.

Every condition, including scratch SAC, uses this schedule:

1. First `Delta - B` interactions: ordinary replay collection and SAC updates
   according to `learning_starts`, warmup and update frequencies.
2. Last `B` interactions: the policy is frozen and acts stochastically; no
   actor, critic, mixing coefficient or entropy-temperature update occurs.
3. Projection/consolidation uses those stored states without more environment
   interaction. Non-distillation parameter baselines normally discard the tail
   buffer; they still receive exactly the same training/frozen-tail allocation.

A separate environment performs periodic monitoring. Monitoring and retention
evaluation **do** consume simulator interactions; they do not update the policy
unless test adaptation is explicitly enabled. `interaction_budget.json` records
Delta, the optimization-phase count, frozen-tail count and monitoring count.
Retention JSON includes evaluation/adaptation interaction matrices. Do not call
this a total-simulator-interaction budget including all evaluations.

With Delta=300000 and B=10000, the allocation is 290000+10000, not 300000+10000.
The optimization phase includes initial collection steps with no optimizer
updates. Buffers are capped by `max_distill_buffer`; the same sampled indices
are applied to every row-aligned metadata array.

## 4. Sigmoid history mass and pool search

The mixing logits remain a softmax. Only the separate history-mass scalar changes:

    alpha_i = softmax(scale * logits)_i
    mass = sigmoid(raw_mass)

A finite initial raw value `log(0.95 / 0.05)` gives mass 0.95 during adaptation.
During historical-only warmup, the effective mass is overridden to **exactly 1**
and its gradient is disabled. This avoids approximating 1 using a huge saturated
sigmoid parameter. The existing double-well penalty remains
`lambda * mass**2 * (1-mass)**2`. The old unconstrained gate remains opt-in for
parameter-mode ablations, but is rejected for a probability mixture.

The historical-only phase now includes a singleton pool. Its mixing-weight
gradient is naturally zero because there is only one choice. Warmup freezes the
new head/residual and encoder using `grad=None`, also preventing optimizer
momentum from moving them. Critics and SAC's entropy temperature still train;
"alpha only" refers to policy adaptation, not every optimizer in SAC.

Updated defaults:

| Hyperparameter | Value | Meaning |
|---|---:|---|
| `learning_starts` | 5000 | Updates begin after initial replay collection |
| `random_actions_end` | 5000 | The policy, rather than uniform random actions, starts acting |
| `alpha_warmup_steps` | 5000 | Warmup ends at step `learning_starts + alpha_warmup_steps` |

For weight-delta variants, approximately steps 5000-10000 now have a
historical-mixture behavior policy. SAC still samples a replay buffer that
contains earlier random transitions: **this is policy-controlled collection,
not an on-policy-only optimizer**. Set `random_actions_end=10000` for the old
random-through-warmup experiment. Setting it to 0 uses the initialized policy
from the first interaction, including on the root task. No new RL algorithm or
replay-filtering method was silently added.

## 5. What policy-space composition means

The policies are tanh-squashed diagonal Gaussians. Averaging their parameters,
means, standard deviations, or sampled actions are different operations. This
implementation mixes **whole action distributions**, with one component sampled
for the entire action vector at each decision:

    pi_mix(a | s) = sum_i alpha_i * pi_i(a | s)

Weights are task-level learnable scalars, not a new state-conditioned router.
All components share the same action bounds and squashing transform.

### Combined policy-space adaptation

The parameter-space branch still uses `mass * sum_i alpha_i theta_i + v`.
The policy-space branch uses a probability-preserving analogue:

    pi(a | s) = mass * sum_i alpha_i pi_i(a | s)
                + (1 - mass) * pi_new(a | s)

`pi_new` is a complete trainable head, initialized from the newest historical
entry. It is not a standalone all-zero residual network. During warmup it is
excluded entirely. After warmup the finite sigmoid gate enables gradients to
both reuse and the novel expert. This is an explicit architectural adaptation,
not an algebraic identity with the parameter-space residual equation.

### Baseline/no-mass policy-space adaptation

Classic stored deltas are first reconstructed as complete base-plus-delta
heads. A current trainable residual is applied to each expert before mixing:

    pi(a | s) = sum_i alpha_i pi_(theta(entry_i) + v)(a | s)

A bare residual is never treated as an executable policy. The no-mass full-head
ablation follows the same rule with complete stored heads.

### Correct SAC density and gradients

`policy_composition.py` computes the mixture's marginal log density using
`logsumexp(log_weight + Gaussian_log_probability)` and subtracts the tanh and
action-scaling Jacobian. It does not substitute the log probability of the
selected component or a moment-matched Gaussian.

Categorical component selection is not reparameterizable. For the actor update,
the code enumerates the small component set and draws a reparameterized Gaussian
sample from each component:

    sum_k w_k E_(a~pi_k)[temperature * log pi_mix(a|s) - min(Q1,Q2)(s,a)]

Both outer weights and mixture-density terms remain differentiable. Environment
and target-critic action sampling use the exact categorical/Gaussian mixture.
A moment-matched summary exists only for compatibility/diagnostics; it is not
used for sampling, action likelihoods or the SAC actor loss.

Runtime cost increases: inference evaluates all active heads; the actor loss
also evaluates Q for each component and compares samples against every component
for the mixture density. This is roughly O(K) head/Q work and O(K^2) density
work, where K is bounded by pool capacity plus one.

### Bounded storage requires projection

An exact ensemble generally cannot be serialized as one MLP. Saving nested
ensembles would silently make memory grow with the whole task history.
Therefore each non-root policy-space task projects its active mixture into one
Gaussian policy head using frozen-tail states, before inserting it into the
bounded pool. No extra environment interaction is used.

The projection objective is the weighted component-to-student forward KL:

    sum_k w_k KL(pi_k || q)

It has the same student-dependent term as `KL(pi_mix || q)`; their difference
is constant in the student. It is **not numerically equal** to mixture KL.
Diagnostics therefore use `projection_*_component_kl` names. A validation split
selects the best initialization/epoch, including the original initialization.
`projection_epochs` and `projection_max_samples` control this additional work.

A single Gaussian can lose multimodality, and finite fitting introduces error.
Projection does not guarantee exact behavior preservation. Inspect its logged
validation loss and actual returns. For classic storage, the projected complete
head is stored relative to the fixed base. There are no recursively nested
expert lists in pool entries.

## 6. Requested KL behavior preserved

The existing `_select_behavioral_pair`, `_select_cosine_pair`, and `_distill_pair`
functions are unchanged in the modern common implementation. In particular the
behavioral selector still evaluates symmetric Gaussian KL on balanced combined
parent states; it was **not** replaced with the draft's alternative equation.

The new insertion projection is separate from those existing pair operations.
Existing pair-distillation `distill_test_kl` is still evaluated on the split used
for best-epoch selection. Treat it as validation-selected held-out diagnostics,
not an independent test estimate. This was documented rather than changing the
pair-distillation procedure contrary to the requested scope.

## 7. Additional consistency fixes

The observation skip connection and condition-specific alpha scale remain
available, but both default off. Thus default methods use the same head input
architecture and plain-softmax scaling. The shared encoder learns on the root
task, then is frozen by default. `--train-shared` remains an ablation; encoder
movement changes the behavior of historical entries and introduces extra
condition-dependent regularization, so it is not the recommended main comparison.

Frozen evaluation is now consistent across all families. The default evaluates
the finalized pool, with uniform mixing logits, zero current residual, history
mass one and no task-based slot selection. `--frozen-eval-policy snapshot`
instead evaluates the exact pre-insertion active-policy snapshot. Mixture
snapshots preserve every active component and its weights, not averaged tensors.

`--eval-action-mode deterministic` uses the weighted sum of squashed component
means for policy mixtures. It is not the true expectation of a squashed Gaussian
or necessarily a mode of a multimodal mixture. `stochastic` evaluates the actual
sampled mixture policy. Use the same mode for every compared method and report it.
Monitoring evaluation preserves the training Torch RNG state.

Run manifests now include composition/projection settings and new source files.
Old checkpoints and metric caches are deliberately invalidated. Linux shell
line endings are fixed. The original overview/presets are retained with explicit
historical warnings instead of being presented as verified results for this patch.

## 8. MetaWorld paper and new Hopper suite

The referenced paper is **Continual Knowledge Adaptation for Reinforcement
Learning**, arXiv:2510.19314, version 2 (20 January 2026). Its Appendix C.1 lists
ten MetaWorld v2 tasks repeated twice. `mw_paper10` adds that ordered task list:
hammer, push-wall, faucet-close, push-back, stick-pull, handle-press-side, push,
shelf-place, window-close, peg-unplug-side. The original easy4/easy6/smoke2 and
legacy7 options remain. Paper10 and legacy7 request non-frozen reset randomization.

This is a **task-list option, not an exact reproduction**. The retained SAC/head
architecture, 150k default MetaWorld budget, horizon, distribution bounds and
new evaluation protocol are not all the paper's configuration. MetaWorld API
fallbacks can also choose v3 environments; the adapter records the resolved API
but real construction must be checked on the installed package. Do not compare
numbers as an exact v2 replication without pinning and matching the full setup.

Hopper is a new extension using Gymnasium Hopper-v5. Six tasks retain target
velocity 1.5 and vary body mass/inertia, foot friction, joint damping or motor
strength. Each appears twice. The native health rules are retained; reward is
healthy reward minus velocity error minus control cost. These perturbation
choices are new design choices, not tasks recovered from the archive or paper.
No real Hopper rollout was available in the review environment.

Primary references used to inspect interfaces and the supplied baseline:
- https://arxiv.org/abs/2510.19314
- https://arxiv.org/html/2510.19314v2 (Appendices C.1 and D)
- https://gymnasium.farama.org/environments/mujoco/hopper/

## Replay-trained policy student variant

The updated code adds `--policy-student-replay` for the combined policy-space condition. The execution Gaussian mixture controls the environment, while the standalone current expert is optimized from replay with SAC; alpha and alpha-mass are then optimized in a separate routing step. The new condition is labeled `combined_policy_student`, stores the standalone novel expert directly, and leaves ordinary `combined_policy` unchanged. See `POLICY_STUDENT_REPLAY.md` for the precise update order and equations.
