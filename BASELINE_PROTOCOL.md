# Baseline ports and comparison protocol

## Source of truth and scope

The main project is the uploaded `Continual-RL-Project(4).zip`. The donor is
`Continual-RL-Project-narges.zip`. These are ports into the supplied main
benchmark, not a claim that published reference results have been reproduced.
The main SAC/projection/lineage implementations and old shell presets remain
as supplied except for the explicitly listed compression-flag wiring/fix and
identity/cache bookkeeping. The old `baseline` is still CKA-RL.

The donor contains conflicting versions of some names. In particular,
`four_baselines/.../ft_n.py` is single-actor fine tuning, while PointMaze's
FT-N has multiple heads with a changing encoder. Neither preserves a full
past actor as in the FT-N/full-model-preservation convention used here.
The donor's MaskNet variants also differ: a HAT-like learned gating version
versus fixed-random-weight supermasks. This integration selects the latter,
from `pointmaze/src/baselines/mask_modules.py`. Do not silently relabel these
as reproductions of the other donor variants.

The module origins and hashes are in `paper_runs/DONOR_PROVENANCE.json`.
Relevant donor source is also retained under `paper_runs/vendor/reference`.
The main reference for the full-model FT-N distinction is the baseline
section of https://arxiv.org/html/2510.19314v1 .

## Implemented training mechanisms

| CLI name | Mechanism | Retained state / caveat |
|---|---|---|
| `baseline` | Existing classic CKA residual composition and cosine/weight averaging | Existing code, not a new reimplementation |
| `ft_n` | Continue fine tuning the complete latest actor; retain a frozen complete actor after every occurrence | Full-model preservation, growing memory; not FT-1 |
| `prognet` | Add a column with lateral hidden/output adapters; freeze old columns | Donor ProgressiveColumn, growing memory |
| `packnet` | Full-actor iterative magnitude pruning, reserve weights per occurrence, retrain current mask | Donor equal-share allocation, keep fraction 1.0, retrain fraction 0.3. Weights and Adam-induced changes to protected entries are blocked |
| `masknet` | Frozen signed-random head weights, learn per-occurrence supermask scores; learn combinations of historical scores | Donor supermask variant; encoder learned on root and then frozen; scores grow with occurrences |
| `crelus` | Continual fine tuning with concatenated positive/negative ReLU features in the actor head | Donor head-level CReLU; no preserved policy bank; encoder remains trainable |
| `componet` | Input/output attention over frozen earlier modules plus a trainable internal module | Donor continuous-action port; raw mean/log-std vector composition, not ETHOS's probability mixture. Shared encoder frozen after root; flat storage avoids duplicate nested trees |
| `cbpnet` | Continual backprop generate-and-test replacement of low-utility actor-head units | Donor head-level mechanism, replacement rate 1e-4, maturity 100, decay .99; unit statistics persist and replaced-unit Adam moments reset |
| `combined_policy` | Existing exact policy-distribution routing, final projection and bounded compression | Main method |

The shared SAC loop is ported from donor `four_baselines/half-cheetah/baselines/common/sac_core.py`.
It uses the **main** environment constructors, squashed-Gaussian utilities,
raw observations, twin-Q architecture, CSV logger and task-budget helper.
Critics, optimizers and SAC replay reset at boundaries to match the main
benchmark; this differs from donor variants that carry critics across tasks.
The two critics are shared across actor actions, not one critic per expert.

The existing script's total steps, batch size, policy/Q learning rates, gamma,
tau, entropy-temperature settings, initial random steps and evaluation schedule
are carried over. The existing frozen tail is spent without optimization,
inside the same per-occurrence budget for all methods. Baselines without
compression do not collect a fake distillation buffer. New baselines use
replay size 1e6, actor frequency 2 (with two actor updates when scheduled),
target frequency 1, and head width 128; ProgressiveColumn uses width 256.
These defaults and architecture-specific mechanisms are recorded in manifests.

History-specific flags such as `alpha_mass_lr`, `alpha_mass_reg`, lineage
balancing or `--no-train-shared` do not redefine the external baselines.
For example, making FT-N's whole encoder permanently frozen would no longer
be full-network fine tuning. The existing CKA/ETHOS interpretation of all
these flags is unchanged.

An occurrence creates a column/mask/module even on a recurring task. Real
`task_id` is not used during training to retrieve an old column/mask; the known
sequence boundary determines the slot. This follows the no-task-ID input
comparison setting rather than giving selected baselines oracle task routing.

## Evaluation: distinguish common-protocol ports from native/oracle results

The default `--baseline-eval-protocol reward_route` uses the main benchmark's
**frozen-library, reward-only retrieval** protocol. For FT-N, ProgNet, PackNet,
MaskNet and CompoNet, candidate policies are constructed from knowledge still
retained in the current checkpoint. Only new temporary simplex routing logits
are fitted for `test_adapt_steps`, using the same immediate-reward REINFORCE
heuristic and learning rate as the main evaluator. Policy weights are frozen,
historical mass is one, and task ID does not select the best policy.
Deterministic evaluation uses weighted squashed means, as in the main code;
stochastic evaluation samples a component and action from the actual mixture.

CReLUs and CbpNet retain a single current actor, so they cannot recover a bank
of earlier snapshots. They consume the same adaptation interaction budget but
have no routing weights to train. Their weights never update at retention time.

This routing wrapper is an **explicit evaluation adaptation** for the new
baselines, not their authors' original native evaluation protocol. Unbounded
methods retain more policy parameters than bounded ETHOS. Report memory and
information-access differences; do not imply that all methods use the same
fixed policy-pool capacity.

Optional diagnostics:

* `--baseline-eval-protocol native`: use the newest retained component/mask
  corresponding to the evaluated task, when available. This uses task identity
  (oracle selection), with zero adaptation. Future/unseen tasks use the latest
  available component. Outputs are under `plots/baseline_native/<suite>/`.
* `--baseline-eval-protocol latest`: latest retained component only, no adaptation.
  Outputs are under `plots/baseline_latest/<suite>/`.

These alternatives never overwrite the default `plots/<suite>` results, and
are not automatically mixed into the standard collector's table.

## Metrics

PERF is unchanged: per-seed average of the explicit final active-policy
success/return scalars over **all occurrences**, then mean/population standard
deviation across seeds, exactly as the supplied collector did.
A_N/FG/BWT use the existing main functions on post-finalization retention
cells. The new baseline evaluator reuses identical diagonal/final cells rather
than reevaluating them. FT uses first unseen encounters after position zero,
the same periodic test tags and the same main AUC formulas. All new methods
use a **common plain-SAC reference**, not a newly trained algorithm-specific
scratch reference. This should be stated alongside FT because architectures
differ across methods.

Missing checkpoints or corrupt files are reported, not converted to zero
performance. Evaluation logs and `seed_status.json` record excluded seeds;
per-metric `_n` columns show how many results remain. This is failure handling,
not an acceptable way to silently discard low-performing runs from a paper.
Rerun failed seeds where feasible and disclose missing results.

AntDir's success is custom; raw return is the primary diagnostic. Its
unnormalized `FT_return_auc_delta` is a separate metric, not a normalized FT.

## Compression ablations

`--merge-ablation kl_merge` is the unchanged default for combined_policy.
`random_merge` draws an unordered pair uniformly without replacement, then
uses the exact same buffer sampling/distillation path.
`kl_discard` finds the minimum-KL pair and discards one member with probability
1/2. The surviving policy and its own buffer are unmodified; discarded lineage
is not relabeled as preserved. **Task-end projection remains enabled**: this
ablation removes overflow distillation only, isolating pool consolidation.

Lineage balancing, pool capacity and initial alpha warmup use the existing
flags. The optional no-merge case sets capacity to the sequence length: it is
a larger-memory diagnostic for compression loss, not a same-memory competitor.
The paper's KL certificate does not assert that pairwise KL always orders
realized return loss; these runs are empirical tests, not proof of that claim.
