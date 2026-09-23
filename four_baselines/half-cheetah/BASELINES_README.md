# Continual-RL baselines

All four baselines plus the shared infrastructure. Nothing in
`experiment_identity.SOURCE_CANDIDATES` was modified, so every CKA-RL
checkpoint you have already trained stays valid.

## Files

```
baselines/
├── __init__.py                 REGISTRY, task-aware/task-blind declaration
├── ft_n.py                     FT-N      sequential fine-tuning (task-blind)
├── prognet.py                  ProgNet   one frozen column per task
├── packnet.py                  PackNet   pruning + binary masks
├── masknet.py                  MaskNet   task-conditioned gating
└── common/
    ├── lifecycle.py            ContinualAgent protocol + TaskContext
    ├── sac_core.py             the SAC loop, shared by all four
    ├── masks.py                pruning, ownership bookkeeping, task gates
    └── snapshot.py             the policy_snapshot.pt contract

run_baseline.py                 entry point: one task, one subprocess
baseline_identity.py            manifests and resumability, baseline-scoped
baseline_defaults.py            THE ONLY suite-specific file
baseline_smoke.py               end-to-end verification, minutes on CPU
```

Everything except `baseline_defaults.py` is **byte-identical** between
`half-cheetah/` and `metaworld/`, verified by diff. Suite-specific behaviour is
reached only through `metrics.ERROR_KEY`, `metrics.EPISODIC_SUCCESS` and
`tasks.get_task`, so no baseline file names an environment.

## The four methods

| | oracle | capacity | forgetting | encoder |
|---|---|---|---|---|
| FT-N | no | one network, shared | catastrophic, by design | trains throughout |
| ProgNet | yes | +1 column per task | exactly zero | frozen after root |
| PackNet | yes | fixed, carved up | exactly zero | frozen after root |
| MaskNet | yes | fixed, gated | proportional to claim | frozen after root |

FT-N is the lower bracket and faces exactly the information set your method
faces. The other three get the oracle and are the upper bracket. Your method
sits between them **without** the oracle, which is the claim.

The encoder column is worth a sentence in the paper: FT-N lets it keep training
because that is what naive fine-tuning *is*, while the three task-aware methods
freeze it after the root task because it sits upstream of every column, mask and
gate, and letting it drift would defeat each of their guarantees at once. This
matches your method's own `train_shared=False` default.

## Parity, and how each part is enforced

| Guarantee | Mechanism |
|---|---|
| Same environments and wrappers | `sac_core` builds envs only via `tasks.get_task` |
| Same budget | the same `training_protocol.TaskBudget(Δ, B)` object |
| Same frozen tail | B interactions spent under a frozen policy, collecting nothing |
| Same action distribution | `policy_composition.sample_action` / `sac_actor_objective` |
| Same architecture | `shared_arch.shared` encoder, `hidden_dim=128` heads |
| Same evaluator | `policy_snapshot.pt` loads into `cka_rl.FrozenCkaPolicy` |
| Same replay semantics | fresh SB3 buffer per subprocess, as in `run_sac.py` |

The evaluator row is the important one. Every baseline emits exactly the
snapshot dict `FrozenCkaPolicy` reads in its `composition_space == "parameter"`
branch, so the retention matrix is computed by **your method's own evaluation
code**:

```python
from checkpoint_evaluation import evaluate
evaluate(run_dir, suite, task_id, episodes, seed, device,
         frozen_policy="snapshot",      # required for baselines
         adapt_steps=0,                 # FrozenCkaPolicy has no alpha to adapt
         error_key=ERROR_KEY, episodic_success=EPISODIC_SUCCESS)
```

`checkpoint_evaluation.py` needed **no changes** for this, including for
ProgNet — see below.

## Three things worth knowing before you read the code

**ProgNet flattens exactly.** A network with lateral connections looks like it
cannot be written as the plain two-layer head the snapshot format requires. It
can. Every column's hidden layer is ReLU of a linear map of the *same* encoder
output, and ReLU is elementwise, so stacking the columns is one wider layer:

```
H     = [h_0 ; … ; h_k] = relu( [W1_0 ; … ; W1_k] z + [b1_0 ; … ; b1_k] )
out_k = [L_k0 | … | L_k(k-1) | W2_k] H + b2_k
```

Not an approximation — the same function, rewritten, as a two-layer head of
width `(k+1) × hidden_dim`. That is why `effective_hidden_dim()` exists and why
the lateral adapters are a **linear** low-rank bottleneck rather than the
paper's nonlinear one: a nonlinearity there would make the composition three
layers deep and no two-layer head would reproduce it. That deviation belongs in
the paper's baseline description.

**Parameter isolation needs two hooks, not one.** Masking a gradient is not
enough under Adam: a parameter with zero current gradient still moves while it
carries momentum. So `before_optimizer_step` masks and `after_optimizer_step`
copies protected values back bit-exactly. The second is what makes the
guarantee real. This applies to PackNet's committed weights *and* to MaskNet's
gate logits, which all live in one tensor with one row per task and therefore
cannot be protected with `requires_grad` at all.

**MaskNet's gate initialisation decides whether the method works.** A constant
positive init looks friendlier but saturates every gate open once the slope
anneals: task 0 claims the entire backbone and every later task is frozen
solid, with no error and no obviously broken curve. Logits therefore start near
zero with a small random spread. If the later tasks of a long sequence come out
starved anyway, turn on `--masknet-sparsity-reg`.

## Verification

```bash
python3 baseline_smoke.py --method all     # all four, ~10 min on CPU
python3 baseline_smoke.py --method packnet # just one
```

Runs a 3-position chain `(0, 1, 0)` per method — the repeat at position 2
exercises the capacity-reuse path — and checks seven things. Six are
bookkeeping. The seventh is the one that earns its runtime:

**Check 5, parameter isolation.** Position 0 trains task 0 and saves its policy.
Position 1 then trains task 1. The check reloads the chain state as it stood
*after* position 1, re-selects task 0, re-exports its head, and compares against
what task 0 actually saved. ProgNet and PackNet must come back at exactly zero.
MaskNet's drift is reported but not asserted, because its protection scales with
how strongly each unit was claimed. FT-N is asserted to be **nonzero** — if
sequential fine-tuning came back unchanged, the check would be vacuous for
everyone.

Every failure mode these architectures have is silent: nothing raises, the loss
still falls, and the damage surfaces only as a retention matrix that looks like
a property of the method. Measuring the invariant directly is the only way to
tell.

### What has and has not been verified here

Verified statically: every file compiles; all 58 cross-module symbol imports
resolve against the real repo source in both suite folders; the snapshot dict
contains every key `FrozenCkaPolicy` reads in the parameter branch; all four
agents implement every abstract method and override none of the forbidden ones;
no unused imports; the twelve shared files are byte-identical across folders.

**Not verified: execution.** `torch` could not be installed in the authoring
sandbox, so the three new baselines have never been run. FT-N has — it passed
the full smoke test on both `halfcheetah_vel` and `mw_easy4`, which is what
validates the shared infrastructure underneath all four.

So `python3 baseline_smoke.py --method all` is the first thing to run.

## Kaggle

Set before any env import, matching `job.sh`:

```python
os.environ["MUJOCO_GL"] = "egl"          # "osmesa" if EGL errors appear
os.environ["PYOPENGL_PLATFORM"] = "egl"
```

For Meta-World, install order is load-bearing: MuJoCo first, then Meta-World at
the pinned commit with `--no-deps`, then `requirements.txt` with the
`metaworld`/`mujoco` lines filtered out. Otherwise pip resolves Meta-World's
stale pins and downgrades torch/numpy/gymnasium underneath the run.

Use **T4**, not P100: the installed torch wheel has no sm_60 kernels, and the
failure surfaces deep inside training rather than at setup.
