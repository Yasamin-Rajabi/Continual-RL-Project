# Continual-RL baselines

Step 2 deliverable: common infrastructure plus FT-N. ProgNet, PackNet and
MaskNet slot into the same contract and are next.

## Files

```
baselines/
├── __init__.py                 REGISTRY, task-aware/task-blind declaration
├── ft_n.py                     FT-N: sequential fine-tuning (implemented)
├── prognet.py                  (next)
├── packnet.py                  (next)
├── masknet.py                  (next)
└── common/
    ├── lifecycle.py            ContinualAgent protocol + TaskContext
    ├── sac_core.py             the SAC loop, shared by all four baselines
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

## Nothing in SOURCE_CANDIDATES was touched

`experiment_identity.SOURCE_CANDIDATES` fingerprints `run_sac.py`, `cka_rl.py`,
`policy_composition.py`, `policy_space.py`, `training_protocol.py`,
`knowledge_pools.py`, `shared_arch.py`, `policy_utils.py`, `tasks.py`,
`<suite>_envs.py` and `analysis_logging.py`. All eleven are unmodified, so
**every CKA-RL checkpoint you have already trained stays valid.** The baselines
import those modules; they never edit them.

`baseline_identity.py` keeps a separate `SOURCE_CANDIDATES` for the baseline
files plus the shared modules both sides genuinely depend on, so the two
fingerprints move independently.

## Parity, and how each part is enforced

| Guarantee | Mechanism |
|---|---|
| Same environments and wrappers | `sac_core` builds envs only via `tasks.get_task` |
| Same budget | the same `training_protocol.TaskBudget(Delta, B)` object |
| Same frozen tail | B interactions are spent under a frozen policy, collecting nothing |
| Same action distribution | `policy_composition.sample_action` / `sac_actor_objective` |
| Same architecture | `shared_arch.shared` encoder, `hidden_dim=128` heads |
| Same evaluator | `policy_snapshot.pt` loads into `cka_rl.FrozenCkaPolicy` |
| Same replay semantics | fresh SB3 buffer per subprocess, as in `run_sac.py` |

The evaluator row is the important one. Baselines emit exactly the snapshot dict
`FrozenCkaPolicy` reads in its `composition_space == "parameter"` branch, so
the retention matrix is computed by **the method's own evaluation code**, not by
baseline code:

```python
from checkpoint_evaluation import evaluate
evaluate(run_dir, suite, task_id, episodes, seed, device,
         frozen_policy="snapshot",      # required for baselines
         adapt_steps=0,                 # FrozenCkaPolicy has no alpha to adapt
         error_key=ERROR_KEY, episodic_success=EPISODIC_SUCCESS)
```

`checkpoint_evaluation.py` needed **no changes** for this.

## Approved design decisions, as implemented

1. **Task oracle.** `baselines.TASK_AWARE_METHODS` is `("prognet", "packnet",
   "masknet")`. FT-N declares `task_aware = False` and asserts it in
   `on_task_start`. The flag is written into every `run_manifest.json`, so a run
   can never be silently reinterpreted as the other kind.
2. **Repeated tasks.** Capacity is keyed by `task_id`, never `seq_idx`.
   `ContinualAgent.allocate_slice` raises if a task is ever reassigned to a
   different slice. `TaskContext.first_encounter` drives allocate-vs-reuse.
3. **Shared critic.** One critic for the whole sequence, carried between
   subprocesses through `stash_critic_state`. Only actor-side capacity differs
   between the four baselines.
4. **PackNet retrain inside the budget.** `on_phase_boundary(step, budget)` is
   called every optimization step with the `TaskBudget`, so the prune/retrain
   split is carved out of `Delta - B`, never added on top.
5. **Budget.** `baseline_defaults.py` mirrors `run_sac.py` per folder:
   300k steps / B=10k / seeds (101, 102) for HalfCheetah, 150k / B=10k /
   seeds (1, 2, 3) for Meta-World.

## Two correctness details worth knowing

**Raw log-std.** A baseline policy's `forward(obs)` returns *unbounded*
log-std. `policy_composition.components()` applies `bound_log_std` itself,
exactly as it does for `CkaRlAgent`. Bounding inside the policy would apply the
tanh squash twice.

**Parameter isolation needs two hooks, not one.** Masking a gradient is not
sufficient under Adam: a parameter with zero current gradient still moves if its
momentum buffers are non-zero. `before_optimizer_step` masks gradients and
`after_optimizer_step` copies protected weights back bit-exactly. PackNet and
ProgNet rely on the second one for actual correctness.

## Verification

```bash
python3 baseline_smoke.py --method ft_n
```

Runs a 3-position chain `(0, 1, 0)` — the repeat at position 2 exercises the
capacity-reuse path — then checks: checkpoints complete and resume-detectable,
a deliberately altered config correctly detected as stale, `FrozenCkaPolicy`
reproducing each trained network to 1e-5, the method's evaluator running on a
baseline checkpoint, and `scalars.csv` carrying the tags `metrics.py` integrates.

Minutes on CPU, no GPU needed.

### What has and has not been verified here

Verified statically in this sandbox: every file compiles; all 39 cross-module
symbol imports resolve against the real repo source in both suite folders; the
snapshot dict contains every key `FrozenCkaPolicy` reads in the parameter branch
(`mixture_weights` is policy-branch only); `TaskBudget`'s field names match;
FT-N implements every abstract method and overrides none of the forbidden ones;
the nine shared files are byte-identical across folders.

**Not verified: actual execution.** PyPI was unreachable from the sandbox
(HTTP 403), so torch, gymnasium, mujoco and stable-baselines3 could not be
installed and the training loop has never been run. `baseline_smoke.py` is the
first thing to run on Kaggle or your own machine.

## Kaggle

Set before any env import, matching `job.sh`:

```python
os.environ["MUJOCO_GL"] = "egl"          # "osmesa" if EGL errors appear
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
```

For Meta-World, install order is load-bearing: MuJoCo first, then Meta-World at
the pinned commit with `--no-deps`, then `requirements.txt` with the
`metaworld`/`mujoco` lines filtered out. Otherwise pip resolves Meta-World's
stale pins and downgrades torch/numpy/gymnasium underneath the run. The existing
`run_kaggle.sh` already does this; the baseline runner will reuse it.
