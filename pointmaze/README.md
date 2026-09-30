# PointMaze continual RL benchmark

Ten-task continual RL on a kinematic PointMaze, with our method and eight
baselines trained under one identical protocol.

```
python pointmaze_smoke.py                  # environment (seconds, no torch)
python verify_static.py && python verify_flags.py
python baseline_smoke.py --method all      # every method end to end (minutes)
python run_continual_benchmark.py --methods Ours CKA-RL FT-N --seeds 1
```

On Kaggle, open `pointmaze_kaggle.ipynb` and edit only the `RUN` list in cell 5.

---

## 1. The environment

Pure NumPy physics behind a Gymnasium `Env`. No MuJoCo, no gymnasium-robotics,
no JAX. **~30 µs per step (≈32k steps/s)**, so wall clock is dominated by SAC
updates, not by simulation.

| | |
|---|---|
| observation | `[x, y, vx, vy]` + 8 ray distances = **12 floats**. The goal is **not** in the observation. |
| action | 2-D continuous, `tanh`-squashed |
| reward | `-d_geo / 10`, where `d_geo` is the shortest path **through the maze** |
| horizon | 300 steps |
| success | latched at termination, aggregated with **max** over the episode |

### Why the ray sensors exist

They are identical in every task of a suite, so the shared encoder has a
genuinely task-independent quantity to learn (local maze geometry) while the
policy heads carry goal-specific behavior. With a bare `[x, y, vx, vy]`
observation, "freeze the encoder after task 0" would be an empty statement and
the encoder-sharing axis of the study would measure nothing.

### Why the reward is geodesic

Both standard PointMaze rewards are wrong for this study:

* **sparse** (1 at the goal) leaves SAC near chance on a maze this branchy, so
  every method scores the same and the benchmark measures nothing;
* **dense** (`exp(-d)`) is positive at every step, so the agent is *punished*
  for ending the episode by reaching the goal, and its reward gradient points
  straight into walls.

The geodesic form has no local optimum on a wall, improves with reaching the
goal sooner, and is `<= 0` everywhere — which keeps `RETURN_UPPER_BOUND = 0`
exact, and the forward-transfer algebra in `metrics.py` valid without a fudge
factor. It is tabulated and bilinearly interpolated, so it is continuous
(measured max step along a corridor: 0.011) rather than a staircase.

### Verified, not assumed

`pointmaze_smoke.py` runs the real physics and checks all of it:

```
pointmaze_goal:     every task solved, 45-66 steps of 300, mean return -20.2
pointmaze_goal_dyn: every task solved, 45-67 steps of 300, mean return -20.3
random policy:      -196.8            (wide, learnable gap)
reward never positive .................. 6000 steps, 0 violations
agent never inside a wall .............. 0 violations
reset(seed) + 50 steps reproduce exactly
geodesic monotone along a corridor ..... max step 0.0112
speed .................................. 28.9 us/step
```

---

## 2. The ten tasks

Three route families from one hub: **A** south-west, **B** north-east,
**C** south-east. Measured structure:

| | shared prefix of the two optimal paths |
|---|---|
| two goals in the **same** family | 8–11 cells |
| two goals in **different** families | **exactly 1** (the hub) |

The orders are `A B C A B C A B C A` — **no two consecutive tasks share a
family**, and `tasks.validate_suite()` asserts it.

This is the most deliberate choice in the benchmark, so it is stated plainly:
the task that just finished is *never* the closest task to the one starting.
Any method that carries forward only its most recent solution (FT-N, and any
warm start) gets the worst available prior at every single boundary, while a
method holding a **pool** of past policies always still has a same-family entry
to draw on. That is the situation a knowledge pool is built for. It is an
experimental design, asserted in code, not a thumb on the scale — and the
honest way to read any result is "on a task stream with recurring, non-adjacent
structure", not "in general".

Ten tasks against a **pool of five** forces exactly five merges, so what is
measured is the *quality of the merge decision*, not raw capacity. The same
ten tasks press on the fixed-capacity baselines in documented ways: PackNet
must fit ten disjoint subnetworks into one network, ProgNet carries ten actor
columns, and MaskNet's gates become ten-way.

A second suite, `pointmaze_goal_dyn`, adds a per-task 2×2 actuator matrix
(period 4) against the family cycle (period 3), so the two task axes are
deliberately uncorrelated.

---

## 3. Methods

| method | mechanism | gets the task index? |
|---|---|---|
| **Ours** | condition 4 (combined): weight-delta vectors + alpha-mass + behavioural-KL distillation merge, composed in **policy** space | **no** |
| Ours-parameter | the same condition in parameter space (isolates the composition-space axis) | no |
| CKA-RL | the original method: classic CKA vectors, cosine-selected arithmetic merge | no |
| FT-N | fine-tune everything, one output head per task | yes |
| ProgNet | one frozen column per task + lateral adapters | yes |
| PackNet | iterative magnitude pruning into per-task subnetworks (see capacity note) | yes |
| MaskNet | fixed random weights, learned per-task supermasks | yes |
| CReLUs | CReLU activations to slow plasticity loss | no |
| CompoNet | attention-based composition of frozen previous policies | yes |
| CbpNet | continual backprop (generate-and-test on hidden units) | no |

Five baselines receive the task index by construction — that is how those
methods are defined, and they are given it. **Our method does not**: it must
infer which stored knowledge is relevant from behavior alone. This asymmetry
favours the baselines and is recorded here rather than buried.

### PackNet capacity

PackNet is the one method that must pack every task into a **single fixed
network**, while ProgNet grows a column per task. At equal width PackNet would
carry roughly a tenth of ProgNet's parameters over a ten-task chain, so it runs
at a wider network by default (`--packnet-width 384`) as the parameter-matched
comparison.

Allocation is **equal share**: every task reserves `keep_frac / N` of the whole
network. With the defaults that is 10% each and the network ends fully used:

| width | total maskable weights | per task | last task |
|---|---|---|---|
| 256 (same as everyone else) | 135,168 | 13,516 | 13,516 (10.0%) |
| **384 (default)** | **301,056** | **30,105** | **30,105 (10.0%)** |

Equal share is deliberately *not* the textbook geometric rule, which keeps a
fraction of whatever is still free. Over ten tasks that rule decays as
`(1-keep_frac)^k` and starves the tail: at `keep_frac=0.75` the tenth task
receives **one weight**. `--packnet-capacity-mode geometric` is available for
faithfulness, and `capacity_report()` prints the exact per-task counts (with a
warning if the tail would be starved) at task 0, so a bad configuration is
visible immediately rather than inferred from a poor score at task 10.

One bug worth naming, since it was live in the ported version: `prune()` ranked
a tensor that had reserved weights zeroed out, so once the free pool got
smaller than the allocation it began handing out positions **earlier tasks
already owned**, destroying them. Selection is now restricted to genuinely free
positions.

### Continuous-action adaptation of the method

The Atari version leans on a convenience: a mixture of categoricals *is* a
categorical, so policy-space composition costs nothing. For continuous actions
that shortcut is gone. What replaces it:

* the executing policy is an exact `tanh`-squashed **mixture of Gaussians**
  (`policy_utils.SquashedGaussianMixture`) — sampled and scored exactly, not
  approximated;
* merge-pair selection and pair distillation compare *single* stored heads, so
  they use the **closed-form** diagonal-Gaussian KL — no sampling noise enters
  the merge decision;
* policy-space insertion fits one head to a whole mixture, which has no closed
  form, so it minimizes a Monte-Carlo **cross-entropy** against samples drawn
  once from the mixture. That has the same minimizer as `KL(mixture || head)`,
  since the two differ only by the mixture's own entropy.

Because `tanh` is invertible and KL is invariant under invertible maps,
`KL(tanh#p ‖ tanh#q) = KL(p ‖ q)`: the closed-form pre-squash KL *is* the
behavioral KL between executed policies. All of this was checked numerically
against Monte-Carlo (closed-form KL matches MC to 5e-4; mixture density
integrates to 1.00000000; the projection objective is minimized exactly at
`q = p`).

---

## 4. Protocol

* `total_timesteps` is Δ, the whole per-task budget. The frozen tail **B is
  inside Δ**, never additional — otherwise methods that retain replay states
  would silently get more environment interaction than methods that do not.
* Every method spends the same B, whether or not it keeps the states.
* Evaluation never updates the policy; its interactions are counted separately
  in `interaction_budget.json`.
* Evaluation episode seeds are fixed per task and independent of the training
  seed, so a continual run and its scratch reference are scored on identical
  episodes.
* Scratch references use **seed 201**, deliberately disjoint from the continual
  seeds. A forward-transfer denominator that shares a seed with its numerator
  is not an independent reference.

### Which matrix cells are computed

A full performance matrix is 10 × 10 = 100 evaluations per method per seed, and
**90 of those cells are never read** by any reported metric. Only two families
matter:

```
diagonal  P[i, i]     peak on a task (used by forgetting and forward transfer)
last row  P[last, i]  retention at the end of the chain
```

`metrics.cell_is_needed()` is that rule, in one predicate, shared by the runner
and the analysis. It is a ~5× reduction in evaluation time and stored results
and changes no reported number.

**On the diagonal, test-time adaptation is switched off.** The checkpoint has
just finished training that exact task; letting it re-tune its routing there
would inflate the peak that forgetting is measured against and make forgetting
look smaller than it is.

### Test-time adaptation budget: 6000 steps

`--test-adapt-steps` is environment interactions spent adapting **only the
routing weights** of a composed policy — the stored experts never move. So it
measures "can the method retrieve the right piece of what it already knows",
not "can it relearn the task".

6000 = **20 episodes × the 300-step horizon**, chosen to be defensible:

* MAML's RL protocol adapts with **20 rollouts per gradient step** for its
  locomotion tasks; its **2D-navigation** task — which is what PointMaze is —
  uses 40 samples per step with up to 4 updates. 20 rollouts sits at the low
  end of that published range.
* PEARL-style few-shot evaluation adapts from a handful of trajectories.
* It is **6%** of the 100k per-task training budget: far too little to be
  mistaken for retraining.

Methods with no routing weights (including `Ours-parameter` and `CKA-RL`, which
collapse their pool into one head) get no adaptation, and the result records
`adapted: false` rather than pretending otherwise.

---

## 5. Layout

```
pointmaze_env.py          physics, tabulated geometry, geodesic reward
tasks.py                  the 10 tasks, both suites, validate_suite()
shared_arch.py            MLP encoder, twin critic
policy_utils.py           squashed Gaussian + exact Gaussian mixture, KLs
policy_composition.py     stacked multi-head forward, mixture construction
policy_space.py           policy-space composition and storage projection
knowledge_pools.py        bounded pool, merging, lineage-balanced buffers
cka_rl.py                 the agent: pool + merge selection + distillation
run_sac_continual.py      SAC trainer for Ours / Ours-parameter / CKA-RL
baselines/                the seven other methods + their SAC trainer
training_protocol.py      TaskBudget (B inside Δ) and warmup logic
replay_buffer.py          preallocated SAC replay
checkpoint_evaluation.py  scoring + routing-only test-time adaptation
metrics.py                sparse performance matrix, forgetting, transfer
scratch_baselines.py      from-scratch references (seed 201)
run_continual_benchmark.py  chain orchestration, resume, identity checks
plots.py                  figures
pointmaze_smoke.py        environment tests (no torch)
baseline_smoke.py         end-to-end test of every method
verify_static.py          imports resolve
verify_flags.py           every CLI flag passed between scripts exists
pointmaze_kaggle.ipynb    Kaggle notebook: pick methods, resume across sessions
```

## 6. Resuming across sessions

Every task writes `manifest.json`. A relaunched chain skips any position whose
checkpoint exists **and** matches on every identity field the orchestrator
controls, so a session that dies at task 7 resumes at task 7.

`--max-seconds` makes the runner stop launching *new* tasks near the session
limit, so a session ends with consistent resumable state instead of being
killed mid-write.

To split work, give each person a different `RUN` list: each name is an
independent chain under its own `save_root/suite/tag/method/seed_*` path. At
the end, attach both output datasets to one session and the scoring cell picks
up everything it finds.

## 7. What is and is not tested here

**Tested by running it:** the whole environment and task suite (physics,
reward, sensors, determinism, speed, solvability, family structure); the
Gaussian/mixture/KL math, checked numerically against Monte-Carlo; checkpoint
identity and resume logic; that every cross-module import resolves and every
CLI flag passed between scripts exists; notebook JSON validity.

**Not tested here:** anything requiring torch or gymnasium to execute. PyPI was
blocked in the environment this was built in (HTTP 403), so no SAC step has
actually run against this code. **Run `baseline_smoke.py --method all` on
Kaggle before committing a session to a full run** — it exercises construction,
a task boundary, the merge, the projection, saving, reloading and scoring for
all ten methods in a few minutes.

## Sources

- [Point Maze — Gymnasium-Robotics](https://robotics.farama.org/envs/maze/point_maze/)
- [MAML (Finn et al., 2017)](https://arxiv.org/pdf/1703.03400) — RL adaptation budgets
- [PEARL (Rakelly et al., 2019)](https://arxiv.org/pdf/1903.08254) — few-shot meta-RL evaluation
- [Continual World (Wołczyk et al., 2021)](https://ar5iv.labs.arxiv.org/html/2105.10919) — CRL metric conventions
- [COOM (Tomilin et al., 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/d61d9f4fe4357296cb658795fd7999f0-Paper-Datasets_and_Benchmarks.pdf) — CRL benchmark protocol
