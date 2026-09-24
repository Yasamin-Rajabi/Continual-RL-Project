# Baselines, compression ablations and unified Kuma submission

This integration starts from the uploaded `Continual-RL-Project(4).zip` and
uses baseline components from `Continual-RL-Project-narges.zip`.
Read **BASELINE_PROTOCOL.md** before treating the baseline table as a published
reproduction: it explains the donor naming conflicts, explicit ports, memory
budgets and default task-blind evaluation wrapper.

## What did not change

All existing `.sh` files from the uploaded main project are byte-for-byte
unchanged. Their hyperparameters, resource requests, images, task suites,
seeds, paths and comments are the source for the additive batch launcher.
HalfCheetah/Walker2D environments, actor/critic updates, projection, lineage
balancing, pool-size behavior and alpha-warmup implementation are unchanged.
MetaWorld, Hopper and the old `ant/` velocity benchmark were not modified.

There are necessary Python changes in HalfCheetah and Walker2D to forward the
compression flag, randomly choose the discarded member, record identity and
cache the new setting. A narrowly verified compatibility mapping accepts the
uploaded default-code checkpoints, not arbitrary historical source edits.
See `INTEGRATION_CHANGES.json` for byte hashes and modified-file lists.

## Install

Extract the changed-files ZIP into the **repository root**, or use the full
project ZIP. Never extract the changed-files ZIP inside `half-cheetah/` itself.
Do not install packages into the host or alter the SIF for this patch. Use the
existing PyTorch/Gymnasium/MuJoCo/SB3/TensorBoard/tyro container. The scripts
use `/opt/conda/bin/python` inside your image and stdlib-only Python on Kuma's
login node. A missing container dependency is an error to investigate, not a
reason to silently install different versions inside every worker.

From Kuma:

```bash
cd "$HOME/Cont/Continual-RL-Project"
bash job_paper.sh --help
```

## 1. Smoke test before a large batch

Run environment construction/reset/step tests in the container:

```bash
PROJECT_ROOT="$HOME/Cont/Continual-RL-Project"
IMAGE="$HOME/containers/ethos_crl_torch280_mj237.sif"
APPTAINERENV_PYTHONNOUSERSITE=1 apptainer exec \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" "$IMAGE" \
  /opt/conda/bin/python "$PROJECT_ROOT/paper_runs/smoke_real.py"
```

In a CPU/GPU allocation, optionally add `--train --evaluate` to exercise real
short task chains, serialization, reuse, scratch, retention and FT. Use `--nv`
inside a GPU allocation. For example, first test one environment:

```bash
APPTAINERENV_PYTHONNOUSERSITE=1 apptainer exec --nv \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" "$IMAGE" \
  /opt/conda/bin/python "$PROJECT_ROOT/paper_runs/smoke_real.py" \
  --environments AntDir --train --evaluate
```

It uses a fresh temporary directory, 128 steps/occurrence, three occurrences,
small buffers and one-episode evaluation. These are **test-only** settings;
production defaults are not modified. All requested new baselines plus
CKA-RL/ETHOS are tested; combined_policy also exercises random_merge and
kl_discard. This is a mechanics smoke test, not meaningful RL performance.
Actual MuJoCo/GPU testing was not possible in the development sandbox; see
INTEGRATION_VALIDATION.md for the tests that were actually run there.

## 2. Preview the experiment grid (no jobs submitted)

```bash
bash job_paper.sh \
  --environments half-cheetah Walker2D AntDir \
  --groups main kl lineage pool warmup \
  --comment paper_v1
```

The current presets plan up to 51 workers per environment before reuse:
48 method/ablation seed workers and 3 scratch seed workers. This is a large
batch (153 workers for all three); preview and restrict it as necessary.
All Slurm jobs retain the name `causal`. Existing memory, CPU, GPU, time,
partition, QoS and node-exclusion directives come from that environment's
`job.sh`, not newly hard-coded overrides.

Selected methods are exactly:

    baseline ft_n prognet packnet masknet crelus componet cbpnet combined_policy

No combined parameter-space, distil_only, weight_only or policy-student run is
scheduled by this launcher. Existing old variants remain available in old code.

Ablation groups:

| Group | Cases |
|---|---|
| main | These eight baselines and combined_policy reference |
| kl | Reference; random_merge + distill; min-KL pair + random member discard |
| lineage | Reference plus the opposite value of the existing lineage flag |
| pool | Capacities from `--pool-sizes` (default 3,5,8) |
| warmup | Initial historical alpha-adaptation steps from `--warmup-steps` (default 0,5000,10000) |
| no_merge | Optional theory diagnostic: capacity equals sequence length |

A case identical to the reference is deduplicated. Pool and warmup settings
are changes only in those explicitly selected ablation cases. The reference
keeps the existing shell values, including mass regularization and entropy.
`--groups all` additionally includes no_merge; no_merge is NOT in the default
grid because it spends more memory. It isolates compression loss, not a
matched-memory improvement.

For a smaller first batch:

```bash
bash job_paper.sh --environments half-cheetah \
  --groups main kl --methods baseline combined_policy --comment paper_v1
```

## 3. Submit

Repeat the preview with `--submit`:

```bash
bash job_paper.sh \
  --environments half-cheetah Walker2D AntDir \
  --groups main kl lineage pool warmup \
  --comment paper_v1 --submit
```

Scratch and continual seeds run independently and concurrently. There are no
automatic Slurm dependencies. Scratch workers train only missing compatible
reference checkpoints. Training workers do not require scratch to be finished.
Each continual `(method, case, mode, seed)` has its own allocation.

The launcher freezes the extracted settings into JSON under
`<EXPERIMENT_ROOT>/.paper_suite/` before `sbatch`. Editing job.sh while a job
is queued therefore does not change that job's arguments. Source code itself
is not snapshotted: do not edit training Python files during an active batch.

The default preflight runs a CPU-only container process before allocation:
valid complete runs are skipped without submitting another GPU job. Existing
incompatible completed data is not removed: use a fresh `--comment` or the
matching settings. `--no-precheck` defers checks to workers, it does not disable
the worker's identity guards. `--force-retrain` is refused by this safe launcher.

A repeat submission checks this launcher's receipts against the user's active
`squeue` jobs and avoids duplicates. Locks also protect seed writes and scratch
reference reservation. **Do not concurrently submit a legacy job.sh and this
launcher to the same output paths**: legacy scripts do not participate in
these new locks or job receipts.

## Existing run names and safe reuse

The canonical format remains:

    <root>/main/<method>_<mode>[_comment]/
        agents/<suite>/<condition>/seed_N/seq_K/<run_name>/
        runs/<suite>/<condition>/seed_N/seq_K/<run_name>/
        analysis/...
        plots/<suite>/...

Existing CKA/ETHOS run-name leaves are unchanged. New methods replace the
`cka-rl` leaf token by their method name, retaining the same parent structure.
Nothing moves, renames or deletes old experiment folders.

Example new folders with `--comment paper_v1`:

    combined_policy_deterministic_paper_v1
    combined_policy_deterministic_paper_v1_random_merge
    combined_policy_deterministic_paper_v1_kl_discard
    combined_policy_deterministic_paper_v1_lineage_off
    combined_policy_deterministic_paper_v1_pool3
    combined_policy_deterministic_paper_v1_warmup0
    ft_n_deterministic_paper_v1
    prognet_deterministic_paper_v1
    ...

When a requested new folder is empty, the launcher can find an older sibling
with a different comment and reuse it **only if completed checkpoint identities
match**. The log prints `[reuse-equivalent]` and records both requested and
resolved paths in the job spec. This allows an existing `_am0_lineage` run to
serve as a matching reference without copying it. `--no-reuse-equivalent`
restricts reuse to the exact requested folder. A label alone never establishes
compatibility, and an occupied incompatible requested folder is not bypassed.

Resume is at a completed **task-occurrence** boundary, not inside an unfinished
SAC update loop. Before retrying an incomplete task, its partial outputs are
moved into `.paper_suite/incomplete_attempts/`; they are not deleted or mixed
with the new TensorBoard event stream. Complete descendants of a missing or
changed parent are not silently reused.

## Scratch reference reuse

Reference selection ignores history-only knobs that a scratch root policy
does not use: pool capacity, pair selection, lineage, historical alpha warmup,
and mass regularization/LR. Genuine root SAC settings remain checked,
including environment, steps, optimizers, architecture, random-action/start
schedule, frozen-tail budget, evaluation cadence and action mode.

Matching existing canonical references are reused. If canonical reference
settings genuinely differ, they remain intact and a new matched reference is
placed under:

    <root>/scratch/<mode>/references/<core_settings_hash>/

The selected reference is recorded and linked at the usual
`<RUN_ROOT>/runs/scratch` when evaluation starts. An existing link to another
reference is never silently repointed. Complete checkpoints with missing
learning curves are reported; they are not declared usable for FT. This
selection policy is for training/reference reuse. The main post-hoc FT
functions still read saved curves without requiring current runtime/source
fingerprints to match a historical metrics process.

The external baselines use the same common plain SAC scratch reference. They
do not each cause an additional scratch grid. To run scratch alone:

```bash
bash job_paper.sh --environments half-cheetah Walker2D AntDir --phase scratch --submit
```

## 4. Evaluate after both scratch and continual jobs finish

Use the **same** environments, groups, comment and explicit overrides:

```bash
bash job_paper.sh \
  --environments half-cheetah Walker2D AntDir \
  --groups main kl lineage pool warmup \
  --comment paper_v1 --phase eval --submit
```

This does not train policies. Existing main-method evaluation remains the
existing post-finalization, alpha-only retrieval pipeline. New baselines use
the documented common-protocol evaluator in BASELINE_PROTOCOL.md. Valid seed
counts and excluded seeds are reported. A case with no ready seed/reference
is skipped at preflight instead of publishing a zero-valued result.

Each `(method, case, mode)` gets one evaluation job over its valid seeds;
evaluation is NOT parallelized per seed here. Main-method complete caches
are reused. The new-baseline evaluator saves each completed checkpoint/task
cell atomically, so it can resume at cell boundaries; this does not add
mid-adaptation resume to the original main evaluator.

The optional `native` and `latest` baseline evaluation protocols write to
separate subdirectories, never over the default results. Do not combine their
numbers with reward_route results as one evaluation protocol.

## 5. Collect paper tables (no environments, no policy updates)

```bash
bash job_paper.sh --environments half-cheetah Walker2D AntDir --phase collect --submit
```

Here `--submit` means execute the collector inside the container; it does not
allocate an H100. The supplied collector reads the canonical metric CSVs and
all matching method/comment folders. It writes the same per-seed/occurrence
PERF tables and plots. AntDir additionally reports the explicitly raw
`FT_return_auc_delta`; its normalized FT_return is deliberately NaN.

## Explicit new experiment overrides

These do not edit job.sh; repeat them at evaluation time:

```bash
bash job_paper.sh --environments half-cheetah --groups main kl \
  --methods combined_policy --set alpha-entropy-reg=0.001 \
  --comment aent1e-3 --submit
```

Environment-specific override:

```bash
bash job_paper.sh --environments AntDir --groups main \
  --env-set AntDir:total-timesteps=500000 --comment steps500k --submit
```

Changing an environment's global `EXPERIMENT_ROOT`, `BASE_STORAGE` or
`ETHOS_IMAGE` follows the same environment-variable overrides already present
in its job.sh. Avoid globally setting one EXPERIMENT_ROOT for unrelated
families unless you deliberately want them under a shared root.

## Validation and limitations

See INTEGRATION_VALIDATION.md. The development tests cover real Torch
optimization, lifecycle, protected weights, serialization, compatibility,
all three compression modes, mock-environment training/evaluation/CSV paths,
reference reservation and preservation of old shell files. They do not
establish benchmark performance, GPU throughput, package compatibility in an
unseen container, or exact reproduction of the published baseline tables.
