# Commented runs and source-lineage balancing

## 1. Put ablations in separate main-run folders

`job.sh` and `job_eval.sh` accept:

```bash
bash job.sh --comment amass
```

The comment is appended only to the **main** run folder and SLURM log name. Scratch
baselines stay at the existing shared paths.

Examples:

```text
main/combined_deterministic/          # original run
main/combined_deterministic_amass/    # --comment amass
main/combined_policy_stochastic_amass/
```

Allowed comment characters are letters, digits, `.`, `_`, and `-`.

For the alpha-mass regularizer ablation, edit `COMMON_ARGS` in `job.sh`:

```bash
--alpha-mass-reg 0
```

then submit:

```bash
bash job.sh --comment amass
```

Do not change or rerun scratch merely because `alpha_mass_reg` changed; scratch uses
no historical alpha mass.

If you later use `job_eval.sh` for a commented run, give the same comment and make
its training-semantic settings match the training job (for example the same
`--alpha-mass-reg` and lineage-balancing flag):

```bash
bash job_eval.sh --comment amass
```

## 2. Source-lineage-balanced merge ablation

The new benchmark/run_sac flag is:

```bash
--balance-source-lineages
```

The control/current behavior is explicitly:

```bash
--no-balance-source-lineages
```

When enabled, the code uses each stored row's `source_ids` (unique continual
sequence occurrence, not semantic task ID) in all three places where recursive
merge snowballing can erase old sources:

1. **Behavioral-KL pair selection:** the per-slot reference-state budget is sampled
   as evenly as possible across original source lineages in that slot.
2. **Distillation:** the total distillation budget is sampled as evenly as possible
   across the union of original source lineages from the two selected parents.
   Each row is still taught by the immediate parent policy that owns that lineage.
   Train/validation splitting is also stratified by source lineage.
3. **Retained merged buffer:** if the merged buffer must be truncated to
   `max_distill_buffer`, rows are retained as evenly as possible across original
   source lineages instead of assigning roughly half the buffer to each immediate
   parent.

If one source lineage has fewer rows than its equal quota, all of its rows are kept
and the unused quota is redistributed across the remaining lineages.

Example ablation:

```bash
# in COMMON_ARGS
--balance-source-lineages

bash job.sh --comment lineage
```

To combine it with the zero double-well coefficient:

```bash
--alpha-mass-reg 0
--balance-source-lineages
```

and:

```bash
bash job.sh --comment amass_lineage
```

The flag is part of checkpoint training identity. A `True` lineage-balanced run
will therefore never silently reuse a checkpoint trained with the old behavior.
Default-off runs remain backward-compatible with checkpoints produced immediately
before this ablation was added.

## 3. Collect paper metrics for a commented run

`collect_paper_metrics.py` accepts the same comment:

```bash
python collect_paper_metrics.py \
  --experiment-root "$ROOT" \
  --suite halfcheetah_wind_vel \
  --eval-mode deterministic \
  --comment amass
```

Its default outputs are tagged too, e.g.:

```text
paper_metrics_deterministic_amass.csv
paper_metrics_per_seed_deterministic_amass.csv
paper_PERF_occurrences_deterministic_amass.csv
paper_PERF_return_deterministic_amass.png
paper_PERF_success_deterministic_amass.png
```


## Separate alpha-mass learning rate

`--alpha-mass-lr` controls only the raw historical-vs-novel mass parameter `g`
where `m = sigmoid(g)`. The within-history alpha logits (and optional alpha
scale) continue to use `--alpha-lr`.

If `--alpha-mass-lr` is omitted, it reuses `--alpha-lr`, exactly preserving the
old optimizer behavior. For the proposed slower-gate diagnostic, use for example:

```bash
--alpha-lr 5e-3
--alpha-mass-lr 3e-4
--alpha-mass-reg 0
```

For the HalfCheetah SLURM scripts the default line is now explicit:

```bash
--alpha-mass-lr 5e-3
```

Change the same line in `job_eval.sh` when evaluating a run trained with a
non-default mass LR; checkpoint identity deliberately treats a different
alpha-mass LR as a different training configuration.
