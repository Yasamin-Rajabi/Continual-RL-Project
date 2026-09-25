# MiniGrid paper integration notes

## What was added

- `minigrid` is a first-class environment in the repository-level `job_paper.sh` launcher.
- Default suite: `doorkey4`, sequence `0 1 2 3 0 2 1 3`, 60k interactions per occurrence.
- Full discrete ETHOS implementation: categorical policy-space routing, categorical KL pair selection/distillation, standalone projection, source-lineage balancing, pool-size/warmup/KL ablations.
- SAC-Discrete critic/actor updates with exact action expectations.
- Shared paper baselines now support both Box and Discrete action spaces.
- Scratch FT references, retention evaluation, A_N/FG/BWT/FT metrics, aggregate collection, and plot/CSV paths follow the same paper pipeline as the continuous environments.
- Compact checkpoint storage and opt-in analysis snapshots match the storage-optimized HalfCheetah workflow.
- One-time cluster dependency bootstrap: `bash minigrid/setup_cluster_env.sh`.

## Compatibility

The continuous-control branches of the shared baseline trainer are unchanged.
For old HalfCheetah/Walker2D/AntDir baseline manifests, `baseline_runner.py`
reproduces the exact pre-MiniGrid source signature, so adding the discrete branch
does not make completed continuous runs appear stale.

## Validation performed in this package

- Python compile check over `minigrid/`, `paper_runs/`, `paper_experiments.py` and collector.
- Shell syntax check for MiniGrid/paper launch scripts.
- `minigrid/sanity_check_pool.py`: all categorical pool/composition/distillation tests pass.
- Paper launcher preview: full `main kl lineage pool warmup` expands to 36 workers (2 scratch + 34 train/ablation workers) for the two paper seeds.
- Pilot eval launcher and collect launcher both resolve MiniGrid paths correctly.
- Synthetic compact-checkpoint test: policy tensors remained bit-identical while retained rollout buffers were removed.
- Existing continuous baseline source signatures were checked against the uploaded pre-MiniGrid project and match exactly for HalfCheetah, Walker2D and AntDir.

## Environment-runtime limitation of this build machine

The local build container used for this integration does not have `gymnasium`
or `minigrid` installed and cannot reach PyPI, so a real MiniGrid rollout could
not be executed here. The cluster setup script therefore performs a real
`doorkey4` instantiation check before submission and the paper worker fails fast
with a clear setup command if the dependency is missing.

## Recommended first run

From the repository root:

```bash
bash minigrid/setup_cluster_env.sh

bash job_paper.sh \
  --environments minigrid \
  --groups main \
  --methods baseline combined_policy \
  --seeds 1 \
  --comment mg_pilot \
  --submit
```

After that pilot finishes, evaluate it with the same selection and `--phase eval`.
If the curves are sensible, launch the complete grid:

```bash
bash job_paper.sh \
  --environments minigrid \
  --groups main kl lineage pool warmup \
  --comment paper_v1 \
  --submit
```
