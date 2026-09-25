# Storage optimization for continual-RL runs

The benchmark now keeps all metric-relevant policy state while avoiding repeated copies of training-only rollout/distillation buffers.

## New default behavior

`run_continual_benchmark.py` now uses two storage controls:

- `--compact-storage` (default: enabled): after sequence checkpoint `k+1` is safely present, checkpoint `k` has only its rollout/distillation buffers removed. Its finalized encoder/policy-pool weights, policy snapshot, manifest, projection/merge metadata, and all scalar logs remain.
- `--no-save-analysis-snapshots` (default at the benchmark layer): do not write `start.pt`, `pre_finalize.pt`, and `post_finalize.pt` tensor snapshots. The scalar analysis metrics are still written to TensorBoard and `scalars.csv`. Merge-lineage plots fall back to metadata in the finalized pool checkpoint.

The newest checkpoint in every chain is deliberately left uncompressed so training can resume or the sequence can be extended. A compacted predecessor is marked by `storage_compacted.json` and the runner refuses to use it as the newest training predecessor.

Scratch forward-transfer baselines also disable task-boundary tensor snapshots because FT is computed from their scalar learning curves.

## Existing results

From an environment directory such as `half-cheetah`, `Walker2D`, `AntDir`, `Hopper`, `metaworld`/`meta-world`, or `atari`, inspect what would be compacted:

```bash
python compact_existing_checkpoints.py --save-root /path/to/RUN_ROOT/agents --dry-run
```

Then compact old checkpoints while keeping the newest checkpoint of each chain fully resumable:

```bash
python compact_existing_checkpoints.py --save-root /path/to/RUN_ROOT/agents
```

If old analysis snapshots also exist and are no longer needed for manual tensor inspection:

```bash
python compact_existing_checkpoints.py \
  --save-root /path/to/RUN_ROOT/agents \
  --analysis-root /path/to/RUN_ROOT/analysis \
  --drop-analysis-snapshots
```

For a completely finished experiment where you only need evaluation/metrics and do not need to resume training, maximum compaction is:

```bash
python compact_existing_checkpoints.py \
  --save-root /path/to/RUN_ROOT/agents \
  --analysis-root /path/to/RUN_ROOT/analysis \
  --include-latest \
  --drop-analysis-snapshots
```

Do **not** use `--include-latest` if you may resume or extend a chain. It removes the latest checkpoint's historical rollout buffers too, which preserves evaluation but not future distillation/merging state.

## What remains available

The compact mode preserves the data needed by the current benchmark for:

- full checkpoint-by-task retention matrices;
- `A_N`, forgetting (FG), backward transfer (BWT), and forward-transfer metrics;
- final return/success/error metrics;
- test-time alpha adaptation with `--frozen-eval-policy pool`;
- merge similarity/distillation diagnostics and lineage plots;
- training curves through scalar logs and `scalars.csv`;
- post-hoc re-evaluation of every old finalized policy under different evaluation seeds/episode counts.

The only information intentionally removed from old checkpoints is the historical raw rollout/distillation memory that has already been inherited by the newer resumable checkpoint, plus optional redundant task-boundary analysis snapshots.
