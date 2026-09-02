# Experiment / ablation flags

The default code favors a stable final implementation, but every small optimization
that can change learning is exposed so it can be disabled in ablations.

## Shared encoder

| Flags | Meaning |
|---|---|
| `--no-train-shared --no-freeze-root-encoder` | Scratch root encoder learns on task 0, then is frozen/reloaded for later tasks. Default without pretraining. |
| `--pretrained-encoder PATH --no-train-shared` | TD-JEPA-style encoder is frozen from task 0 onward. Recommended pretrained setup. |
| `--train-shared` | Encoder is continually fine-tuned; later tasks start from the latest encoder, never reset to the root/pretrained file. |
| `--freeze-root-encoder` | Random-frozen encoder ablation. Mutually exclusive with `--train-shared`. |
| `--encoder-linear-out` | Encoder ends in a linear layer; required by the recommended TD-JEPA objective. The serialized module is validated at load time. |
| `--no-encoder-linear-out` | Original encoder with trailing ReLU. |

The present code is **not an exact reproduction of paper CKA-RL**, because the
knowledge vectors cover the policy heads while `fc` is a separate shared encoder.
A structurally coherent paper-like baseline is therefore:

```bash
--condition-index 1 \
--no-train-shared --no-freeze-root-encoder --encoder-from-base \
--no-encoder-linear-out --no-use-alpha-scale \
--no-collect-cosine-buffers
```

This learns the root representation once, fixes it thereafter, and keeps classic
CKA/cosine/arithmetic merging free of the new stabilizers.

## Four method conditions

1. `baseline`: `classic_cka`, cosine pair selection, arithmetic merge.
2. `distil_only`: `classic_cka`, symmetric-KL pair selection, policy distillation.
3. `weight_only`: `weight_delta`, cosine pair selection, arithmetic merge.
4. `combined`: `weight_delta`, symmetric-KL pair selection, policy distillation.

`--weight-use-alpha-mass/--no-weight-use-alpha-mass` controls whether conditions
3/4 also use learned alpha mass. Disable it to isolate the representation change.

## Small optimizations / legacy switches

| Default | Disable for ablation | Effect |
|---|---|---|
| `--constrain-alpha-mass` | `--no-constrain-alpha-mass` | Positive normalized-softplus mass instead of a scalar that can hit zero/negative. |
| `--distill-select-best-val` | `--no-distill-select-best-val` | Restore best held-out-KL student epoch instead of always keeping the last epoch. |
| `--no-collect-cosine-buffers` | `--collect-cosine-buffers` | Cosine-only modes skip unused post-training rollout states; enable to equalize post-training interaction counts. |
| `--no-use-alpha-scale` | `--use-alpha-scale` | Optional learned global scaling of historical alpha logits. |
| `--autotune --no-autotune-init-from-alpha` | `--autotune-init-from-alpha` | Legacy entropy autotuning starts at alpha=1; optional flag starts at `--alpha`. |

If `--no-autotune` is used, `--alpha` is the fixed SAC entropy coefficient.

## Reproducibility and resuming

Every task checkpoint now contains `run_manifest.json`. A checkpoint is resumable
only if its training configuration, training-source fingerprint, runtime package
versions, pretrained encoder contents, and parent checkpoint identities still
match. Old pre-manifest checkpoints are intentionally considered stale.

Scratch baselines for Forward Transfer must use the same encoder architecture,
pretrained encoder treatment, SAC entropy settings, and training budget as the
continual run. `run_kaggle.sh` creates separate S0/S4 scratch/log/model roots to
prevent accidental cross-use.

## Diagnostics retained for upcoming TODOs

- `source_ids` identify sequence occurrences separately from semantic `task_ids`.
- merge snapshots/logs preserve task-level and source-occurrence lineage.
- KL logs include mean, p95, and max tails for selected merge pairs and distillation.
- parent-balanced replay/distillation is intentionally **not** lineage-balanced yet;
  that behavior is the control for the upcoming exponential-decay investigation.
- critic reset/persistence is intentionally unchanged pending the critic TODO.
