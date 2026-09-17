> **Updated implementation:** read `../IMPLEMENTATION_NOTES.md` and
> `../EVALUATION_GUIDE.md` first. The material below describes the uploaded
> project's older presets; its old defaults and performance claims do not
> validate this revised task-blind/policy-space implementation. Use
> `run_comparison.sh` for the new defaults and train fresh checkpoints.

# Experiment / ablation flags

The default code favors a stable final implementation, but every small optimization
that can change learning is exposed so it can be disabled in ablations.

## Shared encoder

| Flags | Meaning |
|---|---|
| `--no-train-shared --no-freeze-root-encoder` | Scratch root encoder learns on task 0, then is frozen/reloaded for later tasks. Default without pretraining. |
| `--pretrained-encoder PATH --no-train-shared` | Optional legacy/pretraining ablation: freeze the supplied encoder from task 0 onward. |
| `--train-shared` | Friend-workflow encoder rule: learn/fine-tune the encoder continually; later tasks start from the latest encoder, never reset to the root/pretrained file. No pretraining is required. |
| `--freeze-root-encoder` | Random-frozen encoder ablation. Mutually exclusive with `--train-shared`. |
| `--encoder-linear-out` | Encoder ends in a linear layer. Retained for optional pretraining/architecture ablations; serialized modules are validated at load time. |
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


## Friend-method controls merged from latest master

The latest master workflow keeps pretraining support only as an optional legacy/ablation path;
the supplied Kaggle notebook runs from scratch with `--train-shared`.  The following controls
make the new method changes individually switchable:

| Default in condition runner | Ablation | Effect |
|---|---|---|
| `--condition-alpha-scale` | `--no-condition-alpha-scale` | Classic modes learn alpha scale from 1; weight-delta modes use fixed scale 5. Disabling this lets `--use-alpha-scale/--fix-alpha-scale` choose one global rule. |
| `--distill-observation-skip` | `--no-distill-observation-skip` | Distillation modes concatenate raw observation to shared features before the policy heads. |
| `--alpha-lr 5e-3` | change value | Separate learning rate for knowledge-mixture parameters. |
| `--alpha-mass-lr 3e-4` | change value | Optional separate LR for the historical-vs-novel alpha-mass gate. If omitted, it reuses `--alpha-lr` exactly as before. |
| `--alpha-warmup-steps 5000` | `0` | Early weight-delta phase learns historical mixing before the new residual/mass moves. It activates only when at least two historical slots exist. |
| `--alpha-mass-reg 0.05` | `0` | Regularizes effective mass after warmup. |
| `--drift-reg 1.0` | `0` | When `--train-shared` and distillation are both active, penalizes encoder drift on historical observations. |
| `--distill-encoder-lr-mult 0.1` | `1.0` | Slows the shared encoder on later distillation tasks. |
| `--alpha-entropy-reg 0.01` | `0` | Entropy bonus on the **actual scaled** historical mixture during effective weight-delta warmup. |
| `--test-adapt-steps 5000` | `0` | Test-time adaptation of alpha-only mixture parameters for retention/final-row metrics. |

`--test-adapt-steps 0` restores frozen-checkpoint evaluation. FG/BWT now re-evaluate both the
diagonal checkpoint and final checkpoint with the same adaptation protocol and common episode
seeds. Forward transfer remains defined only on first encounters of previously unseen tasks;
repeated encounters are relearning/savings.

## Small optimizations / legacy switches

| Default | Disable for ablation | Effect |
|---|---|---|
| `--constrain-alpha-mass` | `--no-constrain-alpha-mass` | Positive normalized-softplus mass instead of a scalar that can hit zero/negative. |
| `--distill-select-best-val` | `--no-distill-select-best-val` | Restore best held-out-KL student epoch instead of always keeping the last epoch. |
| `--no-balance-source-lineages` | `--balance-source-lineages` | Legacy merge sampling balances immediate parents. Enabling the flag balances behavioral-KL reference states, distillation rows, validation splits, and truncated retained merge buffers across original `source_ids` (sequence occurrences). |
| `--no-collect-cosine-buffers` | `--collect-cosine-buffers` | Cosine-only modes skip unused post-training rollout states; enable to equalize post-training interaction counts. |
| `--no-use-alpha-scale` | `--use-alpha-scale` | Optional learned global scaling of historical alpha logits. |
| `--autotune --no-autotune-init-from-alpha` | `--autotune-init-from-alpha` | Legacy entropy autotuning starts at alpha=1; optional flag starts at `--alpha`. |

If `--no-autotune` is used, `--alpha` is the fixed SAC entropy coefficient.

## Reproducibility and resuming

Every task checkpoint now contains `run_manifest.json`. A checkpoint is resumable
only if its training configuration, training-source fingerprint, runtime package
versions, pretrained encoder contents, and parent checkpoint identities still
match. Old pre-manifest checkpoints are intentionally considered stale.

Scratch baselines for Forward Transfer must use the same encoder treatment, SAC settings,
evaluation cadence, training budget, and **actor-head architecture** as the continual run.
`scratch_baselines.py` therefore caches two variants by default: `plain` for Baseline/Weight-Only
and `distill_skip` for Distill-Only/Combined when observation skip is enabled. The default
`run_kaggle.sh` follows the current no-pretraining `--train-shared` workflow; TD-JEPA remains
available only as an explicit ablation.

## Diagnostics retained for upcoming TODOs

- `source_ids` identify sequence occurrences separately from semantic `task_ids`.
- merge snapshots/logs preserve task-level and source-occurrence lineage.
- KL logs include mean, p95, and max tails for selected merge pairs and distillation.
- `--no-balance-source-lineages` preserves the original immediate-parent-balanced control.
- `--balance-source-lineages` enables the source-occurrence-balanced ablation intended to prevent recursive merge lineages from being exponentially underrepresented.
- critic reset/persistence is intentionally unchanged pending the critic TODO.
