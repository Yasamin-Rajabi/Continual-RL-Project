# ETHOS / CKA-RL MiniGrid DoorKey benchmark

This directory is the discrete-action counterpart of the HalfCheetah paper
pipeline. Use the repository-level `job_paper.sh` for paper experiments.

## Default suite

`doorkey4` uses four MiniGrid tasks with a repeated continual sequence:

| task id | environment | horizon | diagnostic stages |
|---|---|---:|---|
| 0 | `MiniGrid-DoorKey-5x5-v0` | 100 | key -> door -> goal |
| 1 | `MiniGrid-Empty-Random-5x5-v0` | 60 | goal |
| 2 | `MiniGrid-DoorKey-6x6-v0` | 160 | key -> door -> goal |
| 3 | `MiniGrid-Unlock-v0` | 120 | key -> door |

Sequence: `0 1 2 3 0 2 1 3`.

The paper preset uses **60,000 total environment interactions per occurrence**.
The final 5,000-step frozen rollout used to build a distillation buffer is
*inside* that 60k budget, exactly as in the current HalfCheetah protocol.
`success` is the primary normalized continual-RL performance measure; an
auxiliary `task_error = 1 - progress` is logged for diagnostics.

## Discrete analogue of the method

MiniGrid has seven discrete actions. Each stored two-head policy is interpreted
as a categorical policy with

```text
logits(s) = head_a(s) + head_b(s)
```

ETHOS performs a **mixture of categorical policy distributions**, not a mixture
of parameters or action logits. SAC uses the exact expectation over all actions.
Behavioral pair selection and distillation use categorical forward/symmetric KL.
After every task, the routed mixture is projected to a standalone categorical
policy before insertion into the bounded pool. The parameter-space CKA-RL
baseline keeps the corresponding parameter-space composition/merge behavior.

## One-time cluster setup

The MuJoCo container may not contain the third-party `minigrid` package. Run:

```bash
bash minigrid/setup_cluster_env.sh
```

The script first checks the existing container. If MiniGrid is missing, it
installs only `minigrid` and `pygame` under `.paper_deps/minigrid` without
modifying the SIF image, then validates the `doorkey4` environments and runs
the CPU policy-pool sanity suite.

## Recommended short pilot

Before submitting the full grid, run one seed of only CKA-RL and ETHOS:

```bash
bash job_paper.sh \
  --environments minigrid \
  --groups main \
  --methods baseline combined_policy \
  --seeds 1 \
  --comment mg_pilot \
  --submit
```

Then evaluate it:

```bash
bash job_paper.sh \
  --environments minigrid \
  --groups main \
  --methods baseline combined_policy \
  --seeds 1 \
  --comment mg_pilot \
  --phase eval \
  --submit
```

A separate single-task learning pilot is also available from this directory:

```bash
cd minigrid
python pilot_check.py --task-suite doorkey4 --steps 60000
```

The 60k budget is an experimental choice, not a guaranteed convergence claim;
inspect the pilot learning curves before spending the full paper grid.

## Full paper run

This mirrors the groups used for HalfCheetah:

```bash
bash job_paper.sh \
  --environments minigrid \
  --groups main kl lineage pool warmup \
  --comment paper_v1 \
  --submit
```

The full expansion uses the two paper seeds and includes the matched scratch
runs needed for forward-transfer metrics. Preview the jobs by omitting
`--submit`.

After training:

```bash
bash job_paper.sh \
  --environments minigrid \
  --groups main kl lineage pool warmup \
  --comment paper_v1 \
  --phase eval \
  --submit
```

After evaluation jobs finish:

```bash
bash job_paper.sh --environments minigrid --phase collect --submit
```

## Metrics

The same paper pipeline computes:

- `A_N`, FG, BWT from frozen checkpoint evaluation using episodic success;
- `FT_success` from continual-vs-scratch success learning-curve AUC;
- `FT_return` from episodic-return AUC with MiniGrid's return upper bound 1.0;
- final return, success, and `task_error` diagnostics;
- behavioral KL, distillation, routing, merge-lineage, and timing diagnostics.

`scalars.csv` is retained so FT can be recomputed even if TensorBoard event
files are later removed.

## Storage behavior

The same compact-storage policy as HalfCheetah is enabled by default in
`minigrid/job.sh`:

- old sequential checkpoints keep finalized policies/pools but drop obsolete
  rollout/distillation buffers once their successor has been saved;
- the latest checkpoint stays resumable during an active chain;
- large `start.pt`, `pre_finalize.pt`, and `post_finalize.pt` analysis snapshots
  are disabled by default;
- evaluation and metric collection still work from compact checkpoints.

For completed old-style MiniGrid runs, use `compact_existing_checkpoints.py`
just as for the MuJoCo environments.
