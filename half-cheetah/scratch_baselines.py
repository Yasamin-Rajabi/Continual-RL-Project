"""Trains and caches from-scratch, single-task SAC baselines for Forward Transfer.

Forward transfer (survey Eq. 9) needs AUC_i^b: the learning curve of task i
trained ALONE, with no continual history, at the SAME step budget as the
continual run. This script trains and caches exactly that -- once per
(suite, task_id, total_timesteps, seed) combination -- so metrics.py never
retrains a baseline it has already computed; it just reads the cached
TensorBoard logs back.

Run this once, BEFORE computing forward transfer, with the SAME
--total-timesteps you use for the real continual run:

    python3 scratch_baselines.py --task-suites halfcheetah_vel halfcheetah_wind_vel \
        --total-timesteps 300000

Resumable: an already-complete (suite, task_id, seed) combination is
detected via checkpoint_complete() and skipped, not retrained. Uses 3 seeds
by default (101, 102, 103 -- deliberately disjoint from the continual run's
seeds 1/2/3, so nobody mistakes a baseline seed for a continual-run seed).

Everything lands under --save-root (default scratch_models/) for
checkpoints and under runs/scratch/... for TensorBoard logs (the "runs/"
prefix is NOT configurable -- run_sac.py hardcodes it, see
SummaryWriter(f"runs/{args.tag}/{run_name}") in run_sac.py).
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess

from tasks import TASK_SUITES, get_task_name

SCRATCH_SAVE_ROOT = "scratch_models"
DEFAULT_SCRATCH_SEEDS = [101, 102, 103]


def scratch_run_name(suite, task_id, seed):
    return f"{suite}__task_{task_id}__cka-rl__run_sac__{seed}"


def scratch_tag(suite, task_id, total_timesteps, seed):
    return f"scratch/{suite}/task_{task_id}/steps_{total_timesteps}/seed_{seed}"


def scratch_checkpoint_dir(save_root, suite, task_id, total_timesteps, seed):
    return (
        pathlib.Path(save_root) / suite / f"task_{task_id}" / f"steps_{total_timesteps}"
        / f"seed_{seed}" / scratch_run_name(suite, task_id, seed)
    )


def scratch_event_dir(runs_root, suite, task_id, total_timesteps, seed):
    """Where this baseline's TensorBoard log lives -- read by metrics.py to
    compute AUC_i^b. runs_root is normally the literal "runs" (see module
    docstring); passed as a parameter so callers keep one source of truth."""
    return (
        pathlib.Path(runs_root) / scratch_tag(suite, task_id, total_timesteps, seed)
        / scratch_run_name(suite, task_id, seed)
    )


def checkpoint_complete(path):
    required = ["policy_snapshot.pt", "fc.pt", "mean_pool.pt", "logstd_pool.pt"]
    return path.exists() and all((path / name).exists() for name in required)


def train_one_baseline(suite, task_id, total_timesteps, seed, args):
    run_dir = scratch_checkpoint_dir(args.save_root, suite, task_id, total_timesteps, seed)
    if checkpoint_complete(run_dir) and not args.force_retrain:
        print(f"[scratch] {suite}/task_{task_id}/seed_{seed} already complete: {run_dir}")
        return run_dir

    cmd = [
        "python3", "run_sac.py",
        "--model-type=cka-rl",
        f"--task-suite={suite}",
        f"--task-id={task_id}",
        f"--seed={seed}",
        f"--tag={scratch_tag(suite, task_id, total_timesteps, seed)}",
        f"--save-dir={run_dir.parent}",
        f"--analysis-root={args.analysis_root}",
        f"--total-timesteps={total_timesteps}",
        f"--learning-starts={args.learning_starts}",
        f"--random-actions-end={args.random_actions_end}",
        f"--batch-size={args.batch_size}",
        f"--policy-lr={args.policy_lr}",
        f"--q-lr={args.q_lr}",
        f"--gamma={args.gamma}",
        f"--tau={args.tau}",
        f"--pool-size={args.pool_size}",
        f"--eval-every={args.eval_every}",
        f"--num-evals={args.num_evals}",
        f"--distill-extra-steps={args.distill_extra_steps}",
        f"--max-distill-buffer={args.max_distill_buffer}",
        f"--similarity-samples={args.similarity_samples}",
        f"--distill-max-samples={args.distill_max_samples}",
        f"--distill-epochs={args.distill_epochs}",
        f"--distill-lr={args.distill_lr}",
        f"--distill-batch-size={args.distill_batch_size}",
        f"--distill-test-frac={args.distill_test_frac}",
        f"--analysis-log-every={args.analysis_log_every}",
        # A scratch baseline is a lone root task: classic_cka with no
        # distillation and no alpha-mass is the plain, unmodified case --
        # and there's no --prev-units, which is the whole point.
        "--fusion-mode=classic_cka",
        "--no-use-alpha-scale",
        "--no-distillation",
        "--no-use-alpha-mass",
    ]
    if args.cpu:
        cmd.append("--no-cuda")

    print(
        f"\n>>> [scratch] {suite} / task {task_id} ({get_task_name(task_id, suite)}) "
        f"/ seed {seed} / {total_timesteps} steps <<<"
    )
    subprocess.run(cmd, check=True)
    if not checkpoint_complete(run_dir):
        raise RuntimeError(f"scratch baseline finished but checkpoint is incomplete: {run_dir}")
    return run_dir


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--task-suites", nargs="+",
        default=["halfcheetah_vel", "halfcheetah_wind_vel"],
        choices=sorted(TASK_SUITES.keys()),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SCRATCH_SEEDS)
    p.add_argument("--total-timesteps", type=int, default=300_000,
                    help="MUST match the continual run's --total-timesteps for FT to be valid.")
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--random-actions-end", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--q-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--num-evals", type=int, default=5)
    p.add_argument("--distill-extra-steps", type=int, default=10_000)
    p.add_argument("--max-distill-buffer", type=int, default=50_000)
    p.add_argument("--similarity-samples", type=int, default=2_048)
    p.add_argument("--distill-max-samples", type=int, default=20_000)
    p.add_argument("--distill-epochs", type=int, default=8)
    p.add_argument("--distill-lr", type=float, default=3e-4)
    p.add_argument("--distill-batch-size", type=int, default=256)
    p.add_argument("--distill-test-frac", type=float, default=0.2)
    p.add_argument("--analysis-log-every", type=int, default=5_000)
    p.add_argument("--save-root", default=SCRATCH_SAVE_ROOT)
    p.add_argument("--analysis-root", default="analysis_runs_scratch")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    for suite in args.task_suites:
        n_tasks = len(TASK_SUITES[suite])
        for task_id in range(n_tasks):
            for seed in args.seeds:
                train_one_baseline(suite, task_id, args.total_timesteps, seed, args)
    print("\nDone. Scratch baselines cached under:", args.save_root)
    print("(Re-run metrics.py / run_continual_benchmark.py now to use them for Forward Transfer.)")


if __name__ == "__main__":
    main()
