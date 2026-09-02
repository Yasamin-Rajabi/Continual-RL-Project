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
checkpoints and under --runs-root/scratch/... for TensorBoard logs.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

from tasks import TASK_SUITES, get_task_name
from experiment_identity import (
    MANIFEST_NAME,
    checkpoint_matches as identity_checkpoint_matches,
    load_manifest,
)

SCRATCH_SAVE_ROOT = "scratch_models"
DEFAULT_SCRATCH_SEEDS = [101]


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


def scratch_analysis_dir(analysis_root, suite, task_id, total_timesteps, seed):
    return (
        pathlib.Path(analysis_root) / scratch_tag(suite, task_id, total_timesteps, seed)
        / scratch_run_name(suite, task_id, seed)
    )


def checkpoint_complete(path):
    path = pathlib.Path(path)
    required = ["policy_snapshot.pt", "fc.pt", "mean_pool.pt", "logstd_pool.pt", MANIFEST_NAME]
    return path.exists() and all((path / name).exists() for name in required) and load_manifest(path) is not None


def _expected_training_config(suite, task_id, total_timesteps, seed, args):
    return {
        "model_type": "cka-rl",
        "task_suite": suite,
        "task_id": int(task_id),
        "seq_idx": 0,
        "seed": int(seed),
        "cuda": not bool(args.cpu),
        "fusion_mode": "classic_cka",
        "total_timesteps": int(total_timesteps),
        "gamma": float(args.gamma),
        "tau": float(args.tau),
        "batch_size": int(args.batch_size),
        "learning_starts": int(args.learning_starts),
        "random_actions_end": int(args.random_actions_end),
        "policy_lr": float(args.policy_lr),
        "alpha_lr": float(args.alpha_lr),
        "alpha_warmup_steps": int(args.alpha_warmup_steps),
        "q_lr": float(args.q_lr),
        "alpha": float(args.alpha),
        "autotune": bool(args.autotune),
        "autotune_init_from_alpha": bool(args.autotune_init_from_alpha),
        "pool_size": int(args.pool_size),
        "encoder_from_base": bool(args.encoder_from_base),
        "freeze_root_encoder": bool(args.freeze_root_encoder),
        "distillation": False,
        "use_alpha_mass": False,
        "use_alpha_scale": False,
        "fix_alpha_scale": False,
        "alpha_mass_reg": float(args.alpha_mass_reg),
        "drift_reg": float(args.drift_reg),
        "constrain_alpha_mass": bool(args.constrain_alpha_mass),
        "train_shared": bool(args.train_shared),
        "encoder_linear_out": bool(args.encoder_linear_out),
        "distill_observation_skip": bool(args.distill_observation_skip),
        "distill_extra_steps": int(args.distill_extra_steps),
        "collect_cosine_buffers": False,
        "max_distill_buffer": int(args.max_distill_buffer),
        "similarity_samples": int(args.similarity_samples),
        "distill_max_samples": int(args.distill_max_samples),
        "distill_epochs": int(args.distill_epochs),
        "distill_lr": float(args.distill_lr),
        "distill_batch_size": int(args.distill_batch_size),
        "distill_test_frac": float(args.distill_test_frac),
        "distill_select_best_val": bool(args.distill_select_best_val),
    }


def checkpoint_matches(path, suite, task_id, total_timesteps, seed, args):
    expected = _expected_training_config(suite, task_id, total_timesteps, seed, args)
    if not checkpoint_complete(path):
        return False, "checkpoint files or valid run_manifest.json are missing"
    return identity_checkpoint_matches(
        path, expected, pretrained_encoder=args.pretrained_encoder
    )


def train_one_baseline(suite, task_id, total_timesteps, seed, args):
    run_dir = scratch_checkpoint_dir(args.save_root, suite, task_id, total_timesteps, seed)
    event_dir = scratch_event_dir(args.runs_root, suite, task_id, total_timesteps, seed)
    analysis_dir = scratch_analysis_dir(args.analysis_root, suite, task_id, total_timesteps, seed)
    expected = _expected_training_config(suite, task_id, total_timesteps, seed, args)
    if checkpoint_complete(run_dir) and not args.force_retrain:
        matches, reason = identity_checkpoint_matches(
            run_dir, expected, pretrained_encoder=args.pretrained_encoder
        )
        if matches:
            print(f"[scratch] {suite}/task_{task_id}/seed_{seed} already complete: {run_dir}")
            return run_dir
        print(f"[scratch] stale checkpoint ({reason}); retraining: {run_dir}")

    # A forced retrain OR a retry after a partial checkpoint must start with
    # clean TensorBoard/analysis directories. Otherwise EventAccumulator can
    # silently combine scalars from multiple attempts and corrupt AUC/FWT.
    for path in (run_dir, event_dir, analysis_dir):
        if path.exists():
            shutil.rmtree(path)

    cmd = [
        sys.executable, "run_sac.py",
        "--model-type=cka-rl",
        f"--task-suite={suite}",
        f"--task-id={task_id}",
        "--seq-idx=0",
        f"--seed={seed}",
        f"--tag={scratch_tag(suite, task_id, total_timesteps, seed)}",
        f"--save-dir={run_dir.parent}",
        f"--runs-root={args.runs_root}",
        f"--analysis-root={args.analysis_root}",
        f"--total-timesteps={total_timesteps}",
        f"--learning-starts={args.learning_starts}",
        f"--random-actions-end={args.random_actions_end}",
        f"--batch-size={args.batch_size}",
        f"--policy-lr={args.policy_lr}",
        f"--alpha-lr={args.alpha_lr}",
        f"--alpha-mass-reg={args.alpha_mass_reg}",
        f"--alpha-warmup-steps={args.alpha_warmup_steps}",
        f"--drift-reg={args.drift_reg}",
        f"--q-lr={args.q_lr}",
        f"--gamma={args.gamma}",
        f"--tau={args.tau}",
        f"--alpha={args.alpha}",
        "--autotune" if args.autotune else "--no-autotune",
        "--autotune-init-from-alpha" if args.autotune_init_from_alpha else "--no-autotune-init-from-alpha",
        f"--pool-size={args.pool_size}",
        f"--eval-every={args.eval_every}",
        f"--num-evals={args.num_evals}",
        "--distill-observation-skip" if args.distill_observation_skip else "--no-distill-observation-skip",
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
        "--no-fix-alpha-scale",
        "--no-distillation",
        "--no-use-alpha-mass",
        "--constrain-alpha-mass" if args.constrain_alpha_mass else "--no-constrain-alpha-mass",
        "--distill-select-best-val" if args.distill_select_best_val else "--no-distill-select-best-val",
        "--no-collect-cosine-buffers",
        "--train-shared" if args.train_shared else "--no-train-shared",
        "--freeze-root-encoder" if args.freeze_root_encoder else "--no-freeze-root-encoder",
        "--encoder-from-base" if args.encoder_from_base else "--no-encoder-from-base",
        # The encoder configuration MUST match the continual runs these baselines
        # are the denominator for. Forward transfer compares a continual run's AUC
        # against this run's AUC; if the continual runs get a TD-JEPA pretrained
        # encoder and these do not, FT stops measuring the continual mechanism and
        # starts measuring the encoder instead.
        "--encoder-linear-out" if args.encoder_linear_out else "--no-encoder-linear-out",
    ]
    if args.pretrained_encoder:
        cmd.append(f"--pretrained-encoder={args.pretrained_encoder}")
    if args.cpu:
        cmd.append("--no-cuda")

    print(
        f"\n>>> [scratch] {suite} / task {task_id} ({get_task_name(task_id, suite)}) "
        f"/ seed {seed} / {total_timesteps} steps <<<"
    )
    subprocess.run(cmd, check=True)
    if not checkpoint_complete(run_dir):
        raise RuntimeError(f"scratch baseline finished but checkpoint is incomplete: {run_dir}")
    matches, reason = identity_checkpoint_matches(
        run_dir, expected, pretrained_encoder=args.pretrained_encoder
    )
    if not matches:
        raise RuntimeError(f"scratch baseline has unexpected identity: {reason}")
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
    p.add_argument("--alpha-lr", type=float, default=5e-3)
    p.add_argument("--alpha-mass-reg", type=float, default=0.05)
    p.add_argument("--alpha-warmup-steps", type=int, default=5_000)
    p.add_argument("--drift-reg", type=float, default=1.0)
    p.add_argument("--q-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--autotune-init-from-alpha", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--num-evals", type=int, default=5)
    p.add_argument("--distill-observation-skip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--distill-extra-steps", type=int, default=10_000)
    p.add_argument("--max-distill-buffer", type=int, default=50_000)
    p.add_argument("--similarity-samples", type=int, default=2_048)
    p.add_argument("--distill-max-samples", type=int, default=20_000)
    p.add_argument("--distill-epochs", type=int, default=16)
    p.add_argument("--distill-lr", type=float, default=5e-4)
    p.add_argument("--distill-batch-size", type=int, default=256)
    p.add_argument("--distill-test-frac", type=float, default=0.2)
    p.add_argument("--distill-select-best-val", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--analysis-log-every", type=int, default=5_000)
    p.add_argument("--save-root", default=SCRATCH_SAVE_ROOT)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--analysis-root", default="analysis_runs_scratch")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--pretrained-encoder", default=None,
                   help="Must match the continual runs these baselines are the "
                        "denominator for, or forward transfer is meaningless.")
    p.add_argument("--train-shared", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--freeze-root-encoder", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--encoder-from-base", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--encoder-linear-out", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--constrain-alpha-mass", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.train_shared and args.freeze_root_encoder:
        p.error("--train-shared and --freeze-root-encoder are contradictory")
    if args.autotune_init_from_alpha and args.alpha <= 0:
        p.error("--alpha must be > 0 with --autotune-init-from-alpha")
    return args


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
