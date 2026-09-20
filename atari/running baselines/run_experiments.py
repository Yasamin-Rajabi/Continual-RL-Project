"""Run one legacy Atari continual baseline on an explicit task sequence.

Unlike the old runner, this version is repeat-safe: sequence occurrence
(``seq_idx``) and semantic Atari mode are separate, so a sequence such as
0,1,2,0,1 does not overwrite checkpoints or truncate previous-unit lists.
"""
from __future__ import annotations
from task_utils import TASKS

import argparse
import pathlib
import shutil
import subprocess
import sys

from benchmark_protocol import (
    ALL_PREVIOUS_METHODS,
    LATEST_ONLY_METHODS,
    METHODS,
    canonical_env_name,
    canonical_method,
    checkpoint_dir,
    env_id,
    event_dir,
    load_success_thresholds,
    success_threshold,
    task_slot_map,
    unique_in_order,
)


def _bool_flag(name: str, value: bool) -> str:
    return f"--{name}" if value else f"--no-{name}"


def _run_one(args, env, method, seed):
    method = canonical_method(method)
    sequence = [int(x) for x in args.task_sequence]
    slots = task_slot_map(sequence)
    total_task_num = len(slots)
    previous_dirs = []
    latest_by_task = {}

    for seq_idx, task_id in enumerate(sequence):
        run_dir = checkpoint_dir(
            args.save_root, env, args.tag, method, seed, seq_idx, task_id
        )
        log_dir = event_dir(
            args.runs_root, env, args.tag, method, seed, seq_idx, task_id
        )

        if args.force:
            shutil.rmtree(run_dir, ignore_errors=True)
            shutil.rmtree(log_dir, ignore_errors=True)

        # A completed checkpoint is enough for resumable orchestration here;
        # metrics.py performs its own structural checks before evaluation.
        expected_file = run_dir / ("packnet.pt" if method == "PackNet" else "actor.pt")
        if expected_file.exists() and not args.force:
            print(f"[skip] {env}/{method}/seed={seed}/seq={seq_idx}: {run_dir}")
            previous_dirs.append(run_dir)
            latest_by_task[task_id] = run_dir
            continue

        seen_before = task_id in latest_by_task
        if method in ALL_PREVIOUS_METHODS:
            prev_units = list(previous_dirs)
        elif method in LATEST_ONLY_METHODS:
            prev_units = [] if not previous_dirs else [previous_dirs[-1]]
        else:
            prev_units = []

        cmd = [
            sys.executable,
            args.trainer,
            f"--method-type={method}",
            f"--env-id={env_id(env)}",
            f"--mode={task_id}",
            f"--task-id={task_id}",
            f"--seq-idx={seq_idx}",
            f"--task-slot={slots[task_id]}",
            _bool_flag("task-seen-before", seen_before),
            f"--seed={seed}",
            f"--save-dir={run_dir}",
            f"--event-dir={log_dir}",
            f"--total-timesteps={args.total_timesteps}",
            f"--learning-rate={args.learning_rate}",
            f"--num-envs={args.num_envs}",
            f"--num-steps={args.num_steps}",
            _bool_flag("anneal-lr", args.anneal_lr),
            f"--gamma={args.gamma}",
            f"--gae-lambda={args.gae_lambda}",
            f"--num-minibatches={args.num_minibatches}",
            f"--update-epochs={args.update_epochs}",
            _bool_flag("norm-adv", args.norm_adv),
            f"--clip-coef={args.clip_coef}",
            _bool_flag("clip-vloss", args.clip_vloss),
            f"--ent-coef={args.ent_coef}",
            f"--vf-coef={args.vf_coef}",
            f"--max-grad-norm={args.max_grad_norm}",
            f"--eval-every={args.eval_every}",
            f"--num-evals={args.num_evals}",
            f"--eval-action-mode={args.eval_action_mode}",
            f"--total-task-num={total_task_num}",
            f"--num-tasks-learned={len(latest_by_task)}",
            _bool_flag("torch-deterministic", args.torch_deterministic),
            _bool_flag("cuda", not args.cpu),
            _bool_flag("componet-finetune-encoder", args.componet_finetune_encoder),
            f"--alpha-factor={args.alpha_factor}",
            f"--alpha-learning-rate={args.alpha_learning_rate}",
            f"--delta-theta-mode={args.delta_theta_mode}",
            _bool_flag("fuse-encoder", args.fuse_encoder),
            _bool_flag("fuse-actor", args.fuse_actor),
            _bool_flag("reset-actor", args.reset_actor),
            _bool_flag("global-alpha", args.global_alpha),
            f"--alpha-init={args.alpha_init}",
            f"--alpha-major={args.alpha_major}",
            f"--pool-size={args.pool_size}",
        ]

        if args.target_kl is not None:
            cmd.append(f"--target-kl={args.target_kl}")
        threshold = success_threshold(args.success_thresholds, env, task_id)
        if threshold is not None:
            cmd.append(f"--success-threshold={threshold}")
        if prev_units:
            cmd.append("--prev-units")
            cmd.extend(str(path) for path in prev_units)
        if method == "PackNet" and seen_before:
            cmd.append(f"--task-head-dir={latest_by_task[task_id]}")

        print(
            f"\n>>> {env} | {method} | seed {seed} | "
            f"seq {seq_idx}/{len(sequence)-1} | mode {task_id} "
            f"{'(revisit)' if seen_before else '(first encounter)'} <<<"
        )
        subprocess.run(cmd, check=True)
        previous_dirs.append(run_dir)
        latest_by_task[task_id] = run_dir


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--env", default="Freeway", choices=["Freeway", "SpaceInvaders"])
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--task-sequence", nargs="+", type=int, default=None)
    p.add_argument("--tag", default="main")
    p.add_argument("--trainer", default="run_ppo.py")
    p.add_argument("--save-root", default="agents")
    p.add_argument("--runs-root", default="runs")

    # Shared PPO settings: same tuned defaults as current CKA-RL Atari.
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--num-evals", type=int, default=5)
    p.add_argument("--eval-action-mode", choices=["deterministic", "stochastic"], default="deterministic")
    p.add_argument("--success-thresholds-json", default=None)
    p.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cpu", action="store_true")

    # Method-specific knobs retained from the old runner.
    p.add_argument("--componet-finetune-encoder", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--alpha-factor", type=float, default=1e-2)
    p.add_argument("--alpha-learning-rate", type=float, default=2.5e-4)
    p.add_argument("--delta-theta-mode", choices=["T", "TAT"], default="T")
    p.add_argument("--fuse-encoder", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fuse-actor", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reset-actor", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--global-alpha", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--alpha-init", choices=["Randn", "Major", "Uniform"], default="Randn")
    p.add_argument("--alpha-major", type=float, default=0.6)
    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.success_thresholds = load_success_thresholds(args.success_thresholds_json)
    
    env = canonical_env_name(args.env)

    if args.task_sequence is None:
        full_id = env_id(env)  
        args.task_sequence = TASKS.get(full_id)
        if args.task_sequence is None:
            p.error(f"No default task sequence found for environment {env}")

    max_mode = 7 if env == "Freeway" else 9
    bad = [task for task in args.task_sequence if not 0 <= int(task) <= max_mode]
    if bad:
        p.error(f"invalid {env} modes in task sequence: {bad}")
    if args.total_timesteps % args.num_envs != 0:
        p.error("total-timesteps must be divisible by num-envs")
    return args


def main():
    args = parse_args()
    env = canonical_env_name(args.env)
    for seed in args.seeds:
        _run_one(args, env, args.method, int(seed))


if __name__ == "__main__":
    main()
