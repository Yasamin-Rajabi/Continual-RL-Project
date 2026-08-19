"""Four-way continual benchmark for HalfCheetahVel and HalfCheetahWindVel.

The four experimental cases intentionally differ only along two method axes:

    baseline      = classic CKA vectors + arithmetic merge
    distil_only  = classic CKA vectors + KL distillation merge
    weight_only   = weight-delta vectors + alpha-mass + arithmetic merge
    combined      = weight-delta vectors + alpha-mass + KL distillation merge

All four cases use the SAME output-space merge-pair selector: the pair with the
lowest symmetric KL between their full Gaussian policy outputs (mean + log-std)
on a balanced subset of states stored in the knowledge pool. This keeps the
pair-selection rule controlled while isolating what the four cases are meant to
compare.

Besides training curves, this script evaluates every saved checkpoint on all
unique tasks to produce retention matrices, forgetting curves, zero-shot
transfer plots, final per-task results, and merge/distillation diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shutil
import subprocess
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator

from cka_rl import FrozenCkaPolicy
from tasks import DEFAULT_CONTINUAL_SEQUENCE, TASK_SUITES, get_task, get_task_name


CONDITIONS = OrderedDict([
    (
        "baseline",
        {"fusion_mode": "classic_cka", "distillation": False, "use_alpha_mass": False},
    ),
    (
        "distil_only",
        {"fusion_mode": "classic_cka", "distillation": True, "use_alpha_mass": False},
    ),
    (
        "weight_only",
        {"fusion_mode": "weight_delta", "distillation": False, "use_alpha_mass": True},
    ),
    (
        "combined",
        {"fusion_mode": "weight_delta", "distillation": True, "use_alpha_mass": True},
    ),
])

TRAIN_METRICS = OrderedDict([
    ("charts/episodic_return", ("Training episodic return", "train_return")),
    ("charts/test_episodic_return", ("Evaluation return", "eval_return")),
    ("charts/test_velocity_error", ("Evaluation velocity error", "eval_velocity_error")),
    ("charts/test_success", ("Evaluation success fraction", "eval_success")),
    ("losses/actor_loss", ("SAC actor loss", "actor_loss")),
    ("losses/qf_loss", ("SAC critic loss", "critic_loss")),
    ("losses/alpha", ("SAC entropy coefficient", "entropy_alpha")),
    ("analysis/theta/drift_from_task_start_l2", ("Actor drift from task start", "theta_drift")),
    ("analysis/mean/alpha_entropy", ("Historical-mixture alpha entropy", "alpha_entropy")),
    ("analysis/mean/alpha_mass", ("Historical-mixture alpha mass", "alpha_mass")),
])

SEQUENCE_METRICS = OrderedDict([
    ("analysis/merge/symmetric_kl", ("Selected merge-pair symmetric KL", "merge_selected_skl")),
    ("analysis/merge/pairwise_kl_mean", ("Mean pairwise symmetric KL in pool", "merge_pool_mean_skl")),
    ("analysis/merge/pairwise_kl_max", ("Maximum pairwise symmetric KL in pool", "merge_pool_max_skl")),
    ("distillation/policy/distill_train_kl", ("Distillation train KL", "distill_train_kl")),
    ("distillation/policy/distill_test_kl", ("Distillation held-out KL", "distill_test_kl")),
    ("distillation/policy/distill_train_mean_mse", ("Distillation train mean MSE", "distill_train_mean_mse")),
    ("distillation/policy/distill_test_mean_mse", ("Distillation held-out mean MSE", "distill_test_mean_mse")),
    ("distillation/policy/distill_train_logstd_mse", ("Distillation train log-std MSE", "distill_train_logstd_mse")),
    ("distillation/policy/distill_test_logstd_mse", ("Distillation held-out log-std MSE", "distill_test_logstd_mse")),
    ("timing/train_loop_seconds", ("Task training loop time (s)", "train_loop_seconds")),
    ("timing/merge_buffer_seconds", ("Merge-buffer collection time (s)", "merge_buffer_seconds")),
    ("timing/finalize_seconds", ("Finalize / merge time (s)", "finalize_seconds")),
    ("analysis/pool/final_length", ("Final knowledge-pool length", "pool_length")),
    ("analysis/buffer/mean_velocity_error", ("Merge-buffer mean velocity error", "buffer_velocity_error")),
])


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--task-suites", nargs="+",
        default=["halfcheetah_vel", "halfcheetah_wind_vel"],
        choices=sorted(TASK_SUITES.keys()),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--task-sequence", nargs="+", type=int, default=list(DEFAULT_CONTINUAL_SEQUENCE))
    p.add_argument("--total-timesteps", type=int, default=300_000)
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
    p.add_argument("--retention-eval-episodes", type=int, default=3)
    p.add_argument("--distill-extra-steps", type=int, default=10_000)
    p.add_argument("--max-distill-buffer", type=int, default=50_000)
    p.add_argument("--similarity-samples", type=int, default=2_048)
    p.add_argument("--distill-max-samples", type=int, default=20_000)
    p.add_argument("--distill-epochs", type=int, default=8)
    p.add_argument("--distill-lr", type=float, default=3e-4)
    p.add_argument("--distill-batch-size", type=int, default=256)
    p.add_argument("--distill-test-frac", type=float, default=0.2)
    p.add_argument("--analysis-log-every", type=int, default=5_000)
    p.add_argument("--save-root", default="agents_halfcheetah")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--plots-root", default="plots_halfcheetah_continual")
    p.add_argument("--analysis-root", default="analysis_runs")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-retention", action="store_true")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--quick-test", action="store_true")
    args = p.parse_args()

    if args.quick_test:
        # Exercises at least one merge without committing to the full paper run.
        args.task_suites = ["halfcheetah_vel"]
        args.seeds = [1]
        args.task_sequence = [0, 1, 2, 3]
        args.total_timesteps = 20_000
        args.learning_starts = 1_000
        args.random_actions_end = 2_000
        args.pool_size = 2
        args.eval_every = 5_000
        args.num_evals = 1
        args.retention_eval_episodes = 1
        args.distill_extra_steps = 1_000
        args.max_distill_buffer = 4_000
        args.similarity_samples = 256
        args.distill_max_samples = 1_000
        args.distill_epochs = 2
        args.analysis_log_every = 2_000

    return args


def run_name(suite, task_id, seed):
    return f"{suite}__task_{task_id}__cka-rl__run_sac__{seed}"


def event_dir(args, suite, condition, seed, seq_idx, task_id):
    tag = f"{suite}/{condition}/seed_{seed}/seq_{seq_idx}"
    return pathlib.Path(args.runs_root) / tag / run_name(suite, task_id, seed)


def checkpoint_dir(args, suite, condition, seed, seq_idx, task_id):
    return (
        pathlib.Path(args.save_root)
        / suite
        / condition
        / f"seed_{seed}"
        / f"seq_{seq_idx}"
        / run_name(suite, task_id, seed)
    )


def analysis_snapshot_path(args, suite, condition, seed, seq_idx, task_id):
    tag = pathlib.Path(suite) / condition / f"seed_{seed}" / f"seq_{seq_idx}"
    return pathlib.Path(args.analysis_root) / tag / run_name(suite, task_id, seed) / "post_finalize.pt"


def checkpoint_complete(path):
    required = ["policy_snapshot.pt", "fc.pt", "mean_pool.pt", "logstd_pool.pt"]
    return path.exists() and all((path / name).exists() for name in required)


def train_chain(args, suite, condition, cfg, seed):
    previous = []
    for seq_idx, task_id in enumerate(args.task_sequence):
        if task_id < 0 or task_id >= len(TASK_SUITES[suite]):
            raise ValueError(f"task_id {task_id} is invalid for {suite}")

        save_parent = checkpoint_dir(args, suite, condition, seed, seq_idx, task_id).parent
        run_dir = checkpoint_dir(args, suite, condition, seed, seq_idx, task_id)
        tb_dir = event_dir(args, suite, condition, seed, seq_idx, task_id)
        if args.force_retrain:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            if tb_dir.exists():
                shutil.rmtree(tb_dir)

        if checkpoint_complete(run_dir):
            print(f"[{suite}/{condition}/seed={seed}] seq{seq_idx} already complete: {run_dir}")
            previous.append(run_dir)
            continue

        if args.skip_training:
            raise FileNotFoundError(f"Missing checkpoint while --skip-training was set: {run_dir}")

        # Remove partial outputs before a retry, otherwise TensorBoard can mix
        # stale and fresh event files from two different attempts.
        if run_dir.exists():
            shutil.rmtree(run_dir)
        if tb_dir.exists():
            shutil.rmtree(tb_dir)

        tag = f"{suite}/{condition}/seed_{seed}/seq_{seq_idx}"
        cmd = [
            "python3", "run_sac.py",
            "--model-type=cka-rl",
            f"--task-suite={suite}",
            f"--task-id={task_id}",
            f"--seed={seed}",
            f"--tag={tag}",
            f"--save-dir={save_parent}",
            f"--analysis-root={args.analysis_root}",
            f"--total-timesteps={args.total_timesteps}",
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
            f"--fusion-mode={cfg['fusion_mode']}",
            "--no-use-alpha-scale",
            "--distillation" if cfg["distillation"] else "--no-distillation",
            "--use-alpha-mass" if cfg["use_alpha_mass"] else "--no-use-alpha-mass",
        ]
        if args.cpu:
            cmd.append("--no-cuda")

        if previous:
            # run_sac only needs the immutable root and latest continual state.
            prev_args = [previous[0]] if len(previous) == 1 else [previous[0], previous[-1]]
            cmd.append("--prev-units")
            cmd.extend(str(p) for p in prev_args)

        print(
            f"\n>>> {suite} | {condition} | seed {seed} | seq{seq_idx} "
            f"task {task_id}: {get_task_name(task_id, suite)} <<<"
        )
        subprocess.run(cmd, check=True)
        if not checkpoint_complete(run_dir):
            raise RuntimeError(f"Training command finished but checkpoint is incomplete: {run_dir}")
        previous.append(run_dir)
    return previous


def load_scalar(directory, scalar_tag):
    directory = pathlib.Path(directory)
    if not directory.exists():
        return np.empty(0), np.empty(0)
    try:
        ea = event_accumulator.EventAccumulator(
            str(directory), size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
    except Exception:
        return np.empty(0), np.empty(0)
    if scalar_tag not in ea.Tags().get("scalars", []):
        return np.empty(0), np.empty(0)
    events = ea.Scalars(scalar_tag)
    return (
        np.asarray([e.step for e in events], dtype=np.float64),
        np.asarray([e.value for e in events], dtype=np.float64),
    )


def load_continual_scalar(args, suite, condition, seed, scalar_tag):
    xs, ys = [], []
    for seq_idx, task_id in enumerate(args.task_sequence):
        x, y = load_scalar(event_dir(args, suite, condition, seed, seq_idx, task_id), scalar_tag)
        if x.size:
            xs.append(x + seq_idx * (args.total_timesteps + 1))
            ys.append(y)
    if not xs:
        return np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys)


def aggregate_curves(curves, grid_points=1200):
    curves = [(x, y) for x, y in curves if len(x) >= 2]
    if not curves:
        return None
    start = max(float(np.min(x)) for x, _ in curves)
    end = min(float(np.max(x)) for x, _ in curves)
    if end <= start:
        return None
    grid = np.linspace(start, end, grid_points)
    interpolated = []
    for x, y in curves:
        order = np.argsort(x)
        x_sorted, y_sorted = x[order], y[order]
        unique_x, unique_idx = np.unique(x_sorted, return_index=True)
        unique_y = y_sorted[unique_idx]
        interpolated.append(np.interp(grid, unique_x, unique_y))
    arr = np.asarray(interpolated)
    return grid, np.mean(arr, axis=0), np.std(arr, axis=0)


def add_task_boundaries(ax, args):
    for seq_idx in range(1, len(args.task_sequence)):
        ax.axvline(seq_idx * (args.total_timesteps + 1), linewidth=0.5, alpha=0.25)


def plot_training_metrics(args, suite):
    out_dir = pathlib.Path(args.plots_root) / suite / "training_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scalar_tag, (title, filename) in TRAIN_METRICS.items():
        fig, ax = plt.subplots(figsize=(11, 5.5))
        plotted = False
        for condition in CONDITIONS:
            curves = [
                load_continual_scalar(args, suite, condition, seed, scalar_tag)
                for seed in args.seeds
            ]
            agg = aggregate_curves(curves)
            if agg is None:
                continue
            plotted = True
            x, mean, std = agg
            line, = ax.plot(x, mean, label=condition, linewidth=1.7)
            if len(args.seeds) > 1:
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=line.get_color())
        if not plotted:
            plt.close(fig)
            continue
        add_task_boundaries(ax, args)
        ax.set_title(f"{suite}: {title}")
        ax.set_xlabel("Continual environment steps")
        ax.set_ylabel(title)
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / f"{filename}.png", dpi=180)
        plt.close(fig)


def final_scalar(args, suite, condition, seed, seq_idx, task_id, scalar_tag):
    _, values = load_scalar(event_dir(args, suite, condition, seed, seq_idx, task_id), scalar_tag)
    return float(values[-1]) if len(values) else np.nan


def plot_sequence_diagnostics(args, suite):
    out_dir = pathlib.Path(args.plots_root) / suite / "sequence_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(args.task_sequence))
    for scalar_tag, (title, filename) in SEQUENCE_METRICS.items():
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        plotted = False
        for condition in CONDITIONS:
            seed_rows = []
            for seed in args.seeds:
                row = [
                    final_scalar(args, suite, condition, seed, i, task_id, scalar_tag)
                    for i, task_id in enumerate(args.task_sequence)
                ]
                seed_rows.append(row)
            arr = np.asarray(seed_rows, dtype=np.float64)
            if not np.isfinite(arr).any():
                continue
            plotted = True
            valid = np.sum(np.isfinite(arr), axis=0)
            summed = np.nansum(arr, axis=0)
            mean = np.divide(summed, valid, out=np.full(len(x), np.nan), where=valid > 0)
            centered = arr - mean[None, :]
            centered[~np.isfinite(arr)] = np.nan
            std = np.sqrt(
                np.divide(
                    np.nansum(centered ** 2, axis=0),
                    valid,
                    out=np.full(len(x), np.nan),
                    where=valid > 0,
                )
            )
            line, = ax.plot(x, mean, marker="o", markersize=3, label=condition)
            if len(args.seeds) > 1:
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=line.get_color())
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"{suite}: {title}")
        ax.set_xlabel("Task position in continual sequence")
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in args.task_sequence])
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{filename}.png", dpi=180)
        plt.close(fig)


def evaluate_checkpoint(run_dir, suite, task_id, episodes, seed, device):
    env = get_task(task_id, task_suite=suite)
    policy = FrozenCkaPolicy.load(str(run_dir), map_location=device).to(device)
    policy.eval()
    returns, success, velocity_errors = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 * task_id + ep)
        ep_return = 0.0
        ep_success, ep_error = [], []
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = policy(obs_t)
            mid = (env.action_space.high + env.action_space.low) / 2.0
            scale = (env.action_space.high - env.action_space.low) / 2.0
            action = np.tanh(mean[0].cpu().numpy()) * scale + mid
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_success.append(float(info.get("success", np.nan)))
            ep_error.append(float(info.get("velocity_error", np.nan)))
            if terminated or truncated:
                break
        returns.append(ep_return)
        success.append(float(np.nanmean(ep_success)))
        velocity_errors.append(float(np.nanmean(ep_error)))
    env.close()
    return {
        "return": float(np.mean(returns)),
        "success": float(np.nanmean(success)),
        "velocity_error": float(np.nanmean(velocity_errors)),
    }


def retention_cache_path(args, suite, condition, seed):
    return pathlib.Path(args.plots_root) / suite / "retention_data" / f"{condition}_seed_{seed}.json"


def build_retention_matrix(args, suite, condition, seed, device):
    cache = retention_cache_path(args, suite, condition, seed)
    eval_task_ids = sorted(set(args.task_sequence))
    if cache.exists() and not args.force_retrain:
        with open(cache) as f:
            cached = json.load(f)
        if (
            cached.get("suite") == suite
            and cached.get("condition") == condition
            and int(cached.get("seed", -1)) == int(seed)
            and cached.get("sequence") == list(args.task_sequence)
            and cached.get("eval_task_ids") == eval_task_ids
            and int(cached.get("episodes", -1)) == int(args.retention_eval_episodes)
        ):
            return cached

    data = {
        "suite": suite,
        "condition": condition,
        "seed": seed,
        "sequence": list(args.task_sequence),
        "eval_task_ids": eval_task_ids,
        "episodes": int(args.retention_eval_episodes),
        "return": [],
        "success": [],
        "velocity_error": [],
    }
    for seq_idx, trained_task in enumerate(args.task_sequence):
        run_dir = checkpoint_dir(args, suite, condition, seed, seq_idx, trained_task)
        if not checkpoint_complete(run_dir):
            raise FileNotFoundError(run_dir)
        rows = {metric: [] for metric in ("return", "success", "velocity_error")}
        for eval_task in eval_task_ids:
            metrics = evaluate_checkpoint(
                run_dir, suite, eval_task, args.retention_eval_episodes,
                seed + seq_idx * 100_000, device,
            )
            for metric in rows:
                rows[metric].append(metrics[metric])
        for metric in rows:
            data[metric].append(rows[metric])
        print(
            f"retention {suite}/{condition}/seed={seed}: after seq{seq_idx} "
            f"task {trained_task} done"
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(data, f, indent=2)
    return data


def aggregate_retention(seed_payloads, metric):
    arr = np.asarray([payload[metric] for payload in seed_payloads], dtype=np.float64)
    return np.mean(arr, axis=0), np.std(arr, axis=0), arr


def plot_heatmap(matrix, xlabels, ylabels, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    im = ax.imshow(matrix, aspect="auto")
    fig.colorbar(im, ax=ax, label=ylabel)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Checkpoint after sequence position")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def seen_task_indices(sequence, eval_task_ids, stage):
    seen = set(sequence[:stage + 1])
    return [i for i, task_id in enumerate(eval_task_ids) if task_id in seen]


def continual_summary_curves(payload, metric):
    matrix = np.asarray(payload[metric], dtype=np.float64)
    seq = payload["sequence"]
    eval_ids = payload["eval_task_ids"]
    avg_seen, forgetting = [], []
    for stage in range(len(seq)):
        idx = seen_task_indices(seq, eval_ids, stage)
        avg_seen.append(float(np.mean(matrix[stage, idx])))
        fvals = []
        for j in idx:
            task_id = eval_ids[j]
            first_trained = seq.index(task_id)
            best = float(np.max(matrix[first_trained:stage + 1, j]))
            fvals.append(best - float(matrix[stage, j]))
        forgetting.append(float(np.mean(fvals)))
    return np.asarray(avg_seen), np.asarray(forgetting)


def plot_retention(args, suite, all_payloads):
    out_dir = pathlib.Path(args.plots_root) / suite / "retention_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_task_ids = sorted(set(args.task_sequence))
    xlabels = [get_task_name(t, suite) for t in eval_task_ids]
    ylabels = [f"{i}:T{task}" for i, task in enumerate(args.task_sequence)]

    for condition in CONDITIONS:
        payloads = all_payloads[condition]
        for metric in ("return", "velocity_error", "success"):
            mean, _, _ = aggregate_retention(payloads, metric)
            plot_heatmap(
                mean, xlabels, ylabels,
                f"{suite} / {condition}: checkpoint retention ({metric})",
                metric,
                out_dir / f"heatmap_{condition}_{metric}.png",
            )

    # Average performance on all tasks seen so far.
    for metric, ylabel in (
        ("return", "Average return on seen tasks"),
        ("velocity_error", "Average velocity error on seen tasks"),
        ("success", "Average success fraction on seen tasks"),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 5.3))
        for condition in CONDITIONS:
            curves = [continual_summary_curves(p, metric)[0] for p in all_payloads[condition]]
            arr = np.asarray(curves)
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            x = np.arange(len(mean))
            line, = ax.plot(x, mean, marker="o", markersize=3, label=condition)
            if len(args.seeds) > 1:
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=line.get_color())
        ax.set_title(f"{suite}: retention on tasks seen so far")
        ax.set_xlabel("Task position in continual sequence")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"avg_seen_{metric}.png", dpi=180)
        plt.close(fig)

    # Standard max-previous minus current forgetting, using return (higher is better).
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    for condition in CONDITIONS:
        curves = [continual_summary_curves(p, "return")[1] for p in all_payloads[condition]]
        arr = np.asarray(curves)
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        x = np.arange(len(mean))
        line, = ax.plot(x, mean, marker="o", markersize=3, label=condition)
        if len(args.seeds) > 1:
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=line.get_color())
    ax.set_title(f"{suite}: average forgetting (max previous return - current return)")
    ax.set_xlabel("Task position in continual sequence")
    ax.set_ylabel("Average forgetting")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "average_forgetting.png", dpi=180)
    plt.close(fig)

    # Final checkpoint per-task comparison.
    for metric in ("return", "velocity_error", "success"):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(eval_task_ids), dtype=np.float64)
        width = 0.8 / len(CONDITIONS)
        for ci, condition in enumerate(CONDITIONS):
            payloads = all_payloads[condition]
            finals = np.asarray([np.asarray(p[metric])[-1] for p in payloads])
            mean, std = finals.mean(axis=0), finals.std(axis=0)
            pos = x - 0.4 + width / 2.0 + ci * width
            ax.bar(pos, mean, width=width, yerr=std if len(args.seeds) > 1 else None, label=condition)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=35, ha="right")
        ax.set_title(f"{suite}: final checkpoint per-task {metric}")
        ax.set_ylabel(metric)
        ax.legend(loc="best")
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / f"final_per_task_{metric}.png", dpi=180)
        plt.close(fig)

    # Compact final trade-off plots make it easy to see whether a method gets
    # higher return only by forgetting more, or improves both axes.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for condition in CONDITIONS:
        final_returns, final_forgetting = [], []
        for payload in all_payloads[condition]:
            ret = np.asarray(payload["return"], dtype=np.float64)
            _, forgetting = continual_summary_curves(payload, "return")
            final_returns.append(float(ret[-1].mean()))
            final_forgetting.append(float(forgetting[-1]))
        x = np.asarray(final_forgetting)
        y = np.asarray(final_returns)
        ax.errorbar(
            x.mean(), y.mean(),
            xerr=x.std() if len(x) > 1 else None,
            yerr=y.std() if len(y) > 1 else None,
            marker="o", capsize=3, label=condition,
        )
    ax.set_title(f"{suite}: final return / forgetting trade-off")
    ax.set_xlabel("Average forgetting (lower is better)")
    ax.set_ylabel("Final average return (higher is better)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_return_vs_forgetting.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for condition in CONDITIONS:
        final_returns = np.asarray([
            np.asarray(p["return"], dtype=np.float64)[-1].mean()
            for p in all_payloads[condition]
        ])
        final_errors = np.asarray([
            np.asarray(p["velocity_error"], dtype=np.float64)[-1].mean()
            for p in all_payloads[condition]
        ])
        ax.errorbar(
            final_errors.mean(), final_returns.mean(),
            xerr=final_errors.std() if len(final_errors) > 1 else None,
            yerr=final_returns.std() if len(final_returns) > 1 else None,
            marker="o", capsize=3, label=condition,
        )
    ax.set_title(f"{suite}: final return / tracking-error trade-off")
    ax.set_xlabel("Final average velocity error (lower is better)")
    ax.set_ylabel("Final average return (higher is better)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_return_vs_velocity_error.png", dpi=180)
    plt.close(fig)


def plot_merge_lineage(args, suite):
    """Plot which original task buffers are represented in each selected merge."""
    out_dir = pathlib.Path(args.plots_root) / suite / "sequence_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_task_ids = sorted(set(args.task_sequence))
    xlabels = [get_task_name(t, suite) for t in eval_task_ids]
    ylabels = [f"{i}:T{task}" for i, task in enumerate(args.task_sequence)]

    for condition in CONDITIONS:
        seed_matrices = []
        for seed in args.seeds:
            matrix = np.full((len(args.task_sequence), len(eval_task_ids)), np.nan, dtype=np.float64)
            for seq_idx, task_id in enumerate(args.task_sequence):
                path = analysis_snapshot_path(args, suite, condition, seed, seq_idx, task_id)
                if not path.exists():
                    continue
                try:
                    snap = torch.load(path, map_location="cpu", weights_only=False)
                    info = snap["actor"]["mean_headpool"].get("last_merge_info")
                except Exception:
                    continue
                if not info or not info.get("merged_lineage"):
                    continue
                lineage = info["merged_lineage"]
                counts = np.asarray([float(lineage.get(str(t), 0.0)) for t in eval_task_ids])
                total = counts.sum()
                if total > 0:
                    matrix[seq_idx] = counts / total
            seed_matrices.append(matrix)

        stack = np.asarray(seed_matrices)
        valid = np.sum(np.isfinite(stack), axis=0)
        mean = np.divide(
            np.nansum(stack, axis=0), valid,
            out=np.full(stack.shape[1:], np.nan), where=valid > 0,
        )
        if not np.isfinite(mean).any():
            continue
        fig, ax = plt.subplots(figsize=(11.0, 7.0))
        im = ax.imshow(mean, aspect="auto", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, label="Fraction of selected merged buffer")
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("Original task id represented in selected merge")
        ax.set_ylabel("Continual sequence position")
        ax.set_title(f"{suite} / {condition}: selected-merge lineage composition")
        fig.tight_layout()
        fig.savefig(out_dir / f"merge_lineage_{condition}.png", dpi=180)
        plt.close(fig)


def plot_zero_shot(args, suite):
    out_dir = pathlib.Path(args.plots_root) / suite / "sequence_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scalar_tag, title, filename in (
        ("charts/test_episodic_return", "Zero-shot return before training each task", "zero_shot_return"),
        ("charts/test_velocity_error", "Zero-shot velocity error before training each task", "zero_shot_velocity_error"),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 5.3))
        x = np.arange(len(args.task_sequence))
        for condition in CONDITIONS:
            rows = []
            for seed in args.seeds:
                values = []
                for i, task_id in enumerate(args.task_sequence):
                    steps, vals = load_scalar(event_dir(args, suite, condition, seed, i, task_id), scalar_tag)
                    mask = steps == 0
                    values.append(float(vals[np.flatnonzero(mask)[0]]) if np.any(mask) else np.nan)
                rows.append(values)
            arr = np.asarray(rows, dtype=np.float64)
            if not np.isfinite(arr).any():
                continue
            mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
            line, = ax.plot(x, mean, marker="o", markersize=3, label=condition)
            if len(args.seeds) > 1:
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=line.get_color())
        ax.set_title(f"{suite}: {title}")
        ax.set_xlabel("Task position in continual sequence")
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in args.task_sequence])
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{filename}.png", dpi=180)
        plt.close(fig)


def write_summary_csv(args, suite, all_payloads):
    out_path = pathlib.Path(args.plots_root) / suite / "summary_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in CONDITIONS:
        for payload in all_payloads.get(condition, []):
            ret = np.asarray(payload["return"], dtype=np.float64)
            err = np.asarray(payload["velocity_error"], dtype=np.float64)
            succ = np.asarray(payload["success"], dtype=np.float64)
            avg_ret, forgetting = continual_summary_curves(payload, "return")
            avg_err, _ = continual_summary_curves(payload, "velocity_error")
            avg_succ, _ = continual_summary_curves(payload, "success")
            rows.append({
                "suite": suite,
                "condition": condition,
                "seed": payload["seed"],
                "final_avg_return_all_eval_tasks": float(ret[-1].mean()),
                "final_avg_velocity_error_all_eval_tasks": float(err[-1].mean()),
                "final_avg_success_all_eval_tasks": float(succ[-1].mean()),
                "final_avg_return_seen": float(avg_ret[-1]),
                "final_avg_velocity_error_seen": float(avg_err[-1]),
                "final_avg_success_seen": float(avg_succ[-1]),
                "final_average_forgetting": float(forgetting[-1]),
            })
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    pathlib.Path(args.plots_root).mkdir(parents=True, exist_ok=True)
    with open(pathlib.Path(args.plots_root) / "benchmark_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Evaluation device: {device}")
    print(f"Sequence: {args.task_sequence}")

    for suite in args.task_suites:
        print(f"\n================ {suite} ================")
        for condition, cfg in CONDITIONS.items():
            for seed in args.seeds:
                train_chain(args, suite, condition, cfg, seed)

        plot_training_metrics(args, suite)
        plot_sequence_diagnostics(args, suite)
        plot_merge_lineage(args, suite)
        plot_zero_shot(args, suite)

        if not args.skip_retention:
            all_payloads = {condition: [] for condition in CONDITIONS}
            for condition in CONDITIONS:
                for seed in args.seeds:
                    all_payloads[condition].append(
                        build_retention_matrix(args, suite, condition, seed, device)
                    )
            plot_retention(args, suite, all_payloads)
            write_summary_csv(args, suite, all_payloads)

    print(f"\nDone. Plots and cached metrics: {args.plots_root}")


if __name__ == "__main__":
    main()
