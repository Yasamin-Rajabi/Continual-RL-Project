"""Plotting/CSV utilities for the continual Atari CKA-RL benchmark.

This module performs no environment interaction and no metric computation.  It
consumes TensorBoard scalars and JSON payloads produced by metrics.py.
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from atari_tasks import get_task_name
from metrics import (
    analysis_snapshot_path,
    checkpoint_dir,
    event_dir,
    final_scalar,
    load_continual_scalar,
    load_scalar,
)

TRAIN_METRICS = {
    "charts/episodic_return": ("Training full-game episodic return", "train_return"),
    "charts/test_episodic_return": ("Evaluation raw full-game return", "eval_return"),
    "charts/test_success": ("Evaluation threshold success", "eval_success"),
    "losses/policy_loss": ("PPO policy loss", "policy_loss"),
    "losses/value_loss": ("PPO value loss", "value_loss"),
    "losses/entropy": ("Policy entropy", "entropy"),
    "losses/approx_kl": ("PPO approximate KL", "approx_kl"),
    "losses/clipfrac": ("PPO clipped fraction", "clipfrac"),
    "losses/explained_variance": ("PPO explained variance", "explained_variance"),
    "losses/encoder_drift_reg": ("Historical encoder-drift regularizer", "encoder_drift_reg"),
    "analysis/policy/delta_from_task_start_l2": ("Policy-head drift from task start", "policy_delta"),
    "analysis/policy/alpha_entropy": ("Historical-mixture alpha entropy", "alpha_entropy"),
    "analysis/policy/alpha_mass_effective": ("Historical-mixture alpha mass", "alpha_mass"),
}

SEQUENCE_METRICS = {
    "analysis/merge/cosine_similarity": ("Selected merge-pair cosine similarity", "merge_selected_cosine"),
    "analysis/merge/pairwise_cosine_mean": ("Mean pairwise cosine similarity", "merge_pool_mean_cosine"),
    "analysis/merge/pairwise_cosine_max": ("Maximum pairwise cosine similarity", "merge_pool_max_cosine"),
    "analysis/merge/symmetric_kl": ("Selected replay-weighted symmetric KL", "merge_selected_skl"),
    "analysis/merge/pairwise_kl_mean": ("Mean pairwise replay-weighted KL", "merge_pool_mean_skl"),
    "analysis/merge/pairwise_kl_max": ("Maximum pairwise replay-weighted KL", "merge_pool_max_skl"),
    "analysis/merge/selected_state_kl_p95": ("Selected pair directional KL p95", "merge_selected_kl_p95"),
    "analysis/merge/selected_state_kl_max": ("Selected pair directional KL max", "merge_selected_kl_max"),
    "analysis/merge/pool_size_before": ("Pool size before merge", "pool_size_before"),
    "analysis/merge/pool_size_after": ("Pool size after merge", "pool_size_after"),
    "analysis/merge/used_distillation": ("Merge used distillation", "merge_used_distillation"),
    "distillation/policy/distill_train_kl": ("Distillation train KL", "distill_train_kl"),
    "distillation/policy/distill_test_kl": ("Distillation held-out KL", "distill_test_kl"),
    "distillation/policy/distill_train_kl_p95": ("Distillation train KL p95", "distill_train_kl_p95"),
    "distillation/policy/distill_test_kl_p95": ("Distillation held-out KL p95", "distill_test_kl_p95"),
    "distillation/policy/distill_train_kl_max": ("Distillation train KL max", "distill_train_kl_max"),
    "distillation/policy/distill_test_kl_max": ("Distillation held-out KL max", "distill_test_kl_max"),
    "distillation/policy/distill_train_prob_mse": ("Distillation train probability MSE", "distill_train_prob_mse"),
    "distillation/policy/distill_test_prob_mse": ("Distillation held-out probability MSE", "distill_test_prob_mse"),
    "distillation/policy/distill_best_val_kl": ("Distillation best validation KL", "distill_best_val_kl"),
    "distillation/policy/distill_best_epoch": ("Distillation best epoch", "distill_best_epoch"),
    "distillation/policy/distill_selected_val_kl": ("Distillation selected validation KL", "distill_selected_val_kl"),
    "distillation/policy/distill_selected_epoch": ("Distillation selected epoch", "distill_selected_epoch"),
    "timing/train_loop_seconds": ("Task PPO training time (s)", "train_loop_seconds"),
    "timing/merge_buffer_seconds": ("Merge-buffer collection time (s)", "merge_buffer_seconds"),
    "timing/finalize_seconds": ("Finalize / merge time (s)", "finalize_seconds"),
    "analysis/pool/final_length": ("Final policy-pool length", "pool_length"),
    "analysis/buffer/rows": ("Stored reference rows", "buffer_rows"),
    "analysis/encoder/max_drift": ("Frozen encoder max drift", "encoder_max_drift"),
    "analysis/merge/balance_source_lineages": ("Lineage-balanced merge sampling enabled", "merge_balance_source_lineages"),
    "analysis/merge/source_lineages_parent_1": ("Source lineages in merge parent 1", "merge_source_lineages_parent_1"),
    "analysis/merge/source_lineages_parent_2": ("Source lineages in merge parent 2", "merge_source_lineages_parent_2"),
    "analysis/merge/source_lineages_merged": ("Source lineages in retained merged buffer", "merge_source_lineages_merged"),
    "distillation/policy/distill_source_lineages": ("Distillation source-lineage count", "distill_source_lineages"),
    "distillation/policy/distill_balance_source_lineages": ("Lineage-balanced distillation enabled", "distill_balance_source_lineages"),
    "policy/projection_initial_val_mixture_kl": ("Policy projection initial validation KL", "projection_initial_val_kl"),
    "policy/projection_val_mixture_kl": ("Policy projection best validation KL", "projection_val_kl"),
    "policy/projection_train_mixture_kl": ("Policy projection train KL", "projection_train_kl"),
    "policy/projection_best_epoch": ("Policy projection best epoch", "projection_best_epoch"),
    "policy/projection_rows": ("Policy projection rows", "projection_rows"),
    "policy/projection_components": ("Policy projection mixture components", "projection_components"),
    "policy/storage_used_novel_expert": ("Stored standalone novel expert", "storage_used_novel_expert"),
}


def _prepare_curve(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        return x, y
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    _, rev_idx = np.unique(x[::-1], return_index=True)
    keep = x.size - 1 - rev_idx
    keep.sort()
    return x[keep], y[keep]


def aggregate_curves(curves, grid_points=1200):
    clean = []
    for x, y in curves:
        x, y = _prepare_curve(x, y)
        if len(x) >= 2:
            clean.append((x, y))
    if not clean:
        return None
    start = max(float(x.min()) for x, _ in clean)
    end = min(float(x.max()) for x, _ in clean)
    if end <= start:
        return None
    grid = np.linspace(start, end, int(grid_points))
    arr = np.asarray([np.interp(grid, x, y) for x, y in clean])
    return grid, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def add_task_boundaries(ax, args):
    for seq_idx in range(1, len(args.task_sequence)):
        ax.axvline(seq_idx * (args.total_timesteps + 1), linewidth=0.5, alpha=0.25)


def plot_training_metrics(args, suite, conditions):
    out_dir = pathlib.Path(args.plots_root) / suite / "training_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scalar_tag, (title, filename) in TRAIN_METRICS.items():
        fig, ax = plt.subplots(figsize=(11, 5.5))
        plotted = False
        for condition in conditions:
            curves = [
                load_continual_scalar(
                    args.runs_root, suite, condition, seed,
                    args.task_sequence, args.total_timesteps, scalar_tag,
                )
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
        ax.set_xlabel("Continual environment transitions")
        ax.set_ylabel(title)
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / f"{filename}.png", dpi=180)
        plt.close(fig)


def plot_sequence_diagnostics(args, suite, conditions):
    out_dir = pathlib.Path(args.plots_root) / suite / "sequence_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(args.task_sequence))
    for scalar_tag, (title, filename) in SEQUENCE_METRICS.items():
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        plotted = False
        for condition in conditions:
            rows = []
            for seed in args.seeds:
                rows.append([
                    final_scalar(
                        event_dir(args.runs_root, suite, condition, seed, i, task_id),
                        scalar_tag,
                    )
                    for i, task_id in enumerate(args.task_sequence)
                ])
            arr = np.asarray(rows, dtype=np.float64)
            if not np.isfinite(arr).any():
                continue
            plotted = True
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
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


def aggregate_retention(seed_payloads, metric):
    arr = np.asarray([payload[metric] for payload in seed_payloads], dtype=np.float64)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), arr


def plot_heatmap(matrix, xlabels, ylabels, title, ylabel, out_path):
    if not np.isfinite(matrix).any():
        return
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
    seen = set(sequence[: stage + 1])
    return [i for i, task_id in enumerate(eval_task_ids) if task_id in seen]


def continual_summary_curves(payload, metric):
    matrix = np.asarray(payload[metric], dtype=np.float64)
    seq = payload["sequence"]
    eval_ids = payload["eval_task_ids"]
    avg_seen, forgetting = [], []
    for stage in range(len(seq)):
        idx = seen_task_indices(seq, eval_ids, stage)
        vals = matrix[stage, idx]
        avg_seen.append(float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan)
        fvals = []
        for j in idx:
            task_id = eval_ids[j]
            first_trained = seq.index(task_id)
            history = matrix[first_trained : stage + 1, j]
            if not np.isfinite(history).any() or not np.isfinite(matrix[stage, j]):
                continue
            fvals.append(float(np.nanmax(history) - matrix[stage, j]))
        forgetting.append(float(np.mean(fvals)) if fvals else np.nan)
    return np.asarray(avg_seen), np.asarray(forgetting)


def plot_retention(args, suite, conditions, all_payloads):
    out_dir = pathlib.Path(args.plots_root) / suite / "retention_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_task_ids = sorted(set(args.task_sequence))
    xlabels = [get_task_name(t, suite) for t in eval_task_ids]
    ylabels = [f"{i}:T{task}" for i, task in enumerate(args.task_sequence)]

    metric_specs = (("reward", "Raw full-game score"), ("success", "Threshold success"))
    for condition in conditions:
        payloads = all_payloads.get(condition, [])
        if not payloads:
            continue
        for metric, ylabel in metric_specs:
            mean, _, _ = aggregate_retention(payloads, metric)
            plot_heatmap(
                mean, xlabels, ylabels,
                f"{suite} / {condition}: checkpoint retention ({metric})",
                ylabel,
                out_dir / f"heatmap_{condition}_{metric}.png",
            )

    for metric, ylabel in metric_specs:
        fig, ax = plt.subplots(figsize=(10.5, 5.3))
        plotted = False
        for condition in conditions:
            curves = [continual_summary_curves(p, metric)[0] for p in all_payloads.get(condition, [])]
            if not curves:
                continue
            arr = np.asarray(curves, dtype=np.float64)
            if not np.isfinite(arr).any():
                continue
            plotted = True
            mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
            x = np.arange(len(mean))
            line, = ax.plot(x, mean, marker="o", markersize=3, label=condition)
            if len(args.seeds) > 1:
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=line.get_color())
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"{suite}: retention on tasks seen so far")
        ax.set_xlabel("Task position in continual sequence")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"avg_seen_{metric}.png", dpi=180)
        plt.close(fig)

    # Diagnostic best-so-far forgetting on raw score. Survey-exact FG is plotted
    # separately by plot_survey_metrics().
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    plotted = False
    for condition in conditions:
        curves = [continual_summary_curves(p, "reward")[1] for p in all_payloads.get(condition, [])]
        if not curves:
            continue
        arr = np.asarray(curves, dtype=np.float64)
        if not np.isfinite(arr).any():
            continue
        plotted = True
        mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
        x = np.arange(len(mean))
        line, = ax.plot(x, mean, marker="o", markersize=3, label=condition)
        if len(args.seeds) > 1:
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=line.get_color())
    if plotted:
        ax.set_title(f"{suite}: diagnostic best-so-far forgetting (raw score)")
        ax.set_xlabel("Task position in continual sequence")
        ax.set_ylabel("Best-so-far score minus current score")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "average_forgetting_reward.png", dpi=180)
    plt.close(fig)

    for metric, ylabel in metric_specs:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        plotted = False
        x = np.arange(len(eval_task_ids), dtype=np.float64)
        width = 0.8 / max(len(conditions), 1)
        for ci, condition in enumerate(conditions):
            payloads = all_payloads.get(condition, [])
            if not payloads:
                continue
            finals = np.asarray([np.asarray(p[metric], dtype=np.float64)[-1] for p in payloads])
            if not np.isfinite(finals).any():
                continue
            plotted = True
            mean, std = np.nanmean(finals, axis=0), np.nanstd(finals, axis=0)
            pos = x - 0.4 + width / 2.0 + ci * width
            ax.bar(pos, mean, width=width, yerr=std if len(args.seeds) > 1 else None, label=condition)
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=35, ha="right")
        ax.set_title(f"{suite}: final checkpoint per-task {metric}")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / f"final_per_task_{metric}.png", dpi=180)
        plt.close(fig)


def _load_merge_info(args, suite, condition, seed, seq_idx, task_id):
    # Prefer the optional analysis snapshot when present, otherwise use the
    # continuation policy_pool.pt fallback below.
    analysis_root = getattr(args, "analysis_root", None)
    if analysis_root is not None:
        path = analysis_snapshot_path(
            analysis_root, suite, condition, seed, seq_idx, task_id
        )
        if path.exists():
            try:
                snap = torch.load(path, map_location="cpu", weights_only=False)
                info = snap.get("merge_info")
                if info:
                    return info
            except Exception:
                pass
    # Fallback: continuation policy_pool.pt also stores last_merge_info.
    run_dir = checkpoint_dir(args.save_root, suite, condition, seed, seq_idx, task_id)
    pool_path = pathlib.Path(run_dir) / "policy_pool.pt"
    if pool_path.exists():
        try:
            pool = torch.load(pool_path, map_location="cpu", weights_only=False)
            info = getattr(pool, "last_merge_info", None)
            if info:
                return info
        except Exception:
            pass
    return None


def plot_merge_lineage(args, suite, conditions):
    out_dir = pathlib.Path(args.plots_root) / suite / "sequence_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_task_ids = sorted(set(args.task_sequence))
    task_labels = [get_task_name(t, suite) for t in eval_task_ids]
    ylabels = [f"{i}:T{task}" for i, task in enumerate(args.task_sequence)]

    for condition in conditions:
        task_mats, source_mats = [], []
        for seed in args.seeds:
            tmat = np.full((len(args.task_sequence), len(eval_task_ids)), np.nan)
            smat = np.full((len(args.task_sequence), len(args.task_sequence)), np.nan)
            for seq_idx, task_id in enumerate(args.task_sequence):
                info = _load_merge_info(args, suite, condition, seed, seq_idx, task_id)
                if not info:
                    continue
                lineage = info.get("merged_lineage") or {}
                counts = np.asarray([float(lineage.get(str(t), 0.0)) for t in eval_task_ids])
                if counts.sum() > 0:
                    tmat[seq_idx] = counts / counts.sum()
                source = info.get("merged_source_lineage") or {}
                scounts = np.asarray([float(source.get(str(i), 0.0)) for i in range(len(args.task_sequence))])
                if scounts.sum() > 0:
                    smat[seq_idx] = scounts / scounts.sum()
            task_mats.append(tmat)
            source_mats.append(smat)

        for name, stack, labels, xlabel in (
            ("merge_lineage", np.asarray(task_mats), task_labels, "Original task represented in selected merge"),
            ("merge_source_lineage", np.asarray(source_mats), ylabels, "Original sequence occurrence represented in selected merge"),
        ):
            if not np.isfinite(stack).any():
                continue
            mean = np.nanmean(stack, axis=0)
            fig, ax = plt.subplots(figsize=(12, 7))
            im = ax.imshow(mean, aspect="auto", vmin=0.0, vmax=1.0)
            fig.colorbar(im, ax=ax, label="Fraction of selected merged buffer")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=40, ha="right")
            ax.set_yticks(np.arange(len(ylabels)))
            ax.set_yticklabels(ylabels)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Continual sequence position")
            ax.set_title(f"{suite} / {condition}: merge lineage")
            fig.tight_layout()
            fig.savefig(out_dir / f"{name}_{condition}.png", dpi=180)
            plt.close(fig)


def plot_zero_shot(args, suite, conditions):
    out_dir = pathlib.Path(args.plots_root) / suite / "sequence_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scalar_tag, title, filename in (
        ("charts/test_episodic_return", "Zero-shot raw return before training each task", "zero_shot_return"),
        ("charts/test_success", "Zero-shot threshold success before training each task", "zero_shot_success"),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 5.3))
        plotted = False
        x = np.arange(len(args.task_sequence))
        for condition in conditions:
            rows = []
            for seed in args.seeds:
                vals_out = []
                for i, task_id in enumerate(args.task_sequence):
                    steps, vals = load_scalar(
                        event_dir(args.runs_root, suite, condition, seed, i, task_id),
                        scalar_tag,
                    )
                    idx = np.flatnonzero(steps == 0)
                    vals_out.append(float(vals[idx[-1]]) if idx.size else np.nan)
                rows.append(vals_out)
            arr = np.asarray(rows, dtype=np.float64)
            if not np.isfinite(arr).any():
                continue
            plotted = True
            mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
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


def write_summary_csv(args, suite, conditions, all_payloads):
    out_path = pathlib.Path(args.plots_root) / suite / "summary_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in conditions:
        for payload in all_payloads.get(condition, []):
            reward = np.asarray(payload["reward"], dtype=np.float64)
            success = np.asarray(payload["success"], dtype=np.float64)
            avg_reward, forgetting = continual_summary_curves(payload, "reward")
            avg_success, _ = continual_summary_curves(payload, "success")
            rows.append({
                "suite": suite,
                "condition": condition,
                "seed": payload["seed"],
                "final_avg_reward_all_eval_tasks": float(np.nanmean(reward[-1])),
                "final_avg_success_all_eval_tasks": float(np.nanmean(success[-1])) if np.isfinite(success[-1]).any() else np.nan,
                "final_avg_reward_seen": float(avg_reward[-1]),
                "final_avg_success_seen": float(avg_success[-1]) if np.isfinite(avg_success[-1]) else np.nan,
                "final_diagnostic_forgetting_reward": float(forgetting[-1]),
            })
    if not rows:
        return
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_survey_metrics(args, suite, conditions, survey_payloads):
    out_dir = pathlib.Path(args.plots_root) / suite / "survey_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("A_N_reward", "A_N raw score"),
        ("FG_reward", "FG raw score"),
        ("BWT_reward", "BWT raw score"),
        ("FT_reward", "FT relative raw-score AUC"),
        ("A_N_success", "A_N success"),
        ("FG_success", "FG success"),
        ("BWT_success", "BWT success"),
        ("FT_success", "FT normalized success AUC"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    x = np.arange(len(conditions))
    for ax, (key, title) in zip(axes.flat, specs):
        means, stds = [], []
        for condition in conditions:
            vals = [
                float(p[key]) for p in survey_payloads.get(condition, [])
                if p.get(key) is not None and np.isfinite(p.get(key))
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        ax.bar(x, means, yerr=stds if len(args.seeds) > 1 else None, capsize=3)
        ax.axhline(0.0, linewidth=0.6, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    fig.suptitle(f"{suite}: continual-learning metrics")
    fig.tight_layout()
    fig.savefig(out_dir / "survey_metrics_bars.png", dpi=180)
    plt.close(fig)

    for key in (
        "FG_reward_per_position", "BWT_reward_per_position",
        "FT_reward_per_position", "FG_success_per_position",
        "BWT_success_per_position", "FT_success_per_position",
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        plotted = False
        for condition in conditions:
            rows = [p.get(key, []) for p in survey_payloads.get(condition, []) if p.get(key)]
            if not rows:
                continue
            min_len = min(map(len, rows))
            if min_len < 1:
                continue
            arr = np.asarray([r[:min_len] for r in rows], dtype=np.float64)
            if not np.isfinite(arr).any():
                continue
            plotted = True
            mean, std = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
            xpos = np.arange(min_len)
            line, = ax.plot(xpos, mean, marker="o", markersize=3, label=condition)
            if arr.shape[0] > 1:
                ax.fill_between(xpos, mean - std, mean + std, alpha=0.15, color=line.get_color())
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"{suite}: {key}")
        ax.set_xlabel("Eligible sequence position index")
        ax.set_ylabel(key)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}.png", dpi=180)
        plt.close(fig)


def write_survey_metrics_csv(args, suite, conditions, survey_payloads):
    out_path = pathlib.Path(args.plots_root) / suite / "survey_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = (
        "A_N_reward", "FG_reward", "BWT_reward", "FT_reward",
        "A_N_success", "FG_success", "BWT_success", "FT_success",
        "ft_available", "success_thresholds_complete",
        "FT_reward_complete", "FT_success_complete",
    )
    rows = []
    for condition in conditions:
        for payload in survey_payloads.get(condition, []):
            row = {"suite": suite, "condition": condition, "seed": payload["seed"]}
            row.update({key: payload.get(key) for key in keys})
            rows.append(row)
    if not rows:
        return
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
