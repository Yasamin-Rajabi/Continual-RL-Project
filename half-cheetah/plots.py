"""All plotting and table/CSV writing for the continual HalfCheetah benchmark.

Pure drawing: every function here takes already-computed data (or reads
already-logged TensorBoard scalars via metrics.py's path helpers) and writes
PNG/CSV files. No training, no environment rollouts, no checkpoint loading --
see run_continual_benchmark.py (orchestration) and metrics.py (all numeric
computation) for those.

The existing plotting logic (plot_training_metrics, plot_sequence_diagnostics,
plot_merge_lineage, plot_zero_shot, plot_retention, write_summary_csv) is
carried over UNCHANGED from the previous run_continual_benchmark.py -- only
the imports changed (load_scalar/final_scalar/path helpers now come from
metrics.py instead of being defined locally). Two things are new at the
bottom: plot_survey_metrics and write_survey_metrics_csv, for the four
survey metrics (A_N, FG, BWT, FT_success, FT_return) computed in metrics.py.
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from metrics import (
    analysis_snapshot_path,
    event_dir,
    final_scalar,
    load_continual_scalar,
    load_scalar,
)
from tasks import get_task_name

TRAIN_METRICS = {
    "charts/episodic_return": ("Training episodic return", "train_return"),
    "charts/test_episodic_return": ("Evaluation return", "eval_return"),
    "charts/test_velocity_error": ("Evaluation velocity error", "eval_velocity_error"),
    "charts/test_success": ("Evaluation success fraction", "eval_success"),
    "losses/actor_loss": ("SAC actor loss", "actor_loss"),
    "losses/qf_loss": ("SAC critic loss", "critic_loss"),
    "losses/alpha": ("SAC entropy coefficient", "entropy_alpha"),
    "analysis/theta/drift_from_task_start_l2": ("Actor drift from task start", "theta_drift"),
    "analysis/mean/alpha_entropy": ("Historical-mixture alpha entropy", "alpha_entropy"),
    "analysis/mean/alpha_mass": ("Historical-mixture alpha mass", "alpha_mass"),
}

SEQUENCE_METRICS = {
    "analysis/merge/symmetric_kl": ("Selected merge-pair symmetric KL", "merge_selected_skl"),
    "analysis/merge/pairwise_kl_mean": ("Mean pairwise symmetric KL in pool", "merge_pool_mean_skl"),
    "analysis/merge/pairwise_kl_max": ("Maximum pairwise symmetric KL in pool", "merge_pool_max_skl"),
    "distillation/policy/distill_train_kl": ("Distillation train KL", "distill_train_kl"),
    "distillation/policy/distill_test_kl": ("Distillation held-out KL", "distill_test_kl"),
    "distillation/policy/distill_train_mean_mse": ("Distillation train mean MSE", "distill_train_mean_mse"),
    "distillation/policy/distill_test_mean_mse": ("Distillation held-out mean MSE", "distill_test_mean_mse"),
    "distillation/policy/distill_train_logstd_mse": ("Distillation train log-std MSE", "distill_train_logstd_mse"),
    "distillation/policy/distill_test_logstd_mse": ("Distillation held-out log-std MSE", "distill_test_logstd_mse"),
    "timing/train_loop_seconds": ("Task training loop time (s)", "train_loop_seconds"),
    "timing/merge_buffer_seconds": ("Merge-buffer collection time (s)", "merge_buffer_seconds"),
    "timing/finalize_seconds": ("Finalize / merge time (s)", "finalize_seconds"),
    "analysis/pool/final_length": ("Final knowledge-pool length", "pool_length"),
    "analysis/buffer/mean_velocity_error": ("Merge-buffer mean velocity error", "buffer_velocity_error"),
}


# ==========================================================================
# Training-curve plots (unchanged logic; load_scalar/load_continual_scalar
# now come from metrics.py).
# ==========================================================================
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


def plot_training_metrics(args, suite, conditions):
    out_dir = pathlib.Path(args.plots_root) / suite / "training_curves"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scalar_tag, (title, filename) in TRAIN_METRICS.items():
        fig, ax = plt.subplots(figsize=(11, 5.5))
        plotted = False
        for condition in conditions:
            curves = [
                load_continual_scalar(args.runs_root, suite, condition, seed,
                                       args.task_sequence, args.total_timesteps, scalar_tag)
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


def plot_sequence_diagnostics(args, suite, conditions):
    out_dir = pathlib.Path(args.plots_root) / suite / "sequence_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(args.task_sequence))
    for scalar_tag, (title, filename) in SEQUENCE_METRICS.items():
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        plotted = False
        for condition in conditions:
            seed_rows = []
            for seed in args.seeds:
                row = [
                    final_scalar(event_dir(args.runs_root, suite, condition, seed, i, task_id), scalar_tag)
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


# ==========================================================================
# Retention heatmaps / summary curves (unchanged logic).
# ==========================================================================
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
    """NOTE: this is the OLD 'best-so-far minus current' forgetting curve,
    kept as-is for the existing diagnostic plots below. It is intentionally
    DIFFERENT from metrics.compute_fg_bwt's survey-exact FG/BWT (Eq. 8/10),
    which compares only p_i,i (right when first trained) against p_N,i, not
    the best of any intermediate checkpoint. See plot_survey_metrics for the
    survey-exact numbers."""
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


def plot_retention(args, suite, conditions, all_payloads):
    out_dir = pathlib.Path(args.plots_root) / suite / "retention_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_task_ids = sorted(set(args.task_sequence))
    xlabels = [get_task_name(t, suite) for t in eval_task_ids]
    ylabels = [f"{i}:T{task}" for i, task in enumerate(args.task_sequence)]

    for condition in conditions:
        payloads = all_payloads[condition]
        for metric in ("return", "velocity_error", "success"):
            mean, _, _ = aggregate_retention(payloads, metric)
            plot_heatmap(
                mean, xlabels, ylabels,
                f"{suite} / {condition}: checkpoint retention ({metric})",
                metric,
                out_dir / f"heatmap_{condition}_{metric}.png",
            )

    for metric, ylabel in (
        ("return", "Average return on seen tasks"),
        ("velocity_error", "Average velocity error on seen tasks"),
        ("success", "Average success fraction on seen tasks"),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 5.3))
        for condition in conditions:
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

    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    for condition in conditions:
        curves = [continual_summary_curves(p, "return")[1] for p in all_payloads[condition]]
        arr = np.asarray(curves)
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        x = np.arange(len(mean))
        line, = ax.plot(x, mean, marker="o", markersize=3, label=condition)
        if len(args.seeds) > 1:
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=line.get_color())
    ax.set_title(f"{suite}: average forgetting (best-so-far minus current return)")
    ax.set_xlabel("Task position in continual sequence")
    ax.set_ylabel("Average forgetting")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "average_forgetting.png", dpi=180)
    plt.close(fig)

    for metric in ("return", "velocity_error", "success"):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(eval_task_ids), dtype=np.float64)
        width = 0.8 / len(conditions)
        for ci, condition in enumerate(conditions):
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

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for condition in conditions:
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
    for condition in conditions:
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


def plot_merge_lineage(args, suite, conditions):
    """Plot which original task buffers are represented in each selected merge."""
    import torch  # local import: this is the only plot function that touches checkpoints directly

    out_dir = pathlib.Path(args.plots_root) / suite / "sequence_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_task_ids = sorted(set(args.task_sequence))
    xlabels = [get_task_name(t, suite) for t in eval_task_ids]
    ylabels = [f"{i}:T{task}" for i, task in enumerate(args.task_sequence)]

    for condition in conditions:
        seed_matrices = []
        for seed in args.seeds:
            matrix = np.full((len(args.task_sequence), len(eval_task_ids)), np.nan, dtype=np.float64)
            for seq_idx, task_id in enumerate(args.task_sequence):
                path = analysis_snapshot_path(args.analysis_root, suite, condition, seed, seq_idx, task_id)
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


def plot_zero_shot(args, suite, conditions):
    out_dir = pathlib.Path(args.plots_root) / suite / "sequence_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    for scalar_tag, title, filename in (
        ("charts/test_episodic_return", "Zero-shot return before training each task", "zero_shot_return"),
        ("charts/test_velocity_error", "Zero-shot velocity error before training each task", "zero_shot_velocity_error"),
    ):
        fig, ax = plt.subplots(figsize=(10.5, 5.3))
        x = np.arange(len(args.task_sequence))
        for condition in conditions:
            rows = []
            for seed in args.seeds:
                values = []
                for i, task_id in enumerate(args.task_sequence):
                    steps, vals = load_scalar(event_dir(args.runs_root, suite, condition, seed, i, task_id), scalar_tag)
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


def write_summary_csv(args, suite, conditions, all_payloads):
    out_path = pathlib.Path(args.plots_root) / suite / "summary_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in conditions:
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


# ==========================================================================
# NEW: the 4 survey metrics (A_N, FG, BWT, FT_success, FT_return).
# survey_payloads is {condition: [metrics.compute_survey_metrics(...) dict
# per seed]} -- computed in run_continual_benchmark.py, this module only
# draws it.
# ==========================================================================
def plot_survey_metrics(args, suite, conditions, survey_payloads):
    out_dir = pathlib.Path(args.plots_root) / suite / "survey_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("A_N", "Average performance A_N (higher is better)"),
        ("FG", "Forgetting FG (lower is better)"),
        ("BWT", "Backward transfer BWT (higher/positive is better)"),
        ("FT_success", "Forward transfer FT (success-AUC, higher is better)"),
        ("FT_return", "Forward transfer FT (return-based, higher is better)"),
    ]

    fig, axes = plt.subplots(1, len(metric_specs), figsize=(4.6 * len(metric_specs), 5.0))
    if len(metric_specs) == 1:
        axes = [axes]
    x = np.arange(len(conditions))
    for ax, (key, title) in zip(axes, metric_specs):
        means, stds = [], []
        for condition in conditions:
            payloads = survey_payloads.get(condition, [])
            values = [p[key] for p in payloads if p.get(key) is not None and not np.isnan(p[key])]
            means.append(float(np.mean(values)) if values else np.nan)
            stds.append(float(np.std(values)) if len(values) > 1 else 0.0)
        ax.bar(x, means, yerr=stds if len(args.seeds) > 1 else None, capsize=3)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    fig.suptitle(f"{suite}: survey metrics (mean over {len(args.seeds)} seed(s))")
    fig.tight_layout()
    fig.savefig(out_dir / "survey_metrics_bars.png", dpi=180)
    plt.close(fig)

    # Per-position curves for FT/FG/BWT, one figure per metric, all
    # conditions overlaid -- makes it visible whether a metric's average
    # is being driven by a few positions or is consistent across the chain.
    per_position_specs = [
        ("FG_per_position", "Per-position forgetting (excludes last position)"),
        ("BWT_per_position", "Per-position backward transfer (excludes last position)"),
        ("FT_success_per_position", "Per-position forward transfer, success-AUC (excludes first position)"),
        ("FT_return_per_position", "Per-position forward transfer, return-based (excludes first position)"),
    ]
    for key, title in per_position_specs:
        fig, ax = plt.subplots(figsize=(10, 5))
        plotted = False
        for condition in conditions:
            payloads = survey_payloads.get(condition, [])
            rows = [p[key] for p in payloads if p.get(key)]
            if not rows:
                continue
            min_len = min(len(r) for r in rows)
            if min_len == 0:
                continue
            arr = np.asarray([r[:min_len] for r in rows], dtype=np.float64)
            mean = arr.mean(axis=0)
            xpos = np.arange(min_len)
            line, = ax.plot(xpos, mean, marker="o", markersize=3, label=condition)
            if arr.shape[0] > 1:
                ax.fill_between(xpos, mean - arr.std(axis=0), mean + arr.std(axis=0),
                                 alpha=0.15, color=line.get_color())
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"{suite}: {title}")
        ax.set_xlabel("Position within the eligible range")
        ax.set_ylabel(key)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}.png", dpi=180)
        plt.close(fig)


def write_survey_metrics_csv(args, suite, conditions, survey_payloads):
    out_path = pathlib.Path(args.plots_root) / suite / "survey_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in conditions:
        for payload in survey_payloads.get(condition, []):
            rows.append({
                "suite": suite,
                "condition": condition,
                "seed": payload["seed"],
                "A_N": payload["A_N"],
                "FG": payload["FG"],
                "BWT": payload["BWT"],
                "FT_success": payload["FT_success"],
                "FT_return": payload["FT_return"],
            })
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
