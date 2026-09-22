"""Plots for the survey-style Atari baseline metrics produced by metrics.py."""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from benchmark_protocol import METHODS, canonical_env_name, canonical_method, metric_root, unique_in_order


def _load_json(path):
    path = pathlib.Path(path)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return np.nan, np.nan
    return float(arr.mean()), float(arr.std())


def plot_summary(args, env):
    env = canonical_env_name(env)
    out_dir = pathlib.Path(args.plots_root) / env / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("A_N", "Final average performance $A_N$"),
        ("FG", "Forgetting (FG)"),
        ("BWT", "Backward transfer (BWT)"),
        ("FT_return", "Forward transfer on raw return"),
        ("FT_success", "Forward transfer on success"),
        ("A_N_return", "Final average raw return"),
    ]

    method_rows = {}
    for method in args.methods:
        method = canonical_method(method)
        rows = []
        for seed in args.seeds:
            payload = _load_json(metric_root(args.plots_root, env, method, seed) / "survey_metrics.json")
            if payload is not None:
                rows.append(payload)
        method_rows[method] = rows

    for metric, title in metric_specs:
        names, means, stds = [], [], []
        for method, rows in method_rows.items():
            if not rows:
                continue
            mean, std = _mean_std([row.get(metric, np.nan) for row in rows])
            if not np.isfinite(mean):
                continue
            names.append(method)
            means.append(mean)
            stds.append(std)
        if not names:
            continue

        fig, ax = plt.subplots(figsize=(10.5, 5.3))
        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds if len(args.seeds) > 1 else None, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{env}: {title}")
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
        fig.tight_layout()
        fig.savefig(out_dir / f"summary_{metric}.png", dpi=180)
        plt.close(fig)


def _aggregate_retention(payloads, key):
    arrays = [np.asarray(payload[key], dtype=np.float64) for payload in payloads]
    if not arrays:
        return None
    arr = np.asarray(arrays, dtype=np.float64)
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def plot_retention_heatmaps(args, env):
    env = canonical_env_name(env)
    out_dir = pathlib.Path(args.plots_root) / env / "plots" / "retention"
    out_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        method = canonical_method(method)
        payloads = []
        for seed in args.seeds:
            payload = _load_json(metric_root(args.plots_root, env, method, seed) / "retention.json")
            if payload is not None:
                payloads.append(payload)
        if not payloads:
            continue

        sequence = payloads[0]["sequence"]
        eval_ids = payloads[0]["eval_task_ids"]
        ylabels = [f"{i}:T{task}" for i, task in enumerate(sequence)]
        xlabels = [f"T{task}" for task in eval_ids]

        for key, label in (("success", "Success"), ("return", "Raw full-game return")):
            agg = _aggregate_retention(payloads, key)
            if agg is None:
                continue
            mean, _ = agg
            if not np.isfinite(mean).any():
                continue
            fig, ax = plt.subplots(figsize=(10.5, 7.0))
            im = ax.imshow(mean, aspect="auto")
            fig.colorbar(im, ax=ax, label=label)
            ax.set_xticks(np.arange(len(xlabels)))
            ax.set_xticklabels(xlabels)
            ax.set_yticks(np.arange(len(ylabels)))
            ax.set_yticklabels(ylabels)
            ax.set_xlabel("Evaluation task")
            ax.set_ylabel("Checkpoint after sequence position")
            ax.set_title(f"{env} / {method}: retention ({key})")
            fig.tight_layout()
            fig.savefig(out_dir / f"{method}_{key}_heatmap.png", dpi=180)
            plt.close(fig)


def plot_ft_per_position(args, env):
    env = canonical_env_name(env)
    out_dir = pathlib.Path(args.plots_root) / env / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("FT_return", "FT_success"):
        fig, ax = plt.subplots(figsize=(10.5, 5.3))
        plotted = False
        for method in args.methods:
            method = canonical_method(method)
            series = []
            positions = None
            for seed in args.seeds:
                payload = _load_json(metric_root(args.plots_root, env, method, seed) / "survey_metrics.json")
                if payload is None:
                    continue
                values = np.asarray(payload.get(f"{metric}_per_position", []), dtype=np.float64)
                pos = payload.get(f"{metric}_positions", [])
                if values.size:
                    series.append(values)
                    positions = pos
            if not series:
                continue
            min_len = min(len(x) for x in series)
            arr = np.asarray([x[:min_len] for x in series])
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            x = np.arange(min_len)
            label = method
            line, = ax.plot(x, mean, marker="o", label=label)
            if len(series) > 1:
                ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=line.get_color())
            plotted = True

        if not plotted:
            plt.close(fig)
            continue
        ax.axhline(0.0, linewidth=0.8, alpha=0.5)
        ax.set_xlabel("First encounter after initial task")
        ax.set_ylabel(metric)
        ax.set_title(f"{env}: {metric} by first-unseen task position")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}_per_position.png", dpi=180)
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--envs", nargs="+", default=["Freeway", "SpaceInvaders"])
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--plots-root", default="metric_results")
    return p.parse_args()


def main():
    args = parse_args()
    for env in args.envs:
        plot_summary(args, env)
        plot_retention_heatmaps(args, env)
        plot_ft_per_position(args, env)
    print(f"Plots written under: {args.plots_root}")


if __name__ == "__main__":
    main()
