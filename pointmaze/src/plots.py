"""Figures from the saved result JSONs.

Only the cells the protocol actually computes exist in a performance matrix,
so the retention heatmap deliberately shows the diagonal and the final row and
leaves the rest blank rather than imputing values that were never measured.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from metrics import diagonal, final_row


def _require_matplotlib():
    if plt is None:
        raise ImportError("plots.py needs matplotlib; install it or skip plotting")


def load_all(results_root, suite: str) -> Dict[str, List[dict]]:
    """Group every ``*_metrics.json`` under ``results_root`` by method."""
    grouped: Dict[str, List[dict]] = {}
    for path in sorted(pathlib.Path(results_root).glob(f"{suite}__*__metrics.json")):
        with path.open() as f:
            payload = json.load(f)
        grouped.setdefault(payload.get("method", path.stem), []).append(payload)
    return grouped


def plot_headline(grouped, out_path, metric="average_final", ylabel=None):
    """Bar chart of one headline metric, with across-seed standard error."""
    _require_matplotlib()
    methods = sorted(grouped)
    means, errors = [], []
    for method in methods:
        values = [
            m[metric] for m in grouped[method] if np.isfinite(m.get(metric, np.nan))
        ]
        means.append(np.mean(values) if values else np.nan)
        errors.append(
            np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#1f77b4" if not m.startswith("Ours") else "#d62728" for m in methods]
    ax.bar(range(len(methods)), means, yerr=errors, capsize=4, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel(ylabel or metric.replace("_", " "))
    ax.set_title(metric.replace("_", " "))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_retention(chain, out_path):
    """Per-task peak vs final, i.e. what was learned and what survived."""
    _require_matplotlib()
    matrix = chain["performance_matrix"]
    peak = diagonal(matrix)
    final = final_row(matrix)
    tasks = chain["task_sequence"]
    idx = [i for i in range(len(tasks)) if peak[i] is not None and final[i] is not None]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.4
    ax.bar([i - width / 2 for i in idx], [peak[i] for i in idx], width, label="peak (just trained)")
    ax.bar([i + width / 2 for i in idx], [final[i] for i in idx], width, label="final (end of chain)")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"t{tasks[i]}" for i in idx])
    ax.set_ylabel("return")
    ax.set_title(f"{chain['method']} seed {chain['seed']}: retention")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="PointMaze result figures")
    p.add_argument("--results-root", default="results_pointmaze")
    p.add_argument("--suite", default="pointmaze_goal")
    p.add_argument("--out", default="plots_pointmaze")
    args = p.parse_args(argv)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grouped = load_all(args.results_root, args.suite)
    if not grouped:
        print(f"no metrics found under {args.results_root}")
        return 1

    for metric, label in (
        ("average_final", "mean return at end of chain"),
        ("average_peak", "mean return when each task was trained"),
        ("forgetting", "peak minus final (lower is better)"),
        ("forward_transfer", "normalized forward transfer"),
    ):
        try:
            plot_headline(grouped, out / f"{args.suite}__{metric}.png", metric, label)
        except Exception as exc:
            print(f"[warn] could not plot {metric}: {exc}")

    for path in sorted(pathlib.Path(args.results_root).glob(f"{args.suite}__*__chain.json")):
        with path.open() as f:
            chain = json.load(f)
        try:
            plot_retention(chain, out / f"{path.stem}__retention.png")
        except Exception as exc:
            print(f"[warn] could not plot retention for {path.name}: {exc}")

    print(f"wrote figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
