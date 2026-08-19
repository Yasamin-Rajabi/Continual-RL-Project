"""Aggregate the per-seed summary CSVs produced by run_continual_benchmark.py."""
from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plots-root", default="plots_halfcheetah_continual")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = pathlib.Path(args.plots_root)
    rows = []
    for path in root.glob("*/summary_metrics.csv"):
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        raise SystemExit(
            f"No summary_metrics.csv files found under {root}. "
            "Run run_continual_benchmark.py without --skip-retention first."
        )

    numeric = [
        k for k in rows[0]
        if k not in {"suite", "condition", "seed"}
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["condition"])].append(row)

    output_rows = []
    for (suite, condition), group in sorted(grouped.items()):
        out = {"suite": suite, "condition": condition, "n_seeds": len(group)}
        for key in numeric:
            vals = np.asarray([float(r[key]) for r in group], dtype=np.float64)
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
        output_rows.append(out)

    out_path = pathlib.Path(args.output) if args.output else root / "aggregate_summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    for row in output_rows:
        print(f"\n{row['suite']} / {row['condition']} (n={row['n_seeds']})")
        print(
            f"  final return: {row['final_avg_return_all_eval_tasks_mean']:.3f} "
            f"+/- {row['final_avg_return_all_eval_tasks_std']:.3f}"
        )
        print(
            f"  final velocity error: {row['final_avg_velocity_error_all_eval_tasks_mean']:.4f} "
            f"+/- {row['final_avg_velocity_error_all_eval_tasks_std']:.4f}"
        )
        print(
            f"  final forgetting: {row['final_average_forgetting_mean']:.3f} "
            f"+/- {row['final_average_forgetting_std']:.3f}"
        )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
