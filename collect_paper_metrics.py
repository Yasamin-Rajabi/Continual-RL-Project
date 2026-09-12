#!/usr/bin/env python3
"""Aggregate paper-facing continual-RL metrics across seeds.

Reads each method's existing survey_metrics.csv / summary_metrics.csv under
EXPERIMENT_ROOT/main/<method>_<eval_mode>/plots/<suite>/ and writes one compact
CSV with mean/std across seeds. This script never trains or evaluates agents.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev

SURVEY = ("A_N", "FG", "BWT", "FT_success", "FT_return")
SUMMARY = (
    "final_avg_return_all_eval_tasks",
    "final_avg_success_all_eval_tasks",
    "final_avg_return_seen",
    "final_avg_success_seen",
    "final_average_forgetting",
)


def _read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite(rows, key):
    out = []
    for r in rows:
        try:
            x = float(r[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def _stats(values):
    if not values:
        return float("nan"), float("nan"), 0
    return mean(values), (pstdev(values) if len(values) > 1 else 0.0), len(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--eval-mode", choices=("deterministic", "stochastic"), default="deterministic")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = Path(args.experiment_root)
    candidates = sorted((root / "main").glob(f"*_{args.eval_mode}"))
    rows_out = []
    for method_dir in candidates:
        method = method_dir.name[: -(len(args.eval_mode) + 1)]
        base = method_dir / "plots" / args.suite
        survey_rows = _read_rows(base / "survey_metrics.csv")
        summary_rows = _read_rows(base / "summary_metrics.csv")
        if not survey_rows and not summary_rows:
            continue
        row = {"method": method, "eval_mode": args.eval_mode}
        for key in SURVEY:
            mu, sd, n = _stats(_finite(survey_rows, key))
            row[f"{key}_mean"] = mu
            row[f"{key}_std"] = sd
            row[f"{key}_n"] = n
        for key in SUMMARY:
            mu, sd, n = _stats(_finite(summary_rows, key))
            row[f"{key}_mean"] = mu
            row[f"{key}_std"] = sd
            row[f"{key}_n"] = n
        rows_out.append(row)

    if not rows_out:
        raise SystemExit(f"No metric CSVs found under {root/'main'} for mode={args.eval_mode}")

    output = Path(args.output) if args.output else root / f"paper_metrics_{args.eval_mode}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows_out[0].keys())
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    print(output)
    for r in rows_out:
        print(
            f"{r['method']}: "
            f"A_N={r['A_N_mean']:.4g}+-{r['A_N_std']:.3g}, "
            f"FG={r['FG_mean']:.4g}+-{r['FG_std']:.3g}, "
            f"BWT={r['BWT_mean']:.4g}+-{r['BWT_std']:.3g}, "
            f"FT_success={r['FT_success_mean']:.4g}+-{r['FT_success_std']:.3g}, "
            f"FT_return={r['FT_return_mean']:.4g}+-{r['FT_return_std']:.3g}"
        )


if __name__ == "__main__":
    main()
