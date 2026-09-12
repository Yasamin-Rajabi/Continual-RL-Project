#!/usr/bin/env python3
"""Collect paper-facing continual-RL metrics across seeds.

Metric meanings used here
-------------------------
PERF is an acquisition/plasticity metric.  For each continual-sequence
occurrence k, take the performance of the active policy/routing at the end of
training that occurrence, p_k^end, then average over ALL sequence occurrences:

    PERF = (1/N) * sum_k p_k^end

The training code writes exactly these pre-finalization evaluations as
``charts/final_success`` and ``charts/final_return`` in each occurrence's
``scalars.csv``.  The value is evaluated before pool insertion/consolidation,
even though the scalar is written a few lines later.

A_N, FG, BWT and FT are NOT recomputed from PERF here.  They are read from
``survey_metrics.csv`` because they belong to the retention/transfer protocol:

* A_N: final checkpoint/pool evaluated on each unique task, then averaged.
* FG/BWT: checkpoint-retention diagonal versus the final row under the same
  configured retention protocol (e.g. 5k alpha-only retrieval for both sides).
* FT: learning-curve AUC versus matched scratch, first unseen encounters only.

This script never trains or evaluates an agent.  It only reads saved CSVs.

For commented run folders created by ``job.sh --comment NAME`` (for example
``combined_deterministic_amass``), pass the same ``--comment NAME`` here.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _as_float(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _finite(rows, key):
    out = []
    for row in rows:
        x = _as_float(row.get(key))
        if math.isfinite(x):
            out.append(x)
    return out


def _stats(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return float("nan"), float("nan"), 0
    return mean(values), (pstdev(values) if len(values) > 1 else 0.0), len(values)


def _numeric_suffix(path: Path, prefix: str):
    name = path.name
    if not name.startswith(prefix):
        return 10**12
    try:
        return int(name[len(prefix):])
    except ValueError:
        return 10**12


def _last_scalar_from_csv(path: Path, preferred_tag: str, fallback_tag: str):
    """Return the last finite scalar, preferring the explicit final tag.

    ``charts/final_*`` is the canonical PERF source.  ``charts/test_*`` is a
    conservative fallback for older CSV mirrors that may not contain the
    explicit alias; the final test evaluation is the same pre-finalization
    active-policy evaluation in the current trainer.
    """
    last = {preferred_tag: None, fallback_tag: None}
    if not path.exists():
        return float("nan"), None
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tag = row.get("tag")
            if tag not in last:
                continue
            x = _as_float(row.get("value"))
            if math.isfinite(x):
                last[tag] = x
    if last[preferred_tag] is not None:
        return float(last[preferred_tag]), preferred_tag
    if last[fallback_tag] is not None:
        return float(last[fallback_tag]), fallback_tag
    return float("nan"), None


def _task_id_from_run_dir(run_dir: Path):
    m = re.search(r"__task_(-?\d+)__", run_dir.name)
    return int(m.group(1)) if m else None


def _perf_for_method(method_dir: Path, suite: str, method: str, eval_mode: str, survey_rows):
    """Compute per-seed PERF from ALL sequence occurrences.

    Each seed is averaged across its occurrences first.  Paper aggregation is
    then mean/std across seeds, so every random seed receives equal weight.
    """
    requested_seeds = []
    for row in survey_rows:
        try:
            requested_seeds.append(int(row["seed"]))
        except (KeyError, TypeError, ValueError):
            pass
    requested_seeds = sorted(set(requested_seeds))

    run_root = method_dir / "runs" / suite / method
    if not run_root.exists():
        raise RuntimeError(f"PERF run directory not found: {run_root}")

    seed_dirs = sorted(run_root.glob("seed_*"), key=lambda p: _numeric_suffix(p, "seed_"))
    by_seed_dir = {_numeric_suffix(p, "seed_"): p for p in seed_dirs}
    seeds = requested_seeds or sorted(by_seed_dir)

    per_seed = {}
    occurrence_rows = []
    occurrence_counts = []

    for seed in seeds:
        seed_dir = by_seed_dir.get(seed)
        if seed_dir is None:
            raise RuntimeError(f"PERF is missing seed_{seed} under {run_root}")

        seq_dirs = sorted(seed_dir.glob("seq_*"), key=lambda p: _numeric_suffix(p, "seq_"))
        if not seq_dirs:
            raise RuntimeError(f"PERF found no seq_* directories under {seed_dir}")

        success_values = []
        return_values = []
        for seq_dir in seq_dirs:
            seq_idx = _numeric_suffix(seq_dir, "seq_")
            scalar_files = sorted(seq_dir.glob("*/scalars.csv"))
            if len(scalar_files) != 1:
                raise RuntimeError(
                    f"Expected exactly one scalars.csv for {method} seed={seed} seq={seq_idx}, "
                    f"found {len(scalar_files)} under {seq_dir}"
                )
            scalar_path = scalar_files[0]
            run_dir = scalar_path.parent
            task_id = _task_id_from_run_dir(run_dir)

            success, success_tag = _last_scalar_from_csv(
                scalar_path, "charts/final_success", "charts/test_success"
            )
            ret, return_tag = _last_scalar_from_csv(
                scalar_path, "charts/final_return", "charts/test_episodic_return"
            )
            if not (math.isfinite(success) and math.isfinite(ret)):
                raise RuntimeError(
                    f"Missing finite end-of-task PERF scalar(s) in {scalar_path}: "
                    f"success={success}, return={ret}"
                )

            success_values.append(success)
            return_values.append(ret)
            occurrence_rows.append({
                "method": method,
                "eval_mode": eval_mode,
                "seed": seed,
                "seq_idx": seq_idx,
                "task_id": "" if task_id is None else task_id,
                "PERF_success_occurrence": success,
                "PERF_return_occurrence": ret,
                "success_source_tag": success_tag,
                "return_source_tag": return_tag,
                "scalars_csv": str(scalar_path),
            })

        occurrence_counts.append(len(success_values))
        per_seed[seed] = {
            "PERF_success": float(mean(success_values)),
            "PERF_return": float(mean(return_values)),
            "PERF_occurrences": len(success_values),
        }

    if len(set(occurrence_counts)) > 1:
        raise RuntimeError(
            f"Inconsistent number of sequence occurrences for {method}: {occurrence_counts}"
        )

    return per_seed, occurrence_rows


def _plot_perf(rows_out, metric, ylabel, title, output_path: Path):
    rows = [
        row for row in rows_out
        if math.isfinite(_as_float(row.get(f"{metric}_mean")))
    ]
    if not rows:
        print(f"[plot] skipped {metric}: no finite values")
        return

    methods = [row["method"] for row in rows]
    means = [float(row[f"{metric}_mean"]) for row in rows]
    stds = [float(row[f"{metric}_std"]) for row in rows]

    fig_width = max(6.5, 1.15 * len(methods) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    x = list(range(len(methods)))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(output_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--eval-mode", choices=("deterministic", "stochastic"), default="deterministic")
    ap.add_argument("--comment", default="", help="Optional run-folder suffix used by job.sh, e.g. amass for combined_deterministic_amass.")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = Path(args.experiment_root)
    comment = args.comment.strip()
    if comment and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", comment):
        raise SystemExit("--comment may contain only letters, digits, '.', '_' and '-' and must start with a letter/digit")
    run_suffix = f"_{args.eval_mode}" + (f"_{comment}" if comment else "")
    candidates = sorted((root / "main").glob(f"*{run_suffix}"))
    aggregate_rows = []
    per_seed_rows = []
    all_occurrence_rows = []

    for method_dir in candidates:
        method = method_dir.name[:-len(run_suffix)]
        base = method_dir / "plots" / args.suite
        survey_rows = _read_rows(base / "survey_metrics.csv")
        summary_rows = _read_rows(base / "summary_metrics.csv")
        if not survey_rows and not summary_rows:
            continue

        perf_by_seed, occurrence_rows = _perf_for_method(
            method_dir, args.suite, method, args.eval_mode, survey_rows
        )
        all_occurrence_rows.extend(occurrence_rows)

        # Build an auditable per-seed table by joining the survey metrics and
        # acquisition PERF on seed.
        survey_by_seed = {}
        for row in survey_rows:
            try:
                survey_by_seed[int(row["seed"])] = row
            except (KeyError, TypeError, ValueError):
                continue

        summary_by_seed = {}
        for row in summary_rows:
            try:
                summary_by_seed[int(row["seed"])] = row
            except (KeyError, TypeError, ValueError):
                continue

        for seed in sorted(perf_by_seed):
            perf = perf_by_seed[seed]
            survey = survey_by_seed.get(seed, {})
            summary = summary_by_seed.get(seed, {})
            seed_row = {
                "method": method,
                "eval_mode": args.eval_mode,
                "comment": comment,
                "seed": seed,
                "PERF_return": perf["PERF_return"],
                "PERF_success": perf["PERF_success"],
                "PERF_occurrences": perf["PERF_occurrences"],
            }
            for key in SURVEY:
                seed_row[key] = _as_float(survey.get(key))
            # Keep final retention return/success explicit so they cannot be
            # accidentally confused with PERF again.
            seed_row["final_retention_return"] = _as_float(
                summary.get("final_avg_return_all_eval_tasks")
            )
            seed_row["final_retention_success"] = _as_float(
                summary.get("final_avg_success_all_eval_tasks")
            )
            per_seed_rows.append(seed_row)

        row = {"method": method, "eval_mode": args.eval_mode, "comment": comment}

        # Correct PERF: per-seed mean over all task occurrences, then mean/std
        # over seeds.
        for metric in ("PERF_return", "PERF_success"):
            values = [perf_by_seed[s][metric] for s in sorted(perf_by_seed)]
            mu, sd, n = _stats(values)
            row[f"{metric}_mean"] = mu
            row[f"{metric}_std"] = sd
            row[f"{metric}_n"] = n

        # Retention/transfer metrics remain exactly as produced by metrics.py.
        for key in SURVEY:
            mu, sd, n = _stats(_finite(survey_rows, key))
            row[f"{key}_mean"] = mu
            row[f"{key}_std"] = sd
            row[f"{key}_n"] = n

        # Preserve useful final-retention diagnostics under unambiguous names.
        for source, alias in (
            ("final_avg_return_all_eval_tasks", "final_retention_return"),
            ("final_avg_success_all_eval_tasks", "final_retention_success"),
        ):
            mu, sd, n = _stats(_finite(summary_rows, source))
            row[f"{alias}_mean"] = mu
            row[f"{alias}_std"] = sd
            row[f"{alias}_n"] = n

        # Keep the remaining summary diagnostics without relabeling them PERF.
        for key in SUMMARY:
            mu, sd, n = _stats(_finite(summary_rows, key))
            row[f"{key}_mean"] = mu
            row[f"{key}_std"] = sd
            row[f"{key}_n"] = n

        aggregate_rows.append(row)

    if not aggregate_rows:
        raise SystemExit(f"No metric CSVs found under {root/'main'} for mode={args.eval_mode}")

    output_tag = args.eval_mode + (f"_{comment}" if comment else "")
    output = Path(args.output) if args.output else root / f"paper_metrics_{output_tag}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    per_seed_path = output.parent / f"paper_metrics_per_seed_{output_tag}.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    occurrence_path = output.parent / f"paper_PERF_occurrences_{output_tag}.csv"
    with occurrence_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_occurrence_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_occurrence_rows)

    print(output)
    print(per_seed_path)
    print(occurrence_path)
    for row in aggregate_rows:
        print(
            f"{row['method']}: "
            f"PERF_return={row['PERF_return_mean']:.4g}+-{row['PERF_return_std']:.3g}, "
            f"PERF_success={row['PERF_success_mean']:.4g}+-{row['PERF_success_std']:.3g}, "
            f"A_N={row['A_N_mean']:.4g}+-{row['A_N_std']:.3g}, "
            f"FG={row['FG_mean']:.4g}+-{row['FG_std']:.3g}, "
            f"BWT={row['BWT_mean']:.4g}+-{row['BWT_std']:.3g}, "
            f"FT_success={row['FT_success_mean']:.4g}+-{row['FT_success_std']:.3g}, "
            f"FT_return={row['FT_return_mean']:.4g}+-{row['FT_return_std']:.3g}"
        )

    plot_dir = output.parent
    _plot_perf(
        aggregate_rows,
        metric="PERF_return",
        ylabel="Mean end-of-task episodic return",
        title=f"{args.suite}: PERF over task occurrences ({output_tag})",
        output_path=plot_dir / f"paper_PERF_return_{output_tag}.png",
    )
    _plot_perf(
        aggregate_rows,
        metric="PERF_success",
        ylabel="Mean end-of-task success",
        title=f"{args.suite}: PERF over task occurrences ({output_tag})",
        output_path=plot_dir / f"paper_PERF_success_{output_tag}.png",
    )


if __name__ == "__main__":
    main()
