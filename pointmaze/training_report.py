"""Consolidate every per-task training log into one JSON + a set of plots.

The benchmark already logs a great deal -- alpha entropy, alpha-mass, per-pool
routing weights, actor/critic losses, returns, distillation KLs, merge
decisions, timings -- but it scatters them across one ``scalars.csv`` per task
plus a few JSON files per checkpoint, which is why a finished run is hard to
read.  This script gathers all of it into:

    <out>/training_report.json   every curve (downsampled) + per-task summary
    <out>/*.png                  the plots that matter for diagnosis
    stdout                       a short list of the things that look wrong

What to look at first
---------------------
``analysis/routing/regret`` (plotted as *routing regret*) is the measurement
that separates the two failure modes that look identical in the headline
numbers:

    regret ~ 0   the mixture is already routing to the component the critic
                 rates highest, so a bad score for that task is a training
                 ceiling -- more steps, or an easier objective, would help,
                 and changing the routing machinery would not.

    regret >> 0  the mixture is putting its mass somewhere the critic itself
                 rates worse.  That is a routing failure and wants a fix in
                 the alpha/alpha-mass path, not a bigger budget.

Runs that pre-date the routing telemetry simply have no regret curve; every
other plot still works.

Usage:
    python training_report.py                        # defaults, tag=main
    python training_report.py --tag alpha_fix_smoke --methods Ours
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict

MAX_POINTS = 400  # per curve, after downsampling

# Categorical slots, fixed order, never cycled (blue, orange, aqua, yellow).
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# Sequential blue ramp for the ordered "task index" encoding.  Steps 250-700
# only: the lighter end of the ramp is for near-zero magnitude on a heatmap,
# and a 2px line drawn in it is not readable on a light surface.
SEQ = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#e4e3dd"

CURVE_TAGS = {
    "alpha_mass": "analysis/policy/alpha_mass",
    "alpha_mass_raw": "analysis/policy/alpha_mass_raw",
    "alpha_entropy": "analysis/policy/alpha_entropy",
    "alpha_max": "analysis/policy/alpha_max",
    "pool_length": "analysis/policy/pool_length",
    "actor_loss": "losses/actor",
    "critic_loss": "losses/critic",
    "ent_coef": "losses/ent_coef",
    "episodic_return": "charts/episodic_return",
    "episodic_success": "charts/episodic_success",
    "test_return": "charts/test_return",
    "sps": "charts/SPS",
    "routing_regret": "analysis/routing/regret",
    "q_routed": "analysis/routing/q_routed",
    "q_best": "analysis/routing/q_best",
    "q_novel": "analysis/routing/q_novel",
    "q_novel_minus_best_historical": "analysis/routing/q_novel_minus_best_historical",
    "weight_on_argmax": "analysis/routing/weight_on_argmax",
    "own_norm": "analysis/policy/own_norm",
}
SCALAR_TAGS = {
    "zero_shot_return": "charts/zero_shot_return",
    "zero_shot_success": "charts/zero_shot_success",
    "final_return": "charts/final_return",
    "final_success": "charts/final_success",
    "train_loop_seconds": "timing/train_loop_seconds",
    "finalize_seconds": "timing/finalize_seconds",
    "pool_final_length": "analysis/pool/final_length",
}
DISTILL_PREFIX = "distillation/"
MERGE_PREFIX = "analysis/merge/"


# ---------------------------------------------------------------- reading
def read_scalars(csv_path):
    """{tag: [(step, value), ...]} from one scalars.csv, steps ascending."""
    series = defaultdict(list)
    if not csv_path.is_file():
        return series
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            try:
                series[row["tag"]].append((int(row["step"]), float(row["value"])))
            except (TypeError, ValueError, KeyError):
                continue
    for tag in series:
        series[tag].sort(key=lambda p: p[0])
    return series


def downsample(points, limit=MAX_POINTS):
    if len(points) <= limit:
        return [[s, v] for s, v in points]
    stride = math.ceil(len(points) / limit)
    kept = points[::stride]
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return [[s, v] for s, v in kept]


def last_value(series, tag):
    points = series.get(tag)
    return None if not points else points[-1][1]


def first_value(series, tag):
    points = series.get(tag)
    return None if not points else points[0][1]


def collect(runs_root, save_root, results_root, suite, tag, methods, seeds):
    runs_base = pathlib.Path(runs_root) / tag
    report = {"suite": suite, "tag": tag, "methods": {}}
    if not runs_base.is_dir():
        raise FileNotFoundError(f"no run logs under {runs_base}")

    try:
        from tasks import get_task_family, get_task_name
    except Exception:  # running outside the package
        get_task_name = get_task_family = None

    found = defaultdict(set)
    for run_dir in sorted(runs_base.iterdir()):
        # <suite>__task_<id>__<method>__<seed>
        parts = run_dir.name.split("__")
        if len(parts) != 4 or not parts[1].startswith("task_"):
            continue
        found[parts[2]].add(int(parts[3]))

    for method in sorted(found):
        if methods and method not in methods:
            continue
        for seed in sorted(found[method]):
            if seeds and seed not in seeds:
                continue
            entry = {"per_task": {}, "metrics": None}

            metrics_path = (
                pathlib.Path(results_root) / f"{suite}__{method}__seed{seed}__metrics.json"
            )
            if metrics_path.is_file():
                with metrics_path.open() as f:
                    entry["metrics"] = json.load(f)

            for run_dir in sorted(runs_base.glob(f"{suite}__task_*__{method}__{seed}")):
                task_id = int(run_dir.name.split("__")[1].split("_")[1])
                series = read_scalars(run_dir / "scalars.csv")

                curves = {}
                for name, csv_tag in CURVE_TAGS.items():
                    if series.get(csv_tag):
                        curves[name] = downsample(series[csv_tag])
                # Per-pool-member routing weights, however many there are.
                weights = {}
                for csv_tag, points in series.items():
                    if csv_tag.startswith("analysis/policy/alpha_weight_"):
                        weights[csv_tag.rsplit("_", 1)[1]] = downsample(points)
                    elif csv_tag.startswith("analysis/routing/q_component_"):
                        curves.setdefault("q_components", {})[
                            csv_tag.rsplit("_", 1)[1]
                        ] = downsample(points)
                if weights:
                    curves["alpha_weights"] = weights

                summary = {k: last_value(series, t) for k, t in SCALAR_TAGS.items()}
                summary["zero_shot_return"] = first_value(series, SCALAR_TAGS["zero_shot_return"])
                summary["alpha_mass_start"] = first_value(series, CURVE_TAGS["alpha_mass"])
                summary["alpha_mass_end"] = last_value(series, CURVE_TAGS["alpha_mass"])
                summary["alpha_entropy_start"] = first_value(series, CURVE_TAGS["alpha_entropy"])
                summary["alpha_entropy_end"] = last_value(series, CURVE_TAGS["alpha_entropy"])
                summary["routing_regret_end"] = last_value(series, CURVE_TAGS["routing_regret"])
                summary["actor_loss_end"] = last_value(series, CURVE_TAGS["actor_loss"])
                summary["critic_loss_end"] = last_value(series, CURVE_TAGS["critic_loss"])

                distill = {
                    t[len(DISTILL_PREFIX):]: v[-1][1]
                    for t, v in series.items()
                    if t.startswith(DISTILL_PREFIX) and v
                }
                if distill.get("policy/distill_test_kl") is not None and \
                        distill.get("policy/distill_train_kl") is not None:
                    distill["generalization_gap"] = (
                        distill["policy/distill_test_kl"] - distill["policy/distill_train_kl"]
                    )
                merge = {
                    t[len(MERGE_PREFIX):]: v[-1][1]
                    for t, v in series.items()
                    if t.startswith(MERGE_PREFIX) and v
                }

                # merge_info.json carries the non-scalar detail (lineages).
                merge_json = None
                ckpt = (
                    pathlib.Path(save_root) / suite / tag / method / f"seed_{seed}"
                )
                for candidate in sorted(ckpt.glob(f"seq_*/task_{task_id}/merge_info.json")):
                    with candidate.open() as f:
                        merge_json = json.load(f)

                task = {
                    "task_id": task_id,
                    "summary": summary,
                    "curves": curves,
                    "distillation": distill or None,
                    "merge": merge or None,
                    "merge_detail": merge_json,
                }
                if get_task_name is not None:
                    try:
                        task["task_name"] = get_task_name(task_id, suite)
                        task["family"] = get_task_family(task_id, suite)
                    except Exception:
                        pass
                entry["per_task"][str(task_id)] = task

            report["methods"].setdefault(method, {})[f"seed_{seed}"] = entry
    return report


# ---------------------------------------------------------------- plotting
def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _overlay(plt, tasks, curve, out_path, title, ylabel, logy=False):
    """One axes, one line per task, sequential ramp + direct end labels."""
    items = [(tid, t["curves"][curve]) for tid, t in tasks if curve in t["curves"]]
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    for order, (tid, points) in enumerate(items):
        color = SEQ[min(int(tid), len(SEQ) - 1)]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, linewidth=2.0, color=color, label=f"task {tid}")
        if xs:
            ax.annotate(f"{tid}", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(4, 0), fontsize=7, color=MUTED, va="center")
    if logy:
        ax.set_yscale("log")
    _style(ax, title, "environment step", ylabel)
    ax.legend(fontsize=7, frameon=False, ncol=2, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="#fcfcfb")
    plt.close(fig)
    return out_path


def _small_multiples(plt, tasks, curve, out_path, title, ylabel):
    items = [(tid, t["curves"][curve]) for tid, t in tasks if curve in t["curves"]]
    if not items:
        return None
    cols = min(5, len(items))
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(max(7.5, 3.0 * cols), 2.3 * rows),
                             dpi=140, squeeze=False)
    for idx, (tid, points) in enumerate(items):
        ax = axes[idx // cols][idx % cols]
        ax.plot([p[0] for p in points], [p[1] for p in points],
                linewidth=1.8, color=CAT[0])
        _style(ax, f"task {tid}", "", ylabel if idx % cols == 0 else "")
    for idx in range(len(items), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, facecolor="#fcfcfb")
    plt.close(fig)
    return out_path


def _grouped_bars(plt, labels, groups, out_path, title, ylabel):
    """groups: [(name, [values aligned to labels]), ...] -- max 4 series."""
    groups = [(n, v) for n, v in groups if any(x is not None for x in v)]
    if not groups:
        return None
    n_series = len(groups)
    width = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(labels)), 4.6), dpi=140)
    for si, (name, values) in enumerate(groups):
        xs = [i - 0.4 + width * (si + 0.5) for i in range(len(labels))]
        ys = [0 if v is None else v for v in values]
        ax.bar(xs, ys, width=width * 0.92, color=CAT[si % len(CAT)], label=name)
        # Contrast WARN on two of these hues obliges visible labels.
        for x, v in zip(xs, values):
            if v is None:
                continue
            ax.annotate(f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}",
                        (x, v), textcoords="offset points",
                        xytext=(0, 3 if v >= 0 else -10),
                        ha="center", fontsize=6, color=MUTED)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, color=MUTED)
    ax.axhline(0, color=GRID, linewidth=1.0)
    _style(ax, title, "", ylabel)
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="#fcfcfb")
    plt.close(fig)
    return out_path


def _stacked_weights(plt, tasks, out_path):
    items = [(tid, t["curves"]["alpha_weights"]) for tid, t in tasks
             if "alpha_weights" in t["curves"]]
    if not items:
        return None
    cols = min(5, len(items))
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(max(7.5, 3.0 * cols), 2.3 * rows),
                             dpi=140, squeeze=False)
    for idx, (tid, weights) in enumerate(items):
        ax = axes[idx // cols][idx % cols]
        keys = sorted(weights, key=lambda k: int(k))
        steps = [p[0] for p in weights[keys[0]]]
        stack = [[p[1] for p in weights[k]] for k in keys]
        n = min(len(steps), *(len(s) for s in stack))
        ax.stackplot(steps[:n], *[s[:n] for s in stack],
                     colors=[SEQ[min(int(k) * 2, len(SEQ) - 1)] for k in keys],
                     labels=[f"pool {k}" for k in keys], edgecolor="#fcfcfb",
                     linewidth=0.5)
        ax.set_ylim(0, 1)
        _style(ax, f"task {tid}", "", "weight" if idx % cols == 0 else "")
        if idx == 0:
            ax.legend(fontsize=6, frameon=False, labelcolor=MUTED, loc="upper left")
    for idx in range(len(items), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle("Routing weight within the historical pool "
                 "(softmax over alpha, before the alpha-mass gate)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, facecolor="#fcfcfb")
    plt.close(fig)
    return out_path


def make_plots(report, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available; wrote JSON only")
        return []

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for method, seeds in report["methods"].items():
        for seed_key, entry in seeds.items():
            prefix = f"{method}__{seed_key}"
            tasks = sorted(entry["per_task"].items(), key=lambda kv: int(kv[0]))
            labels = [f"t{tid}" for tid, _ in tasks]
            metrics = entry.get("metrics") or {}

            written += [p for p in (
                _overlay(plt, tasks, "alpha_mass", out_dir / f"{prefix}__alpha_mass.png",
                         "alpha-mass (effective) during training — 1.0 = trust the "
                         "historical pool, 0.0 = trust the fresh expert",
                         "alpha_mass", logy=True),
                _overlay(plt, tasks, "alpha_entropy", out_dir / f"{prefix}__alpha_entropy.png",
                         "Entropy of the routing distribution over the historical pool",
                         "entropy (nats)"),
                _overlay(plt, tasks, "routing_regret", out_dir / f"{prefix}__routing_regret.png",
                         "Routing regret: max_k Q_k − Σ_k w_k Q_k  (0 = routing is optimal)",
                         "regret (Q units)"),
                _overlay(plt, tasks, "q_novel_minus_best_historical",
                         out_dir / f"{prefix}__q_novel_advantage.png",
                         "How much better the fresh expert is than the best frozen one "
                         "(critic's own estimate)", "ΔQ"),
                _small_multiples(plt, tasks, "episodic_return",
                                 out_dir / f"{prefix}__return_curves.png",
                                 "Training return per task", "return"),
                _small_multiples(plt, tasks, "actor_loss",
                                 out_dir / f"{prefix}__actor_loss.png",
                                 "Actor loss per task", "loss"),
                _small_multiples(plt, tasks, "critic_loss",
                                 out_dir / f"{prefix}__critic_loss.png",
                                 "Critic loss per task", "loss"),
                _stacked_weights(plt, tasks, out_dir / f"{prefix}__alpha_weights.png"),
            ) if p]

            # Where each task ended up, four ways.
            peak = metrics.get("peak_per_task") or []
            final = metrics.get("final_per_task") or []
            p = _grouped_bars(
                plt, labels,
                [("zero-shot (before training it)",
                  [t["summary"].get("zero_shot_return") for _, t in tasks]),
                 ("end of its own training",
                  [t["summary"].get("final_return") for _, t in tasks]),
                 ("peak (scored, no adaptation)",
                  [peak[i] if i < len(peak) else None for i in range(len(tasks))]),
                 ("final (end of chain, adapted)",
                  [final[i] if i < len(final) else None for i in range(len(tasks))])],
                out_dir / f"{prefix}__per_task_returns.png",
                "Return per task at four moments", "return")
            if p:
                written.append(p)

            # Distillation quality, one group per merge that happened.
            d_labels, d_train, d_test, d_p95 = [], [], [], []
            for tid, t in tasks:
                d = t.get("distillation") or {}
                if d.get("policy/distill_test_kl") is None:
                    continue
                d_labels.append(f"t{tid}")
                d_train.append(d.get("policy/distill_train_kl"))
                d_test.append(d.get("policy/distill_test_kl"))
                d_p95.append(d.get("policy/distill_test_kl_p95"))
            if d_labels:
                p = _grouped_bars(
                    plt, d_labels,
                    [("train KL", d_train), ("test KL", d_test), ("test KL p95", d_p95)],
                    out_dir / f"{prefix}__distillation.png",
                    "Merge quality: KL of the distilled student against its two parents "
                    "(lower is a cleaner merge; test ≫ train means it memorised)",
                    "symmetric KL")
                if p:
                    written.append(p)

            p = _grouped_bars(
                plt, labels,
                [("training loop", [t["summary"].get("train_loop_seconds") for _, t in tasks]),
                 ("finalize / merge", [t["summary"].get("finalize_seconds") for _, t in tasks])],
                out_dir / f"{prefix}__wallclock.png",
                "Wall-clock seconds per task", "seconds")
            if p:
                written.append(p)
    return written


# ---------------------------------------------------------------- summary
def print_summary(report):
    for method, seeds in report["methods"].items():
        for seed_key, entry in seeds.items():
            print(f"\n===== {method} / {seed_key} =====")
            tasks = sorted(entry["per_task"].items(), key=lambda kv: int(kv[0]))
            metrics = entry.get("metrics") or {}
            peak = metrics.get("peak_per_task") or []
            final = metrics.get("final_per_task") or []

            head = (f"{'task':<6}{'name':<26}{'zero_shot':>11}{'own_end':>10}"
                    f"{'peak':>10}{'final':>10}{'a_mass_end':>12}{'regret':>9}")
            print(head)
            print("-" * len(head))
            for tid, t in tasks:
                s = t["summary"]
                i = int(tid)

                def f(v, nd=2):
                    return "n/a" if v is None else f"{v:.{nd}f}"

                print(f"{tid:<6}{(t.get('task_name') or '')[:25]:<26}"
                      f"{f(s.get('zero_shot_return')):>11}{f(s.get('final_return')):>10}"
                      f"{f(peak[i] if i < len(peak) else None):>10}"
                      f"{f(final[i] if i < len(final) else None):>10}"
                      f"{f(s.get('alpha_mass_end'), 6):>12}"
                      f"{f(s.get('routing_regret_end')):>9}")

            flags = []
            for tid, t in tasks:
                s, i = t["summary"], int(tid)
                own = s.get("final_return")
                pk = peak[i] if i < len(peak) else None
                if own is not None and pk is not None and own - pk > 20:
                    flags.append(
                        f"task {tid}: scored peak ({pk:.1f}) is much worse than what "
                        f"training itself reached ({own:.1f}) -> look at the merge/"
                        f"projection step, not the training budget")
                fn = final[i] if i < len(final) else None
                # The final stage's own task cannot have been forgotten yet,
                # which is why metrics.py excludes it from forgetting too.
                last_stage = (metrics.get("num_tasks") or len(tasks)) - 1
                if pk is not None and fn is not None and pk - fn > 50 and i < last_stage:
                    flags.append(
                        f"task {tid}: lost {pk - fn:.0f} return between its peak and the "
                        f"end of the chain -> its expertise is no longer represented in "
                        f"the pool that the final checkpoint routes over")
                regret = s.get("routing_regret_end")
                if regret is not None and regret > 1.0:
                    flags.append(
                        f"task {tid}: routing regret still {regret:.2f} at the end -> "
                        f"the mixture is not routing to the component the critic prefers")
                am = s.get("alpha_mass_end")
                if am is not None and s.get("alpha_mass_start") is not None \
                        and abs(am - s["alpha_mass_start"]) < 1e-3:
                    flags.append(f"task {tid}: alpha_mass never moved ({am:.4f})")
                d = t.get("distillation") or {}
                gap = d.get("generalization_gap")
                if gap is not None and d.get("policy/distill_train_kl"):
                    ratio = d["policy/distill_test_kl"] / max(d["policy/distill_train_kl"], 1e-9)
                    if ratio > 3:
                        flags.append(
                            f"task {tid}: distillation test KL is {ratio:.1f}x the train KL "
                            f"-> the merged head fits its buffer, not the parents' behaviour")
            print("\nflags:" if flags else "\nflags: none")
            for line in flags:
                print(f"  - {line}")


def build_parser():
    p = argparse.ArgumentParser(description="Consolidated PointMaze training report")
    p.add_argument("--runs-root", default="runs_pointmaze")
    p.add_argument("--save-root", default="agents_pointmaze")
    p.add_argument("--results-root", default="results_pointmaze")
    p.add_argument("--suite", default="pointmaze_goal")
    p.add_argument("--tag", default="main")
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--out", default="report_pointmaze")
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = collect(args.runs_root, args.save_root, args.results_root,
                     args.suite, args.tag, args.methods, args.seeds)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "training_report.json"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {json_path}")

    if not args.no_plots:
        for path in make_plots(report, out_dir):
            print(f"wrote {path}")

    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
