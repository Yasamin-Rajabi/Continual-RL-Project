"""Evaluate arbitrary continual Atari CKA-RL checkpoint roots without retraining.

Each custom condition maps to a root containing:
    seq_0/<run_name>/...
    seq_1/<run_name>/...
    ...
TensorBoard logs may live under either <root>/runs/seq_i/<run_name> or a
sibling <root.parent>/runs/seq_i/<run_name>; metrics.py supports both.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch

from atari_tasks import TASK_SUITES, get_continual_sequence
import metrics
import plots
import scratch_baselines


def _load_json_or_path(spec):
    path = pathlib.Path(spec)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return json.loads(spec)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--task-suites", nargs="+", default=["freeway", "space_invaders"],
        choices=sorted(TASK_SUITES.keys()),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument(
        "--task-sequence", nargs="+", type=int, default=None,
        help="Optional sequence override. Otherwise each Atari suite uses its default mode stream.",
    )
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)

    p.add_argument("--retention-eval-episodes", type=int, default=10)
    p.add_argument("--test-adapt-steps", type=int, default=0)
    p.add_argument("--test-adapt-lr", type=float, default=1e-2)
    p.add_argument(
        "--frozen-eval-policy", choices=["pool", "snapshot"], default="pool"
    )
    p.add_argument(
        "--eval-action-mode", choices=["deterministic", "stochastic"],
        default="deterministic",
    )
    p.add_argument(
        "--skip-forward-transfer", action="store_true",
        help="Compute A_N/FG/BWT without scratch denominators; FT remains NaN.",
    )
    p.add_argument("--success-thresholds-json", default=None)

    p.add_argument("--save-root", default="agents_atari_continual")
    p.add_argument("--runs-root", default="runs_atari")
    p.add_argument("--plots-root", default="plots_atari_custom_evaluation")
    p.add_argument("--analysis-root", default="analysis_runs_atari")

    p.add_argument("--skip-retention", action="store_true")
    p.add_argument("--skip-survey-metrics", action="store_true")
    p.add_argument(
        "--scratch-seeds", nargs="+", type=int,
        default=scratch_baselines.DEFAULT_SCRATCH_SEEDS,
    )
    p.add_argument("--scratch-save-root", default=scratch_baselines.SCRATCH_SAVE_ROOT)
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--cpu", action="store_true")

    p.add_argument(
        "--custom-models-json", required=True,
        help=(
            "JSON string or JSON file mapping condition labels to custom model roots, "
            "e.g. '{\"method_A\":\"/path/to/A\",\"method_B\":\"/path/to/B\"}'."
        ),
    )

    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be >= 1")
    if args.total_timesteps < 1 or args.retention_eval_episodes < 1:
        p.error("total_timesteps and retention_eval_episodes must be >= 1")
    if args.test_adapt_steps < 0 or args.test_adapt_lr <= 0:
        p.error("test_adapt_steps must be >= 0 and test_adapt_lr must be > 0")
    if args.test_adapt_steps and args.frozen_eval_policy != "pool":
        p.error("Test-time adaptation requires --frozen-eval-policy pool")

    args.success_thresholds = (
        {} if not args.success_thresholds_json else _load_json_or_path(args.success_thresholds_json)
    )
    args.task_sequence_override = None if args.task_sequence is None else list(args.task_sequence)
    return args


def _usable_scratch_seeds(args, suite):
    """Return scratch seeds with all FT learning curves present.

    Post-hoc FT consumes TensorBoard/CSV curves; checkpoint/runtime identity is
    a training/resume concern and is intentionally not re-checked here.
    """
    if args.skip_forward_transfer:
        return []

    usable = []
    for seed in args.scratch_seeds:
        ok = True
        for _seq_idx, task_id in _first_unseen_positions(args.task_sequence):
            directory = pathlib.Path(
                scratch_baselines.scratch_event_dir(
                    args.runs_root,
                    suite,
                    task_id,
                    args.total_timesteps,
                    seed,
                )
            )
            has_curve = (directory / "scalars.csv").is_file() or any(
                directory.glob("events.out.tfevents.*")
            )
            if not has_curve:
                ok = False
                break
        if ok:
            usable.append(seed)
    return usable


def _first_unseen_positions(task_sequence):
    seen = set()
    result = []
    for seq_idx, task_id in enumerate(task_sequence):
        if task_id not in seen and seq_idx > 0:
            result.append((seq_idx, task_id))
        seen.add(task_id)
    return result


def main():
    args = parse_args()
    pathlib.Path(args.plots_root).mkdir(parents=True, exist_ok=True)

    custom_map = _load_json_or_path(args.custom_models_json)
    if not isinstance(custom_map, dict) or not custom_map:
        raise ValueError("--custom-models-json must contain a non-empty condition->path mapping")
    metrics.set_custom_model_map(custom_map)
    conditions = list(custom_map.keys())

    sequence_by_suite = {
        suite: (
            list(args.task_sequence_override)
            if args.task_sequence_override is not None
            else list(get_continual_sequence(suite, repeats=args.repeats))
        )
        for suite in args.task_suites
    }

    config = dict(vars(args))
    config["custom_models"] = {k: str(v) for k, v in custom_map.items()}
    config["resolved_sequences"] = sequence_by_suite
    config["task_sequence"] = args.task_sequence_override
    with (pathlib.Path(args.plots_root) / "benchmark_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Evaluation device: {device}")
    print(f"Custom conditions: {conditions}")

    for suite in args.task_suites:
        args.task_sequence = list(sequence_by_suite[suite])
        for task_id in args.task_sequence:
            if task_id < 0 or task_id >= len(TASK_SUITES[suite]):
                raise ValueError(f"task_id {task_id} is invalid for {suite}")

        print(f"\n================ {suite} ================")
        print(f"Sequence: {args.task_sequence}")

        plots.plot_sequence_diagnostics(args, suite, conditions)
        plots.plot_merge_lineage(args, suite, conditions)
        plots.plot_zero_shot(args, suite, conditions)

        if not args.skip_retention:
            all_payloads = {condition: [] for condition in conditions}
            for condition in conditions:
                for seed in args.seeds:
                    all_payloads[condition].append(
                        metrics.build_retention_matrix(args, suite, condition, seed, device)
                    )
            plots.plot_retention(args, suite, conditions, all_payloads)
            plots.write_summary_csv(args, suite, conditions, all_payloads)

        if not args.skip_survey_metrics:
            scratch_seeds = _usable_scratch_seeds(args, suite)
            if not scratch_seeds:
                reason = (
                    "--skip-forward-transfer was set"
                    if args.skip_forward_transfer
                    else "no complete scratch learning-curve denominator is available"
                )
                print(
                    f"{reason} for {suite}; A_N/FG/BWT will be computed and "
                    "FT metrics will be NaN."
                )
            survey_payloads = {condition: [] for condition in conditions}
            for condition in conditions:
                for seed in args.seeds:
                    survey_payloads[condition].append(
                        metrics.compute_survey_metrics(
                            args, suite, condition, seed, device,
                            scratch_seeds, args.total_timesteps,
                        )
                    )
            plots.plot_survey_metrics(args, suite, conditions, survey_payloads)
            plots.write_survey_metrics_csv(args, suite, conditions, survey_payloads)

    print(f"\nDone. Custom Atari evaluation outputs: {args.plots_root}")


if __name__ == "__main__":
    main()
