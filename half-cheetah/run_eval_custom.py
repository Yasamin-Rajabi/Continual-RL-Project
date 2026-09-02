"""Custom evaluation and metrics script for arbitrary model paths,
hyperparameters, and custom test-adaptation steps (e.g. warmup steps).
Does not touch training logic; purely for metrics calculation and plotting.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import OrderedDict

import torch

from tasks import DEFAULT_CONTINUAL_SEQUENCE, TASK_SUITES
import metrics
import plots
import scratch_baselines


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--task-suites", nargs="+",
        default=["halfcheetah_vel", "halfcheetah_wind_vel"],
        choices=sorted(TASK_SUITES.keys()),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[101])
    p.add_argument("--task-sequence", nargs="+", type=int, default=list(DEFAULT_CONTINUAL_SEQUENCE))
    p.add_argument("--total-timesteps", type=int, default=80_000)

    # Metric and test-time adaptation controls.
    p.add_argument("--retention-eval-episodes", type=int, default=3)
    p.add_argument("--test-adapt-steps", type=int, default=1000,
                   help="Number of alpha-only adaptation steps before retention/final evaluation; 0 disables it.")
    p.add_argument("--test-adapt-lr", type=float, default=1e-2,
                   help="Learning rate for test-time alpha adaptation.")

    p.add_argument("--save-root", default="agents_halfcheetah")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--plots-root", default="plots_custom_evaluation")
    p.add_argument("--analysis-root", default="analysis_runs")

    p.add_argument("--skip-retention", action="store_true")
    p.add_argument("--skip-survey-metrics", action="store_true")
    p.add_argument(
        "--scratch-seeds", nargs="+", type=int, default=scratch_baselines.DEFAULT_SCRATCH_SEEDS,
    )
    p.add_argument(
        "--scratch-save-root", default=scratch_baselines.SCRATCH_SAVE_ROOT,
    )
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--cpu", action="store_true")

    # Map user-defined condition names to model-root directories.
    p.add_argument(
        "--custom-models-json", required=True,
        help="JSON string or path to JSON file containing custom condition-to-path dictionary, "
             "e.g. '{\"Aware_Seed101\": \"/path/to/aware\", \"Agnostic_Seed101\": \"/path/to/agnostic\"}'"
    )

    args = p.parse_args()
    if args.test_adapt_steps < 0 or args.test_adapt_lr <= 0:
        p.error("--test-adapt-steps must be >=0 and --test-adapt-lr must be >0")
    return args


def main():
    args = parse_args()
    pathlib.Path(args.plots_root).mkdir(parents=True, exist_ok=True)

    custom_path = pathlib.Path(args.custom_models_json)
    if custom_path.exists():
        with open(custom_path) as f:
            custom_map = json.load(f)
    else:
        custom_map = json.loads(args.custom_models_json)

    metrics.set_custom_model_map(custom_map)

    conditions = list(custom_map.keys())
    print(f"Custom evaluation conditions / hyperparameters: {conditions}")

    with open(pathlib.Path(args.plots_root) / "benchmark_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Evaluation device: {device}")
    print(f"Sequence: {args.task_sequence}")
    print(f"Test adaptation warmup steps: {args.test_adapt_steps}")

    for suite in args.task_suites:
        print(f"\n================ {suite} ================")

        # Evaluation only: no training is launched from this script.

        # Diagnostic plots using the custom checkpoint paths.
        plots.plot_sequence_diagnostics(args, suite, conditions)
        plots.plot_zero_shot(args, suite, conditions)

        # Retention matrix with the requested test-time adaptation budget.
        if not args.skip_retention:
            all_payloads = {condition: [] for condition in conditions}
            for condition in conditions:
                for seed in args.seeds:
                    print(f"Building retention matrix for [{condition}] with seed {seed}...")
                    all_payloads[condition].append(
                        metrics.build_retention_matrix(args, suite, condition, seed, device)
                    )
            plots.plot_retention(args, suite, conditions, all_payloads)
            plots.write_summary_csv(args, suite, conditions, all_payloads)

        # Survey metrics: A_N, FG, BWT, and FT.
        if not args.skip_survey_metrics:
            used_task_ids = sorted(set(args.task_sequence))
            missing_baselines = [
                task_id for task_id in used_task_ids
                for seed in args.scratch_seeds
                if not scratch_baselines.checkpoint_complete(
                    scratch_baselines.scratch_checkpoint_dir(
                        args.scratch_save_root, suite, task_id, args.total_timesteps, seed,
                    )
                )
            ]
            if missing_baselines:
                print(
                    f"\n!!! Skipping survey metrics for {suite}: missing scratch baselines for "
                    f"task_id(s) {sorted(set(missing_baselines))}."
                )
            else:
                survey_payloads = {condition: [] for condition in conditions}
                for condition in conditions:
                    for seed in args.seeds:
                        print(f"Computing survey metrics for [{condition}] with seed {seed}...")
                        survey_payloads[condition].append(
                            metrics.compute_survey_metrics(
                                args, suite, condition, seed, device,
                                args.scratch_seeds, args.total_timesteps,
                            )
                        )
                plots.plot_survey_metrics(args, suite, conditions, survey_payloads)
                plots.write_survey_metrics_csv(args, suite, conditions, survey_payloads)

    print(f"\nDone. Custom evaluation plots and cached metrics saved to: {args.plots_root}")


if __name__ == "__main__":
    main()
