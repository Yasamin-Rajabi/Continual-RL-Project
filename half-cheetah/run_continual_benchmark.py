"""Four-way continual benchmark for HalfCheetahVel and HalfCheetahWindVel.

The four experimental cases intentionally differ only along two method axes:

    baseline      = classic CKA vectors + arithmetic merge
    distil_only  = classic CKA vectors + KL distillation merge
    weight_only   = weight-delta vectors + alpha-mass + arithmetic merge
    combined      = weight-delta vectors + alpha-mass + KL distillation merge

All four cases use the SAME output-space merge-pair selector: the pair with the
lowest symmetric KL between their full Gaussian policy outputs (mean + log-std)
on a balanced subset of states stored in the knowledge pool. This keeps the
pair-selection rule controlled while isolating what the four cases are meant to
compare.

This file only ORCHESTRATES: it defines the experiment config (CONDITIONS,
argparse), runs training (train_chain, one subprocess call to run_sac.py per
task), and calls into metrics.py / plots.py for everything else:

  - metrics.py computes every number: the full retention matrix (checkpoint x
    unique task, used by the heatmap-style diagnostic plots) AND the four
    survey metrics -- A_N, FG, BWT, FT (two variants) -- per the CRL survey's
    Eq. 7-10. See metrics.py's module docstring for exact formulas and which
    TensorBoard scalar backs p_i(t).
  - plots.py draws every PNG/CSV from whatever metrics.py computed. No
    training, no environment rollouts, no checkpoint loading happens there.

Forward transfer needs a from-scratch, single-task baseline per unique
task_id -- see scratch_baselines.py, which trains and caches those
separately (run it once before this script, or before calling this script's
survey-metrics step, with the SAME --total-timesteps).
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
from collections import OrderedDict

import torch

from tasks import DEFAULT_CONTINUAL_SEQUENCE, TASK_SUITES, get_task_name
import metrics
import plots
import scratch_baselines


CONDITIONS = OrderedDict([
    (
        "baseline",
        {"fusion_mode": "classic_cka", "distillation": False, "use_alpha_mass": False},
    ),
    (
        "distil_only",
        {"fusion_mode": "classic_cka", "distillation": True, "use_alpha_mass": False},
    ),
    (
        "weight_only",
        {"fusion_mode": "weight_delta", "distillation": False, "use_alpha_mass": True},
    ),
    (
        "combined",
        {"fusion_mode": "weight_delta", "distillation": True, "use_alpha_mass": True},
    ),
])


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--task-suites", nargs="+",
        default=["halfcheetah_vel", "halfcheetah_wind_vel"],
        choices=sorted(TASK_SUITES.keys()),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--task-sequence", nargs="+", type=int, default=list(DEFAULT_CONTINUAL_SEQUENCE))
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--random-actions-end", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--q-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--num-evals", type=int, default=5)
    p.add_argument("--retention-eval-episodes", type=int, default=3)
    p.add_argument("--distill-extra-steps", type=int, default=10_000)
    p.add_argument("--max-distill-buffer", type=int, default=50_000)
    p.add_argument("--similarity-samples", type=int, default=2_048)
    p.add_argument("--distill-max-samples", type=int, default=20_000)
    p.add_argument("--distill-epochs", type=int, default=8)
    p.add_argument("--distill-lr", type=float, default=3e-4)
    p.add_argument("--distill-batch-size", type=int, default=256)
    p.add_argument("--distill-test-frac", type=float, default=0.2)
    p.add_argument("--analysis-log-every", type=int, default=5_000)
    p.add_argument("--save-root", default="agents_halfcheetah")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--plots-root", default="plots_halfcheetah_continual")
    p.add_argument("--analysis-root", default="analysis_runs")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-retention", action="store_true")
    p.add_argument("--skip-survey-metrics", action="store_true")
    p.add_argument(
        "--scratch-seeds", nargs="+", type=int, default=scratch_baselines.DEFAULT_SCRATCH_SEEDS,
        help="Must match the seeds scratch_baselines.py was run with.",
    )
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--train-shared", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--condition-index", type=int, default=0, choices=[0, 1, 2, 3, 4],
        help="0 = run all 4 CONDITIONS. 1-4 = run only that one condition, "
             "by position in CONDITIONS' insertion order "
             "(1=baseline, 2=distil_only, 3=weight_only, 4=combined).",
    )
    p.add_argument("--quick-test", action="store_true")
    args = p.parse_args()

    if args.quick_test:
        # Exercises at least one merge without committing to the full paper run.
        args.task_suites = ["halfcheetah_vel"]
        args.seeds = [1]
        args.task_sequence = [0, 1, 2, 3]
        args.total_timesteps = 20_000
        args.learning_starts = 1_000
        args.random_actions_end = 2_000
        args.pool_size = 2
        args.eval_every = 5_000
        args.num_evals = 1
        args.retention_eval_episodes = 1
        args.distill_extra_steps = 1_000
        args.max_distill_buffer = 4_000
        args.similarity_samples = 256
        args.distill_max_samples = 1_000
        args.distill_epochs = 2
        args.analysis_log_every = 2_000

    return args


# ==========================================================================
# TRAINING: the only thing this file still does directly.
# ==========================================================================
def train_chain(args, suite, condition, cfg, seed):
    previous = []
    for seq_idx, task_id in enumerate(args.task_sequence):
        if task_id < 0 or task_id >= len(TASK_SUITES[suite]):
            raise ValueError(f"task_id {task_id} is invalid for {suite}")

        save_parent = metrics.checkpoint_dir(args.save_root, suite, condition, seed, seq_idx, task_id).parent
        run_dir = metrics.checkpoint_dir(args.save_root, suite, condition, seed, seq_idx, task_id)
        tb_dir = metrics.event_dir(args.runs_root, suite, condition, seed, seq_idx, task_id)
        if args.force_retrain:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            if tb_dir.exists():
                shutil.rmtree(tb_dir)

        if metrics.checkpoint_complete(run_dir):
            print(f"[{suite}/{condition}/seed={seed}] seq{seq_idx} already complete: {run_dir}")
            previous.append(run_dir)
            continue

        if args.skip_training:
            raise FileNotFoundError(f"Missing checkpoint while --skip-training was set: {run_dir}")

        # Remove partial outputs before a retry, otherwise TensorBoard can mix
        # stale and fresh event files from two different attempts.
        if run_dir.exists():
            shutil.rmtree(run_dir)
        if tb_dir.exists():
            shutil.rmtree(tb_dir)

        tag = f"{suite}/{condition}/seed_{seed}/seq_{seq_idx}"
        cmd = [
            "python3", "run_sac.py",
            "--model-type=cka-rl",
            f"--task-suite={suite}",
            f"--task-id={task_id}",
            f"--seed={seed}",
            f"--tag={tag}",
            f"--save-dir={save_parent}",
            f"--analysis-root={args.analysis_root}",
            f"--total-timesteps={args.total_timesteps}",
            f"--learning-starts={args.learning_starts}",
            f"--random-actions-end={args.random_actions_end}",
            f"--batch-size={args.batch_size}",
            f"--policy-lr={args.policy_lr}",
            f"--q-lr={args.q_lr}",
            f"--gamma={args.gamma}",
            f"--tau={args.tau}",
            f"--pool-size={args.pool_size}",
            f"--eval-every={args.eval_every}",
            f"--num-evals={args.num_evals}",
            f"--distill-extra-steps={args.distill_extra_steps}",
            f"--max-distill-buffer={args.max_distill_buffer}",
            f"--similarity-samples={args.similarity_samples}",
            f"--distill-max-samples={args.distill_max_samples}",
            f"--distill-epochs={args.distill_epochs}",
            f"--distill-lr={args.distill_lr}",
            f"--distill-batch-size={args.distill_batch_size}",
            f"--distill-test-frac={args.distill_test_frac}",
            f"--analysis-log-every={args.analysis_log_every}",
            f"--fusion-mode={cfg['fusion_mode']}",
            "--no-use-alpha-scale",
            "--distillation" if cfg["distillation"] else "--no-distillation",
            "--train-shared" if args.train_shared else "--no-train-shared",
            "--use-alpha-mass" if cfg["use_alpha_mass"] else "--no-use-alpha-mass",
        ]
        if args.cpu:
            cmd.append("--no-cuda")

        if previous:
            # run_sac only needs the immutable root and latest continual state.
            prev_args = [previous[0]] if len(previous) == 1 else [previous[0], previous[-1]]
            cmd.append("--prev-units")
            cmd.extend(str(p) for p in prev_args)

        print(
            f"\n>>> {suite} | {condition} | seed {seed} | seq{seq_idx} "
            f"task {task_id}: {get_task_name(task_id, suite)} <<<"
        )
        subprocess.run(cmd, check=True)
        if not metrics.checkpoint_complete(run_dir):
            raise RuntimeError(f"Training command finished but checkpoint is incomplete: {run_dir}")
        previous.append(run_dir)
    return previous


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    args = parse_args()
    pathlib.Path(args.plots_root).mkdir(parents=True, exist_ok=True)
    import json
    with open(pathlib.Path(args.plots_root) / "benchmark_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Evaluation device: {device}")
    print(f"Sequence: {args.task_sequence}")

    all_condition_names = list(CONDITIONS.keys())
    if args.condition_index == 0:
        conditions = all_condition_names
    else:
        conditions = [all_condition_names[args.condition_index - 1]]
    selected_conditions = {name: CONDITIONS[name] for name in conditions}
    print(f"Conditions: {conditions}")

    for suite in args.task_suites:
        print(f"\n================ {suite} ================")
        for condition, cfg in selected_conditions.items():
            for seed in args.seeds:
                train_chain(args, suite, condition, cfg, seed)

        plots.plot_training_metrics(args, suite, conditions)
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
            missing_baselines = [
                task_id for task_id in range(len(TASK_SUITES[suite]))
                for seed in args.scratch_seeds
                if not scratch_baselines.checkpoint_complete(
                    scratch_baselines.scratch_checkpoint_dir(
                        scratch_baselines.SCRATCH_SAVE_ROOT, suite, task_id, args.total_timesteps, seed,
                    )
                )
            ]
            if missing_baselines:
                print(
                    f"\n!!! Skipping survey metrics for {suite}: missing scratch baselines for "
                    f"task_id(s) {sorted(set(missing_baselines))}. Run:\n"
                    f"    python3 scratch_baselines.py --task-suites {suite} "
                    f"--total-timesteps {args.total_timesteps} --seeds {' '.join(map(str, args.scratch_seeds))}\n"
                )
            else:
                survey_payloads = {condition: [] for condition in conditions}
                for condition in conditions:
                    for seed in args.seeds:
                        survey_payloads[condition].append(
                            metrics.compute_survey_metrics(
                                args, suite, condition, seed, device,
                                args.scratch_seeds, args.total_timesteps,
                            )
                        )
                plots.plot_survey_metrics(args, suite, conditions, survey_payloads)
                plots.write_survey_metrics_csv(args, suite, conditions, survey_payloads)

    print(f"\nDone. Plots and cached metrics: {args.plots_root}")


if __name__ == "__main__":
    main()