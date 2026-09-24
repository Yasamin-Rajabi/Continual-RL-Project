"""Orchestrator: run one method's full ten-task chain, then score it.

This file is orchestration only.  It launches one training process per
sequence position, validates that each finished checkpoint has the identity it
asked for, and then calls ``metrics.py``.

Resumability is the point of the design: every task writes a manifest, and a
chain that is re-launched skips any position whose checkpoint already exists
and matches.  A Kaggle session that dies at task 7 therefore resumes at task 7
rather than at task 0, which is what makes a multi-session run practical.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

import torch

from baselines import ALL_METHODS, ALL_PREVIOUS_METHODS, canonical_method, is_ours
from experiment_identity import manifest_matches
from metrics import checkpoint_complete, compute_metrics, evaluate_chain, save_results
from tasks import DEFAULT_SUITE, SUITES, get_continual_sequence, get_task_name, num_tasks, validate_suite

# Our method's reported configuration: condition 4 (combined) = weight-delta
# vectors + alpha-mass + behavioral-KL distillation merge, in policy space.
OURS_CONFIG = {
    "Ours": {
        "fusion_mode": "weight_delta",
        "composition_space": "policy",
        "distillation": True,
        "use_alpha_mass": True,
        "use_alpha_scale": False,
        "fix_alpha_scale": True,
    },
    "Ours-parameter": {
        "fusion_mode": "weight_delta",
        "composition_space": "parameter",
        "distillation": True,
        "use_alpha_mass": True,
        "use_alpha_scale": False,
        "fix_alpha_scale": True,
    },
    # The original CKA-RL: classic CKA vectors, arithmetic (cosine) merge.
    "CKA-RL": {
        "fusion_mode": "classic_cka",
        "composition_space": "parameter",
        "distillation": False,
        "use_alpha_mass": False,
        "use_alpha_scale": True,
        "fix_alpha_scale": False,
    },
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="PointMaze continual benchmark runner",
    )
    p.add_argument("--methods", nargs="+", default=["Ours"])
    p.add_argument("--suite", default=DEFAULT_SUITE, choices=sorted(SUITES))
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--distill-extra-steps", type=int, default=4_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--learning-starts", type=int, default=2_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--policy-frequency", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--num-evals", type=int, default=10)
    p.add_argument("--shared-dim", type=int, default=256)
    p.add_argument("--head-hidden-dim", type=int, default=256)
    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument("--alpha-warmup-steps", type=int, default=5_000)
    p.add_argument("--projection-epochs", type=int, default=16)
    p.add_argument("--projection-max-samples", type=int, default=4_000)
    p.add_argument("--max-distill-buffer", type=int, default=20_000)
    p.add_argument("--distill-max-samples", type=int, default=4_000)
    p.add_argument("--similarity-samples", type=int, default=1_024)
    p.add_argument("--distill-epochs", type=int, default=8)
    p.add_argument("--train-shared", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--packnet-retrain-frac", type=float, default=0.3)
    p.add_argument(
        "--packnet-width", type=int, default=384,
        help="PackNet-only network width. PackNet packs every task into one "
             "network while ProgNet grows a column per task, so equal width "
             "is not a parameter-matched comparison.",
    )
    p.add_argument(
        "--packnet-capacity-mode", default="equal_share",
        choices=["equal_share", "geometric"],
    )
    p.add_argument("--packnet-keep-frac", type=float, default=1.0)

    p.add_argument("--test-adapt-steps", type=int, default=6_000)
    p.add_argument("--test-adapt-lr", type=float, default=1e-2)
    p.add_argument("--scratch-reference", default=None)

    p.add_argument("--save-root", default="agents_pointmaze")
    p.add_argument("--runs-root", default="runs_pointmaze")
    p.add_argument("--results-root", default="results_pointmaze")
    p.add_argument("--analysis-root", default="analysis_pointmaze")
    p.add_argument("--tag", default="main")
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--force", action=argparse.BooleanOptionalAction, default=False,
        help="Retrain positions even when a matching checkpoint already exists.",
    )
    p.add_argument(
        "--skip-eval", action=argparse.BooleanOptionalAction, default=False,
        help="Train only; run the scoring pass separately.",
    )
    p.add_argument(
        "--max-seconds", type=int, default=0,
        help="Stop launching new tasks after this many seconds (0 = no limit). "
             "Lets a session end cleanly with resumable state instead of being killed.",
    )
    return p


def checkpoint_dir(args, method: str, seed: int, seq_idx: int, task_id: int) -> pathlib.Path:
    return (
        pathlib.Path(args.save_root)
        / args.suite
        / args.tag
        / method
        / f"seed_{int(seed)}"
        / f"seq_{int(seq_idx)}"
        / f"task_{int(task_id)}"
    )


def _ours_command(args, method: str, seed: int, seq_idx: int, task_id: int, prev, run_dir):
    cfg = OURS_CONFIG[method]
    cmd = [
        sys.executable, "run_sac_continual.py",
        f"--method={method}",
        f"--suite={args.suite}",
        f"--task-id={task_id}",
        f"--seq-idx={seq_idx}",
        f"--seed={seed}",
        f"--save-dir={run_dir}",
        f"--runs-root={args.runs_root}",
        f"--analysis-root={args.analysis_root}",
        f"--tag={args.tag}/{method}/seed_{seed}",
        f"--total-timesteps={args.total_timesteps}",
        f"--distill-extra-steps={args.distill_extra_steps}",
        f"--learning-rate={args.learning_rate}",
        f"--batch-size={args.batch_size}",
        f"--buffer-size={args.buffer_size}",
        f"--learning-starts={args.learning_starts}",
        f"--gamma={args.gamma}",
        f"--tau={args.tau}",
        f"--policy-frequency={args.policy_frequency}",
        f"--eval-every={args.eval_every}",
        f"--num-evals={args.num_evals}",
        f"--shared-dim={args.shared_dim}",
        f"--head-hidden-dim={args.head_hidden_dim}",
        f"--pool-size={args.pool_size}",
        f"--alpha-warmup-steps={args.alpha_warmup_steps}",
        f"--projection-epochs={args.projection_epochs}",
        f"--projection-max-samples={args.projection_max_samples}",
        f"--max-distill-buffer={args.max_distill_buffer}",
        f"--distill-max-samples={args.distill_max_samples}",
        f"--similarity-samples={args.similarity_samples}",
        f"--distill-epochs={args.distill_epochs}",
        f"--fusion-mode={cfg['fusion_mode']}",
        f"--composition-space={cfg['composition_space']}",
        "--distillation" if cfg["distillation"] else "--no-distillation",
        "--use-alpha-mass" if cfg["use_alpha_mass"] else "--no-use-alpha-mass",
        "--use-alpha-scale" if cfg["use_alpha_scale"] else "--no-use-alpha-scale",
        "--fix-alpha-scale" if cfg["fix_alpha_scale"] else "--no-fix-alpha-scale",
        "--train-shared" if args.train_shared else "--no-train-shared",
        "--cuda" if args.cuda else "--no-cuda",
    ]
    if prev:
        cmd.append("--prev-units")
        cmd.extend(str(p) for p in prev)
    return cmd


def _baseline_command(args, method: str, seed: int, seq_idx: int, task_id: int, prev, run_dir):
    cmd = [
        sys.executable, "-m", "baselines.run_baseline",
        f"--method={method}",
        f"--suite={args.suite}",
        f"--task-id={task_id}",
        f"--seq-idx={seq_idx}",
        f"--seed={seed}",
        f"--save-dir={run_dir}",
        f"--runs-root={args.runs_root}",
        f"--analysis-root={args.analysis_root}",
        f"--tag={args.tag}/{method}/seed_{seed}",
        f"--total-timesteps={args.total_timesteps}",
        f"--distill-extra-steps={args.distill_extra_steps}",
        f"--learning-rate={args.learning_rate}",
        f"--batch-size={args.batch_size}",
        f"--buffer-size={args.buffer_size}",
        f"--learning-starts={args.learning_starts}",
        f"--gamma={args.gamma}",
        f"--tau={args.tau}",
        f"--policy-frequency={args.policy_frequency}",
        f"--eval-every={args.eval_every}",
        f"--num-evals={args.num_evals}",
        f"--shared-dim={args.shared_dim}",
        f"--head-hidden-dim={args.head_hidden_dim}",
        f"--packnet-retrain-frac={args.packnet_retrain_frac}",
        f"--packnet-width={args.packnet_width}",
        f"--packnet-capacity-mode={args.packnet_capacity_mode}",
        f"--packnet-keep-frac={args.packnet_keep_frac}",
        "--cuda" if args.cuda else "--no-cuda",
    ]
    if prev:
        cmd.append("--prev-units")
        cmd.extend(str(p) for p in prev)
    return cmd


def expected_config(args, method: str, seed: int, seq_idx: int, task_id: int) -> dict:
    """The identity fields the finished checkpoint must report back."""
    cfg = {
        "method": method,
        "suite": args.suite,
        "task_id": int(task_id),
        "seq_idx": int(seq_idx),
        "seed": int(seed),
        "total_timesteps": int(args.total_timesteps),
        "distill_extra_steps": int(args.distill_extra_steps),
        "shared_dim": int(args.shared_dim),
        "head_hidden_dim": int(args.head_hidden_dim),
        "learning_rate": float(args.learning_rate),
        "batch_size": int(args.batch_size),
        "gamma": float(args.gamma),
        "tau": float(args.tau),
        "learning_starts": int(args.learning_starts),
        "num_tasks": num_tasks(args.suite),
    }
    if method == "PackNet":
        cfg.update(
            {
                "packnet_width": int(args.packnet_width),
                "packnet_capacity_mode": str(args.packnet_capacity_mode),
                "packnet_keep_frac": float(args.packnet_keep_frac),
                "packnet_retrain_frac": float(args.packnet_retrain_frac),
            }
        )
    if is_ours(method) or method == "CKA-RL":
        cfg.update(OURS_CONFIG[method])
        cfg.update(
            {
                "pool_size": int(args.pool_size),
                "alpha_warmup_steps": int(args.alpha_warmup_steps),
                "projection_epochs": int(args.projection_epochs),
                "projection_max_samples": int(args.projection_max_samples),
                "max_distill_buffer": int(args.max_distill_buffer),
                "distill_max_samples": int(args.distill_max_samples),
                "similarity_samples": int(args.similarity_samples),
                "distill_epochs": int(args.distill_epochs),
                "train_shared": bool(args.train_shared),
            }
        )
    return cfg


def train_chain(args, method: str, seed: int, sequence, deadline=None):
    """Train (or resume) one method's chain; return the checkpoint directories."""
    previous = []
    for seq_idx, task_id in enumerate(sequence):
        run_dir = checkpoint_dir(args, method, seed, seq_idx, task_id)
        prev = (
            previous
            if method in ALL_PREVIOUS_METHODS
            else (previous[-1:] if previous else [])
        )
        cfg = expected_config(args, method, seed, seq_idx, task_id)

        if not args.force and checkpoint_complete(run_dir):
            ok, reason = manifest_matches(run_dir, cfg, parent_dirs=prev)
            if ok:
                print(f"  [skip] seq {seq_idx} task {task_id}: checkpoint already complete")
                previous.append(run_dir)
                continue
            print(f"  [redo] seq {seq_idx} task {task_id}: {reason}")

        if deadline is not None and time.time() > deadline:
            print(
                f"  [stop] time budget reached before seq {seq_idx}; "
                "rerun to resume from here"
            )
            return previous, False

        # Clear partial output so a retry never merges two runs' curves.
        if run_dir.exists():
            shutil.rmtree(run_dir)

        cmd = (
            _ours_command(args, method, seed, seq_idx, task_id, prev, run_dir)
            if (is_ours(method) or method == "CKA-RL")
            else _baseline_command(args, method, seed, seq_idx, task_id, prev, run_dir)
        )
        print(
            f"\n>>> {method} | seed {seed} | seq {seq_idx} | "
            f"task {task_id}: {get_task_name(task_id, args.suite)} <<<"
        )
        subprocess.run(cmd, check=True, cwd=str(pathlib.Path(__file__).resolve().parent))

        if not checkpoint_complete(run_dir):
            raise RuntimeError(f"training finished but checkpoint is incomplete: {run_dir}")
        ok, reason = manifest_matches(run_dir, cfg, parent_dirs=prev)
        if not ok:
            raise RuntimeError(f"checkpoint has unexpected identity: {reason}")
        previous.append(run_dir)

    return previous, True


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_suite(args.suite)

    methods = [canonical_method(m) for m in args.methods]
    for m in methods:
        if m not in ALL_METHODS:
            raise SystemExit(f"unknown method {m}")

    sequence = list(get_continual_sequence(args.suite))
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    results_root = pathlib.Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_seconds if args.max_seconds > 0 else None

    scratch = None
    if args.scratch_reference:
        path = pathlib.Path(args.scratch_reference)
        if path.is_file():
            with path.open() as f:
                scratch = json.load(f)
        else:
            print(f"[warn] scratch reference {path} not found; forward transfer will be NaN")

    summary = {}
    for method in methods:
        for seed in args.seeds:
            print(f"\n===== {method} | seed {seed} | {args.suite} =====")
            dirs, complete = train_chain(args, method, seed, sequence, deadline)
            if not complete:
                summary[f"{method}_seed{seed}"] = {"status": "incomplete", "stages_done": len(dirs)}
                continue
            if args.skip_eval:
                summary[f"{method}_seed{seed}"] = {"status": "trained", "stages_done": len(dirs)}
                continue

            print(f"\n----- scoring {method} seed {seed} -----")
            chain = evaluate_chain(
                [str(d) for d in dirs], sequence, method, args.suite, device,
                episodes=args.num_evals, seed=seed,
                test_adapt_steps=args.test_adapt_steps,
                test_adapt_lr=args.test_adapt_lr,
            )
            chain_path = results_root / f"{args.suite}__{method}__seed{seed}__chain.json"
            save_results(chain_path, chain)
            metrics = compute_metrics(chain, scratch)
            save_results(
                results_root / f"{args.suite}__{method}__seed{seed}__metrics.json", metrics
            )
            summary[f"{method}_seed{seed}"] = {
                "status": "done",
                "average_final": metrics["average_final"],
                "average_peak": metrics["average_peak"],
                "forgetting": metrics["forgetting"],
                "forward_transfer": metrics["forward_transfer"],
                "cells_evaluated": metrics["cells_evaluated"],
                "cells_total": metrics["cells_total"],
            }
            print(json.dumps(summary[f"{method}_seed{seed}"], indent=2))

    save_results(results_root / f"{args.suite}__{args.tag}__summary.json", summary)
    print("\n===== summary =====")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
