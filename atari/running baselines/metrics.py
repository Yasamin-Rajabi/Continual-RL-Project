"""Survey-style continual Atari metrics for all legacy baselines.

Reported metrics
----------------
A_N, FG, BWT
    Computed from fixed-threshold success probabilities, matching the bounded
    performance variable used by the HalfCheetah/Atari survey-style protocol.
    They are NaN unless every task in the sequence has an explicit fixed
    success threshold.

FT_success
    First-encounter forward transfer using success-learning-curve AUC:
        (AUC_continual - AUC_scratch) / (1 - AUC_scratch)

FT_return
    Atari raw-score AUC transfer:
        (AUC_continual - AUC_scratch) / max(|AUC_scratch|, eps)
    This is the Atari-safe counterpart of the HalfCheetah return formula; raw
    Atari scores do not share HalfCheetah's zero upper bound.

For diagnosis, A_N_return / FG_return / BWT_return are also emitted.  Raw score
aggregates must be compared within one game/suite; do not average Freeway and
SpaceInvaders raw scores into one number.

Repeated semantic tasks
-----------------------
The diagonal remains occurrence-specific, while the final row is keyed by the
semantic task ID.  FT uses only the first encounter of each previously unseen
task after sequence position 0; later repeats are relearning/savings, not FT.
"""
from __future__ import annotations
from task_utils import TASKS

import argparse
import csv
import json
import pathlib
from typing import Iterable, Sequence

import numpy as np
import torch
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    event_accumulator = None

from benchmark_protocol import (
    METHODS,
    canonical_env_name,
    canonical_method,
    checkpoint_dir,
    event_dir,
    env_id,
    load_success_thresholds,
    metric_root,
    scratch_event_dir,
    success_threshold,
    unique_in_order,
)
from checkpoint_evaluation import checkpoint_complete, evaluate_checkpoint

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz
_EPS = 1e-8
CACHE_SCHEMA_VERSION = 1


def _load_scalar_csv(directory, tag):
    path = pathlib.Path(directory) / "scalars.csv"
    if not path.exists():
        return np.empty(0), np.empty(0)
    xs, ys = [], []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("tag") != tag:
                    continue
                if row.get("step", "") == "" or row.get("value", "") == "":
                    continue
                xs.append(float(row["step"]))
                ys.append(float(row["value"]))
    except Exception:
        return np.empty(0), np.empty(0)
    return _prepare_curve(xs, ys)


def load_scalar(directory, tag):
    """TensorBoard scalar reader with scalars.csv fallback."""
    directory = pathlib.Path(directory)
    if not directory.exists():
        return np.empty(0), np.empty(0)
    if event_accumulator is not None:
        try:
            ea = event_accumulator.EventAccumulator(
                str(directory), size_guidance={event_accumulator.SCALARS: 0}
            )
            ea.Reload()
            if tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                if events:
                    return _prepare_curve(
                        [event.step for event in events],
                        [event.value for event in events],
                    )
        except Exception:
            pass
    return _load_scalar_csv(directory, tag)


def _prepare_curve(steps, values):
    steps = np.asarray(steps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(steps) & np.isfinite(values)
    steps, values = steps[finite], values[finite]
    if not len(steps):
        return steps, values
    order = np.argsort(steps, kind="stable")
    steps, values = steps[order], values[order]
    # Keep the last event at duplicate x values.
    _, rev_idx = np.unique(steps[::-1], return_index=True)
    keep = steps.size - 1 - rev_idx
    keep.sort()
    return steps[keep], values[keep]


def _auc(steps, values):
    steps, values = _prepare_curve(steps, values)
    if len(steps) < 2:
        return None
    span = float(steps[-1] - steps[0])
    if span <= 0:
        return None
    return float(_TRAPZ(values, steps) / span)


def _first_unseen_positions(task_sequence):
    seen = set()
    out = []
    for seq_idx, task_id in enumerate(task_sequence):
        task_id = int(task_id)
        if seq_idx > 0 and task_id not in seen:
            out.append((seq_idx, task_id))
        seen.add(task_id)
    return out


def _thresholds_complete(thresholds, env, task_ids: Iterable[int]):
    return all(success_threshold(thresholds, env, task_id) is not None for task_id in task_ids)


def _checkpoint_dirs(args, env, method, seed):
    return [
        checkpoint_dir(
            args.save_root,
            env,
            args.tag,
            method,
            seed,
            seq_idx,
            task_id,
        )
        for seq_idx, task_id in enumerate(args.task_sequence)
    ]


def retention_cache_path(args, env, method, seed):
    return metric_root(args.plots_root, env, method, seed) / "retention.json"


def survey_cache_path(args, env, method, seed):
    return metric_root(args.plots_root, env, method, seed) / "survey_metrics.json"


def build_retention_matrix(args, env, method, seed, device):
    env = canonical_env_name(env)
    method = canonical_method(method)
    task_sequence = [int(x) for x in args.task_sequence]
    eval_task_ids = unique_in_order(task_sequence)
    checkpoints = _checkpoint_dirs(args, env, method, seed)

    for seq_idx, path in enumerate(checkpoints):
        if not checkpoint_complete(method, path):
            raise FileNotFoundError(
                f"missing/incomplete {method} checkpoint at seq_idx={seq_idx}: {path}"
            )

    cache = retention_cache_path(args, env, method, seed)
    config = {
        "schema": CACHE_SCHEMA_VERSION,
        "sequence": task_sequence,
        "episodes": int(args.retention_eval_episodes),
        "test_adapt_steps": int(args.test_adapt_steps),
        "test_adapt_lr": float(args.test_adapt_lr),
        "eval_action_mode": str(args.eval_action_mode),
        "success_thresholds": args.success_thresholds,
    }
    if cache.exists() and not args.force:
        try:
            saved = json.loads(cache.read_text())
            if saved.get("cache_config") == config:
                return saved
        except Exception:
            pass

    task_col = {task_id: idx for idx, task_id in enumerate(eval_task_ids)}
    payload = {
        "cache_config": config,
        "env": env,
        "method": method,
        "seed": int(seed),
        "sequence": task_sequence,
        "eval_task_ids": eval_task_ids,
        "return": [],
        "success": [],
        "adaptation_interactions": [],
        "evaluation_interactions": [],
    }

    seen = set()
    for stage_idx, trained_task in enumerate(task_sequence):
        seen.add(trained_task)
        rows = {
            "return": [float("nan")] * len(eval_task_ids),
            "success": [float("nan")] * len(eval_task_ids),
            "adaptation_interactions": [0] * len(eval_task_ids),
            "evaluation_interactions": [0] * len(eval_task_ids),
        }
        for eval_task in eval_task_ids:
            if eval_task not in seen:
                continue

            is_last_stage = (stage_idx == len(task_sequence) - 1)
            is_diagonal = (eval_task == trained_task)
            if not (is_diagonal or is_last_stage):
                continue

            result = evaluate_checkpoint(
                method,
                checkpoints,
                task_sequence,
                stage_idx,
                eval_task,
                args.retention_eval_episodes,
                seed,
                device,
                test_adapt_steps=0 if is_diagonal else args.test_adapt_steps,
                test_adapt_lr=args.test_adapt_lr,
                action_mode=args.eval_action_mode,
                success_threshold=success_threshold(args.success_thresholds, env, eval_task),
                env=env,
            )
            j = task_col[eval_task]
            rows["return"][j] = result["return"]
            rows["success"][j] = result["success"]
            rows["adaptation_interactions"][j] = result["adaptation_interactions"]
            rows["evaluation_interactions"][j] = result["evaluation_interactions"]

        for key in rows:
            payload[key].append(rows[key])
        print(
            f"retention {env}/{method}/seed={seed}: "
            f"after seq{stage_idx} task {trained_task}"
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2, allow_nan=True))
    return payload


def _diagonal_and_final(payload, metric):
    matrix = np.asarray(payload[metric], dtype=np.float64)
    sequence = payload["sequence"]
    eval_ids = payload["eval_task_ids"]
    cols = {task_id: i for i, task_id in enumerate(eval_ids)}
    diagonal = {
        str(seq_idx): float(matrix[seq_idx, cols[int(task_id)]])
        for seq_idx, task_id in enumerate(sequence)
    }
    final = {
        str(task_id): float(matrix[-1, cols[int(task_id)]])
        for task_id in eval_ids
    }
    return diagonal, final


def compute_A_N(final_row):
    values = np.asarray(list(final_row.values()), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def compute_fg_bwt(diagonal, final_row, task_sequence, prefix=""):
    fg, bwt = [], []
    positions = []
    for seq_idx in range(len(task_sequence) - 1):
        task_id = int(task_sequence[seq_idx])
        p_ii = float(diagonal.get(str(seq_idx), np.nan))
        p_Ni = float(final_row.get(str(task_id), np.nan))
        if not np.isfinite(p_ii) or not np.isfinite(p_Ni):
            continue
        fg.append(max(p_ii - p_Ni, 0.0))
        bwt.append(p_Ni - p_ii)
        positions.append({"seq_idx": seq_idx, "task_id": task_id})
    lead = f"{prefix}_" if prefix else ""
    return {
        f"{lead}FG": float(np.mean(fg)) if fg else float("nan"),
        f"{lead}BWT": float(np.mean(bwt)) if bwt else float("nan"),
        f"{lead}FG_per_position": fg,
        f"{lead}BWT_per_position": bwt,
        f"{lead}FG_BWT_positions": positions,
    }


def _scratch_auc(args, env, method, task_id, tag, scratch_seeds):
    values = []
    for scratch_seed in scratch_seeds:
        directory = scratch_event_dir(
            args.scratch_runs_root,
            env,
            method,
            task_id,
            args.scratch_total_timesteps,
            scratch_seed,
        )
        x, y = load_scalar(directory, tag)
        value = _auc(x, y)
        if value is not None and np.isfinite(value):
            values.append(value)
    return None if not values else float(np.mean(values))


def compute_forward_transfer(args, env, method, seed):
    positions = _first_unseen_positions(args.task_sequence)
    scratch_seeds = [int(x) for x in args.scratch_seeds]

    ft_return = []
    ft_success = []
    return_positions = []
    success_positions = []

    thresholds_complete = _thresholds_complete(
        args.success_thresholds,
        env,
        unique_in_order(args.task_sequence),
    )

    for seq_idx, task_id in positions:
        directory = event_dir(
            args.runs_root,
            env,
            args.tag,
            method,
            seed,
            seq_idx,
            task_id,
        )

        x, y = load_scalar(directory, "charts/test_episodic_return")
        current = _auc(x, y)
        baseline = _scratch_auc(
            args,
            env,
            method,
            task_id,
            "charts/test_episodic_return",
            scratch_seeds,
        )
        if current is not None and baseline is not None:
            ft_return.append((current - baseline) / max(abs(baseline), _EPS))
            return_positions.append({"seq_idx": seq_idx, "task_id": task_id})

        if thresholds_complete:
            sx, sy = load_scalar(directory, "charts/test_success")
            current_success = _auc(sx, sy)
            baseline_success = _scratch_auc(
                args,
                env,
                method,
                task_id,
                "charts/test_success",
                scratch_seeds,
            )
            if (
                current_success is not None
                and baseline_success is not None
                and baseline_success < 1.0 - _EPS
            ):
                ft_success.append(
                    (current_success - baseline_success) / (1.0 - baseline_success)
                )
                success_positions.append({"seq_idx": seq_idx, "task_id": task_id})

    return {
        "FT_return": float(np.mean(ft_return)) if ft_return else float("nan"),
        "FT_return_per_position": ft_return,
        "FT_return_positions": return_positions,
        "FT_success": float(np.mean(ft_success)) if ft_success else float("nan"),
        "FT_success_per_position": ft_success,
        "FT_success_positions": success_positions,
        "FT_success_complete": bool(thresholds_complete and len(ft_success) == len(positions)),
    }


def compute_survey_metrics(args, env, method, seed, device):
    env = canonical_env_name(env)
    method = canonical_method(method)
    retention = build_retention_matrix(args, env, method, seed, device)

    return_diag, return_final = _diagonal_and_final(retention, "return")
    success_diag, success_final = _diagonal_and_final(retention, "success")

    thresholds_complete = _thresholds_complete(
        args.success_thresholds,
        env,
        unique_in_order(args.task_sequence),
    )

    return_fg_bwt = compute_fg_bwt(return_diag, return_final, args.task_sequence, "return")
    if thresholds_complete:
        success_fg_bwt = compute_fg_bwt(success_diag, success_final, args.task_sequence)
        A_N = compute_A_N(success_final)
    else:
        A_N = float("nan")
        success_fg_bwt = {
            "FG": float("nan"),
            "BWT": float("nan"),
            "FG_per_position": [],
            "BWT_per_position": [],
            "FG_BWT_positions": [],
        }

    ft = (
        compute_forward_transfer(args, env, method, seed)
        if args.scratch_seeds and not args.skip_forward_transfer
        else {
            "FT_return": float("nan"),
            "FT_return_per_position": [],
            "FT_return_positions": [],
            "FT_success": float("nan"),
            "FT_success_per_position": [],
            "FT_success_positions": [],
            "FT_success_complete": False,
        }
    )

    result = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "env": env,
        "method": method,
        "seed": int(seed),
        "sequence": list(map(int, args.task_sequence)),
        "success_thresholds_complete": bool(thresholds_complete),
        "test_adapt_steps": int(args.test_adapt_steps),
        "test_adapt_lr": float(args.test_adapt_lr),
        "eval_action_mode": args.eval_action_mode,
        # Primary bounded family, same names as HalfCheetah.
        "A_N": A_N,
        **success_fg_bwt,
        **ft,
        # Complementary Atari raw-return family.
        "A_N_return": compute_A_N(return_final),
        **return_fg_bwt,
        "p_diagonal_success": success_diag,
        "p_final_row_success": success_final,
        "p_diagonal_return": return_diag,
        "p_final_row_return": return_final,
    }

    path = survey_cache_path(args, env, method, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=True))
    return result


def _aggregate_rows(rows, keys):
    result = {}
    for key in keys:
        values = np.asarray([row.get(key, np.nan) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[key] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std()) if finite.size else float("nan"),
            "n": int(finite.size),
        }
    return result


def compute_all(args):
    device = args.device
    if str(device).lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    all_rows = {}
    for env in args.envs:
        env = canonical_env_name(env)
        all_rows[env] = {}
        for method in args.methods:
            method = canonical_method(method)
            rows = []
            for seed in args.seeds:
                rows.append(compute_survey_metrics(args, env, method, int(seed), device))
            all_rows[env][method] = rows

        summary = {
            method: _aggregate_rows(
                rows,
                ["A_N", "FG", "BWT", "FT_return", "FT_success", "A_N_return", "return_FG", "return_BWT"],
            )
            for method, rows in all_rows[env].items()
        }
        out = pathlib.Path(args.plots_root) / env / "summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, allow_nan=True))
    return all_rows


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--envs", nargs="+", default=["Freeway", "SpaceInvaders"])
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--task-sequence", nargs="+", type=int, default=None)
    p.add_argument("--tag", default="main")
    p.add_argument("--save-root", default="agents")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--scratch-runs-root", default="runs")
    p.add_argument("--scratch-total-timesteps", type=int, default=1_000_000)
    p.add_argument("--scratch-seeds", nargs="+", type=int, default=[101, 102, 103])
    p.add_argument("--plots-root", default="metric_results")
    p.add_argument("--retention-eval-episodes", type=int, default=10)
    p.add_argument("--test-adapt-steps", type=int, default=50_000)
    p.add_argument("--test-adapt-lr", type=float, default=1e-2)
    p.add_argument("--eval-action-mode", choices=["deterministic", "stochastic"], default="deterministic")
    p.add_argument("--success-thresholds-json", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--skip-forward-transfer", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.success_thresholds = load_success_thresholds(args.success_thresholds_json)
    if args.test_adapt_steps < 0 or args.test_adapt_lr <= 0:
        p.error("test adaptation settings are invalid")
    if args.retention_eval_episodes < 1:
        p.error("retention-eval-episodes must be >= 1")

    if args.task_sequence is None:
        first_env = canonical_env_name(args.envs[0])
        full_id = env_id(first_env)
        args.task_sequence = TASKS.get(full_id, [0, 1, 2, 3, 4, 0, 1, 2, 5, 6])

    return args


if __name__ == "__main__":
    compute_all(parse_args())
