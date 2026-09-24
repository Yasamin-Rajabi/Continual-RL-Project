"""Continual-RL metrics for the PointMaze benchmark.

The performance matrix ``P[stage, task]`` is the score of the checkpoint saved
after training stage ``stage`` when evaluated on ``task``.  Three quantities
are derived from it:

    Final performance   mean_i P[last, i]
    Forgetting          mean_i ( P[i, i] - P[last, i] )
    Forward transfer    mean_i ( P[i, i] - scratch_i ) / ( UB - scratch_i )

Which cells are actually computed
---------------------------------
A full matrix is 10 stages x 10 tasks = 100 evaluations per method per seed,
and 90 of those cells are never read by any of the three metrics above.  Only
two families of cell matter:

    the diagonal    P[i, i]     -- peak performance on a task, used by both
                                   forgetting and forward transfer
    the last row    P[last, i]  -- retention at the end of the chain

so everything else is skipped.  That is a 5x reduction in evaluation time and
in stored results, and it changes no reported number.

On the diagonal the checkpoint has *just* finished training that exact task,
so test-time adaptation is switched off there: letting a method re-tune its
routing on the task it just trained would inflate the peak that forgetting is
measured against, and would make forgetting look smaller than it is.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Optional

import numpy as np

from checkpoint_evaluation import (
    DEFAULT_TEST_ADAPT_LR,
    DEFAULT_TEST_ADAPT_STEPS,
    evaluate_checkpoint,
)
from pointmaze_env import RETURN_UPPER_BOUND

# Sentinel for a cell the protocol deliberately never computes.
SKIPPED = None


def checkpoint_complete(run_dir) -> bool:
    run_dir = pathlib.Path(run_dir)
    if not run_dir.is_dir():
        return False
    if not (run_dir / "manifest.json").is_file():
        return False
    return (run_dir / "agent.pt").is_file() or (run_dir / "policy_snapshot.pt").is_file()


def cell_is_needed(stage_idx: int, eval_task: int, trained_task: int, n_stages: int) -> bool:
    """True for the diagonal and for the final row; False otherwise.

    This is the whole memory/time optimization, in one predicate, so that the
    runner and the analysis agree on exactly which cells exist.
    """
    is_last_stage = stage_idx == n_stages - 1
    is_diagonal = eval_task == trained_task
    return bool(is_diagonal or is_last_stage)


def evaluate_chain(
    checkpoint_dirs: List[str],
    task_sequence: List[int],
    method: str,
    suite: str,
    device,
    *,
    episodes: int = 10,
    seed: int = 1,
    test_adapt_steps: int = DEFAULT_TEST_ADAPT_STEPS,
    test_adapt_lr: float = DEFAULT_TEST_ADAPT_LR,
    action_mode: str = "deterministic",
    verbose: bool = True,
) -> Dict:
    """Evaluate only the cells the metrics need."""
    n_stages = len(checkpoint_dirs)
    if n_stages != len(task_sequence):
        raise ValueError("checkpoint_dirs and task_sequence must have equal length")

    matrix: List[List[Optional[float]]] = [
        [SKIPPED for _ in range(n_stages)] for _ in range(n_stages)
    ]
    success: List[List[Optional[float]]] = [
        [SKIPPED for _ in range(n_stages)] for _ in range(n_stages)
    ]
    adaptation_steps_used = 0
    evaluated = 0

    for stage_idx, run_dir in enumerate(checkpoint_dirs):
        trained_task = int(task_sequence[stage_idx])
        for eval_idx, eval_task in enumerate(task_sequence):
            eval_task = int(eval_task)
            if not cell_is_needed(stage_idx, eval_task, trained_task, n_stages):
                continue

            is_diagonal = eval_task == trained_task
            # No adaptation on the diagonal: the checkpoint just trained this
            # task, so adapting there would inflate the peak that forgetting
            # is measured against.
            steps = 0 if is_diagonal else int(test_adapt_steps)

            result = evaluate_checkpoint(
                run_dir,
                method,
                suite,
                eval_task,
                episodes=episodes,
                seed=seed,
                device=device,
                adapt_steps=steps,
                adapt_lr=test_adapt_lr,
                action_mode=action_mode,
            )
            matrix[stage_idx][eval_idx] = result["return"]
            success[stage_idx][eval_idx] = result["success"]
            adaptation_steps_used += result["adaptation_interactions"]
            evaluated += 1
            if verbose:
                print(
                    f"  stage {stage_idx} -> task {eval_task}: "
                    f"return={result['return']:.2f} success={result['success']:.2f}"
                    f"{' (diag)' if is_diagonal else ''}"
                )

    return {
        "method": method,
        "suite": suite,
        "seed": int(seed),
        "task_sequence": [int(t) for t in task_sequence],
        "performance_matrix": matrix,
        "success_matrix": success,
        "cells_evaluated": int(evaluated),
        "cells_total": int(n_stages * n_stages),
        "test_adapt_steps": int(test_adapt_steps),
        "test_adapt_lr": float(test_adapt_lr),
        "total_adaptation_interactions": int(adaptation_steps_used),
        "episodes_per_cell": int(episodes),
    }


def diagonal(matrix) -> List[Optional[float]]:
    return [matrix[i][i] for i in range(len(matrix))]


def final_row(matrix) -> List[Optional[float]]:
    return list(matrix[-1])


def compute_metrics(chain: Dict, scratch: Optional[Dict] = None) -> Dict:
    """Derive the three headline metrics from a (sparse) performance matrix."""
    matrix = chain["performance_matrix"]
    success = chain.get("success_matrix")
    n = len(matrix)

    diag = diagonal(matrix)
    last = final_row(matrix)
    diag_ok = [v for v in diag if v is not None]
    last_ok = [v for v in last if v is not None]

    out = {
        "method": chain.get("method"),
        "suite": chain.get("suite"),
        "seed": chain.get("seed"),
        "num_tasks": n,
        "peak_per_task": diag,
        "final_per_task": last,
        "average_peak": float(np.mean(diag_ok)) if diag_ok else float("nan"),
        "average_final": float(np.mean(last_ok)) if last_ok else float("nan"),
    }

    # Forgetting: how much of the peak was lost by the end of the chain.
    # The final stage's own task cannot be forgotten yet, so it is excluded.
    losses = [
        diag[i] - last[i]
        for i in range(n - 1)
        if diag[i] is not None and last[i] is not None
    ]
    out["forgetting"] = float(np.mean(losses)) if losses else float("nan")
    out["forgetting_per_task"] = losses

    if success:
        s_diag = [success[i][i] for i in range(n)]
        s_last = list(success[-1])
        out["average_peak_success"] = float(
            np.mean([v for v in s_diag if v is not None])
        ) if any(v is not None for v in s_diag) else float("nan")
        out["average_final_success"] = float(
            np.mean([v for v in s_last if v is not None])
        ) if any(v is not None for v in s_last) else float("nan")

    # Forward transfer against an independent from-scratch reference.
    if scratch is not None:
        ft = []
        for i in range(n):
            task = chain["task_sequence"][i]
            ref = scratch.get("per_task", {}).get(str(int(task)))
            if ref is None or diag[i] is None:
                continue
            denominator = RETURN_UPPER_BOUND - float(ref)
            if abs(denominator) < 1e-9:
                continue
            ft.append((diag[i] - float(ref)) / denominator)
        out["forward_transfer"] = float(np.mean(ft)) if ft else float("nan")
        out["forward_transfer_per_task"] = ft
        out["scratch_seed"] = scratch.get("seed")
    else:
        out["forward_transfer"] = float("nan")
        out["forward_transfer_per_task"] = []

    out["cells_evaluated"] = chain.get("cells_evaluated")
    out["cells_total"] = chain.get("cells_total")
    out["test_adapt_steps"] = chain.get("test_adapt_steps")
    return out


def aggregate_seeds(per_seed: List[Dict]) -> Dict:
    """Mean and standard error across seeds for the headline metrics."""
    keys = (
        "average_peak",
        "average_final",
        "forgetting",
        "forward_transfer",
        "average_peak_success",
        "average_final_success",
    )
    out = {"num_seeds": len(per_seed), "seeds": [m.get("seed") for m in per_seed]}
    for key in keys:
        values = [m[key] for m in per_seed if key in m and np.isfinite(m.get(key, np.nan))]
        if values:
            out[key] = float(np.mean(values))
            out[f"{key}_sem"] = (
                float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            )
        else:
            out[key] = float("nan")
            out[f"{key}_sem"] = float("nan")
    return out


def save_results(path, payload) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_results(path):
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PointMaze continual metrics")
    p.add_argument("--results", nargs="+", required=True, help="chain result JSON files")
    p.add_argument("--scratch", default=None, help="scratch reference JSON")
    p.add_argument("--out", default=None, help="where to write the aggregate")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    scratch = load_results(args.scratch) if args.scratch else None
    per_seed = [compute_metrics(load_results(path), scratch) for path in args.results]
    aggregate = aggregate_seeds(per_seed)
    payload = {"per_seed": per_seed, "aggregate": aggregate}
    print(json.dumps(aggregate, indent=2))
    if args.out:
        save_results(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
