"""All numeric-metric computation for the continual HalfCheetah benchmark.

Pure computation + JSON caching: reads TensorBoard scalars, evaluates saved
checkpoints, returns numbers. No matplotlib anywhere in this file -- see
plots.py for all drawing, which consumes exactly the dict/JSON structures
this module produces.

This module owns the path-construction helpers (run_name, event_dir,
checkpoint_dir, ...) so it has NO dependency on run_continual_benchmark.py;
run_continual_benchmark.py imports these back FROM here instead, to avoid a
circular import (run_continual_benchmark -> metrics -> run_continual_benchmark).

SURVEY METRICS (p_i(t) = charts/test_success throughout, already in [0,1],
periodically evaluated -- NOT the noisier training-time charts/success, and
NOT charts/episodic_return, which is unbounded and needs no [0,1] range to
begin with):

- A_N            : mean final-checkpoint success across every unique task
                   in the sequence (survey Eq. 7, final value A_N).
- FG  (forgetting): mean over i=0..len(seq)-2 of max(p_i,i - p_N,i, 0)
                    (survey Eq. 8) -- LAST position excluded (forgetting
                    relative to itself at the final step is trivially 0).
- BWT (backward)  : mean over the same range of (p_N,i - p_i,i), signed,
                    no floor (survey Eq. 10).
- FT  (forward)   : two variants, both averaged over occurrences i=1..N-1
                    (survey Eq. 9 -- FIRST position excluded, since it has
                    no continual history to transfer from):
      FT_success  : the literal survey formula, AUC over test_success in
                    [0,1] vs. a from-scratch baseline's AUC.
      FT_return   : algebraic reduction of the same formula assuming
                    r_max=0 (true here: reward = -|v_error| - ctrl_cost is
                    always <= 0), which cancels the unknown r_min and
                    reduces to FT_i = 1 - R_i/R_i^b using RAW
                    charts/test_episodic_return integrals -- no [0,1]
                    normalization needed. Both p_i,i and p_N,i (for FG/BWT)
                    and the FT baselines come from checkpoints/logs that
                    already exist -- p_i,i is read directly from each
                    position's own TensorBoard log (free), and p_N,i is
                    evaluated ONCE per UNIQUE task_id against the FINAL
                    checkpoint (not once per occurrence -- duplicate
                    task_ids share the same environment, hence the same
                    value).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator

from cka_rl import FrozenCkaPolicy
from tasks import get_task
import scratch_baselines as scratch

# NumPy 2.0 removed np.trapz in favor of np.trapezoid; NumPy <2.0 only has
# np.trapz. Picking whichever exists at import time keeps this file working
# regardless of which NumPy version is installed (e.g. on Kaggle vs. locally).
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ==========================================================================
# Path helpers (the single source of truth -- plots.py and
# run_continual_benchmark.py both import these from here).
# ==========================================================================
def run_name(suite, task_id, seed):
    return f"{suite}__task_{task_id}__cka-rl__run_sac__{seed}"


def event_dir(runs_root, suite, condition, seed, seq_idx, task_id):
    tag = f"{suite}/{condition}/seed_{seed}/seq_{seq_idx}"
    return pathlib.Path(runs_root) / tag / run_name(suite, task_id, seed)


def checkpoint_dir(save_root, suite, condition, seed, seq_idx, task_id):
    return (
        pathlib.Path(save_root) / suite / condition / f"seed_{seed}"
        / f"seq_{seq_idx}" / run_name(suite, task_id, seed)
    )


def analysis_snapshot_path(analysis_root, suite, condition, seed, seq_idx, task_id):
    tag = pathlib.Path(suite) / condition / f"seed_{seed}" / f"seq_{seq_idx}"
    return pathlib.Path(analysis_root) / tag / run_name(suite, task_id, seed) / "post_finalize.pt"


def checkpoint_complete(path):
    required = ["policy_snapshot.pt", "fc.pt", "mean_pool.pt", "logstd_pool.pt"]
    return path.exists() and all((path / name).exists() for name in required)


def load_scalar(directory, scalar_tag):
    directory = pathlib.Path(directory)
    if not directory.exists():
        return np.empty(0), np.empty(0)
    try:
        ea = event_accumulator.EventAccumulator(
            str(directory), size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
    except Exception:
        return np.empty(0), np.empty(0)
    if scalar_tag not in ea.Tags().get("scalars", []):
        return np.empty(0), np.empty(0)
    events = ea.Scalars(scalar_tag)
    return (
        np.asarray([e.step for e in events], dtype=np.float64),
        np.asarray([e.value for e in events], dtype=np.float64),
    )


def final_scalar(directory, scalar_tag):
    _, values = load_scalar(directory, scalar_tag)
    return float(values[-1]) if len(values) else float("nan")


def load_continual_scalar(runs_root, suite, condition, seed, task_sequence, total_timesteps, scalar_tag):
    """Concatenates one scalar across the whole continual chain onto one
    x-axis (each task's local steps offset by its position) -- used by
    plots.py for the during-training curves."""
    xs, ys = [], []
    for seq_idx, task_id in enumerate(task_sequence):
        x, y = load_scalar(event_dir(runs_root, suite, condition, seed, seq_idx, task_id), scalar_tag)
        if x.size:
            xs.append(x + seq_idx * (total_timesteps + 1))
            ys.append(y)
    if not xs:
        return np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys)


# ==========================================================================
# Checkpoint evaluation (shared by the full retention matrix below and the
# cheap diagonal/last-row-only survey metrics further down).
# ==========================================================================
def evaluate_checkpoint(run_dir, suite, task_id, episodes, seed, device):
    env = get_task(task_id, task_suite=suite)
    policy = FrozenCkaPolicy.load(str(run_dir), map_location=device).to(device)
    policy.eval()
    returns, success, velocity_errors = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 * task_id + ep)
        ep_return = 0.0
        ep_success, ep_error = [], []
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = policy(obs_t)
            mid = (env.action_space.high + env.action_space.low) / 2.0
            scale = (env.action_space.high - env.action_space.low) / 2.0
            action = np.tanh(mean[0].cpu().numpy()) * scale + mid
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_success.append(float(info.get("success", np.nan)))
            ep_error.append(float(info.get("velocity_error", np.nan)))
            if terminated or truncated:
                break
        returns.append(ep_return)
        success.append(float(np.nanmean(ep_success)))
        velocity_errors.append(float(np.nanmean(ep_error)))
    env.close()
    return {
        "return": float(np.mean(returns)),
        "success": float(np.nanmean(success)),
        "velocity_error": float(np.nanmean(velocity_errors)),
    }


# ==========================================================================
# FULL retention matrix (unchanged logic from before -- kept for the
# existing heatmap/sequence-diagnostic plots, which want every checkpoint x
# every unique task, not just the diagonal + final row).
# ==========================================================================
def retention_cache_path(args, suite, condition, seed):
    return pathlib.Path(args.plots_root) / suite / "retention_data" / f"{condition}_seed_{seed}.json"


def build_retention_matrix(args, suite, condition, seed, device):
    cache = retention_cache_path(args, suite, condition, seed)
    eval_task_ids = sorted(set(args.task_sequence))
    if cache.exists() and not args.force_retrain:
        with open(cache) as f:
            cached = json.load(f)
        if (
            cached.get("suite") == suite
            and cached.get("condition") == condition
            and int(cached.get("seed", -1)) == int(seed)
            and cached.get("sequence") == list(args.task_sequence)
            and cached.get("eval_task_ids") == eval_task_ids
            and int(cached.get("episodes", -1)) == int(args.retention_eval_episodes)
        ):
            return cached

    data = {
        "suite": suite,
        "condition": condition,
        "seed": seed,
        "sequence": list(args.task_sequence),
        "eval_task_ids": eval_task_ids,
        "episodes": int(args.retention_eval_episodes),
        "return": [],
        "success": [],
        "velocity_error": [],
    }
    for seq_idx, trained_task in enumerate(args.task_sequence):
        run_dir = checkpoint_dir(args.save_root, suite, condition, seed, seq_idx, trained_task)
        if not checkpoint_complete(run_dir):
            raise FileNotFoundError(run_dir)
        rows = {metric: [] for metric in ("return", "success", "velocity_error")}
        for eval_task in eval_task_ids:
            metrics = evaluate_checkpoint(
                run_dir, suite, eval_task, args.retention_eval_episodes,
                seed + seq_idx * 100_000, device,
            )
            for metric in rows:
                rows[metric].append(metrics[metric])
        for metric in rows:
            data[metric].append(rows[metric])
        print(
            f"retention {suite}/{condition}/seed={seed}: after seq{seq_idx} "
            f"task {trained_task} done"
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(data, f, indent=2)
    return data


# ==========================================================================
# Survey metrics: A_N, FG, BWT (cheap -- diagonal is free, last row is
# evaluated once per UNIQUE task, not once per occurrence).
# ==========================================================================
def compute_p_diagonal(args, suite, condition, seed):
    """p_{i,i}: performance right when position i's own task just finished
    training. Read directly from that position's TensorBoard log
    (charts/test_success) -- no new evaluation, this is already logged.
    Keys are strings (not int) so this survives a JSON cache round-trip
    unchanged -- JSON always serializes dict keys as strings."""
    diagonal = {}
    for seq_idx, task_id in enumerate(args.task_sequence):
        directory = event_dir(args.runs_root, suite, condition, seed, seq_idx, task_id)
        diagonal[str(seq_idx)] = final_scalar(directory, "charts/test_success")
    return diagonal


def compute_p_final_row(args, suite, condition, seed, device):
    """p_{N,i}: the FINAL checkpoint's performance on each UNIQUE task_id in
    the sequence. Evaluated once per unique task_id, not once per
    occurrence -- a repeated task_id is the same environment, so it shares
    the same value regardless of which position(s) it appeared at. Keys
    are strings, same reasoning as compute_p_diagonal."""
    final_seq_idx = len(args.task_sequence) - 1
    final_task_id = args.task_sequence[final_seq_idx]
    final_run_dir = checkpoint_dir(args.save_root, suite, condition, seed, final_seq_idx, final_task_id)
    if not checkpoint_complete(final_run_dir):
        raise FileNotFoundError(final_run_dir)

    row = {}
    for task_id in sorted(set(args.task_sequence)):
        result = evaluate_checkpoint(
            final_run_dir, suite, task_id, args.retention_eval_episodes,
            seed + 500_000, device,
        )
        row[str(task_id)] = result["success"]
    return row


def compute_fg_bwt(diagonal, final_row, task_sequence):
    """FG_i = max(p_i,i - p_N,i, 0), BWT_i = p_N,i - p_i,i (signed, no
    floor). Both averaged over seq_idx = 0 .. len(task_sequence)-2 -- the
    LAST position is excluded (survey Eq. 8/10: sums run to N-1 terms over
    N tasks, and forgetting/backward-transfer of the final task relative to
    itself is trivially zero). diagonal/final_row are keyed by STRING (see
    compute_p_diagonal/compute_p_final_row), whether freshly computed or
    reloaded from the JSON cache."""
    fg_values, bwt_values = [], []
    for seq_idx in range(len(task_sequence) - 1):
        task_id = task_sequence[seq_idx]
        p_ii = diagonal.get(str(seq_idx), float("nan"))
        p_Ni = final_row.get(str(task_id), float("nan"))
        if np.isnan(p_ii) or np.isnan(p_Ni):
            continue
        bwt_values.append(p_Ni - p_ii)
        fg_values.append(max(p_ii - p_Ni, 0.0))
    return {
        "FG": float(np.mean(fg_values)) if fg_values else float("nan"),
        "BWT": float(np.mean(bwt_values)) if bwt_values else float("nan"),
        "FG_per_position": fg_values,
        "BWT_per_position": bwt_values,
    }


def compute_A_N(final_row):
    """Survey Eq. 7's final value A_N: mean final-checkpoint performance
    over every unique task encountered in the sequence."""
    values = list(final_row.values())
    return float(np.mean(values)) if values else float("nan")


# ==========================================================================
# Survey metric: FT (forward transfer), two variants.
# ==========================================================================
def _auc(steps, values):
    """Time-average of values(t) over the observed step range -- normalizes
    by whatever range the data actually spans, so eval points that don't
    start exactly at 0 (they start at --eval-every) don't bias the result."""
    if steps.size < 2:
        return None
    span = float(steps[-1] - steps[0])
    if span <= 0:
        return None
    return float(_trapz(values, steps) / span)


def compute_forward_transfer_success(args, suite, condition, seed, scratch_seeds, scratch_total_timesteps):
    """Survey Eq. 9, literal: p_i(t) = charts/test_success, already in
    [0,1]. Averaged over occurrences i=1..N-1 (0-indexed: seq_idx=1..end),
    the FIRST occurrence excluded (no continual history to transfer from
    yet, so FT_0 is not meaningful)."""
    per_position = []
    for seq_idx in range(1, len(args.task_sequence)):
        task_id = args.task_sequence[seq_idx]
        run_steps, run_values = load_scalar(
            event_dir(args.runs_root, suite, condition, seed, seq_idx, task_id),
            "charts/test_success",
        )
        auc = _auc(run_steps, run_values)
        if auc is None:
            continue

        baseline_aucs = []
        for b_seed in scratch_seeds:
            b_dir = scratch.scratch_event_dir(args.runs_root, suite, task_id, scratch_total_timesteps, b_seed)
            b_steps, b_values = load_scalar(b_dir, "charts/test_success")
            b_auc = _auc(b_steps, b_values)
            if b_auc is not None:
                baseline_aucs.append(b_auc)
        if not baseline_aucs:
            continue
        auc_b = float(np.mean(baseline_aucs))
        if auc_b >= 1.0:
            continue  # a perfect baseline leaves no headroom -- 1-AUC_b would divide by zero
        per_position.append((auc - auc_b) / (1.0 - auc_b))

    return {
        "FT_success": float(np.mean(per_position)) if per_position else float("nan"),
        "FT_success_per_position": per_position,
    }


def compute_forward_transfer_return(args, suite, condition, seed, scratch_seeds, scratch_total_timesteps):
    """Algebraic reduction FT_i = 1 - R_i/R_i^b (see module docstring for
    the derivation), using RAW charts/test_episodic_return integrals -- no
    [0,1] normalization needed since r_min cancels out of the ratio. Same
    i=1..N-1 range as compute_forward_transfer_success."""
    per_position = []
    for seq_idx in range(1, len(args.task_sequence)):
        task_id = args.task_sequence[seq_idx]
        run_steps, run_values = load_scalar(
            event_dir(args.runs_root, suite, condition, seed, seq_idx, task_id),
            "charts/test_episodic_return",
        )
        if run_steps.size < 2:
            continue
        r_i = float(_trapz(run_values, run_steps))

        baseline_integrals = []
        for b_seed in scratch_seeds:
            b_dir = scratch.scratch_event_dir(args.runs_root, suite, task_id, scratch_total_timesteps, b_seed)
            b_steps, b_values = load_scalar(b_dir, "charts/test_episodic_return")
            if b_steps.size >= 2:
                baseline_integrals.append(float(_trapz(b_values, b_steps)))
        if not baseline_integrals:
            continue
        r_i_b = float(np.mean(baseline_integrals))
        if r_i_b == 0.0:
            continue
        per_position.append(1.0 - (r_i / r_i_b))

    return {
        "FT_return": float(np.mean(per_position)) if per_position else float("nan"),
        "FT_return_per_position": per_position,
    }


# ==========================================================================
# Orchestrator: computes + caches everything above for one (suite,
# condition, seed).
# ==========================================================================
def survey_metrics_cache_path(args, suite, condition, seed):
    return pathlib.Path(args.plots_root) / suite / "survey_metrics" / f"{condition}_seed_{seed}.json"


def compute_survey_metrics(args, suite, condition, seed, device, scratch_seeds, scratch_total_timesteps):
    cache = survey_metrics_cache_path(args, suite, condition, seed)
    if cache.exists() and not args.force_retrain:
        with open(cache) as f:
            cached = json.load(f)
        if (
            cached.get("suite") == suite
            and cached.get("condition") == condition
            and int(cached.get("seed", -1)) == int(seed)
            and cached.get("sequence") == list(args.task_sequence)
        ):
            return cached

    diagonal = compute_p_diagonal(args, suite, condition, seed)
    final_row = compute_p_final_row(args, suite, condition, seed, device)
    fg_bwt = compute_fg_bwt(diagonal, final_row, args.task_sequence)
    ft_success = compute_forward_transfer_success(args, suite, condition, seed, scratch_seeds, scratch_total_timesteps)
    ft_return = compute_forward_transfer_return(args, suite, condition, seed, scratch_seeds, scratch_total_timesteps)

    result = {
        "suite": suite,
        "condition": condition,
        "seed": seed,
        "sequence": list(args.task_sequence),
        "A_N": compute_A_N(final_row),
        **fg_bwt,
        **ft_success,
        **ft_return,
        "p_diagonal": diagonal,
        "p_final_row": final_row,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(result, f, indent=2)
    return result
