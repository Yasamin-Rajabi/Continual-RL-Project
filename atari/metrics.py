"""Numeric metrics for the continual Atari CKA-RL benchmark.

This is the Atari/PPO counterpart of the HalfCheetah metrics module.  The
retention/evaluation logic is intentionally kept as close as possible to the
HalfCheetah implementation, while replacing assumptions that are specific to
continuous actions and non-positive HalfCheetah rewards.

Evaluation protocol
-------------------
* Checkpoints are evaluated on RAW full-game Atari score:
    clip_reward=False, episodic_life=False.
* The policy is categorical; deterministic checkpoint evaluation uses argmax.
* ALE does not provide a native success flag.  If the benchmark supplies a
  fixed raw-score threshold for every task, success for an episode is
      1[raw_episode_score >= threshold].
* Checkpoint diagonal and final-row evaluations use identical episode seeds and
  identical optional test-time alpha adaptation.

Survey-style metrics
--------------------
When complete success thresholds are configured, success is in [0,1] and is the
cleanest Atari analogue of the HalfCheetah survey metric p_i(t):

  A_N_success : mean final-checkpoint success over unique tasks.
  FG_success  : mean max(p_i,i - p_N,i, 0) over all non-final positions.
  BWT_success : mean (p_N,i - p_i,i) over all non-final positions.
  FT_success  : normalized AUC improvement over from-scratch learning curves,
                (AUC - AUC_b) / (1 - AUC_b), on first encounters only.

Raw Atari score is also reported as a complementary metric family:

  A_N_reward, FG_reward, BWT_reward

and an auxiliary relative forward-transfer score

  FT_reward = (AUC_reward - AUC_reward_b) / |AUC_reward_b|.

FT_reward is NOT the literal bounded-performance survey formula; it is a
relative raw-score AUC improvement.  The HalfCheetah shortcut
1 - R/R_b is invalid for Atari because Atari scores are not constrained to be
non-positive.

Forward transfer is computed only on the first encounter of each previously
unseen task after sequence position 0.  Repeated tasks are relearning/savings,
not forward transfer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from typing import Iterable, Sequence

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

import scratch_baselines as scratch
from experiment_identity import (
    MANIFEST_NAME,
    checkpoint_matches as _identity_checkpoint_matches,
    checkpoint_signature,
    load_manifest,
)


_trapz = getattr(np, "trapezoid", None) or np.trapz


CACHE_SCHEMA_VERSION = 14
_EPS = 1e-8

# Optional custom checkpoint roots used by run_eval_custom.py.  Normal benchmark
# training leaves this empty.
_CUSTOM_MODEL_MAP = {}


def set_custom_model_map(mapping):
    global _CUSTOM_MODEL_MAP
    _CUSTOM_MODEL_MAP = {str(k): pathlib.Path(v) for k, v in dict(mapping).items()}


def _is_custom_condition(condition) -> bool:
    return str(condition) in _CUSTOM_MODEL_MAP


def _is_custom_checkpoint_path(path) -> bool:
    path = pathlib.Path(path).resolve()
    for root in _CUSTOM_MODEL_MAP.values():
        try:
            path.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


# ============================================================================
# Basic helpers / paths
# ============================================================================
def _threshold(args, suite, task_id):
    mapping = getattr(args, "success_thresholds", None) or {}
    row = mapping.get(suite, {})
    value = row.get(str(task_id), row.get(task_id))
    return None if value is None else float(value)


def _success_thresholds_complete(args, suite, task_ids: Iterable[int]) -> bool:
    """True only if every requested task has an explicit fixed threshold."""
    return all(_threshold(args, suite, int(task_id)) is not None for task_id in task_ids)


def run_name(suite, task_id, seed):
    return f"{suite}__task_{task_id}__cka-rl__run_ppo__{seed}"


def event_dir(runs_root, suite, condition, seed, seq_idx, task_id):
    if _is_custom_condition(condition):
        base = _CUSTOM_MODEL_MAP[str(condition)]
        # Preferred layout: a sibling ``runs`` tree with seq_i/run_name.
        sibling_runs = base.parent / "runs"
        if sibling_runs.exists():
            return sibling_runs / f"seq_{seq_idx}" / run_name(suite, task_id, seed)
        # Fallback: logs stored below the custom model root itself.
        return base / "runs" / f"seq_{seq_idx}" / run_name(suite, task_id, seed)
    tag = f"{suite}/{condition}/seed_{seed}/seq_{seq_idx}"
    return pathlib.Path(runs_root) / tag / run_name(suite, task_id, seed)


def checkpoint_dir(save_root, suite, condition, seed, seq_idx, task_id):
    if _is_custom_condition(condition):
        return (
            _CUSTOM_MODEL_MAP[str(condition)]
            / f"seq_{seq_idx}"
            / run_name(suite, task_id, seed)
        )
    return (
        pathlib.Path(save_root)
        / suite
        / condition
        / f"seed_{seed}"
        / f"seq_{seq_idx}"
        / run_name(suite, task_id, seed)
    )


def analysis_snapshot_path(analysis_root, suite, condition, seed, seq_idx, task_id):
    run = run_name(suite, task_id, seed)
    if _is_custom_condition(condition):
        base = _CUSTOM_MODEL_MAP[str(condition)]
        # Support either a sibling analysis directory or an analysis directory
        # nested under the supplied custom root.
        candidates = [
            base.parent / "analysis" / f"seq_{seq_idx}" / run / "post_finalize.pt",
            base / "analysis" / f"seq_{seq_idx}" / run / "post_finalize.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    tag = pathlib.Path(suite) / str(condition) / f"seed_{seed}" / f"seq_{seq_idx}"
    return pathlib.Path(analysis_root) / tag / run / "post_finalize.pt"


def checkpoint_complete(path):
    path = pathlib.Path(path)
    core = ["policy_snapshot.pt", "fc.pt", "policy_pool.pt"]
    if not (path.exists() and all((path / name).exists() for name in core)):
        return False
    if _is_custom_checkpoint_path(path):
        # Custom evaluation intentionally accepts project checkpoints that may
        # predate manifests; the three model files above are still mandatory.
        return True
    return (path / MANIFEST_NAME).exists() and load_manifest(path) is not None


def checkpoint_matches(path, expected_mapping, *, parent_dirs=(), pretrained_encoder=None):
    if not checkpoint_complete(path):
        return False, "checkpoint files are missing or manifest is invalid"
    if _is_custom_checkpoint_path(path) and load_manifest(path) is None:
        return True, "custom checkpoint without manifest"
    return _identity_checkpoint_matches(
        path,
        expected_mapping,
        parent_dirs=parent_dirs,
        pretrained_encoder=pretrained_encoder,
    )


# ============================================================================
# TensorBoard scalar readers
# ============================================================================
def _load_scalar_csv(directory, scalar_tag):
    """Fallback reader for the scalars.csv mirror written next to TensorBoard."""
    path = pathlib.Path(directory) / "scalars.csv"
    if not path.exists():
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    steps, values = [], []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("tag") != scalar_tag:
                    continue
                step = row.get("step", "")
                value = row.get("value", "")
                if step == "" or value == "":
                    continue
                steps.append(float(step))
                values.append(float(value))
    except Exception:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    if not steps:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    order = np.argsort(np.asarray(steps, dtype=np.float64), kind="stable")
    return (
        np.asarray(steps, dtype=np.float64)[order],
        np.asarray(values, dtype=np.float64)[order],
    )


def load_scalar(directory, scalar_tag):
    """Read TensorBoard first; fall back to the CSV scalar mirror."""
    directory = pathlib.Path(directory)
    if not directory.exists():
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    try:
        ea = event_accumulator.EventAccumulator(
            str(directory), size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        if scalar_tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(scalar_tag)
            if events:
                return (
                    np.asarray([e.step for e in events], dtype=np.float64),
                    np.asarray([e.value for e in events], dtype=np.float64),
                )
    except Exception:
        pass

    return _load_scalar_csv(directory, scalar_tag)


def final_scalar(directory, scalar_tag):
    _, values = load_scalar(directory, scalar_tag)
    return float(values[-1]) if values.size else float("nan")


def load_continual_scalar(
    runs_root,
    suite,
    condition,
    seed,
    task_sequence,
    total_timesteps,
    scalar_tag,
):
    """Concatenate a scalar across the continual stream on one x-axis."""
    xs, ys = [], []
    for seq_idx, task_id in enumerate(task_sequence):
        x, y = load_scalar(
            event_dir(runs_root, suite, condition, seed, seq_idx, task_id),
            scalar_tag,
        )
        if x.size:
            xs.append(x + seq_idx * (total_timesteps + 1))
            ys.append(y)
    if not xs:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.concatenate(xs), np.concatenate(ys)


def _prepare_curve(steps, values):
    """Sort a TensorBoard curve and keep the last value for duplicate steps.

    run_ppo_continual.py can evaluate exactly at the final periodic-evaluation
    step and then evaluate once more after training.  TensorBoard can therefore
    contain duplicate x-values.  Removing duplicates makes AUC deterministic
    and prevents accidental dependence on event ordering.
    """
    steps = np.asarray(steps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(steps) & np.isfinite(values)
    steps, values = steps[finite], values[finite]
    if steps.size == 0:
        return steps, values

    order = np.argsort(steps, kind="stable")
    steps, values = steps[order], values[order]

    # Reverse-unique -> last occurrence of every step, then restore order.
    _, rev_idx = np.unique(steps[::-1], return_index=True)
    keep = steps.size - 1 - rev_idx
    keep.sort()
    return steps[keep], values[keep]


def _auc(steps, values):
    """Time-average a learning curve over its observed step interval."""
    steps, values = _prepare_curve(steps, values)
    if steps.size < 2:
        return None
    span = float(steps[-1] - steps[0])
    if span <= 0:
        return None
    return float(_trapz(values, steps) / span)


# ============================================================================
# Checkpoint evaluation
# ============================================================================
def evaluate_checkpoint(
    run_dir,
    suite,
    task_id,
    episodes,
    seed,
    device,
    *,
    frozen_policy="pool",
    action_mode="deterministic",
    success_threshold=None,
):
    """Evaluate one checkpoint using the shared Atari checkpoint evaluator."""
    from checkpoint_evaluation import evaluate

    return evaluate(
        run_dir,
        suite,
        task_id,
        episodes,
        seed,
        device,
        frozen_policy=frozen_policy,
        action_mode=action_mode,
        success_threshold=success_threshold,
    )


def adapt_and_evaluate_checkpoint(
    run_dir,
    suite,
    task_id,
    episodes,
    seed,
    device,
    *,
    adapt_steps=0,
    adapt_lr=1e-2,
    frozen_policy="pool",
    action_mode="deterministic",
    success_threshold=None,
):
    """Evaluate with optional, explicitly-counted test-time mixture adaptation."""
    from checkpoint_evaluation import evaluate

    return evaluate(
        run_dir,
        suite,
        task_id,
        episodes,
        seed,
        device,
        adapt_steps=adapt_steps,
        adapt_lr=adapt_lr,
        frozen_policy=frozen_policy,
        action_mode=action_mode,
        success_threshold=success_threshold,
    )


def _evaluate_metric_checkpoint(args, run_dir, suite, task_id, episodes, seed, device):
    return adapt_and_evaluate_checkpoint(
        run_dir,
        suite,
        task_id,
        episodes,
        seed,
        device,
        adapt_steps=int(getattr(args, "test_adapt_steps", 0)),
        adapt_lr=float(getattr(args, "test_adapt_lr", 1e-2)),
        frozen_policy=getattr(args, "frozen_eval_policy", "pool"),
        action_mode=getattr(args, "eval_action_mode", "deterministic"),
        success_threshold=_threshold(args, suite, task_id),
    )


# ============================================================================
# Cache identity
# ============================================================================
def _benchmark_cache_config(args):
    """Configuration knobs that materially determine cached Atari metrics."""
    keys = (
        "composition_spaces",
        "policy_student_replay",
        "projection_epochs",
        "projection_max_samples",
        "frozen_eval_policy",
        "eval_action_mode",
        "skip_forward_transfer",
        "task_sequence",
        "save_root",
        "runs_root",
        "analysis_root",
        "plots_root",
        "total_timesteps",
        "eval_every",
        "num_evals",
        "retention_eval_episodes",
        "test_adapt_steps",
        "test_adapt_lr",
        "success_thresholds",
        "learning_rate",
        "num_envs",
        "num_steps",
        "anneal_lr",
        "gamma",
        "gae_lambda",
        "num_minibatches",
        "update_epochs",
        "norm_adv",
        "clip_coef",
        "clip_vloss",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
        "torch_deterministic",
        "pool_size",
        "alpha_init",
        "alpha_major",
        "alpha_factor",
        "fix_alpha",
        "alpha_learning_rate",
        "alpha_mass_learning_rate",
        "alpha_warmup_steps",
        "alpha_entropy_reg",
        "alpha_mass_reg",
        "constrain_alpha_mass",
        "condition_alpha_scale",
        "use_alpha_scale",
        "fix_alpha_scale",
        "weight_use_alpha_mass",
        "encoder_from_base",
        "train_shared",
        "freeze_root_encoder",
        "pretrained_encoder",
        "shared_dim",
        "head_hidden_dim",
        "distill_encoder_lr_mult",
        "drift_reg",
        "distill_extra_steps",
        "collect_cosine_buffers",
        "max_distill_buffer",
        "similarity_samples",
        "balance_source_lineages",
        "distill_max_samples",
        "distill_epochs",
        "distill_lr",
        "distill_batch_size",
        "distill_test_frac",
        "distill_select_best_val",
    )

    config = {}
    for key in keys:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if key == "task_sequence":
            config["sequence"] = list(value)
        elif key == "alpha_mass_learning_rate" and value is None:
            config[key] = getattr(args, "alpha_learning_rate", None)
        elif isinstance(value, pathlib.Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def _metric_checkpoint_signature(path):
    sig = checkpoint_signature(path)
    if sig is not None:
        return sig
    path = pathlib.Path(path)
    if not _is_custom_checkpoint_path(path):
        return None
    h = hashlib.sha256()
    found = 0
    for name in ("policy_snapshot.pt", "fc.pt", "policy_pool.pt"):
        file_path = path / name
        if not file_path.exists():
            continue
        found += 1
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest() if found else None


def _continual_checkpoint_signatures(args, suite, condition, seed):
    return [
        _metric_checkpoint_signature(
            checkpoint_dir(args.save_root, suite, condition, seed, seq_idx, task_id)
        )
        for seq_idx, task_id in enumerate(args.task_sequence)
    ]


def _scratch_checkpoint_signatures(
    args,
    suite,
    scratch_seeds: Sequence[int],
    scratch_total_timesteps: int,
):
    save_root = getattr(args, "scratch_save_root", scratch.SCRATCH_SAVE_ROOT)
    result = {}
    for task_id in sorted(set(args.task_sequence)):
        for seed in scratch_seeds:
            run_dir = scratch.scratch_checkpoint_dir(
                save_root,
                suite,
                task_id,
                scratch_total_timesteps,
                seed,
            )
            result[f"task_{task_id}/seed_{seed}"] = checkpoint_signature(run_dir)
    return result


def _validate_scratch_checkpoints(
    args,
    suite,
    scratch_seeds: Sequence[int],
    scratch_total_timesteps: int,
):
    """Require only the scratch learning curves consumed by Forward Transfer.

    Training/resume identity checks belong to scratch_baselines.py. Post-hoc
    metric computation must not suppress A_N/FG/BWT because a scratch manifest
    is missing, stale, or from a different runtime.
    """
    if not scratch_seeds or _CUSTOM_MODEL_MAP:
        return

    missing = []
    for _seq_idx, task_id in _first_unseen_positions(args.task_sequence):
        for scratch_seed in scratch_seeds:
            curve_dir = pathlib.Path(
                scratch.scratch_event_dir(
                    args.runs_root,
                    suite,
                    task_id,
                    scratch_total_timesteps,
                    scratch_seed,
                )
            )
            has_curve_file = (curve_dir / "scalars.csv").is_file() or any(
                curve_dir.glob("events.out.tfevents.*")
            )
            if not has_curve_file:
                missing.append(
                    f"task {task_id}, seed {scratch_seed}: "
                    f"no scratch learning-curve file found ({curve_dir})"
                )

    if missing:
        joined = "\n  - ".join(missing)
        raise RuntimeError(
            "Forward-transfer scratch baselines are missing:\n  - " + joined
        )


# ============================================================================
# Retention matrix
# ============================================================================
def retention_cache_path(args, suite, condition, seed):
    return (
        pathlib.Path(args.plots_root)
        / suite
        / "retention_data"
        / f"{condition}_seed_{seed}.json"
    )


def build_retention_matrix(args, suite, condition, seed, device):
    cache = retention_cache_path(args, suite, condition, seed)
    eval_task_ids = sorted(set(args.task_sequence))
    expected_config = _benchmark_cache_config(args)
    expected_signatures = _continual_checkpoint_signatures(
        args, suite, condition, seed
    )

    if cache.exists() and not getattr(args, "force_retrain", False):
        with cache.open() as f:
            cached = json.load(f)
        if (
            cached.get("cache_schema_version") == CACHE_SCHEMA_VERSION
            and cached.get("suite") == suite
            and cached.get("condition") == condition
            and int(cached.get("seed", -1)) == int(seed)
            and cached.get("cache_config") == expected_config
            and cached.get("checkpoint_signatures") == expected_signatures
            and cached.get("eval_task_ids") == eval_task_ids
            and int(cached.get("episodes", -1))
            == int(args.retention_eval_episodes)
        ):
            return cached

    data = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_config": expected_config,
        "checkpoint_signatures": expected_signatures,
        "suite": suite,
        "condition": condition,
        "seed": int(seed),
        "sequence": list(args.task_sequence),
        "eval_task_ids": eval_task_ids,
        "episodes": int(args.retention_eval_episodes),
        "frozen_eval_policy": getattr(args, "frozen_eval_policy", "pool"),
        "eval_action_mode": getattr(args, "eval_action_mode", "deterministic"),
        "test_adapt_steps_per_checkpoint_task": int(
            getattr(args, "test_adapt_steps", 0)
        ),
        "reward": [],
        "return": [],
        "success": [],
        "evaluation_interactions": [],
        "adaptation_interactions": [],
    }

    for seq_idx, trained_task in enumerate(args.task_sequence):
        run_dir = checkpoint_dir(
            args.save_root, suite, condition, seed, seq_idx, trained_task
        )
        if not checkpoint_complete(run_dir):
            raise FileNotFoundError(run_dir)

        rows = {
            "reward": [],
            "success": [],
            "evaluation_interactions": [],
            "adaptation_interactions": [],
        }
        for eval_task in eval_task_ids:
            result = _evaluate_metric_checkpoint(
                args,
                run_dir,
                suite,
                eval_task,
                args.retention_eval_episodes,
                seed,
                device,
            )
            rows["reward"].append(result["reward"])
            rows["success"].append(result["success"])
            rows["evaluation_interactions"].append(
                int(result.get("evaluation_interactions", 0))
            )
            rows["adaptation_interactions"].append(
                int(result.get("adaptation_interactions", 0))
            )

        data["reward"].append(rows["reward"])
        data["return"].append(list(rows["reward"]))
        data["success"].append(rows["success"])
        data["evaluation_interactions"].append(
            rows["evaluation_interactions"]
        )
        data["adaptation_interactions"].append(
            rows["adaptation_interactions"]
        )

        print(
            f"retention {suite}/{condition}/seed={seed}: "
            f"after seq{seq_idx} task {trained_task} done"
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        json.dump(data, f, indent=2)
    return data


# ============================================================================
# Diagonal + final row and A_N / FG / BWT
# ============================================================================
def _compute_diagonal_all(args, suite, condition, seed, device):
    """Evaluate each just-trained checkpoint ONCE and retain both metrics."""
    reward, success = {}, {}
    for seq_idx, task_id in enumerate(args.task_sequence):
        run_dir = checkpoint_dir(
            args.save_root, suite, condition, seed, seq_idx, task_id
        )
        if not checkpoint_complete(run_dir):
            raise FileNotFoundError(run_dir)
        result = _evaluate_metric_checkpoint(
            args,
            run_dir,
            suite,
            task_id,
            args.retention_eval_episodes,
            seed,
            device,
        )
        reward[str(seq_idx)] = result["reward"]
        success[str(seq_idx)] = result["success"]
    return {"reward": reward, "success": success}


def _compute_final_row_all(args, suite, condition, seed, device):
    """Evaluate the final checkpoint once on every unique task."""
    final_seq_idx = len(args.task_sequence) - 1
    final_task_id = args.task_sequence[final_seq_idx]
    final_run_dir = checkpoint_dir(
        args.save_root,
        suite,
        condition,
        seed,
        final_seq_idx,
        final_task_id,
    )
    if not checkpoint_complete(final_run_dir):
        raise FileNotFoundError(final_run_dir)

    reward, success = {}, {}
    for task_id in sorted(set(args.task_sequence)):
        result = _evaluate_metric_checkpoint(
            args,
            final_run_dir,
            suite,
            task_id,
            args.retention_eval_episodes,
            seed,
            device,
        )
        reward[str(task_id)] = result["reward"]
        success[str(task_id)] = result["success"]
    return {"reward": reward, "success": success}


def compute_fg_bwt(diagonal, final_row, task_sequence, prefix):
    """Compute forgetting and signed backward transfer for one scalar metric."""
    fg_values, bwt_values = [], []
    per_position = []

    for seq_idx in range(len(task_sequence) - 1):
        task_id = task_sequence[seq_idx]
        p_ii = diagonal.get(str(seq_idx), float("nan"))
        p_Ni = final_row.get(str(task_id), float("nan"))
        if not np.isfinite(p_ii) or not np.isfinite(p_Ni):
            continue

        bwt = float(p_Ni - p_ii)
        fg = float(max(p_ii - p_Ni, 0.0))
        bwt_values.append(bwt)
        fg_values.append(fg)
        per_position.append(
            {
                "seq_idx": int(seq_idx),
                "task_id": int(task_id),
                "p_ii": float(p_ii),
                "p_Ni": float(p_Ni),
                "FG": fg,
                "BWT": bwt,
            }
        )

    return {
        f"FG_{prefix}": float(np.mean(fg_values)) if fg_values else float("nan"),
        f"BWT_{prefix}": float(np.mean(bwt_values)) if bwt_values else float("nan"),
        f"FG_{prefix}_per_position": fg_values,
        f"BWT_{prefix}_per_position": bwt_values,
        f"{prefix}_retention_positions": per_position,
    }


def compute_A_N(final_row):
    values = [float(v) for v in final_row.values() if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def compute_p_diagonal(args, suite, condition, seed, device, metric="success"):
    """Public compatibility wrapper analogous to the HalfCheetah API.

    ``metric`` may be ``"success"`` or ``"reward"``.  The default remains
    success because that is the bounded survey-style performance measure when
    fixed thresholds are configured.
    """
    if metric not in ("reward", "success"):
        raise ValueError("metric must be 'reward' or 'success'")
    return _compute_diagonal_all(args, suite, condition, seed, device)[metric]


def compute_p_final_row(args, suite, condition, seed, device, metric="success"):
    """Public compatibility wrapper analogous to the HalfCheetah API."""
    if metric not in ("reward", "success"):
        raise ValueError("metric must be 'reward' or 'success'")
    return _compute_final_row_all(args, suite, condition, seed, device)[metric]


def _nan_fg_bwt(prefix):
    return {
        f"FG_{prefix}": float("nan"),
        f"BWT_{prefix}": float("nan"),
        f"FG_{prefix}_per_position": [],
        f"BWT_{prefix}_per_position": [],
        f"{prefix}_retention_positions": [],
    }


# ============================================================================
# Forward transfer
# ============================================================================
def _first_unseen_positions(task_sequence):
    seen = set()
    result = []
    for seq_idx, task_id in enumerate(task_sequence):
        if task_id not in seen and seq_idx > 0:
            result.append((seq_idx, task_id))
        seen.add(task_id)
    return result


def compute_forward_transfer_success(
    args,
    suite,
    condition,
    seed,
    scratch_seeds,
    scratch_total_timesteps,
):
    """Survey-style normalized forward transfer on thresholded success AUC."""
    eligible = _first_unseen_positions(args.task_sequence)
    if not scratch_seeds or not eligible:
        return {
            "FT_success": float("nan"),
            "FT_success_per_position": [],
            "FT_success_positions": [],
            "FT_success_complete": False,
        }

    # Do not silently average success over only the subset of tasks for which a
    # threshold happened to be supplied.  Aggregate success metrics are valid
    # only when the definition is fixed for every FT-eligible task.
    if not _success_thresholds_complete(args, suite, [task_id for _, task_id in eligible]):
        return {
            "FT_success": float("nan"),
            "FT_success_per_position": [],
            "FT_success_positions": [],
            "FT_success_complete": False,
        }

    per_position, positions = [], []
    complete = True

    for seq_idx, task_id in eligible:
        run_steps, run_values = load_scalar(
            event_dir(args.runs_root, suite, condition, seed, seq_idx, task_id),
            "charts/test_success",
        )
        auc = _auc(run_steps, run_values)
        if auc is None:
            complete = False
            continue

        baseline_aucs = []
        for b_seed in scratch_seeds:
            b_steps, b_values = load_scalar(
                scratch.scratch_event_dir(
                    args.runs_root,
                    suite,
                    task_id,
                    scratch_total_timesteps,
                    b_seed,
                ),
                "charts/test_success",
            )
            b_auc = _auc(b_steps, b_values)
            if b_auc is None:
                complete = False
                continue
            baseline_aucs.append(b_auc)

        if len(baseline_aucs) != len(scratch_seeds):
            complete = False
            continue

        auc_b = float(np.mean(baseline_aucs))
        if auc_b >= 1.0 - 1e-12:
            # No headroom remains in the survey normalization.
            complete = False
            continue

        value = float((auc - auc_b) / (1.0 - auc_b))
        per_position.append(value)
        positions.append(
            {
                "seq_idx": int(seq_idx),
                "task_id": int(task_id),
                "auc_continual": float(auc),
                "auc_scratch": float(auc_b),
                "FT_success": value,
            }
        )

    complete = complete and len(per_position) == len(eligible)
    return {
        "FT_success": (
            float(np.mean(per_position)) if complete and per_position else float("nan")
        ),
        "FT_success_per_position": per_position,
        "FT_success_positions": positions,
        "FT_success_complete": bool(complete),
    }


def compute_forward_transfer_reward(
    args,
    suite,
    condition,
    seed,
    scratch_seeds,
    scratch_total_timesteps,
):
    """Relative improvement in time-averaged RAW Atari score AUC.

    This is deliberately not presented as the literal bounded survey formula.
    It is an auxiliary scale-relative raw-score metric:

        (AUC_continual - AUC_scratch) / |AUC_scratch|.
    """
    eligible = _first_unseen_positions(args.task_sequence)
    if not scratch_seeds or not eligible:
        return {
            "FT_reward": float("nan"),
            "FT_reward_per_position": [],
            "FT_reward_positions": [],
            "FT_reward_complete": False,
        }

    per_position, positions = [], []
    complete = True

    for seq_idx, task_id in eligible:
        run_steps, run_values = load_scalar(
            event_dir(args.runs_root, suite, condition, seed, seq_idx, task_id),
            "charts/test_episodic_return",
        )
        auc = _auc(run_steps, run_values)
        if auc is None:
            complete = False
            continue

        baseline_aucs = []
        for b_seed in scratch_seeds:
            b_steps, b_values = load_scalar(
                scratch.scratch_event_dir(
                    args.runs_root,
                    suite,
                    task_id,
                    scratch_total_timesteps,
                    b_seed,
                ),
                "charts/test_episodic_return",
            )
            b_auc = _auc(b_steps, b_values)
            if b_auc is None:
                complete = False
                continue
            baseline_aucs.append(b_auc)

        if len(baseline_aucs) != len(scratch_seeds):
            complete = False
            continue

        auc_b = float(np.mean(baseline_aucs))
        denom = abs(auc_b)
        if denom < _EPS:
            complete = False
            continue

        value = float((auc - auc_b) / denom)
        per_position.append(value)
        positions.append(
            {
                "seq_idx": int(seq_idx),
                "task_id": int(task_id),
                "auc_continual": float(auc),
                "auc_scratch": float(auc_b),
                "FT_reward": value,
            }
        )

    complete = complete and len(per_position) == len(eligible)
    return {
        "FT_reward": (
            float(np.mean(per_position)) if complete and per_position else float("nan")
        ),
        "FT_reward_per_position": per_position,
        "FT_reward_positions": positions,
        "FT_reward_complete": bool(complete),
    }


# ============================================================================
# Survey metric orchestration / cache
# ============================================================================
def survey_metrics_cache_path(args, suite, condition, seed):
    return (
        pathlib.Path(args.plots_root)
        / suite
        / "survey_metrics"
        / f"{condition}_seed_{seed}.json"
    )


def compute_survey_metrics(
    args,
    suite,
    condition,
    seed,
    device,
    scratch_seeds,
    scratch_total_timesteps,
):
    scratch_seeds = [int(x) for x in scratch_seeds]

    ft_available = (
        bool(scratch_seeds)
        and not bool(getattr(args, "skip_forward_transfer", False))
    )
    if ft_available:
        try:
            _validate_scratch_checkpoints(
                args,
                suite,
                scratch_seeds,
                scratch_total_timesteps,
            )
        except RuntimeError as exc:
            print(
                f"[metrics] A_N/FG/BWT will be computed; "
                f"FT unavailable: {exc}"
            )
            ft_available = False

    cache = survey_metrics_cache_path(args, suite, condition, seed)
    cache_config = {
        **_benchmark_cache_config(args),
        "ft_available": bool(ft_available),
        "retention_eval_episodes": int(args.retention_eval_episodes),
        "scratch_seeds": scratch_seeds,
        "scratch_total_timesteps": int(scratch_total_timesteps),
        "scratch_save_root": str(
            getattr(args, "scratch_save_root", scratch.SCRATCH_SAVE_ROOT)
        ),
    }

    continual_signatures = _continual_checkpoint_signatures(
        args, suite, condition, seed
    )
    scratch_signatures = _scratch_checkpoint_signatures(
        args,
        suite,
        scratch_seeds,
        scratch_total_timesteps,
    )

    if cache.exists() and not getattr(args, "force_retrain", False):
        with cache.open() as f:
            cached = json.load(f)
        if (
            cached.get("cache_schema_version") == CACHE_SCHEMA_VERSION
            and cached.get("suite") == suite
            and cached.get("condition") == condition
            and int(cached.get("seed", -1)) == int(seed)
            and cached.get("cache_config") == cache_config
            and cached.get("checkpoint_signatures") == continual_signatures
            and cached.get("scratch_checkpoint_signatures") == scratch_signatures
        ):
            return cached

    diagonal = _compute_diagonal_all(
        args, suite, condition, seed, device
    )
    final_row = _compute_final_row_all(
        args, suite, condition, seed, device
    )

    reward_diag = diagonal["reward"]
    reward_final = final_row["reward"]
    success_diag = diagonal["success"]
    success_final = final_row["success"]

    reward_fg_bwt = compute_fg_bwt(
        reward_diag, reward_final, args.task_sequence, "reward"
    )

    unique_tasks = sorted(set(args.task_sequence))
    success_complete = _success_thresholds_complete(
        args, suite, unique_tasks
    )
    if success_complete:
        A_N_success = compute_A_N(success_final)
        success_fg_bwt = compute_fg_bwt(
            success_diag, success_final, args.task_sequence, "success"
        )
    else:
        A_N_success = float("nan")
        success_fg_bwt = _nan_fg_bwt("success")

    if ft_available:
        ft_reward = compute_forward_transfer_reward(
            args,
            suite,
            condition,
            seed,
            scratch_seeds,
            scratch_total_timesteps,
        )
        ft_success = compute_forward_transfer_success(
            args,
            suite,
            condition,
            seed,
            scratch_seeds,
            scratch_total_timesteps,
        )
    else:
        ft_reward = {
            "FT_reward": float("nan"),
            "FT_reward_per_position": [],
            "FT_reward_positions": [],
            "FT_reward_complete": False,
        }
        ft_success = {
            "FT_success": float("nan"),
            "FT_success_per_position": [],
            "FT_success_positions": [],
            "FT_success_complete": False,
        }

    result = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_config": cache_config,
        "checkpoint_signatures": continual_signatures,
        "scratch_checkpoint_signatures": scratch_signatures,
        "suite": suite,
        "condition": condition,
        "seed": int(seed),
        "sequence": list(args.task_sequence),
        "ft_available": bool(ft_available),
        "success_thresholds_complete": bool(success_complete),
        "success_thresholds": getattr(args, "success_thresholds", {}),

        # Complementary raw-score family. Keep suites separate: Freeway and
        # SpaceInvaders raw scores must not be pooled into one global average.
        "A_N_reward": compute_A_N(reward_final),
        **reward_fg_bwt,
        **ft_reward,

        # Bounded survey-style family, only defined with complete thresholds.
        "A_N_success": A_N_success,
        **success_fg_bwt,
        **ft_success,

        "p_diagonal_reward": reward_diag,
        "p_final_row_reward": reward_final,
        "p_diagonal_success": success_diag,
        "p_final_row_success": success_final,
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        json.dump(result, f, indent=2)
    return result

