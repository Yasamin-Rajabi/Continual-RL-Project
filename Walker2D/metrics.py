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
- FT  (forward)   : two variants, averaged only over FIRST encounters of
                    previously unseen tasks after the initial stream task. A
                    repeated task is relearning/savings, not forward transfer.
      FT_success  : the literal survey formula, AUC over test_success in
                    [0,1] vs. a from-scratch baseline's AUC.
      FT_return   : (AUC_return - AUC_scratch) / (U - AUC_scratch),
                    with the suite-specific achievable return upper bound U.
                    FG/BWT use matching frozen/adapted checkpoint evaluations.
                    FT comes from monitored active-policy learning curves.
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator

from cka_rl import FrozenCkaPolicy
from policy_utils import bound_log_std
from tasks import get_task
import scratch_baselines as scratch
from experiment_identity import (
    MANIFEST_NAME,
    checkpoint_matches as _identity_checkpoint_matches,
    checkpoint_signature,
    load_manifest,
)

# NumPy 2.0 removed np.trapz in favor of np.trapezoid; NumPy <2.0 only has
# np.trapz. Picking whichever exists at import time keeps this file working
# regardless of which NumPy version is installed (e.g. on Kaggle vs. locally).
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ==========================================================================
# Optional custom checkpoint mapping used by run_eval_custom.py. Normal
# training leaves this empty, so resumable benchmark paths are unchanged.
# ==========================================================================
_CUSTOM_MODEL_MAP = {}


def set_custom_model_map(mapping):
    global _CUSTOM_MODEL_MAP
    _CUSTOM_MODEL_MAP = {str(k): pathlib.Path(v) for k, v in dict(mapping).items()}


def _is_custom_checkpoint_path(path):
    path = pathlib.Path(path).resolve()
    for root in _CUSTOM_MODEL_MAP.values():
        try:
            path.relative_to(root.resolve())
            return True
        except ValueError:
            pass
    return False


# ==========================================================================
# Path helpers (the single source of truth -- plots.py and
# run_continual_benchmark.py both import these from here).
# ==========================================================================
def run_name(suite, task_id, seed):
    return f"{suite}__task_{task_id}__cka-rl__run_sac__{seed}"


def event_dir(runs_root, suite, condition, seed, seq_idx, task_id):
    if condition in _CUSTOM_MODEL_MAP:
        custom_base = _CUSTOM_MODEL_MAP[condition]
        sibling_runs = custom_base.parent / "runs"
        if sibling_runs.exists():
            return sibling_runs / f"seq_{seq_idx}" / run_name(suite, task_id, seed)
        return custom_base / f"seq_{seq_idx}" / run_name(suite, task_id, seed)
    tag = f"{suite}/{condition}/seed_{seed}/seq_{seq_idx}"
    return pathlib.Path(runs_root) / tag / run_name(suite, task_id, seed)


def checkpoint_dir(save_root, suite, condition, seed, seq_idx, task_id):
    if condition in _CUSTOM_MODEL_MAP:
        return _CUSTOM_MODEL_MAP[condition] / f"seq_{seq_idx}" / run_name(suite, task_id, seed)
    return (
        pathlib.Path(save_root) / suite / condition / f"seed_{seed}"
        / f"seq_{seq_idx}" / run_name(suite, task_id, seed)
    )


def analysis_snapshot_path(analysis_root, suite, condition, seed, seq_idx, task_id):
    tag = pathlib.Path(suite) / condition / f"seed_{seed}" / f"seq_{seq_idx}"
    return pathlib.Path(analysis_root) / tag / run_name(suite, task_id, seed) / "post_finalize.pt"


def checkpoint_complete(path):
    path = pathlib.Path(path)
    core = ["policy_snapshot.pt", "fc.pt", "mean_pool.pt", "logstd_pool.pt"]
    if not (path.exists() and all((path / name).exists() for name in core)):
        return False
    # Custom evaluation intentionally supports older/arbitrary checkpoints that
    # predate run_manifest.json. Normal training/resume still requires it.
    if _is_custom_checkpoint_path(path):
        return True
    return (path / MANIFEST_NAME).exists() and load_manifest(path) is not None


def checkpoint_matches(path, expected_mapping, *, parent_dirs=(), pretrained_encoder=None):
    """Validate the complete training identity of a resumable checkpoint."""
    if not checkpoint_complete(path):
        return False, "checkpoint files or valid run_manifest.json are missing"
    return _identity_checkpoint_matches(
        path, expected_mapping, parent_dirs=parent_dirs,
        pretrained_encoder=pretrained_encoder,
    )


CACHE_SCHEMA_VERSION = 7
ERROR_KEY = "velocity_error"
EPISODIC_SUCCESS = False
RETURN_UPPER_BOUND = 1000.0


def _benchmark_cache_config(args):
    """Configuration knobs that materially determine cached metrics."""
    keys = (
        "composition_spaces", "projection_epochs", "projection_max_samples",
        "frozen_eval_policy", "eval_action_mode", "skip_forward_transfer",
        "task_sequence", "save_root", "runs_root", "analysis_root",
        "total_timesteps", "learning_starts", "random_actions_end",
        "batch_size", "policy_lr", "alpha_lr", "alpha_mass_lr", "alpha_warmup_steps", "alpha_entropy_reg",
        "distill_encoder_lr_mult", "q_lr", "gamma", "tau", "alpha",
        "autotune", "autotune_init_from_alpha", "pool_size", "eval_every",
        "num_evals", "test_adapt_steps", "test_adapt_lr", "distill_observation_skip",
        "distill_extra_steps", "collect_cosine_buffers",
        "max_distill_buffer", "similarity_samples", "balance_source_lineages", "distill_max_samples",
        "distill_epochs", "distill_lr", "distill_batch_size",
        "distill_test_frac", "distill_select_best_val", "train_shared",
        "freeze_root_encoder", "encoder_from_base", "pretrained_encoder",
        "encoder_linear_out", "condition_alpha_scale", "use_alpha_scale", "fix_alpha_scale",
        "weight_use_alpha_mass", "alpha_mass_reg", "drift_reg", "constrain_alpha_mass",
    )
    config = {}
    for key in keys:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if key == "task_sequence":
            config["sequence"] = list(value)
        elif key == "alpha_mass_lr" and value is None:
            config[key] = float(getattr(args, "alpha_lr"))
        elif isinstance(value, pathlib.Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


def _continual_checkpoint_signatures(args, suite, condition, seed):
    signatures = []
    for seq_idx, task_id in enumerate(args.task_sequence):
        run_dir = checkpoint_dir(args.save_root, suite, condition, seed, seq_idx, task_id)
        signatures.append(checkpoint_signature(run_dir))
    return signatures


def _scratch_variant(args, condition):
    # Custom-model evaluation cannot infer architecture from an arbitrary label;
    # expose one explicit choice there. Normal benchmark conditions are known.
    if _CUSTOM_MODEL_MAP:
        return getattr(args, "scratch_variant", "plain")
    return scratch.variant_for_condition(
        condition, bool(getattr(args, "distill_observation_skip", False))
    )


def _scratch_checkpoint_signatures(args, suite, condition, scratch_seeds, scratch_total_timesteps):
    save_root = getattr(args, "scratch_save_root", scratch.SCRATCH_SAVE_ROOT)
    variant = _scratch_variant(args, condition)
    result = {}
    for task_id in sorted(set(args.task_sequence)):
        for seed in scratch_seeds:
            run_dir = scratch.scratch_checkpoint_dir(
                save_root, suite, task_id, scratch_total_timesteps, seed, variant
            )
            result[f"{variant}/task_{task_id}/seed_{seed}"] = checkpoint_signature(run_dir)
    return result


def _validate_scratch_checkpoints(
    args, suite, condition, seed, scratch_seeds, scratch_total_timesteps
):
    """Require only that the scratch learning-curve files exist.

    Forward Transfer consumes saved TensorBoard/CSV learning curves. Do not
    compare manifests, package/runtime versions, source identity, training
    configuration, or checkpoint compatibility here. Those checks belong to
    training/resume, not post-hoc metric computation.
    """
    del seed  # FT scratch availability does not depend on the continual seed.
    if _CUSTOM_MODEL_MAP:
        return

    variant = _scratch_variant(args, condition)
    missing = []

    for _seq_idx, task_id in _first_unseen_positions(args.task_sequence):
        for scratch_seed in scratch_seeds:
            curve_dir = pathlib.Path(scratch.scratch_event_dir(
                args.runs_root, suite, task_id, scratch_total_timesteps,
                scratch_seed, variant
            ))
            has_curve_file = (curve_dir / "scalars.csv").is_file() or any(
                curve_dir.glob("events.out.tfevents.*")
            )
            if not has_curve_file:
                missing.append(
                    f"task {task_id}, seed {scratch_seed}: no scratch learning-curve "
                    f"file found ({curve_dir})"
                )

    if missing:
        joined = "\n  - ".join(missing)
        raise RuntimeError(
            "Forward-transfer scratch baselines are missing:\n  - " + joined
        )

def _load_scalar_csv(directory, scalar_tag):
    """Fallback reader for the scalars.csv mirror written next to TensorBoard."""
    path = pathlib.Path(directory) / "scalars.csv"
    if not path.exists():
        return np.empty(0), np.empty(0)
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
        return np.empty(0), np.empty(0)
    if not steps:
        return np.empty(0), np.empty(0)
    order = np.argsort(np.asarray(steps, dtype=np.float64), kind="stable")
    return (
        np.asarray(steps, dtype=np.float64)[order],
        np.asarray(values, dtype=np.float64)[order],
    )


def load_scalar(directory, scalar_tag):
    directory = pathlib.Path(directory)
    if not directory.exists():
        return np.empty(0), np.empty(0)
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
    # TensorBoard remains the primary source. CSV is a byte-simple backup and
    # makes post-hoc FT robust if an event file is missing/corrupted.
    return _load_scalar_csv(directory, scalar_tag)


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
def evaluate_checkpoint(run_dir, suite, task_id, episodes, seed, device, *,
                        frozen_policy="pool", action_mode="deterministic"):
    from checkpoint_evaluation import evaluate
    return evaluate(run_dir, suite, task_id, episodes, seed, device,
                    frozen_policy=frozen_policy, action_mode=action_mode,
                    error_key=ERROR_KEY, episodic_success=EPISODIC_SUCCESS)


def adapt_and_evaluate_checkpoint(run_dir, suite, task_id, episodes, seed, device, *,
                                  adapt_steps=0, adapt_lr=1e-2, frozen_policy="pool",
                                  action_mode="deterministic"):
    from checkpoint_evaluation import evaluate
    return evaluate(run_dir, suite, task_id, episodes, seed, device,
                    adapt_steps=adapt_steps, adapt_lr=adapt_lr,
                    frozen_policy=frozen_policy, action_mode=action_mode,
                    error_key=ERROR_KEY, episodic_success=EPISODIC_SUCCESS)


def _evaluate_metric_checkpoint(args, run_dir, suite, task_id, episodes, seed, device):
    return adapt_and_evaluate_checkpoint(
        run_dir, suite, task_id, episodes, seed, device,
        adapt_steps=int(getattr(args, "test_adapt_steps", 0)),
        adapt_lr=float(getattr(args, "test_adapt_lr", 1e-2)),
        frozen_policy=getattr(args, "frozen_eval_policy", "pool"),
        action_mode=getattr(args, "eval_action_mode", "deterministic"))


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
        expected_config = _benchmark_cache_config(args)
        expected_signatures = _continual_checkpoint_signatures(args, suite, condition, seed)
        if (
            cached.get("cache_schema_version") == CACHE_SCHEMA_VERSION
            and cached.get("suite") == suite
            and cached.get("condition") == condition
            and int(cached.get("seed", -1)) == int(seed)
            and cached.get("cache_config") == expected_config
            and cached.get("checkpoint_signatures") == expected_signatures
            and cached.get("eval_task_ids") == eval_task_ids
            and int(cached.get("episodes", -1)) == int(args.retention_eval_episodes)
        ):
            return cached

    data = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_config": _benchmark_cache_config(args),
        "checkpoint_signatures": _continual_checkpoint_signatures(args, suite, condition, seed),
        "suite": suite,
        "condition": condition,
        "seed": seed,
        "sequence": list(args.task_sequence),
        "eval_task_ids": eval_task_ids,
        "episodes": int(args.retention_eval_episodes),
        "frozen_eval_policy": getattr(args, "frozen_eval_policy", "pool"),
        "eval_action_mode": getattr(args, "eval_action_mode", "deterministic"),
        "test_adapt_steps_per_checkpoint_task": int(getattr(args, "test_adapt_steps", 0)),
        "return": [],
        "success": [],
        "evaluation_interactions": [],
        "adaptation_interactions": [],
        "velocity_error": [],
    }
    for seq_idx, trained_task in enumerate(args.task_sequence):
        run_dir = checkpoint_dir(args.save_root, suite, condition, seed, seq_idx, trained_task)
        if not checkpoint_complete(run_dir):
            raise FileNotFoundError(run_dir)
        rows = {metric: [] for metric in ("return", "success", "velocity_error", "evaluation_interactions", "adaptation_interactions")}
        for eval_task in eval_task_ids:
            metrics = _evaluate_metric_checkpoint(
                args, run_dir, suite, eval_task, args.retention_eval_episodes,
                seed, device,
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
# Survey metrics: A_N, FG, BWT. The diagonal and final row are both
# re-evaluated from checkpoints under the same test-adaptation protocol.
# ==========================================================================
def compute_p_diagonal(args, suite, condition, seed, device):
    """p_{i,i}: evaluate each just-trained checkpoint on its own task.

    This uses the SAME test-time adaptation settings, deterministic policy
    evaluation, episode count, and base episode seeds as p_{N,i}. Reading the
    training-time TensorBoard scalar would compare an unadapted diagonal with an
    adapted final row whenever --test-adapt-steps > 0.
    """
    diagonal = {}
    for seq_idx, task_id in enumerate(args.task_sequence):
        run_dir = checkpoint_dir(args.save_root, suite, condition, seed, seq_idx, task_id)
        if not checkpoint_complete(run_dir):
            raise FileNotFoundError(run_dir)
        result = _evaluate_metric_checkpoint(
            args, run_dir, suite, task_id, args.retention_eval_episodes, seed, device
        )
        diagonal[str(seq_idx)] = result["success"]
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
        result = _evaluate_metric_checkpoint(
            args, final_run_dir, suite, task_id, args.retention_eval_episodes,
            seed, device,
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


def _first_unseen_positions(task_sequence):
    """Positions eligible for forward transfer: first encounter of each new task.

    Position 0 is excluded because there is no prior continual experience. Later
    repetitions are relearning/savings and must not be mislabeled as FT.
    """
    seen = set()
    result = []
    for seq_idx, task_id in enumerate(task_sequence):
        if task_id not in seen and seq_idx > 0:
            result.append((seq_idx, task_id))
        seen.add(task_id)
    return result


def compute_forward_transfer_success(args, suite, condition, seed, scratch_seeds, scratch_total_timesteps):
    """Survey Eq. 9, literal: p_i(t) = charts/test_success, already in
    [0,1]. Averaged only over first encounters of previously unseen tasks
    after sequence position 0. Repeated tasks measure relearning/savings, not FT."""
    per_position = []
    per_position_index = []
    scratch_variant = _scratch_variant(args, condition)
    for seq_idx, task_id in _first_unseen_positions(args.task_sequence):
        run_steps, run_values = load_scalar(
            event_dir(args.runs_root, suite, condition, seed, seq_idx, task_id),
            "charts/test_success",
        )
        auc = _auc(run_steps, run_values)
        if auc is None:
            continue

        baseline_aucs = []
        for b_seed in scratch_seeds:
            b_dir = scratch.scratch_event_dir(
                args.runs_root, suite, task_id, scratch_total_timesteps, b_seed, scratch_variant
            )
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
        per_position_index.append({"seq_idx": int(seq_idx), "task_id": int(task_id)})

    return {
        "FT_success": float(np.mean(per_position)) if per_position else float("nan"),
        "FT_success_per_position": per_position,
        "FT_success_positions": per_position_index,
    }


def compute_forward_transfer_return(args, suite, condition, seed, scratch_seeds, scratch_total_timesteps):
    """Return-AUC transfer with a valid environment-specific upper bound.

    (AUC_return - AUC_scratch) / (return_upper_bound - AUC_scratch).
    This is 1 - R/R_b only for HalfCheetah (upper bound zero). Walker2D
    and Hopper have a survival reward, so the zero-bound shortcut is invalid.
    MetaWorld uses its retained shaped-reward bound, 10 * 200 = 2000.
    """
    per_position, positions = [], []
    variant = _scratch_variant(args, condition)
    for seq_idx, task_id in _first_unseen_positions(args.task_sequence):
        steps, values = load_scalar(event_dir(args.runs_root, suite, condition, seed, seq_idx, task_id),
                                    "charts/test_episodic_return")
        auc = _auc(steps, values)
        if auc is None:
            continue
        baselines = []
        for baseline_seed in scratch_seeds:
            directory = scratch.scratch_event_dir(args.runs_root, suite, task_id,
                                                 scratch_total_timesteps, baseline_seed, variant)
            bs, bv = load_scalar(directory, "charts/test_episodic_return")
            value = _auc(bs, bv)
            if value is not None and np.isfinite(value):
                baselines.append(value)
        if not baselines:
            continue
        reference = float(np.mean(baselines))
        denominator = RETURN_UPPER_BOUND - reference
        if denominator <= 1e-12:
            continue
        per_position.append((auc - reference) / denominator)
        positions.append({"seq_idx": int(seq_idx), "task_id": int(task_id)})
    return {"FT_return": float(np.mean(per_position)) if per_position else float("nan"),
            "FT_return_per_position": per_position, "FT_return_positions": positions,
            "FT_return_upper_bound": RETURN_UPPER_BOUND}


# ==========================================================================
# Orchestrator: computes + caches everything above for one (suite,
# condition, seed).
# ==========================================================================
def survey_metrics_cache_path(args, suite, condition, seed):
    return pathlib.Path(args.plots_root) / suite / "survey_metrics" / f"{condition}_seed_{seed}.json"


def compute_survey_metrics(args, suite, condition, seed, device, scratch_seeds, scratch_total_timesteps):
    # Forward transfer only requires the expected scratch learning-curve
    # files to exist. Provenance/runtime compatibility is a training/resume
    # concern and is intentionally not re-checked during post-hoc metrics.
    ft_available = not bool(getattr(args, "skip_forward_transfer", False))
    if ft_available:
        try:
            _validate_scratch_checkpoints(args, suite, condition, seed, scratch_seeds, scratch_total_timesteps)
        except RuntimeError as exc:
            print(f"[metrics] A_N/FG/BWT will be computed; FT unavailable: {exc}")
            ft_available = False

    cache = survey_metrics_cache_path(args, suite, condition, seed)
    if cache.exists() and not args.force_retrain:
        with open(cache) as f:
            cached = json.load(f)
        expected_config = {
            **_benchmark_cache_config(args),
            "ft_available": ft_available,
            "retention_eval_episodes": int(args.retention_eval_episodes),
            "scratch_seeds": [int(x) for x in scratch_seeds],
            "scratch_total_timesteps": int(scratch_total_timesteps),
            "scratch_save_root": str(getattr(args, "scratch_save_root", scratch.SCRATCH_SAVE_ROOT)),
        }
        expected_checkpoint_signatures = _continual_checkpoint_signatures(args, suite, condition, seed)
        expected_scratch_signatures = _scratch_checkpoint_signatures(
            args, suite, condition, scratch_seeds, scratch_total_timesteps
        )
        if (
            cached.get("cache_schema_version") == CACHE_SCHEMA_VERSION
            and cached.get("suite") == suite
            and cached.get("condition") == condition
            and int(cached.get("seed", -1)) == int(seed)
            and cached.get("cache_config") == expected_config
            and cached.get("checkpoint_signatures") == expected_checkpoint_signatures
            and cached.get("scratch_checkpoint_signatures") == expected_scratch_signatures
        ):
            return cached

    diagonal = compute_p_diagonal(args, suite, condition, seed, device)
    final_row = compute_p_final_row(args, suite, condition, seed, device)
    fg_bwt = compute_fg_bwt(diagonal, final_row, args.task_sequence)
    ft_success = (compute_forward_transfer_success(args, suite, condition, seed, scratch_seeds, scratch_total_timesteps)
                  if ft_available else {"FT_success": float("nan"), "FT_success_per_position": []})
    ft_return = (compute_forward_transfer_return(args, suite, condition, seed, scratch_seeds, scratch_total_timesteps)
                 if ft_available else {"FT_return": float("nan"), "FT_return_per_position": []})

    result = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_config": {
            **_benchmark_cache_config(args),
            "ft_available": ft_available,
            "retention_eval_episodes": int(args.retention_eval_episodes),
            "scratch_seeds": [int(x) for x in scratch_seeds],
            "scratch_total_timesteps": int(scratch_total_timesteps),
            "scratch_save_root": str(getattr(args, "scratch_save_root", scratch.SCRATCH_SAVE_ROOT)),
        },
        "checkpoint_signatures": _continual_checkpoint_signatures(args, suite, condition, seed),
        "scratch_checkpoint_signatures": _scratch_checkpoint_signatures(
            args, suite, condition, scratch_seeds, scratch_total_timesteps
        ),
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
