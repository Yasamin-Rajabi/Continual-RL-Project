"""All numeric-metric computation for the continual Meta-World benchmark.

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
      FT_return   : algebraic reduction of the same formula assuming
                    r_max=0 -- NOT valid on Meta-World, whose rewards are
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


CACHE_SCHEMA_VERSION = 4


def _benchmark_cache_config(args):
    """Configuration knobs that materially determine cached metrics."""
    keys = (
        "task_sequence", "save_root", "runs_root", "analysis_root",
        "total_timesteps", "learning_starts", "random_actions_end",
        "batch_size", "policy_lr", "alpha_lr", "alpha_warmup_steps", "q_lr", "gamma", "tau", "alpha",
        "autotune", "autotune_init_from_alpha", "pool_size", "eval_every",
        "num_evals", "test_adapt_steps", "test_adapt_lr", "distill_observation_skip",
        "distill_extra_steps", "collect_cosine_buffers",
        "max_distill_buffer", "similarity_samples", "distill_max_samples",
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


def _scratch_checkpoint_signatures(args, suite, scratch_seeds, scratch_total_timesteps):
    save_root = getattr(args, "scratch_save_root", scratch.SCRATCH_SAVE_ROOT)
    result = {}
    for task_id in sorted(set(args.task_sequence)):
        for seed in scratch_seeds:
            run_dir = scratch.scratch_checkpoint_dir(
                save_root, suite, task_id, scratch_total_timesteps, seed
            )
            result[f"task_{task_id}/seed_{seed}"] = checkpoint_signature(run_dir)
    return result


def _validate_scratch_checkpoints(args, suite, scratch_seeds, scratch_total_timesteps):
    """Fail fast if FT would use missing or configuration-mismatched baselines.

    Cache signatures alone prevent stale JSON reuse, but direct metric calls must
    also reject a baseline trained with different encoder/SAC settings.
    """
    # Arbitrary custom-model evaluation cannot reconstruct the training CLI of
    # externally supplied checkpoints. Keep the normal benchmark strict, but
    # allow run_eval_custom.py to use user-supplied scratch baselines explicitly.
    if _CUSTOM_MODEL_MAP:
        return
    save_root = getattr(args, "scratch_save_root", scratch.SCRATCH_SAVE_ROOT)
    problems = []
    for task_id in sorted(set(args.task_sequence)):
        for seed in scratch_seeds:
            run_dir = scratch.scratch_checkpoint_dir(
                save_root, suite, task_id, scratch_total_timesteps, seed
            )
            matches, reason = scratch.checkpoint_matches(
                run_dir, suite, task_id, scratch_total_timesteps, seed, args
            )
            if not matches:
                problems.append(
                    f"task {task_id}, seed {seed}: {reason} ({run_dir})"
                )
    if problems:
        joined = "\n  - ".join(problems)
        raise RuntimeError(
            "Forward-transfer scratch baselines are missing or incompatible. "
            "Retrain them with scratch_baselines.py using the same training/encoder "
            f"settings as the continual run:\n  - {joined}"
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
    returns, success, task_errors = [], [], []
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
            ep_error.append(float(info.get("task_error", np.nan)))
            if terminated or truncated:
                break
        returns.append(ep_return)
        success.append(float(np.nanmean(ep_success)))
        task_errors.append(float(np.nanmean(ep_error)))
    env.close()
    return {
        "return": float(np.mean(returns)),
        "success": float(np.nanmean(success)),
        "task_error": float(np.nanmean(task_errors)),
    }


def adapt_and_evaluate_checkpoint(
    run_dir, suite, task_id, episodes, seed, device, *, adapt_steps=5_000, adapt_lr=1e-2
):
    """Evaluate after optional test-time adaptation of knowledge-mixture scalars.

    This integrates the friend's alpha-adaptation evaluator while keeping the
    normal frozen-checkpoint evaluator available with adapt_steps=0. Head/encoder
    weights stay frozen; only alpha, an enabled learnable alpha-scale, and
    alpha-mass are adapted.
    """
    if adapt_steps <= 0:
        return evaluate_checkpoint(run_dir, suite, task_id, episodes, seed, device)

    run_dir = pathlib.Path(run_dir)
    required = [run_dir / name for name in ("policy_snapshot.pt", "fc.pt", "mean_pool.pt", "logstd_pool.pt")]
    if not all(path.exists() for path in required):
        return evaluate_checkpoint(run_dir, suite, task_id, episodes, seed, device)

    snapshot = torch.load(run_dir / "policy_snapshot.pt", map_location=device, weights_only=False)
    mean_pool_data = torch.load(run_dir / "mean_pool.pt", map_location="cpu", weights_only=False)
    fusion_mode = getattr(mean_pool_data, "fusion_mode", "classic_cka")
    use_alpha_mass = bool(getattr(mean_pool_data, "use_alpha_mass", False))
    constrain_alpha_mass = bool(getattr(mean_pool_data, "constrain_alpha_mass", True))

    saved_scale = getattr(mean_pool_data, "alpha_scale", None)
    saved_scale_value = None if saved_scale is None else float(saved_scale.detach().cpu().reshape(-1)[0])
    saved_scale_trainable = bool(saved_scale is not None and saved_scale.requires_grad)
    fix_alpha_scale = bool(
        saved_scale is not None and (not saved_scale_trainable)
        and saved_scale_value is not None and abs(saved_scale_value - 5.0) < 1e-6
    )
    use_alpha_scale = bool(saved_scale_trainable and not fix_alpha_scale)

    from cka_rl import CkaRlAgent
    # A finalized pool no longer has a semantically valid pre-finalize alpha
    # vector. Start adaptation deterministically from a uniform mixture instead
    # of reusing stale/reindexed logits or random initialization.
    agent = CkaRlAgent(
        obs_dim=int(snapshot["obs_dim"]),
        act_dim=int(snapshot["act_dim"]),
        base_dir=None,
        latest_dir=str(run_dir),
        alpha_init="Uniform",
        fusion_mode=fusion_mode,
        use_alpha_mass=use_alpha_mass,
        constrain_alpha_mass=constrain_alpha_mass,
        use_alpha_scale=use_alpha_scale,
        fix_alpha_scale=fix_alpha_scale,
        distillation=bool(snapshot.get("distillation", False)),
        distill_observation_skip=bool(snapshot.get("distill_observation_skip", True)),
        encoder_linear_out=bool(snapshot.get("encoder_linear_out", False)),
        train_shared=False,
    ).to(device)

    # Remember intended alpha-scale trainability BEFORE freezing everything.
    adapt_scale = use_alpha_scale
    for param in agent.parameters():
        param.requires_grad_(False)

    adapt_params = []
    if agent.alpha is not None and agent.alpha.numel() > 1:
        agent.alpha.requires_grad_(True)
        adapt_params.append(agent.alpha)
    if adapt_scale and agent.alpha_scale is not None:
        agent.alpha_scale.requires_grad_(True)
        adapt_params.append(agent.alpha_scale)
    if use_alpha_mass and agent.alpha_mass is not None:
        agent.alpha_mass.requires_grad_(True)
        adapt_params.append(agent.alpha_mass)

    env = get_task(task_id, task_suite=suite)
    if adapt_params:
        optimizer = torch.optim.Adam(adapt_params, lr=adapt_lr)
        torch.manual_seed(int(seed))
        obs, _ = env.reset(seed=seed)
        for _ in range(int(adapt_steps)):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mean, raw_log_std = agent(obs_t)
            std = bound_log_std(raw_log_std).exp()
            dist = torch.distributions.Normal(mean, std)
            sampled = dist.sample()
            action = torch.tanh(sampled)[0].detach().cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Friend's test-time rule: one-step REINFORCE on the mixture scalars.
            # The sampled action is treated as the score-function sample.
            loss = -dist.log_prob(sampled).sum(dim=-1) * float(reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

    agent.eval()
    returns, success, task_errors = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 * task_id + ep)
        ep_return = 0.0
        ep_success, ep_error = [], []
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent(obs_t)
            action = torch.tanh(mean[0]).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_success.append(float(info.get("success", np.nan)))
            ep_error.append(float(info.get("task_error", np.nan)))
            if terminated or truncated:
                break
        returns.append(ep_return)
        success.append(float(np.nanmean(ep_success)))
        task_errors.append(float(np.nanmean(ep_error)))
    env.close()
    return {
        "return": float(np.mean(returns)),
        "success": float(np.nanmean(success)),
        "task_error": float(np.nanmean(task_errors)),
    }


def _evaluate_metric_checkpoint(args, run_dir, suite, task_id, episodes, seed, device):
    return adapt_and_evaluate_checkpoint(
        run_dir, suite, task_id, episodes, seed, device,
        adapt_steps=int(getattr(args, "test_adapt_steps", 0)),
        adapt_lr=float(getattr(args, "test_adapt_lr", 1e-2)),
    )


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
        "return": [],
        "success": [],
        "task_error": [],
    }
    for seq_idx, trained_task in enumerate(args.task_sequence):
        run_dir = checkpoint_dir(args.save_root, suite, condition, seed, seq_idx, trained_task)
        if not checkpoint_complete(run_dir):
            raise FileNotFoundError(run_dir)
        rows = {metric: [] for metric in ("return", "success", "task_error")}
        for eval_task in eval_task_ids:
            metrics = _evaluate_metric_checkpoint(
                args, run_dir, suite, eval_task, args.retention_eval_episodes,
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
        result = _evaluate_metric_checkpoint(
            args, final_run_dir, suite, task_id, args.retention_eval_episodes,
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
        per_position_index.append({"seq_idx": int(seq_idx), "task_id": int(task_id)})

    return {
        "FT_success": float(np.mean(per_position)) if per_position else float("nan"),
        "FT_success_per_position": per_position,
        "FT_success_positions": per_position_index,
    }


def _mean_normalised(values, steps, denom):
    """Mean of values/denom over the logged step range, clipped to [0,1].

    Meta-World returns are positive and bounded by MAX_REWARD_PER_STEP*HORIZON,
    so dividing by that bound puts episodic return on the same [0,1] scale the
    survey's forward-transfer formula assumes for success rates.
    """
    span = float(steps[-1] - steps[0])
    if span <= 0.0:
        return None
    return float(_trapz(np.clip(values / denom, 0.0, 1.0), steps)) / span


def compute_forward_transfer_return(args, suite, condition, seed, scratch_seeds, scratch_total_timesteps):
    """Survey Eq. 9 applied to episodic return.

    The HalfCheetah version used the algebraic shortcut FT_i = 1 - R_i/R_i^b,
    valid only when r_max = 0 (there reward was -|velocity_error| - ctrl_cost,
    always <= 0, so r_max cancelled). Meta-World rewards are POSITIVE, shaped
    into roughly [0, 10] per step, so that shortcut is wrong here -- it would
    invert the sign of the metric.

    Instead we normalise return onto [0,1] with the known bound
    MAX_REWARD_PER_STEP * HORIZON and apply the SAME literal formula as
    compute_forward_transfer_success. Identical code path, identical meaning,
    no assumption about the sign of the reward.

    On Meta-World this is strictly weaker than FT_success, since success is the
    benchmark's own criterion and return is only a shaped proxy. Report
    FT_success as the primary number.
    """
    from metaworld_envs import MAX_EPISODE_RETURN
    per_position = []
    per_position_index = []
    for seq_idx, task_id in _first_unseen_positions(args.task_sequence):
        run_steps, run_values = load_scalar(
            event_dir(args.runs_root, suite, condition, seed, seq_idx, task_id),
            "charts/test_episodic_return",
        )
        if run_steps.size < 2:
            continue
        # Mean normalised performance over the run, i.e. the trapezoid integral
        # divided by its own step span. Each curve is normalised by ITS OWN span
        # so a continual run and a scratch baseline stay comparable even if one
        # logged a slightly different step range.
        auc = _mean_normalised(run_values, run_steps, MAX_EPISODE_RETURN)
        if auc is None:
            continue

        baseline_aucs = []
        for b_seed in scratch_seeds:
            b_dir = scratch.scratch_event_dir(args.runs_root, suite, task_id, scratch_total_timesteps, b_seed)
            b_steps, b_values = load_scalar(b_dir, "charts/test_episodic_return")
            if b_steps.size >= 2:
                b_auc = _mean_normalised(b_values, b_steps, MAX_EPISODE_RETURN)
                if b_auc is not None:
                    baseline_aucs.append(b_auc)
        if not baseline_aucs:
            continue
        auc_b = float(np.mean(baseline_aucs))
        if auc_b >= 1.0:
            continue
        per_position.append((auc - auc_b) / (1.0 - auc_b))
        per_position_index.append({"seq_idx": int(seq_idx), "task_id": int(task_id)})

    return {
        "FT_return": float(np.mean(per_position)) if per_position else float("nan"),
        "FT_return_per_position": per_position,
        "FT_return_positions": per_position_index,
    }


# ==========================================================================
# Orchestrator: computes + caches everything above for one (suite,
# condition, seed).
# ==========================================================================
def survey_metrics_cache_path(args, suite, condition, seed):
    return pathlib.Path(args.plots_root) / suite / "survey_metrics" / f"{condition}_seed_{seed}.json"


def compute_survey_metrics(args, suite, condition, seed, device, scratch_seeds, scratch_total_timesteps):
    # Forward transfer is undefined unless its from-scratch denominator was
    # trained under the same SAC/encoder configuration. Validate before even
    # considering a cached metric file so direct callers cannot silently mix
    # incompatible experiments.
    _validate_scratch_checkpoints(
        args, suite, scratch_seeds, scratch_total_timesteps
    )

    cache = survey_metrics_cache_path(args, suite, condition, seed)
    if cache.exists() and not args.force_retrain:
        with open(cache) as f:
            cached = json.load(f)
        expected_config = {
            **_benchmark_cache_config(args),
            "retention_eval_episodes": int(args.retention_eval_episodes),
            "scratch_seeds": [int(x) for x in scratch_seeds],
            "scratch_total_timesteps": int(scratch_total_timesteps),
            "scratch_save_root": str(getattr(args, "scratch_save_root", scratch.SCRATCH_SAVE_ROOT)),
        }
        expected_checkpoint_signatures = _continual_checkpoint_signatures(args, suite, condition, seed)
        expected_scratch_signatures = _scratch_checkpoint_signatures(
            args, suite, scratch_seeds, scratch_total_timesteps
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

    diagonal = compute_p_diagonal(args, suite, condition, seed)
    final_row = compute_p_final_row(args, suite, condition, seed, device)
    fg_bwt = compute_fg_bwt(diagonal, final_row, args.task_sequence)
    ft_success = compute_forward_transfer_success(args, suite, condition, seed, scratch_seeds, scratch_total_timesteps)
    ft_return = compute_forward_transfer_return(args, suite, condition, seed, scratch_seeds, scratch_total_timesteps)

    result = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_config": {
            **_benchmark_cache_config(args),
            "retention_eval_episodes": int(args.retention_eval_episodes),
            "scratch_seeds": [int(x) for x in scratch_seeds],
            "scratch_total_timesteps": int(scratch_total_timesteps),
            "scratch_save_root": str(getattr(args, "scratch_save_root", scratch.SCRATCH_SAVE_ROOT)),
        },
        "checkpoint_signatures": _continual_checkpoint_signatures(args, suite, condition, seed),
        "scratch_checkpoint_signatures": _scratch_checkpoint_signatures(
            args, suite, scratch_seeds, scratch_total_timesteps
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
