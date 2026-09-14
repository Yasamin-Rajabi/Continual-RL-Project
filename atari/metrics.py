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

import hashlib
import json
import pathlib
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator

from atari_tasks import get_task
from cka_rl import CkaRlAgent, FrozenCkaPolicy
import scratch_baselines as scratch
from experiment_identity import (
    MANIFEST_NAME,
    checkpoint_matches as _identity_checkpoint_matches,
    checkpoint_signature,
    load_manifest,
)


_trapz = getattr(np, "trapezoid", None) or np.trapz


def _evaluation_seed(task_id: int, episode: int) -> int:
    """Fixed per-task evaluation seeds, independent of training/scratch seed."""
    return 10_000 + 10_000 * int(task_id) + int(episode)
CACHE_SCHEMA_VERSION = 13
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
def _torch_load(path, map_location=None):
    kwargs = {} if map_location is None else {"map_location": map_location}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:  # PyTorch versions predating weights_only=
        return torch.load(path, **kwargs)


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
def load_scalar(directory, scalar_tag):
    directory = pathlib.Path(directory)
    if not directory.exists():
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    try:
        ea = event_accumulator.EventAccumulator(
            str(directory), size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
    except Exception:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    if scalar_tag not in ea.Tags().get("scalars", []):
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    events = ea.Scalars(scalar_tag)
    return (
        np.asarray([e.step for e in events], dtype=np.float64),
        np.asarray([e.value for e in events], dtype=np.float64),
    )


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
# Evaluation environment / checkpoint evaluation
# ============================================================================
def _eval_env(suite, task_id):
    # Training uses reward clipping + EpisodicLife.  Paper metrics should use
    # the real full-game score instead.
    return get_task(
        task_id,
        task_suite=suite,
        clip_reward=False,
        episodic_life=False,
    )


@torch.no_grad()
def evaluate_checkpoint(
    run_dir,
    suite,
    task_id,
    episodes,
    seed,
    device,
    success_threshold=None,
):
    """Evaluate the exact pre-finalize policy snapshot deterministically."""
    if episodes < 1:
        raise ValueError("episodes must be >= 1")

    env = _eval_env(suite, task_id)
    policy = FrozenCkaPolicy.load(str(run_dir), map_location=device).to(device)
    policy.eval()

    returns, successes = [], []
    try:
        for ep in range(int(episodes)):
            obs, _ = env.reset(seed=_evaluation_seed(task_id, ep))
            ep_return = 0.0
            while True:
                x = (
                    torch.as_tensor(obs, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    / 255.0
                )
                logits = policy(x)
                action = int(torch.argmax(logits, dim=-1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                if terminated or truncated:
                    break

            returns.append(ep_return)
            if success_threshold is not None:
                successes.append(float(ep_return >= float(success_threshold)))
    finally:
        env.close()

    reward = float(np.mean(returns))
    success = float(np.mean(successes)) if successes else float("nan")
    return {"reward": reward, "return": reward, "success": success}


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
    success_threshold=None,
):
    """Optionally adapt only the knowledge-mixture scalars, then evaluate.

    The Atari analogue of the HalfCheetah evaluator uses a categorical
    REINFORCE score-function update.  Encoder/head parameters stay frozen;
    alpha, a learned alpha-scale, and alpha-mass are the only possible adapted
    parameters.
    """
    if adapt_steps < 0:
        raise ValueError("adapt_steps must be >= 0")
    if adapt_lr <= 0:
        raise ValueError("adapt_lr must be > 0")
    if adapt_steps <= 0:
        return evaluate_checkpoint(
            run_dir,
            suite,
            task_id,
            episodes,
            seed,
            device,
            success_threshold,
        )

    run_dir = pathlib.Path(run_dir)
    required = [run_dir / name for name in ("policy_snapshot.pt", "fc.pt", "policy_pool.pt")]
    if not all(path.exists() for path in required):
        return evaluate_checkpoint(
            run_dir,
            suite,
            task_id,
            episodes,
            seed,
            device,
            success_threshold,
        )

    snapshot = _torch_load(run_dir / "policy_snapshot.pt", map_location=device)
    pool_data = _torch_load(run_dir / "policy_pool.pt", map_location="cpu")

    fusion_mode = getattr(pool_data, "fusion_mode", "classic_cka")
    use_alpha_mass = bool(getattr(pool_data, "use_alpha_mass", False))
    constrain_alpha_mass = bool(getattr(pool_data, "constrain_alpha_mass", True))
    hidden_dim = int(getattr(pool_data, "hidden_dim", 128))
    pool_size = int(getattr(pool_data, "pool_size", max(len(getattr(pool_data, "pool", [])), 2)))

    saved_scale = getattr(pool_data, "alpha_scale", None)
    saved_scale_value = (
        None
        if saved_scale is None
        else float(saved_scale.detach().cpu().reshape(-1)[0])
    )
    saved_scale_trainable = bool(saved_scale is not None and saved_scale.requires_grad)
    fix_alpha_scale = bool(
        saved_scale is not None
        and not saved_scale_trainable
        and saved_scale_value is not None
        and abs(saved_scale_value - 5.0) < 1e-6
    )
    use_alpha_scale = bool(saved_scale_trainable and not fix_alpha_scale)

    # Finalization changes pool topology, so the alpha vector saved inside the
    # full pool is not an authoritative representation of the just-trained
    # pre-finalize policy.  As in the HalfCheetah evaluator, adaptation starts
    # from a deterministic uniform mixture over the finalized pool.
    agent = CkaRlAgent(
        obs_shape=tuple(snapshot["obs_shape"]),
        act_dim=int(snapshot["act_dim"]),
        shared_dim=int(snapshot.get("shared_dim", 512)),
        hidden_dim=hidden_dim,
        pool_size=pool_size,
        base_dir=None,
        latest_dir=str(run_dir),
        alpha_init="Uniform",
        fusion_mode=fusion_mode,
        use_alpha_mass=use_alpha_mass,
        constrain_alpha_mass=constrain_alpha_mass,
        use_alpha_scale=use_alpha_scale,
        fix_alpha_scale=fix_alpha_scale,
        distillation=bool(getattr(pool_data, "distillation", False)),
        train_shared=False,
    ).to(device)

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

    env = _eval_env(suite, task_id)
    try:
        if adapt_params:
            optimizer = torch.optim.Adam(adapt_params, lr=float(adapt_lr))
            torch.manual_seed(int(seed))
            obs, _ = env.reset(seed=int(seed))

            for _ in range(int(adapt_steps)):
                x = (
                    torch.as_tensor(obs, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    / 255.0
                )
                logits = agent(x)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))

                # One-step categorical REINFORCE on mixture scalars only.
                loss = -dist.log_prob(action).mean() * float(reward)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if terminated or truncated:
                    obs, _ = env.reset()
                else:
                    obs = next_obs

        agent.eval()
        returns, successes = [], []
        for ep in range(int(episodes)):
            obs, _ = env.reset(seed=_evaluation_seed(task_id, ep))
            ep_return = 0.0
            while True:
                x = (
                    torch.as_tensor(obs, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    / 255.0
                )
                with torch.no_grad():
                    logits = agent(x)
                action = int(torch.argmax(logits, dim=-1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                if terminated or truncated:
                    break

            returns.append(ep_return)
            if success_threshold is not None:
                successes.append(float(ep_return >= float(success_threshold)))
    finally:
        env.close()

    reward = float(np.mean(returns))
    success = float(np.mean(successes)) if successes else float("nan")
    return {"reward": reward, "return": reward, "success": success}


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
        success_threshold=_threshold(args, suite, task_id),
    )


# ============================================================================
# Cache identity
# ============================================================================
def _benchmark_cache_config(args):
    """Metric-relevant settings not already represented by checkpoint files."""
    keys = (
        "task_sequence",
        "save_root",
        "runs_root",
        "plots_root",
        "total_timesteps",
        "eval_every",
        "num_evals",
        "retention_eval_episodes",
        "test_adapt_steps",
        "test_adapt_lr",
        "success_thresholds",
        # Include major architecture/training knobs too.  Checkpoint signatures
        # already fingerprint these, but explicit cache metadata makes JSON
        # outputs self-describing and guards older manifests.
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
        "pool_size",
        "alpha_init",
        "alpha_major",
        "alpha_factor",
        "fix_alpha",
        "alpha_learning_rate",
        "alpha_warmup_steps",
        "alpha_entropy_reg",
        "alpha_mass_reg",
        "constrain_alpha_mass",
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
        "distill_max_samples",
        "distill_epochs",
        "distill_lr",
        "distill_batch_size",
        "distill_test_frac",
        "distill_select_best_val",
        "condition_alpha_scale",
        "use_alpha_scale",
        "fix_alpha_scale",
        "weight_use_alpha_mass",
        "torch_deterministic",
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
    """Strictly validate every scratch seed/task requested for FT.

    An empty scratch_seeds sequence is allowed: A_N/FG/BWT remain computable and
    FT metrics will simply be NaN.
    """
    if not scratch_seeds:
        return
    if _CUSTOM_MODEL_MAP:
        # Arbitrary external/custom model labels do not provide enough training
        # CLI identity to prove scratch parity.  run_eval_custom.py therefore
        # treats the explicitly supplied scratch baselines as user-authorized.
        return

    save_root = getattr(args, "scratch_save_root", scratch.SCRATCH_SAVE_ROOT)
    problems = []
    for task_id in sorted(set(args.task_sequence)):
        for seed in scratch_seeds:
            run_dir = scratch.scratch_checkpoint_dir(
                save_root,
                suite,
                task_id,
                scratch_total_timesteps,
                seed,
            )
            matches, reason = scratch.checkpoint_matches(
                run_dir,
                suite,
                task_id,
                scratch_total_timesteps,
                seed,
                args,
            )
            if not matches:
                problems.append(
                    f"task {task_id}, seed {seed}: {reason} ({run_dir})"
                )

    if problems:
        joined = "\n  - ".join(problems)
        raise RuntimeError(
            "Forward-transfer scratch baselines are missing or incompatible. "
            "Use scratch_baselines.py with the same PPO/encoder/evaluation "
            f"settings as the continual run:\n  - {joined}"
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
    expected_signatures = _continual_checkpoint_signatures(args, suite, condition, seed)

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
        "reward": [],
        "return": [],  # compatibility alias; numerically identical to reward
        "success": [],
    }

    for seq_idx, trained_task in enumerate(args.task_sequence):
        run_dir = checkpoint_dir(
            args.save_root, suite, condition, seed, seq_idx, trained_task
        )
        if not checkpoint_complete(run_dir):
            raise FileNotFoundError(run_dir)

        row_reward, row_success = [], []
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
            row_reward.append(result["reward"])
            row_success.append(result["success"])

        data["reward"].append(row_reward)
        data["return"].append(list(row_reward))
        data["success"].append(row_success)
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

    # Direct callers get the same protection as the benchmark orchestrator.
    # Empty scratch_seeds is explicitly allowed so A_N/FG/BWT can still be
    # computed when no FT denominator is available.
    _validate_scratch_checkpoints(
        args,
        suite,
        scratch_seeds,
        scratch_total_timesteps,
    )

    cache = survey_metrics_cache_path(args, suite, condition, seed)
    cache_config = {
        **_benchmark_cache_config(args),
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

    diagonal = _compute_diagonal_all(args, suite, condition, seed, device)
    final_row = _compute_final_row_all(args, suite, condition, seed, device)

    reward_diag = diagonal["reward"]
    reward_final = final_row["reward"]
    success_diag = diagonal["success"]
    success_final = final_row["success"]

    reward_fg_bwt = compute_fg_bwt(
        reward_diag, reward_final, args.task_sequence, "reward"
    )

    unique_tasks = sorted(set(args.task_sequence))
    success_complete = _success_thresholds_complete(args, suite, unique_tasks)
    if success_complete:
        A_N_success = compute_A_N(success_final)
        success_fg_bwt = compute_fg_bwt(
            success_diag, success_final, args.task_sequence, "success"
        )
    else:
        A_N_success = float("nan")
        success_fg_bwt = _nan_fg_bwt("success")

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

    result = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_config": cache_config,
        "checkpoint_signatures": continual_signatures,
        "scratch_checkpoint_signatures": scratch_signatures,
        "suite": suite,
        "condition": condition,
        "seed": int(seed),
        "sequence": list(args.task_sequence),
        "success_thresholds_complete": bool(success_complete),
        "success_thresholds": getattr(args, "success_thresholds", {}),

        # Raw-score family (complementary; not normalized across arbitrary games).
        "A_N_reward": compute_A_N(reward_final),
        **reward_fg_bwt,
        **ft_reward,

        # Bounded threshold-success family; primary survey-style Atari analogue.
        "A_N_success": A_N_success,
        **success_fg_bwt,
        **ft_success,

        # Exact values used to derive retention metrics.
        "p_diagonal_reward": reward_diag,
        "p_final_row_reward": reward_final,
        "p_diagonal_success": success_diag,
        "p_final_row_success": success_final,
    }

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as f:
        json.dump(result, f, indent=2)
    return result
