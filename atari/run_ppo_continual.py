"""PPO trainer for the bounded categorical CKA-RL Atari agent.

This is the Atari/PPO counterpart of run_sac.py.  The RL algorithm differs,
but the continual-learning principles are intentionally preserved:

- zero-shot evaluation at task start;
- root/base + current contribution + bounded historical pool;
- learned alpha reuse and weight-delta warmup;
- optional trainable shared encoder with slower LR and historical feature-drift
  regularization in distillation conditions;
- post-training representative-state collection;
- exact-policy snapshot BEFORE pool topology changes;
- set_base() on the root task, finalize() on later tasks;
- cosine or replay-weighted symmetric categorical-KL merge selection;
- categorical KL distillation when enabled;
- detailed merge/lineage/pool diagnostics and start/pre/post-finalize snapshots.

Training uses clipped-reward EpisodicLife Atari preprocessing.  Evaluation uses
raw full-game rewards (clip_reward=False, episodic_life=False).
"""
from __future__ import annotations

import copy
import inspect
import json
import os
import pathlib
import random
import time
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from atari_tasks import TASK_SUITES, get_task, get_task_name
from cka_rl import CkaRlAgent
from experiment_identity import write_manifest

_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


@dataclass
class Args:
    model_type: Literal["cka-rl"] = "cka-rl"
    task_suite: Literal["freeway", "space_invaders"] = "freeway"
    task_id: int = 0
    # Unique occurrence index in the continual sequence.  task_id may repeat;
    # seq_idx preserves occurrence-level buffer lineage.
    seq_idx: int = 0
    prev_units: Tuple[pathlib.Path, ...] = ()
    save_dir: str = "agents_atari/debug"
    runs_root: str = "runs_atari"
    tag: str = "debug"

    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    # ------------------------------------------------------------------
    # PPO settings: same family as the project's legacy CleanRL Atari PPO.
    # ------------------------------------------------------------------
    total_timesteps: int = 1_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    eval_every: int = 50_000
    num_evals: int = 5
    # ALE has no native success flag.  If set:
    # success = 1[raw full-game score >= threshold].
    success_threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # Knowledge-pool / four-condition knobs.
    # ------------------------------------------------------------------
    fusion_mode: Literal["classic_cka", "weight_delta"] = "classic_cka"
    pool_size: int = 5
    alpha_init: Literal["Randn", "Major", "Uniform"] = "Randn"
    alpha_major: float = 0.6
    alpha_factor: float = 1e-3
    fix_alpha: bool = False
    alpha_learning_rate: float = 2.5e-4
    alpha_warmup_steps: int = 5_000
    alpha_entropy_reg: float = 0.01
    alpha_mass_reg: float = 0.05
    use_alpha_scale: bool = False
    fix_alpha_scale: bool = False
    use_alpha_mass: bool = False
    constrain_alpha_mass: bool = True

    # ------------------------------------------------------------------
    # Shared CNN encoder.
    # ------------------------------------------------------------------
    encoder_from_base: bool = True
    train_shared: bool = False
    freeze_root_encoder: bool = False
    pretrained_encoder: Optional[str] = None
    shared_dim: int = 512
    head_hidden_dim: int = 128
    distill_encoder_lr_mult: float = 0.1
    # Same principle as HalfCheetah: if a historical encoder is allowed to move
    # in a distillation condition, preserve its representation on old states.
    drift_reg: float = 1.0

    # ------------------------------------------------------------------
    # Merge / distillation.
    # ------------------------------------------------------------------
    distillation: bool = True
    collect_cosine_buffers: bool = False
    # Number of TOTAL stored transitions, not vector-env steps.
    distill_extra_steps: int = 2_000
    max_distill_buffer: int = 5_000
    similarity_samples: int = 512
    distill_max_samples: int = 2_000
    distill_epochs: int = 8
    distill_lr: float = 3e-4
    distill_batch_size: int = 256
    distill_test_frac: float = 0.2
    distill_select_best_val: bool = True

    # ------------------------------------------------------------------
    # Analysis / diagnostics, mirroring run_sac.py principles.
    # ------------------------------------------------------------------
    analysis_log_every: int = 5_000
    save_analysis_snapshots: bool = True
    analysis_root: str = "analysis_runs_atari"


def make_train_env(task_id: int, task_suite: str):
    def thunk():
        return get_task(
            task_id,
            task_suite=task_suite,
            clip_reward=True,
            episodic_life=True,
        )

    return thunk


def make_vector_env(args):
    kwargs = {}
    # Gymnasium >=1.0 exposes autoreset_mode. SAME_STEP matches the vector-env
    # semantics assumed by CleanRL-style PPO rollouts: after a terminal/life
    # boundary, the returned next observation is already the reset observation
    # while final_info retains the completed episode record.
    if "autoreset_mode" in inspect.signature(gym.vector.SyncVectorEnv).parameters:
        autoreset = getattr(gym.vector, "AutoresetMode", None)
        if autoreset is not None:
            kwargs["autoreset_mode"] = autoreset.SAME_STEP
    return gym.vector.SyncVectorEnv(
        [make_train_env(args.task_id, args.task_suite) for _ in range(args.num_envs)],
        **kwargs,
    )


def _evaluation_seed(task_id: int, episode: int) -> int:
    """Fixed per-task episode seeds, independent of the training seed.

    This is important for FT: continual seed 1 and scratch seed 101 should be
    evaluated on the same episode initializations.
    """
    return 10_000 + 10_000 * int(task_id) + int(episode)


@torch.no_grad()
def evaluate(agent, args, device, global_step, writer=None):
    """Deterministic evaluation on raw full-game Atari score."""
    env = get_task(
        args.task_id,
        task_suite=args.task_suite,
        clip_reward=False,
        episodic_life=False,
    )
    returns = []
    successes = []
    for ep in range(args.num_evals):
        obs, _ = env.reset(seed=_evaluation_seed(args.task_id, ep))
        ep_return = 0.0
        while True:
            x = (
                torch.as_tensor(obs, dtype=torch.float32, device=device)
                .unsqueeze(0)
                / 255.0
            )
            logits = agent(x)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            if terminated or truncated:
                break
        returns.append(ep_return)
        if args.success_threshold is not None:
            successes.append(float(ep_return >= float(args.success_threshold)))
    env.close()

    result = {
        "reward": float(np.mean(returns)),
        "return": float(np.mean(returns)),
        "success": float(np.mean(successes)) if successes else float("nan"),
    }
    if writer is not None:
        writer.add_scalar("charts/test_episodic_return", result["reward"], global_step)
        if np.isfinite(result["success"]):
            writer.add_scalar("charts/test_success", result["success"], global_step)
    return result


def _log_finished_episodes(writer, infos, global_step, success_threshold=None):
    """Log training-time full-game episode records across Gymnasium layouts.

    RecordEpisodeStatistics sits inside EpisodicLife/ClipReward in atari_envs.py,
    so its episode return is the raw full-game score rather than the clipped
    learning reward.  This log is diagnostic only; FT uses periodic test curves.
    """

    def log_episode(ep_return, ep_length):
        writer.add_scalar("charts/episodic_return", float(ep_return), global_step)
        writer.add_scalar("charts/episodic_length", float(ep_length), global_step)
        if success_threshold is not None:
            writer.add_scalar(
                "charts/success",
                float(float(ep_return) >= float(success_threshold)),
                global_step,
            )

    if "final_info" in infos and not isinstance(infos["final_info"], dict):
        final_infos = infos["final_info"]
        mask = infos.get("_final_info", np.ones(len(final_infos), dtype=bool))
        for idx, enabled in enumerate(mask):
            if not enabled or final_infos[idx] is None:
                continue
            fi = final_infos[idx]
            if "episode" in fi:
                log_episode(fi["episode"]["r"], fi["episode"]["l"])
        return

    if "final_info" in infos and isinstance(infos["final_info"], dict):
        fi = infos["final_info"]
        mask = infos.get("_final_info", np.ones(1, dtype=bool))
        for idx, enabled in enumerate(mask):
            if not enabled:
                continue
            if "episode" in fi:
                ep_r = np.asarray(fi["episode"]["r"])[idx]
                ep_l = np.asarray(fi["episode"]["l"])[idx]
                log_episode(ep_r, ep_l)
        return

    if "episode" in infos:
        r = np.atleast_1d(infos["episode"]["r"])
        l = np.atleast_1d(infos["episode"]["l"])
        mask = infos.get("_episode", np.ones(len(r), dtype=bool))
        for idx, enabled in enumerate(mask):
            if enabled:
                log_episode(r[idx], l[idx])


def collect_merge_buffer(agent, envs, target_rows, task_id, seq_idx, device, seed):
    """Collect raw uint8 reference frames for behavioral KL/distillation."""
    if target_rows <= 0:
        return None, 0.0
    obs, _ = envs.reset(seed=seed)
    obs_parts, action_parts = [], []
    start = time.time()
    rows = 0
    agent.eval()
    while rows < target_rows:
        raw = np.asarray(obs)
        x = torch.as_tensor(raw, dtype=torch.float32, device=device) / 255.0
        with torch.no_grad():
            action = agent.action_distribution(x).sample()
        action_np = action.cpu().numpy()
        obs_parts.append(raw.astype(np.uint8, copy=True))
        action_parts.append(action_np.astype(np.int16, copy=True))
        obs, _, _, _, _ = envs.step(action_np)
        rows += raw.shape[0]

    obs_arr = np.concatenate(obs_parts, axis=0)[:target_rows]
    action_arr = np.concatenate(action_parts, axis=0)[:target_rows]
    buffer = {
        "obs": obs_arr.astype(np.uint8, copy=False),
        "actions": action_arr.astype(np.int16, copy=False),
        "task_ids": np.full(target_rows, int(task_id), dtype=np.int16),
        "source_ids": np.full(target_rows, int(seq_idx), dtype=np.int16),
    }
    return buffer, time.time() - start


def _effective_policy_vector(agent: CkaRlAgent) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat(
            [tensor.reshape(-1) for tensor in agent.policy_pool._effective()], dim=0
        )


def _module_l2_norm(module: nn.Module) -> float:
    with torch.no_grad():
        params = [p.detach().reshape(-1) for p in module.parameters()]
        if not params:
            return 0.0
        return float(torch.cat(params).norm().item())


def _save_analysis_snapshot(
    path,
    stage,
    step,
    args,
    agent,
    *,
    include_effective=True,
    task_start_policy=None,
):
    """Save Atari equivalents of start/pre_finalize/post_finalize snapshots."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": str(stage),
        "step": int(step),
        "args": asdict(args),
        "encoder_state_dict": {
            k: v.detach().cpu().clone() for k, v in agent.fc.state_dict().items()
        },
        "critic_state_dict": {
            k: v.detach().cpu().clone() for k, v in agent.critic.state_dict().items()
        },
        "pool_length": int(agent.policy_pool.pool_length()),
        "alpha": None if agent.alpha is None else agent.alpha.detach().cpu().clone(),
        "alpha_scale": (
            None
            if agent.alpha_scale is None
            else agent.alpha_scale.detach().cpu().clone()
        ),
        "alpha_mass": (
            None if agent.alpha_mass is None else agent.alpha_mass.detach().cpu().clone()
        ),
        "merge_info": agent.get_merge_info(),
        "distill_metrics": agent.get_distill_metrics(),
    }
    if include_effective:
        payload["effective_policy"] = _effective_policy_vector(agent).detach().cpu()
        if task_start_policy is not None:
            current = payload["effective_policy"]
            start = task_start_policy.detach().cpu()
            payload["effective_policy_delta_l2"] = float((current - start).norm())
    torch.save(payload, path)


def _log_analysis_state(writer, step, agent, task_start_policy):
    """Lightweight periodic Atari training-state diagnostics."""
    writer.add_scalar("analysis/encoder/l2_norm", _module_l2_norm(agent.fc), step)
    writer.add_scalar("analysis/critic/l2_norm", _module_l2_norm(agent.critic), step)
    writer.add_scalar(
        "analysis/pool/current_length", int(agent.policy_pool.pool_length()), step
    )

    current = _effective_policy_vector(agent)
    writer.add_scalar("analysis/policy/effective_l2_norm", float(current.norm()), step)
    if task_start_policy is not None and current.numel() == task_start_policy.numel():
        writer.add_scalar(
            "analysis/policy/delta_from_task_start_l2",
            float((current - task_start_policy).norm()),
            step,
        )

    if agent.alpha is not None:
        scale = agent.alpha_scale if agent.alpha_scale is not None else 1.0
        probs = torch.softmax(agent.alpha.detach() * scale.detach(), dim=0)
        entropy = -(probs * (probs + 1e-12).log()).sum()
        writer.add_scalar("analysis/policy/alpha_entropy", float(entropy), step)
        writer.add_scalar("analysis/policy/alpha_max", float(probs.max()), step)

    if agent.alpha_mass is not None:
        writer.add_scalar(
            "analysis/policy/alpha_mass_raw", float(agent.alpha_mass.detach().item()), step
        )
        writer.add_scalar(
            "analysis/policy/alpha_mass_effective",
            float(agent.policy_pool.effective_alpha_mass().detach().item()),
            step,
        )


def validate_args(args):
    if args.task_id < 0 or args.task_id >= len(TASK_SUITES[args.task_suite]):
        raise ValueError(f"invalid task_id={args.task_id} for {args.task_suite}")
    if args.pool_size < 2:
        raise ValueError("pool_size must be >=2")
    if args.fusion_mode == "classic_cka" and args.use_alpha_mass:
        raise ValueError("alpha_mass is only valid with fusion_mode='weight_delta'")
    if args.use_alpha_scale and args.fix_alpha_scale:
        raise ValueError("use_alpha_scale and fix_alpha_scale are mutually exclusive")
    if args.train_shared and args.freeze_root_encoder:
        raise ValueError("train_shared and freeze_root_encoder are contradictory")
    if args.learning_rate <= 0 or args.alpha_learning_rate <= 0:
        raise ValueError("learning rates must be > 0")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if args.clip_coef <= 0 or args.max_grad_norm <= 0:
        raise ValueError("clip_coef and max_grad_norm must be > 0")
    if args.ent_coef < 0 or args.vf_coef < 0:
        raise ValueError("ent_coef and vf_coef must be >= 0")
    if args.target_kl is not None and args.target_kl <= 0:
        raise ValueError("target_kl must be > 0 when provided")
    if args.distill_encoder_lr_mult <= 0:
        raise ValueError("distill_encoder_lr_mult must be > 0")
    if args.drift_reg < 0 or args.alpha_entropy_reg < 0 or args.alpha_mass_reg < 0:
        raise ValueError("drift/alpha regularizers must be >= 0")
    if args.total_timesteps < 1 or args.num_envs < 1 or args.num_steps < 1:
        raise ValueError("total_timesteps, num_envs and num_steps must be >= 1")
    if args.num_minibatches < 1 or args.update_epochs < 1:
        raise ValueError("num_minibatches and update_epochs must be >= 1")
    batch_size = args.num_envs * args.num_steps
    if batch_size % args.num_minibatches != 0:
        raise ValueError("num_envs*num_steps must be divisible by num_minibatches")
    if args.total_timesteps < batch_size:
        raise ValueError(
            "total_timesteps must be at least one PPO rollout "
            f"({batch_size} environment steps)"
        )
    if args.eval_every < 0 or args.num_evals < 1:
        raise ValueError("eval_every must be >=0 and num_evals must be >=1")
    if args.alpha_warmup_steps < 0:
        raise ValueError("alpha_warmup_steps must be >=0")
    if args.similarity_samples < 2:
        raise ValueError("similarity_samples must be >=2")
    if args.max_distill_buffer < 2:
        raise ValueError("max_distill_buffer must be >=2")
    if args.distill_max_samples < 2:
        raise ValueError("distill_max_samples must be >=2")
    if args.distill_batch_size < 1:
        raise ValueError("distill_batch_size must be >=1")
    if not 0 <= args.distill_test_frac < 1:
        raise ValueError("distill_test_frac must be in [0,1)")
    if args.distillation and args.distill_epochs < 1:
        raise ValueError("distill_epochs must be >=1 when distillation is enabled")
    if (args.distillation or args.collect_cosine_buffers) and args.distill_extra_steps <= 0:
        raise ValueError(
            "distill_extra_steps must be >0 when a merge/reference buffer is collected"
        )
    if args.distill_extra_steps < 0:
        raise ValueError("distill_extra_steps must be >=0")
    if args.analysis_log_every < 0:
        raise ValueError("analysis_log_every must be >=0")


def main():
    args = tyro.cli(Args)
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.task_suite}__task_{args.task_id}__cka-rl__run_ppo__{args.seed}"
    task_name = get_task_name(args.task_id, args.task_suite)
    event_dir = pathlib.Path(args.runs_root) / args.tag / run_name
    analysis_dir = pathlib.Path(args.analysis_root) / args.tag / run_name
    writer = SummaryWriter(str(event_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )
    logger.info(f"run={run_name} | task={task_name} | device={device}")

    envs = make_vector_env(args)
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise TypeError("Atari PPO requires a Discrete action space")
    obs_shape = tuple(envs.single_observation_space.shape)
    act_dim = int(envs.single_action_space.n)
    envs.single_action_space.seed(args.seed)

    base_dir = str(args.prev_units[0]) if args.prev_units else None
    latest_dir = str(args.prev_units[-1]) if args.prev_units else None
    agent = CkaRlAgent(
        obs_shape=obs_shape,
        act_dim=act_dim,
        base_dir=base_dir,
        latest_dir=latest_dir,
        pool_size=args.pool_size,
        alpha_init=args.alpha_init,
        alpha_major=args.alpha_major,
        alpha_factor=args.alpha_factor,
        fix_alpha=args.fix_alpha,
        use_alpha_scale=args.use_alpha_scale,
        fix_alpha_scale=args.fix_alpha_scale,
        use_alpha_mass=args.use_alpha_mass,
        constrain_alpha_mass=args.constrain_alpha_mass,
        encoder_from_base=args.encoder_from_base,
        distillation=args.distillation,
        fusion_mode=args.fusion_mode,
        max_distill_buffer=args.max_distill_buffer,
        distill_test_frac=args.distill_test_frac,
        distill_select_best_val=args.distill_select_best_val,
        distill_epochs=args.distill_epochs,
        distill_lr=args.distill_lr,
        distill_batch_size=args.distill_batch_size,
        distill_max_samples=args.distill_max_samples,
        similarity_samples=args.similarity_samples,
        hidden_dim=args.head_hidden_dim,
        shared_dim=args.shared_dim,
        train_shared=args.train_shared,
        freeze_root_encoder=args.freeze_root_encoder,
        pretrained_encoder=args.pretrained_encoder,
    ).to(device)
    agent.log_alphas()

    # ------------------------------------------------------------------
    # Historical representation stabilization for train_shared=True.
    # This is the CNN analogue of the HalfCheetah drift regularizer.
    # ------------------------------------------------------------------
    old_fc = None
    past_obs_pool = None
    if args.seq_idx > 0 and args.train_shared and args.distillation:
        old_fc = copy.deepcopy(agent.fc).to(device)
        old_fc.eval()
        for p in old_fc.parameters():
            p.requires_grad_(False)

        past_obs = [
            entry["buffer"]["obs"]
            for entry in agent.policy_pool.pool
            if entry.get("buffer") is not None
            and "obs" in entry["buffer"]
            and len(entry["buffer"]["obs"]) > 0
        ]
        if past_obs:
            past_obs_pool = np.concatenate(past_obs, axis=0)
            logger.info(
                f"encoder-drift reference pool: {len(past_obs_pool)} historical frames"
            )

    # Separate alpha optimizer, as in the legacy Atari CKA code.
    alpha_params = [
        p
        for n, p in agent.named_parameters()
        if p.requires_grad and "alpha" in n
    ]
    nonalpha_named = [
        (n, p)
        for n, p in agent.named_parameters()
        if p.requires_grad and "alpha" not in n
    ]
    encoder_ids = {id(p) for p in agent.fc.parameters()}
    encoder_params = [p for _, p in nonalpha_named if id(p) in encoder_ids]
    other_params = [p for _, p in nonalpha_named if id(p) not in encoder_ids]

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": args.learning_rate})
    if encoder_params:
        encoder_lr = args.learning_rate
        if args.prev_units and args.distillation and args.train_shared:
            encoder_lr *= args.distill_encoder_lr_mult
        groups.append({"params": encoder_params, "lr": encoder_lr})
    if not groups:
        raise RuntimeError("No non-alpha PPO parameters are trainable")

    optimizer = optim.Adam(groups, eps=1e-5)
    initial_group_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    alpha_optimizer = (
        optim.Adam(alpha_params, lr=args.alpha_learning_rate, eps=1e-5)
        if alpha_params
        else None
    )

    # Verify that the default/frozen continual encoder really remains fixed.
    encoder_should_be_frozen = bool(
        not args.train_shared
        and (
            args.pretrained_encoder is not None
            or latest_dir is not None
            or args.freeze_root_encoder
        )
    )
    encoder_fingerprint = None
    if encoder_should_be_frozen:
        if not all(not p.requires_grad for p in agent.fc.parameters()):
            raise RuntimeError("encoder should be frozen but some parameters require grad")
        optimizer_ids = {
            id(p)
            for group in optimizer.param_groups
            for p in group["params"]
        }
        leaked = [p for p in agent.fc.parameters() if id(p) in optimizer_ids]
        if leaked:
            raise RuntimeError("frozen encoder parameters leaked into PPO optimizer")
        with torch.no_grad():
            encoder_fingerprint = torch.cat(
                [p.detach().reshape(-1) for p in agent.fc.parameters()]
            ).clone()

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    actual_total_timesteps = num_iterations * batch_size
    if actual_total_timesteps != args.total_timesteps:
        logger.warning(
            f"requested total_timesteps={args.total_timesteps}, but PPO uses complete "
            f"rollouts of {batch_size} steps; actual_total_timesteps={actual_total_timesteps}"
        )
    writer.add_scalar("timing/requested_total_timesteps", args.total_timesteps, 0)
    writer.add_scalar("timing/actual_total_timesteps", actual_total_timesteps, 0)

    obs_buf = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    global_step = 0
    next_eval = args.eval_every if args.eval_every > 0 else None
    next_analysis = args.analysis_log_every if args.analysis_log_every > 0 else None
    start_time = time.time()

    task_start_policy = _effective_policy_vector(agent).detach().clone()
    if args.save_analysis_snapshots:
        _save_analysis_snapshot(
            analysis_dir / "start.pt",
            "start",
            0,
            args,
            agent,
            include_effective=True,
            task_start_policy=task_start_policy,
        )
    _log_analysis_state(writer, 0, agent, task_start_policy)

    # Zero-shot/task-start evaluation for transfer diagnostics.
    evaluate(agent, args, device, 0, writer)

    # Keep the most recent losses available for logging.
    pg_loss = torch.tensor(float("nan"), device=device)
    v_loss = torch.tensor(float("nan"), device=device)
    entropy_loss = torch.tensor(float("nan"), device=device)
    approx_kl = torch.tensor(float("nan"), device=device)
    old_approx_kl = torch.tensor(float("nan"), device=device)
    drift_loss = None
    alpha_entropy = None
    mass_loss = None

    for iteration in tqdm(range(1, num_iterations + 1)):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / max(num_iterations, 1)
            for group, initial_lr in zip(optimizer.param_groups, initial_group_lrs):
                group["lr"] = frac * initial_lr
            if alpha_optimizer is not None:
                alpha_optimizer.param_groups[0]["lr"] = (
                    frac * args.alpha_learning_rate
                )

        # --------------------------------------------------------------
        # On-policy rollout.
        # --------------------------------------------------------------
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs / 255.0
                )
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value.flatten()

            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            _log_finished_episodes(
                writer, infos, global_step, args.success_threshold
            )
            next_done_np = np.logical_or(terminations, truncations)
            rewards_buf[step] = torch.as_tensor(
                reward, dtype=torch.float32, device=device
            )
            next_obs = torch.as_tensor(
                next_obs_np, dtype=torch.float32, device=device
            )
            next_done = torch.as_tensor(
                next_done_np, dtype=torch.float32, device=device
            )

        # --------------------------------------------------------------
        # GAE / returns.
        # --------------------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs / 255.0).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = (
                    rewards_buf[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values_buf[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma
                    * args.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1,) + obs_shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1).long()
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        inds = np.arange(batch_size)
        clipfracs = []

        # --------------------------------------------------------------
        # PPO updates.
        # --------------------------------------------------------------
        for _epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb = inds[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb] / 255.0, b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    )

                mb_adv = b_advantages[mb]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv
                    * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    ),
                ).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb]) ** 2
                    v_clipped_pred = b_values[mb] + torch.clamp(
                        newvalue - b_values[mb],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_clipped = (v_clipped_pred - b_returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + args.vf_coef * v_loss
                )

                # Historical encoder drift regularization: same continual
                # principle as run_sac.py, with uint8 Atari frames normalized
                # exactly as the policy input.
                drift_loss = None
                if (
                    args.distillation
                    and old_fc is not None
                    and past_obs_pool is not None
                    and args.drift_reg > 0
                ):
                    drift_count = min(len(mb), len(past_obs_pool))
                    drift_idx = np.random.randint(
                        0, len(past_obs_pool), size=drift_count
                    )
                    s_past = (
                        torch.as_tensor(
                            past_obs_pool[drift_idx],
                            dtype=torch.float32,
                            device=device,
                        )
                        / 255.0
                    )
                    with torch.no_grad():
                        phi_old = old_fc(s_past)
                    phi_curr = agent.fc(s_past)
                    drift_loss = args.drift_reg * F.mse_loss(phi_curr, phi_old)
                    loss = loss + drift_loss

                # Preserve HalfCheetah weight-delta mixture warmup principle.
                mixture_warmup = bool(
                    args.fusion_mode == "weight_delta"
                    and global_step < args.alpha_warmup_steps
                    and agent.alpha is not None
                    and agent.alpha.numel() > 1
                )
                alpha_entropy = None
                if mixture_warmup and args.alpha_entropy_reg > 0:
                    scale = (
                        agent.alpha_scale
                        if agent.alpha_scale is not None
                        else 1.0
                    )
                    probs = torch.softmax(agent.alpha * scale, dim=0)
                    alpha_entropy = -(
                        probs * torch.log(probs + 1e-8)
                    ).sum()
                    loss = loss - args.alpha_entropy_reg * alpha_entropy

                mass_loss = None
                if (
                    not mixture_warmup
                    and agent.alpha_mass is not None
                    and agent.alpha_mass.requires_grad
                    and args.alpha_mass_reg > 0
                ):
                    eff_mass = agent.policy_pool.effective_alpha_mass()
                    mass_loss = (
                        args.alpha_mass_reg
                        * (eff_mass**2)
                        * ((eff_mass - 1.0) ** 2)
                    )
                    loss = loss + mass_loss.mean()

                optimizer.zero_grad(set_to_none=True)
                if alpha_optimizer is not None:
                    alpha_optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # During the weight-delta warmup, learn historical mixture
                # coefficients before allowing the new policy residual/mass to move.
                if mixture_warmup:
                    for p in (
                        agent.policy_pool.own_l0_weight,
                        agent.policy_pool.own_l0_bias,
                        agent.policy_pool.own_l2_weight,
                        agent.policy_pool.own_l2_bias,
                    ):
                        if p.grad is not None:
                            p.grad.zero_()
                    if (
                        agent.alpha_mass is not None
                        and agent.alpha_mass.grad is not None
                    ):
                        agent.alpha_mass.grad.zero_()

                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                if alpha_optimizer is not None:
                    alpha_optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # --------------------------------------------------------------
        # PPO / continual diagnostics.
        # --------------------------------------------------------------
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y
        )

        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        if alpha_optimizer is not None:
            writer.add_scalar(
                "charts/alpha_learning_rate",
                alpha_optimizer.param_groups[0]["lr"],
                global_step,
            )
        writer.add_scalar("losses/value_loss", float(v_loss.item()), global_step)
        writer.add_scalar("losses/policy_loss", float(pg_loss.item()), global_step)
        writer.add_scalar("losses/entropy", float(entropy_loss.item()), global_step)
        writer.add_scalar(
            "losses/old_approx_kl", float(old_approx_kl.item()), global_step
        )
        writer.add_scalar("losses/approx_kl", float(approx_kl.item()), global_step)
        writer.add_scalar(
            "losses/clipfrac",
            float(np.mean(clipfracs)) if clipfracs else float("nan"),
            global_step,
        )
        writer.add_scalar(
            "losses/explained_variance", float(explained_var), global_step
        )
        if drift_loss is not None:
            writer.add_scalar(
                "losses/encoder_drift_reg", float(drift_loss.item()), global_step
            )
        if alpha_entropy is not None:
            writer.add_scalar(
                "losses/knowledge_alpha_entropy",
                float(alpha_entropy.item()),
                global_step,
            )
        if mass_loss is not None:
            writer.add_scalar(
                "losses/alpha_mass_reg", float(mass_loss.mean().item()), global_step
            )
        writer.add_scalar(
            "charts/SPS",
            int(global_step / max(time.time() - start_time, 1e-9)),
            global_step,
        )

        if next_eval is not None and global_step >= next_eval:
            evaluate(agent, args, device, global_step, writer)
            while next_eval <= global_step:
                next_eval += args.eval_every

        if next_analysis is not None and global_step >= next_analysis:
            _log_analysis_state(writer, global_step, agent, task_start_policy)
            writer.add_scalar("analysis/checkpoint_marker", 1.0, global_step)
            while next_analysis <= global_step:
                next_analysis += args.analysis_log_every

    # ------------------------------------------------------------------
    # End-of-task checks and exact policy evaluation.
    # ------------------------------------------------------------------
    train_loop_seconds = time.time() - start_time
    writer.add_scalar("timing/train_loop_seconds", train_loop_seconds, global_step)
    logger.info(
        f"TRAIN_LOOP_SECONDS={train_loop_seconds:.2f} | "
        f"steps={global_step} | SPS={global_step / max(train_loop_seconds, 1e-9):.2f}"
    )

    if encoder_should_be_frozen:
        with torch.no_grad():
            current_encoder = torch.cat(
                [p.detach().reshape(-1) for p in agent.fc.parameters()]
            )
            encoder_max_drift = float(
                (current_encoder - encoder_fingerprint).abs().max().item()
            )
        writer.add_scalar(
            "analysis/encoder/max_drift", encoder_max_drift, global_step
        )
        if encoder_max_drift != 0.0:
            raise RuntimeError(
                f"frozen shared encoder drifted by {encoder_max_drift}"
            )

    _log_analysis_state(writer, global_step, agent, task_start_policy)
    final_eval = evaluate(agent, args, device, global_step, writer)

    run_dir = pathlib.Path(args.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Exact just-trained policy is saved BEFORE pool topology changes.
    agent.save_policy_snapshot(str(run_dir))

    # Collect representative states after training, without changing weights.
    needs_buffer = bool(args.distillation or args.collect_cosine_buffers)
    merge_buffer = None
    buffer_seconds = 0.0
    if needs_buffer:
        merge_buffer, buffer_seconds = collect_merge_buffer(
            agent,
            envs,
            args.distill_extra_steps,
            args.task_id,
            args.seq_idx,
            device,
            args.seed + 77_777,
        )
    writer.add_scalar("timing/merge_buffer_seconds", buffer_seconds, global_step)
    writer.add_scalar(
        "analysis/buffer/rows",
        0 if merge_buffer is None else len(merge_buffer["obs"]),
        global_step,
    )
    agent.set_own_buffer(merge_buffer)

    if args.save_analysis_snapshots:
        _save_analysis_snapshot(
            analysis_dir / "pre_finalize.pt",
            "pre_finalize",
            global_step,
            args,
            agent,
            include_effective=True,
            task_start_policy=task_start_policy,
        )

    t_finalize = time.time()
    if args.prev_units:
        agent.finalize()
    else:
        agent.set_base()
    finalize_seconds = time.time() - t_finalize
    writer.add_scalar("timing/finalize_seconds", finalize_seconds, global_step)
    writer.add_scalar(
        "analysis/pool/final_length", agent.policy_pool.pool_length(), global_step
    )

    # Detailed merge diagnostics, matching run_sac.py's auditability.
    info = agent.get_merge_info()
    if info:
        scalar_keys = (
            "idx1",
            "idx2",
            "similarity_states",
            "cosine_similarity",
            "pairwise_cosine_min",
            "pairwise_cosine_mean",
            "pairwise_cosine_max",
            "symmetric_kl",
            "pairwise_kl_min",
            "pairwise_kl_mean",
            "pairwise_kl_max",
            "selected_state_kl_p95",
            "selected_state_kl_max",
            "pool_size_before",
            "pool_size_after",
        )
        for key in scalar_keys:
            value = info.get(key)
            if value is not None and np.isscalar(value):
                writer.add_scalar(f"analysis/merge/{key}", float(value), global_step)
        if "used_distillation" in info:
            writer.add_scalar(
                "analysis/merge/used_distillation",
                float(bool(info["used_distillation"])),
                global_step,
            )

        lineage = {
            "task_ids": {
                "parent_1": info.get("parent_1_lineage", {}),
                "parent_2": info.get("parent_2_lineage", {}),
                "merged": info.get("merged_lineage", {}),
            },
            "source_ids": {
                "parent_1": info.get("parent_1_source_lineage", {}),
                "parent_2": info.get("parent_2_source_lineage", {}),
                "merged": info.get("merged_source_lineage", {}),
            },
        }
        writer.add_text(
            "analysis/merge/lineage", json.dumps(lineage, sort_keys=True), global_step
        )
        writer.add_text(
            "analysis/merge/info", json.dumps(info, sort_keys=True), global_step
        )
        logger.info(f"MERGE_LINEAGE={json.dumps(lineage, sort_keys=True)}")

    # Pool slot norms are useful for checking growth/degeneracy, especially in
    # weight_delta conditions.
    with torch.no_grad():
        for i, entry in enumerate(agent.policy_pool.pool):
            vec = torch.cat([entry[k].reshape(-1) for k in _HEAD_KEYS])
            writer.add_scalar(
                f"analysis/pool/norm_slot_{i}", float(vec.norm().item()), global_step
            )
        if agent.alpha_mass is not None:
            raw_mass = float(agent.alpha_mass.detach().item())
            effective_mass = float(
                agent.policy_pool.effective_alpha_mass().detach().item()
            )
            writer.add_scalar("analysis/alpha_mass_raw", raw_mass, global_step)
            writer.add_scalar("analysis/alpha_mass", effective_mass, global_step)
            writer.add_scalar(
                "analysis/alpha_mass_effective", effective_mass, global_step
            )

    for name, value in agent.get_distill_metrics().items():
        if value is not None:
            writer.add_scalar(f"distillation/{name}", float(value), global_step)
            logger.info(f"distillation/{name}={value}")

    writer.add_scalar(
        "charts/final_episodic_return", final_eval["reward"], global_step
    )
    writer.add_scalar("charts/final_return", final_eval["reward"], global_step)
    if np.isfinite(final_eval["success"]):
        writer.add_scalar("charts/final_success", final_eval["success"], global_step)

    if args.save_analysis_snapshots:
        _save_analysis_snapshot(
            analysis_dir / "post_finalize.pt",
            "post_finalize",
            global_step,
            args,
            agent,
            # After finalize, pool topology changed and the pre-finalize alpha
            # vector is no longer the authoritative current-policy mixture.
            include_effective=False,
            task_start_policy=task_start_policy,
        )

    agent.save(str(run_dir))
    manifest = write_manifest(
        run_dir,
        vars(args),
        parent_dirs=args.prev_units,
        pretrained_encoder=args.pretrained_encoder,
    )
    logger.info(f"saved {run_dir} | signature={manifest['run_signature']}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
