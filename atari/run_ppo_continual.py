"""Continual Atari PPO trainer for bounded categorical CKA-RL.

Shared continual-learning semantics mirror the corrected HalfCheetah runner:
- Delta is ``total_timesteps`` and the frozen tail B is INSIDE Delta;
- exact task-start / pre-finalize / post-finalize analysis snapshots;
- parameter or exact policy-space composition;
- optional policy-student variant with explicit behavior-policy provenance;
- fixed-capacity pool finalization, projection, lineage-aware merging/distillation;
- CSV-mirrored TensorBoard scalars and reproducible checkpoint manifests.

Atari-specific behavior remains unchanged where it should:
- clipped-reward + EpisodicLife preprocessing for PPO training;
- raw full-game score for evaluation;
- categorical policies and a task-local PPO value head;
- raw uint8 retained frame stacks, normalized only when re-encoded.
"""
from __future__ import annotations

import copy
import gc
import json
import os
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from loguru import logger
from torch.distributions import Categorical
from tqdm import tqdm

from analysis_logging import effective_theta_vector, log_training_state, save_task_snapshot
from atari_tasks import TASK_SUITES, get_task, get_task_name
from cka_rl import CkaRlAgent
from csv_summary_writer import CsvSummaryWriter
from experiment_identity import write_manifest
from training_protocol import TaskBudget, bounded_buffer, mixture_warmup_active

_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


@dataclass
class Args:
    model_type: Literal["cka-rl"] = "cka-rl"
    task_suite: Literal["freeway", "space_invaders"] = "freeway"
    task_id: int = 0
    seq_idx: int = 0
    prev_units: Tuple[pathlib.Path, ...] = ()
    save_dir: str = "agents_atari/debug"
    runs_root: str = "runs_atari"
    tag: str = "debug"

    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    # PPO.
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
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    eval_every: int = 50_000
    num_evals: int = 5
    eval_action_mode: Literal["deterministic", "stochastic"] = "deterministic"
    success_threshold: Optional[float] = None

    # Continual composition / storage.
    fusion_mode: Literal["classic_cka", "weight_delta"] = "classic_cka"
    composition_space: Literal["parameter", "policy"] = "parameter"
    policy_student_replay: bool = False
    projection_epochs: int = 16
    projection_max_samples: int = 20_000

    pool_size: int = 5
    alpha_init: Literal["Randn", "Major", "Uniform"] = "Randn"
    alpha_major: float = 0.6
    alpha_factor: float = 1e-3
    fix_alpha: bool = False
    alpha_learning_rate: float = 2.5e-4
    alpha_mass_learning_rate: Optional[float] = None
    alpha_warmup_steps: int = 5_000
    alpha_entropy_reg: float = 0.01
    alpha_mass_reg: float = 0.05
    use_alpha_scale: bool = False
    fix_alpha_scale: bool = False
    use_alpha_mass: bool = False
    constrain_alpha_mass: bool = True

    # Shared CNN encoder.
    encoder_from_base: bool = True
    train_shared: bool = False
    freeze_root_encoder: bool = False
    pretrained_encoder: Optional[str] = None
    shared_dim: int = 512
    head_hidden_dim: int = 512
    distill_encoder_lr_mult: float = 0.1
    drift_reg: float = 1.0

    # Frozen tail / merge / distillation.
    distill_buffer_steps: Optional[int] = None
    """Compatibility alias for distill_extra_steps."""
    distillation: bool = True
    collect_cosine_buffers: bool = False
    distill_extra_steps: int = 2_000
    """Final B Atari transitions INSIDE total_timesteps; never extra interactions."""
    max_distill_buffer: int = 5_000
    similarity_samples: int = 512
    balance_source_lineages: bool = False
    distill_max_samples: int = 2_000
    distill_epochs: int = 8
    distill_lr: float = 3e-4
    distill_batch_size: int = 256
    distill_test_frac: float = 0.2
    distill_select_best_val: bool = True

    # Analysis.
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


def make_vector_env(args, num_envs: Optional[int] = None):
    count = int(args.num_envs if num_envs is None else num_envs)
    return gym.vector.SyncVectorEnv(
        [make_train_env(args.task_id, args.task_suite) for _ in range(count)]
    )


def _evaluation_seed(task_id: int, episode: int) -> int:
    return 10_000 + 10_000 * int(task_id) + int(episode)


@torch.no_grad()
def evaluate(agent, args, device, global_step, writer=None):
    """Evaluate exact categorical behavior on raw full-game Atari score."""
    env = get_task(
        args.task_id,
        task_suite=args.task_suite,
        clip_reward=False,
        episodic_life=False,
    )
    returns, successes = [], []
    eval_steps = 0

    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            for ep in range(int(args.num_evals)):
                seed = _evaluation_seed(args.task_id, ep)
                torch.manual_seed(seed)
                obs, _ = env.reset(seed=seed)
                ep_return = 0.0
                while True:
                    x = (
                        torch.as_tensor(obs, dtype=torch.float32, device=device)
                        .unsqueeze(0)
                        .div_(255.0)
                    )
                    dist = agent.action_distribution(x)
                    if args.eval_action_mode == "stochastic":
                        action = int(dist.sample().item())
                    else:
                        action = int(dist.probs.argmax(dim=-1).item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    eval_steps += 1
                    ep_return += float(reward)
                    if terminated or truncated:
                        break
                returns.append(ep_return)
                if args.success_threshold is not None:
                    successes.append(
                        float(ep_return >= float(args.success_threshold))
                    )
    finally:
        env.close()

    agent.evaluation_env_steps = getattr(agent, "evaluation_env_steps", 0) + eval_steps
    result = {
        "reward": float(np.mean(returns)),
        "return": float(np.mean(returns)),
        "success": float(np.mean(successes)) if successes else float("nan"),
        "evaluation_interactions": int(eval_steps),
    }
    if writer is not None:
        writer.add_scalar("charts/test_episodic_return", result["reward"], global_step)
        if np.isfinite(result["success"]):
            writer.add_scalar("charts/test_success", result["success"], global_step)
    return result


def _log_finished_episodes(writer, infos, global_step, success_threshold=None):
    """Log full-game episode records across Gymnasium vector-info layouts."""

    def log_episode(ep_return, ep_length):
        ep_return = float(np.asarray(ep_return).reshape(-1)[0])
        ep_length = float(np.asarray(ep_length).reshape(-1)[0])
        writer.add_scalar("charts/episodic_return", ep_return, global_step)
        writer.add_scalar("charts/episodic_length", ep_length, global_step)
        if success_threshold is not None:
            writer.add_scalar(
                "charts/success",
                float(ep_return >= float(success_threshold)),
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
            if enabled and "episode" in fi:
                log_episode(
                    np.asarray(fi["episode"]["r"])[idx],
                    np.asarray(fi["episode"]["l"])[idx],
                )
        return

    if "episode" in infos:
        r = np.atleast_1d(infos["episode"]["r"])
        l = np.atleast_1d(infos["episode"]["l"])
        mask = infos.get("_episode", np.ones(len(r), dtype=bool))
        for idx, enabled in enumerate(mask):
            if enabled:
                log_episode(r[idx], l[idx])


def collect_merge_buffer(agent, envs, steps, task_id, seq_idx, device, seed):
    """Collect exactly ``steps`` frozen-policy transitions as raw uint8 stacks.

    Callers use a one-environment vector env, so one loop iteration is exactly
    one environment transition. This avoids silently exceeding B because of
    vector-env parallelism.

    The destination arrays are preallocated up front and written into row by
    row; the previous version appended per-step chunks to a Python list and
    then ``np.concatenate``d the whole thing, which briefly doubled the
    memory cost of this buffer (list of chunks + the concatenated copy) for
    no benefit, since the final row count is already known exactly.
    """
    if steps <= 0:
        return None, 0.0
    if envs.num_envs != 1:
        raise ValueError("frozen-tail collection must use exactly one environment")

    steps = int(steps)
    obs, _ = envs.reset(seed=seed)
    start = time.time()
    agent.eval()

    obs_arr = None
    action_arr = np.empty(steps, dtype=np.int16)
    for i in range(steps):
        raw = np.asarray(obs)
        if obs_arr is None:
            obs_arr = np.empty((steps,) + raw.shape[1:], dtype=np.uint8)
        x = torch.as_tensor(raw, dtype=torch.float32, device=device).div_(255.0)
        with torch.no_grad():
            action = agent.action_distribution(x).sample()
        action_np = action.cpu().numpy()
        obs_arr[i] = raw[0]
        action_arr[i] = action_np[0]
        obs, _, _, _, _ = envs.step(action_np)

    buffer = {
        "obs": obs_arr,
        "actions": action_arr,
        "task_ids": np.full(steps, int(task_id), dtype=np.int32),
        "source_ids": np.full(steps, int(seq_idx), dtype=np.int32),
    }
    return buffer, time.time() - start


class HistoricalFrameSampler:
    """Uniform sampling over historical pool frames WITHOUT concatenating them.

    The drift regularizer used to sample from ``np.concatenate(all pool
    buffers)``, which materialized a second full copy of every historical
    replay frame across the whole pool (pool_size * buffer rows of 4x84x84
    uint8) and kept it resident for the entire task. Sampling directly from
    the original per-slot arrays is mathematically identical (uniform over
    the same underlying rows) and allocates nothing beyond the sampled batch.
    """

    def __init__(self, arrays):
        self.arrays = [a for a in arrays if a is not None and len(a) > 0]
        if not self.arrays:
            raise ValueError("HistoricalFrameSampler requires at least one non-empty array")
        sizes = np.asarray([len(a) for a in self.arrays], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        self.total = int(self.offsets[-1])
        self.row_shape = tuple(self.arrays[0].shape[1:])
        self.dtype = self.arrays[0].dtype

    def __len__(self):
        return self.total

    def sample(self, n: int) -> np.ndarray:
        n = int(n)
        flat = np.random.randint(0, self.total, size=n)
        slot = np.searchsorted(self.offsets, flat, side="right") - 1
        local = flat - self.offsets[slot]
        out = np.empty((n,) + self.row_shape, dtype=self.dtype)
        for s in np.unique(slot):
            mask = slot == s
            out[mask] = self.arrays[int(s)][local[mask]]
        return out


def _drift_penalty(agent, old_fc, drift_sampler, batch_size, device, coefficient):
    if old_fc is None or drift_sampler is None or coefficient <= 0:
        return None
    count = min(int(batch_size), len(drift_sampler))
    states = (
        torch.as_tensor(drift_sampler.sample(count), dtype=torch.float32, device=device)
        .div_(255.0)
    )
    with torch.no_grad():
        old_features = old_fc(states)
    current_features = agent.fc(states)
    return float(coefficient) * F.mse_loss(current_features, old_features)


def _rss_gib() -> float:
    """Current resident set size in GiB, or nan when /proc is unavailable."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except Exception:
        return float("nan")


def _log_rss(writer, stage: str, step: int) -> float:
    """Record host RSS so an OOM kill can be located in the run timeline."""
    rss = _rss_gib()
    if np.isfinite(rss):
        writer.add_scalar(f"memory/rss_gib/{stage}", rss, step)
        logger.info(f"RSS[{stage}]={rss:.2f} GiB")
    return rss


def _value_loss(args, newvalue, oldvalue, returns):
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_unclipped = (newvalue - returns) ** 2
        v_clipped_pred = oldvalue + torch.clamp(
            newvalue - oldvalue, -args.clip_coef, args.clip_coef
        )
        v_clipped = (v_clipped_pred - returns) ** 2
        return 0.5 * torch.max(v_unclipped, v_clipped).mean()
    return 0.5 * ((newvalue - returns) ** 2).mean()


def _ppo_policy_loss(args, newlogprob, oldlogprob, advantage):
    logratio = newlogprob - oldlogprob
    ratio = logratio.exp()
    pg = torch.max(
        -advantage * ratio,
        -advantage * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
    ).mean()
    return pg, logratio, ratio


def validate_args(args):
    if args.distill_buffer_steps is not None:
        args.distill_extra_steps = int(args.distill_buffer_steps)

    if args.task_id < 0 or args.task_id >= len(TASK_SUITES[args.task_suite]):
        raise ValueError(f"invalid task_id={args.task_id} for {args.task_suite}")
    budget = TaskBudget(args.total_timesteps, args.distill_extra_steps)

    if args.pool_size < 2:
        raise ValueError("pool_size must be >= 2")
    if args.num_envs < 1 or args.num_steps < 1:
        raise ValueError("num_envs and num_steps must be >= 1")
    if budget.training < args.num_envs:
        raise ValueError("Delta-B must contain at least one vector-environment step")
    if budget.training % args.num_envs != 0:
        raise ValueError(
            "Delta-B must be divisible by num_envs so PPO can use exactly the "
            "requested environment-transition budget without overshoot"
        )
    if args.num_minibatches < 1 or args.update_epochs < 1:
        raise ValueError("num_minibatches and update_epochs must be >= 1")
    if args.num_envs * args.num_steps < args.num_minibatches:
        raise ValueError("num_minibatches cannot exceed a full PPO rollout")
    if args.learning_rate <= 0 or args.alpha_learning_rate <= 0:
        raise ValueError("learning rates must be > 0")
    if args.alpha_mass_learning_rate is not None and args.alpha_mass_learning_rate <= 0:
        raise ValueError("alpha_mass_learning_rate must be > 0 when specified")
    if not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        raise ValueError("gamma must be in (0,1] and gae_lambda in [0,1]")
    if args.clip_coef <= 0 or args.max_grad_norm <= 0:
        raise ValueError("clip_coef and max_grad_norm must be > 0")
    if args.ent_coef < 0 or args.vf_coef < 0:
        raise ValueError("ent_coef and vf_coef must be >= 0")
    if args.target_kl is not None and args.target_kl <= 0:
        raise ValueError("target_kl must be > 0 when specified")
    if args.eval_every < 0 or args.num_evals < 1:
        raise ValueError("eval_every must be >= 0 and num_evals must be >= 1")
    if args.analysis_log_every < 0:
        raise ValueError("analysis_log_every must be >= 0")

    if args.fusion_mode == "classic_cka" and args.use_alpha_mass:
        raise ValueError("alpha_mass is only valid with fusion_mode='weight_delta'")
    if args.use_alpha_scale and args.fix_alpha_scale:
        raise ValueError("use_alpha_scale and fix_alpha_scale are mutually exclusive")
    if args.train_shared and args.freeze_root_encoder:
        raise ValueError("train_shared and freeze_root_encoder are contradictory")
    if args.alpha_warmup_steps < 0:
        raise ValueError("alpha_warmup_steps must be >= 0")
    if args.alpha_entropy_reg < 0 or args.alpha_mass_reg < 0 or args.drift_reg < 0:
        raise ValueError("alpha/drift regularizers must be >= 0")
    if args.distill_encoder_lr_mult <= 0:
        raise ValueError("distill_encoder_lr_mult must be > 0")

    if args.composition_space == "policy":
        if budget.frozen_tail < 2:
            raise ValueError("policy composition requires B >= 2 for storage projection")
        if args.projection_epochs < 1 or args.projection_max_samples < 2:
            raise ValueError("policy composition requires a nonempty projection budget")
        if args.use_alpha_mass and not args.constrain_alpha_mass:
            raise ValueError("policy mixtures require constrained sigmoid alpha-mass")
    if args.policy_student_replay:
        if args.composition_space != "policy":
            raise ValueError("policy_student_replay requires composition_space='policy'")
        if args.fusion_mode != "weight_delta" or not args.use_alpha_mass:
            raise ValueError("policy_student_replay requires weight_delta with alpha-mass")
        if not args.distillation:
            raise ValueError("policy_student_replay is the combined distillation variant")

    if args.similarity_samples < 2 or args.max_distill_buffer < 2 or args.distill_max_samples < 2:
        raise ValueError("similarity/distillation sample budgets must be >= 2")
    if args.distill_batch_size < 1:
        raise ValueError("distill_batch_size must be >= 1")
    if not 0 <= args.distill_test_frac < 1:
        raise ValueError("distill_test_frac must be in [0,1)")
    if args.distillation and args.distill_epochs < 1:
        raise ValueError("distill_epochs must be >= 1 when distillation is enabled")
    if (
        args.distillation
        or args.collect_cosine_buffers
        or args.composition_space == "policy"
    ) and budget.frozen_tail < 1:
        raise ValueError("this configuration requires B >= 1")

    return budget


def _optimizer_parameter_groups(agent, args):
    encoder_ids = {id(p) for p in agent.fc.parameters()}
    route_params = []
    if agent.alpha is not None and agent.alpha.requires_grad:
        route_params.append(agent.alpha)
    if agent.alpha_scale is not None and agent.alpha_scale.requires_grad:
        route_params.append(agent.alpha_scale)
    mass_params = []
    if agent.alpha_mass is not None and agent.alpha_mass.requires_grad:
        mass_params.append(agent.alpha_mass)
    alpha_ids = {id(p) for p in route_params + mass_params}

    encoder_params, other_params = [], []
    for p in agent.parameters():
        if not p.requires_grad or id(p) in alpha_ids:
            continue
        if id(p) in encoder_ids:
            encoder_params.append(p)
        else:
            other_params.append(p)

    encoder_lr = args.learning_rate
    if args.prev_units and args.distillation and args.train_shared:
        encoder_lr *= args.distill_encoder_lr_mult

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": args.learning_rate})
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr})
    if not groups:
        raise RuntimeError("No non-routing PPO parameters are trainable")

    main_optimizer = optim.Adam(groups, betas=(0.9, 0.999), eps=1e-5)
    route_optimizer = (
        optim.Adam(route_params, lr=args.alpha_learning_rate, betas=(0.9, 0.999), eps=1e-5)
        if route_params
        else None
    )
    mass_lr = (
        args.alpha_learning_rate
        if args.alpha_mass_learning_rate is None
        else args.alpha_mass_learning_rate
    )
    mass_optimizer = (
        optim.Adam(mass_params, lr=mass_lr, betas=(0.9, 0.999), eps=1e-5)
        if mass_params
        else None
    )
    return {
        "main": main_optimizer,
        "route": route_optimizer,
        "mass": mass_optimizer,
        "encoder_params": encoder_params,
        "other_params": other_params,
        "route_params": route_params,
        "mass_params": mass_params,
        "initial_main_lrs": [float(group["lr"]) for group in main_optimizer.param_groups],
        "mass_lr": float(mass_lr),
    }


def _anneal_optimizers(opts, args, progress):
    if not args.anneal_lr:
        return
    frac = max(0.0, 1.0 - float(progress))
    for group, initial_lr in zip(opts["main"].param_groups, opts["initial_main_lrs"]):
        group["lr"] = frac * initial_lr
    if opts["route"] is not None:
        opts["route"].param_groups[0]["lr"] = frac * args.alpha_learning_rate
    if opts["mass"] is not None:
        opts["mass"].param_groups[0]["lr"] = frac * opts["mass_lr"]


def _zero_optimizers(opts):
    opts["main"].zero_grad(set_to_none=True)
    if opts["route"] is not None:
        opts["route"].zero_grad(set_to_none=True)
    if opts["mass"] is not None:
        opts["mass"].zero_grad(set_to_none=True)


def _step_routing_optimizers(opts):
    if opts["route"] is not None:
        opts["route"].step()
    if opts["mass"] is not None:
        opts["mass"].step()


def main():
    args = tyro.cli(Args)
    budget = validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.task_suite}__task_{args.task_id}__cka-rl__run_ppo__{args.seed}"
    task_name = get_task_name(args.task_id, args.task_suite)
    event_dir = pathlib.Path(args.runs_root) / args.tag / run_name
    analysis_dir = pathlib.Path(args.analysis_root) / args.tag / run_name
    writer = CsvSummaryWriter(str(event_dir))
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
        balance_source_lineages=args.balance_source_lineages,
        hidden_dim=args.head_hidden_dim,
        shared_dim=args.shared_dim,
        train_shared=args.train_shared,
        freeze_root_encoder=args.freeze_root_encoder,
        pretrained_encoder=args.pretrained_encoder,
        composition_space=args.composition_space,
        projection_epochs=args.projection_epochs,
        projection_max_samples=args.projection_max_samples,
        policy_student_replay=args.policy_student_replay,
    ).to(device)

    # Optional historical encoder stabilization.
    old_fc = None
    drift_sampler = None
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
            # Zero-copy view over the pool slots; see HistoricalFrameSampler.
            drift_sampler = HistoricalFrameSampler(past_obs)
            logger.info(
                f"encoder-drift reference pool: {len(drift_sampler)} historical frames "
                "(sampled in place, not concatenated)"
            )
        del past_obs

    opts = _optimizer_parameter_groups(agent, args)

    # Verify frozen encoder really remains fixed.
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
            for group in opts["main"].param_groups
            for p in group["params"]
        }
        if any(id(p) in optimizer_ids for p in agent.fc.parameters()):
            raise RuntimeError("frozen encoder parameters leaked into PPO optimizer")
        with torch.no_grad():
            encoder_fingerprint = torch.cat(
                [p.detach().reshape(-1) for p in agent.fc.parameters()]
            ).clone()

    # Max-size buffers; the final PPO rollout may be shorter so Delta-B is exact.
    # Frames are stored as uint8 and normalized only at point of use: a 4x
    # reduction of the rollout buffer and of every host->device transfer,
    # with bit-identical inputs to the network.
    obs_buf = torch.zeros(
        (args.num_steps, args.num_envs) + obs_shape, dtype=torch.uint8, device=device
    )
    actions_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(np.asarray(next_obs_np), device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    global_step = 0
    next_eval = args.eval_every if args.eval_every > 0 else None
    next_analysis = args.analysis_log_every if args.analysis_log_every > 0 else None
    start_time = time.time()

    agent.set_mixture_warmup(
        mixture_warmup_active(
            0, 0, args.alpha_warmup_steps, args.fusion_mode, agent.policy_pool.pool_length()
        )
    )
    theta_task_start = effective_theta_vector(agent).detach().clone()
    if args.save_analysis_snapshots:
        save_task_snapshot(
            analysis_dir / "start.pt",
            "start",
            0,
            args,
            agent,
            include_effective=True,
            include_critic=True,
        )
    log_training_state(writer, 0, agent, theta_task_start)
    evaluate(agent, args, device, 0, writer)

    last_metrics = {
        "pg_loss": float("nan"),
        "value_loss": float("nan"),
        "entropy": float("nan"),
        "approx_kl": float("nan"),
        "old_approx_kl": float("nan"),
        "clipfrac": float("nan"),
        "novel_policy_loss": float("nan"),
        "mixture_policy_loss": float("nan"),
    }
    last_drift = None
    last_alpha_entropy = None
    last_mass_loss = None

    total_vector_steps = budget.training // args.num_envs
    completed_vector_steps = 0
    pbar = tqdm(total=budget.training)

    while completed_vector_steps < total_vector_steps:
        remaining_vector_steps = total_vector_steps - completed_vector_steps
        rollout_steps = min(args.num_steps, remaining_vector_steps)

        # Do not let one PPO rollout straddle the routing-warmup boundary.
        # Otherwise the behavior-policy provenance in one rollout would mix
        # historical-only and post-warmup policies.
        if mixture_warmup_active(
            global_step,
            0,
            args.alpha_warmup_steps,
            args.fusion_mode,
            agent.policy_pool.pool_length(),
        ):
            transitions_to_boundary = max(args.alpha_warmup_steps - global_step, 1)
            vector_steps_to_boundary = max(
                1, (transitions_to_boundary + args.num_envs - 1) // args.num_envs
            )
            rollout_steps = min(rollout_steps, vector_steps_to_boundary)

        current_batch = rollout_steps * args.num_envs
        progress = global_step / max(budget.training, 1)
        _anneal_optimizers(opts, args, progress)

        mixture_warmup = mixture_warmup_active(
            global_step,
            0,
            args.alpha_warmup_steps,
            args.fusion_mode,
            agent.policy_pool.pool_length(),
        )
        agent.set_mixture_warmup(mixture_warmup)

        # ---------------------------- rollout ----------------------------
        for step in range(rollout_steps):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.float() / 255.0)
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value.flatten()

            next_obs_np, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            global_step += args.num_envs
            completed_vector_steps += 1
            pbar.update(args.num_envs)
            _log_finished_episodes(writer, infos, global_step, args.success_threshold)

            next_done_np = np.logical_or(terminations, truncations)
            rewards_buf[step] = torch.as_tensor(
                reward, dtype=torch.float32, device=device
            )
            next_obs = torch.as_tensor(np.asarray(next_obs_np), device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)

        # ----------------------------- GAE -------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs.float() / 255.0).reshape(1, -1)
            advantages = torch.zeros((rollout_steps, args.num_envs), device=device)
            lastgaelam = 0
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
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
                    + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf[:rollout_steps]

        b_obs = obs_buf[:rollout_steps].reshape((-1,) + obs_shape)
        b_old_logprobs = logprobs_buf[:rollout_steps].reshape(-1)
        b_actions = actions_buf[:rollout_steps].reshape(-1).long()
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf[:rollout_steps].reshape(-1)
        inds = np.arange(current_batch)
        clipfracs = []

        # ----------------------------- PPO -------------------------------
        stop_for_kl = False
        for _epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            minibatches = [x for x in np.array_split(inds, min(args.num_minibatches, current_batch)) if len(x)]
            for mb_np in minibatches:
                mb = torch.as_tensor(mb_np, dtype=torch.long, device=device)
                mb_adv = b_advantages[mb]
                if args.norm_adv and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                last_drift = None
                last_alpha_entropy = None
                last_mass_loss = None

                if args.policy_student_replay:
                    # The rollout was generated by the EXECUTION MIXTURE. The
                    # novel student therefore uses the recorded mixture
                    # log-probability as its behavior-policy denominator; it is
                    # never treated as though the student generated the action.
                    newvalue = agent.get_value(b_obs[mb].float() / 255.0)
                    v_loss = _value_loss(args, newvalue, b_values[mb], b_returns[mb])

                    if not mixture_warmup:
                        novel_logits = agent.novel_policy_logits(b_obs[mb].float() / 255.0)
                        novel_dist = Categorical(logits=novel_logits)
                        novel_logprob = novel_dist.log_prob(b_actions[mb])
                        novel_pg, _, _ = _ppo_policy_loss(
                            args, novel_logprob, b_old_logprobs[mb], mb_adv
                        )
                        novel_entropy = novel_dist.entropy().mean()
                        main_loss = novel_pg - args.ent_coef * novel_entropy + args.vf_coef * v_loss
                        last_drift = _drift_penalty(
                            agent, old_fc, drift_sampler, len(mb_np), device, args.drift_reg
                        )
                        if last_drift is not None:
                            main_loss = main_loss + last_drift
                    else:
                        novel_pg = None
                        novel_entropy = None
                        main_loss = args.vf_coef * v_loss

                    opts["main"].zero_grad(set_to_none=True)
                    main_loss.backward()
                    if mixture_warmup:
                        for p in opts["encoder_params"]:
                            p.grad = None
                    nn.utils.clip_grad_norm_(
                        [p for group in opts["main"].param_groups for p in group["params"]],
                        args.max_grad_norm,
                    )
                    opts["main"].step()

                    # Routing update against the actual stored behavior mixture.
                    dist = agent.routing_action_distribution(b_obs[mb].float() / 255.0)
                    mix_logprob = dist.log_prob(b_actions[mb])
                    mix_pg, logratio, ratio = _ppo_policy_loss(
                        args, mix_logprob, b_old_logprobs[mb], mb_adv
                    )
                    mix_entropy = dist.entropy().mean()
                    route_loss = mix_pg - args.ent_coef * mix_entropy

                    if mixture_warmup and args.alpha_entropy_reg > 0 and agent.alpha is not None:
                        scale = agent.alpha_scale if agent.alpha_scale is not None else 1.0
                        probs = torch.softmax(agent.alpha * scale, dim=0)
                        last_alpha_entropy = -(probs * torch.log(probs + 1e-8)).sum()
                        route_loss = route_loss - args.alpha_entropy_reg * last_alpha_entropy
                    if (
                        not mixture_warmup
                        and agent.alpha_mass is not None
                        and agent.alpha_mass.requires_grad
                        and args.alpha_mass_reg > 0
                    ):
                        eff_mass = agent.policy_pool.effective_alpha_mass()
                        last_mass_loss = args.alpha_mass_reg * (eff_mass ** 2) * ((eff_mass - 1.0) ** 2)
                        route_loss = route_loss + last_mass_loss.mean()

                    if opts["route"] is not None or opts["mass"] is not None:
                        if opts["route"] is not None:
                            opts["route"].zero_grad(set_to_none=True)
                        if opts["mass"] is not None:
                            opts["mass"].zero_grad(set_to_none=True)
                        route_loss.backward()
                        routing_params = opts["route_params"] + opts["mass_params"]
                        if routing_params:
                            nn.utils.clip_grad_norm_(routing_params, args.max_grad_norm)
                        _step_routing_optimizers(opts)

                    pg_loss = mix_pg
                    entropy_loss = mix_entropy
                    last_metrics["novel_policy_loss"] = (
                        float(novel_pg.detach()) if novel_pg is not None else float("nan")
                    )
                    last_metrics["mixture_policy_loss"] = float(mix_pg.detach())
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb].float() / 255.0, b_actions[mb]
                    )
                    pg_loss, logratio, ratio = _ppo_policy_loss(
                        args, newlogprob, b_old_logprobs[mb], mb_adv
                    )
                    v_loss = _value_loss(args, newvalue, b_values[mb], b_returns[mb])
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                    last_drift = _drift_penalty(
                        agent, old_fc, drift_sampler, len(mb_np), device, args.drift_reg
                    )
                    if last_drift is not None:
                        loss = loss + last_drift
                    if mixture_warmup and args.alpha_entropy_reg > 0 and agent.alpha is not None:
                        scale = agent.alpha_scale if agent.alpha_scale is not None else 1.0
                        probs = torch.softmax(agent.alpha * scale, dim=0)
                        last_alpha_entropy = -(probs * torch.log(probs + 1e-8)).sum()
                        loss = loss - args.alpha_entropy_reg * last_alpha_entropy
                    if (
                        not mixture_warmup
                        and agent.alpha_mass is not None
                        and agent.alpha_mass.requires_grad
                        and args.alpha_mass_reg > 0
                    ):
                        eff_mass = agent.policy_pool.effective_alpha_mass()
                        last_mass_loss = args.alpha_mass_reg * (eff_mass ** 2) * ((eff_mass - 1.0) ** 2)
                        loss = loss + last_mass_loss.mean()

                    _zero_optimizers(opts)
                    loss.backward()
                    if mixture_warmup:
                        for p in (
                            agent.policy_pool.own_l0_weight,
                            agent.policy_pool.own_l0_bias,
                            agent.policy_pool.own_l2_weight,
                            agent.policy_pool.own_l2_bias,
                            *opts["encoder_params"],
                        ):
                            p.grad = None
                        if agent.alpha_mass is not None:
                            agent.alpha_mass.grad = None
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    opts["main"].step()
                    _step_routing_optimizers(opts)

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )
                if args.target_kl is not None and approx_kl > args.target_kl:
                    stop_for_kl = True
                    break
            if stop_for_kl:
                break

        # --------------------------- diagnostics -------------------------
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        last_metrics.update(
            {
                "pg_loss": float(pg_loss.detach()),
                "value_loss": float(v_loss.detach()),
                "entropy": float(entropy_loss.detach()),
                "approx_kl": float(approx_kl.detach()),
                "old_approx_kl": float(old_approx_kl.detach()),
                "clipfrac": float(np.mean(clipfracs)) if clipfracs else float("nan"),
            }
        )

        writer.add_scalar("charts/learning_rate", opts["main"].param_groups[0]["lr"], global_step)
        if opts["route"] is not None:
            writer.add_scalar("charts/alpha_learning_rate", opts["route"].param_groups[0]["lr"], global_step)
        if opts["mass"] is not None:
            writer.add_scalar("charts/alpha_mass_learning_rate", opts["mass"].param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", last_metrics["value_loss"], global_step)
        writer.add_scalar("losses/policy_loss", last_metrics["pg_loss"], global_step)
        writer.add_scalar("losses/entropy", last_metrics["entropy"], global_step)
        writer.add_scalar("losses/old_approx_kl", last_metrics["old_approx_kl"], global_step)
        writer.add_scalar("losses/approx_kl", last_metrics["approx_kl"], global_step)
        writer.add_scalar("losses/clipfrac", last_metrics["clipfrac"], global_step)
        writer.add_scalar("losses/explained_variance", float(explained_var), global_step)
        if args.policy_student_replay:
            if np.isfinite(last_metrics["novel_policy_loss"]):
                writer.add_scalar("losses/novel_policy_loss", last_metrics["novel_policy_loss"], global_step)
            if np.isfinite(last_metrics["mixture_policy_loss"]):
                writer.add_scalar("losses/mixture_weight_policy_loss", last_metrics["mixture_policy_loss"], global_step)
        if last_drift is not None:
            writer.add_scalar("losses/encoder_drift_reg", float(last_drift.detach()), global_step)
        if last_alpha_entropy is not None:
            writer.add_scalar("losses/knowledge_alpha_entropy", float(last_alpha_entropy.detach()), global_step)
        if last_mass_loss is not None:
            writer.add_scalar("losses/alpha_mass_reg", float(last_mass_loss.mean().detach()), global_step)
        writer.add_scalar("charts/SPS", int(global_step / max(time.time() - start_time, 1e-9)), global_step)

        if next_eval is not None and global_step >= next_eval:
            evaluate(agent, args, device, global_step, writer)
            while next_eval is not None and next_eval <= global_step:
                next_eval += args.eval_every
        if next_analysis is not None and global_step >= next_analysis:
            log_training_state(writer, global_step, agent, theta_task_start)
            while next_analysis is not None and next_analysis <= global_step:
                next_analysis += args.analysis_log_every

    pbar.close()
    if global_step != budget.training:
        raise RuntimeError(
            f"optimization budget mismatch: got {global_step}, expected {budget.training}"
        )

    train_loop_seconds = time.time() - start_time
    writer.add_scalar("timing/train_loop_seconds", train_loop_seconds, budget.total)
    logger.info(
        f"TRAIN_LOOP_SECONDS={train_loop_seconds:.2f} | "
        f"optimization_transitions={budget.training}"
    )

    # The merge/distillation/projection phase below is the memory peak of the
    # whole run (it touches every retained pool buffer at once). Nothing from
    # the PPO loop is needed any more, so release it first.
    del old_fc, drift_sampler, obs_buf, actions_buf, logprobs_buf, rewards_buf, dones_buf, values_buf
    old_fc = None
    drift_sampler = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log_rss(writer, "post_train_loop", budget.training)

    if encoder_should_be_frozen:
        with torch.no_grad():
            current_encoder = torch.cat([p.detach().reshape(-1) for p in agent.fc.parameters()])
            encoder_max_drift = float((current_encoder - encoder_fingerprint).abs().max().item())
        writer.add_scalar("analysis/encoder/max_drift", encoder_max_drift, budget.total)
        if encoder_max_drift != 0.0:
            raise RuntimeError(f"frozen shared encoder drifted by {encoder_max_drift}")

    # Exact pre-tail active policy at the end of the optimization phase.
    agent.set_mixture_warmup(
        mixture_warmup_active(
            budget.training,
            0,
            args.alpha_warmup_steps,
            args.fusion_mode,
            agent.policy_pool.pool_length(),
        )
    )
    evaluate(agent, args, device, budget.training, writer)
    log_training_state(writer, budget.training, agent, theta_task_start)

    # Every condition consumes the same frozen B interactions; only some retain them.
    needs_buffer = bool(
        args.distillation
        or args.collect_cosine_buffers
        or args.composition_space == "policy"
    )
    tail_buffer = None
    buffer_seconds = 0.0
    if budget.frozen_tail:
        tail_envs = make_vector_env(args, num_envs=1)
        try:
            tail_buffer, buffer_seconds = collect_merge_buffer(
                agent,
                tail_envs,
                budget.frozen_tail,
                args.task_id,
                args.seq_idx,
                device,
                args.seed + 123_456,
            )
        finally:
            tail_envs.close()
    _log_rss(writer, "post_collect_merge_buffer", budget.total)
    merge_buffer = bounded_buffer(tail_buffer, args.max_distill_buffer) if needs_buffer else None

    final_step = budget.total
    writer.add_scalar("timing/merge_buffer_seconds", buffer_seconds, final_step)
    writer.add_scalar("analysis/buffer/rows", 0 if merge_buffer is None else len(merge_buffer["obs"]), final_step)
    writer.add_scalar("budget/optimization_phase_env_steps", budget.training, final_step)
    writer.add_scalar("budget/frozen_tail_env_steps", budget.frozen_tail, final_step)
    writer.add_scalar("budget/total_learning_env_steps", budget.total, final_step)

    final_eval = evaluate(agent, args, device, final_step, writer)
    agent.set_own_buffer(merge_buffer)

    run_dir = pathlib.Path(args.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Exact active policy before pool topology changes.
    agent.save_policy_snapshot(str(run_dir))
    with (run_dir / "interaction_budget.json").open("w") as f:
        json.dump(
            {
                "Delta": budget.total,
                "optimization_phase_steps": budget.training,
                "frozen_tail_steps": budget.frozen_tail,
                "monitor_evaluation_steps": getattr(agent, "evaluation_env_steps", 0),
                "evaluation_updates_policy": False,
            },
            f,
            indent=2,
        )

    if args.save_analysis_snapshots:
        save_task_snapshot(
            analysis_dir / "pre_finalize.pt",
            "pre_finalize",
            final_step,
            args,
            agent,
            include_effective=True,
            include_critic=True,
        )

    t_finalize = time.time()
    _log_rss(writer, "pre_finalize", final_step)
    if args.prev_units:
        agent.finalize()
    else:
        agent.set_base()
    gc.collect()
    finalize_seconds = time.time() - t_finalize
    _log_rss(writer, "post_finalize", final_step)
    writer.add_scalar("timing/finalize_seconds", finalize_seconds, final_step)
    writer.add_scalar("analysis/pool/final_length", agent.policy_pool.pool_length(), final_step)

    for key, value in getattr(agent, "last_projection_metrics", {}).items():
        if value is not None:
            writer.add_scalar(key, float(value), final_step)
    with (run_dir / "projection_metrics.json").open("w") as f:
        json.dump(getattr(agent, "last_projection_metrics", {}), f, indent=2)

    info = agent.get_merge_info()
    if info:
        for key in (
            "idx1", "idx2", "similarity_states",
            "cosine_similarity", "pairwise_cosine_min", "pairwise_cosine_mean", "pairwise_cosine_max",
            "symmetric_kl", "pairwise_kl_min", "pairwise_kl_mean", "pairwise_kl_max",
            "selected_state_kl_p95", "selected_state_kl_max",
            "pool_size_before", "pool_size_after",
        ):
            value = info.get(key)
            if value is not None and np.isscalar(value):
                writer.add_scalar(f"analysis/merge/{key}", float(value), final_step)
        writer.add_scalar("analysis/merge/used_distillation", float(bool(info.get("used_distillation", False))), final_step)
        writer.add_scalar("analysis/merge/balance_source_lineages", float(bool(info.get("balance_source_lineages", False))), final_step)
        writer.add_scalar("analysis/merge/source_lineages_parent_1", len(info.get("parent_1_source_lineage", {})), final_step)
        writer.add_scalar("analysis/merge/source_lineages_parent_2", len(info.get("parent_2_source_lineage", {})), final_step)
        writer.add_scalar("analysis/merge/source_lineages_merged", len(info.get("merged_source_lineage", {})), final_step)
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
        writer.add_text("analysis/merge/lineage", json.dumps(lineage, sort_keys=True), final_step)
        writer.add_text("analysis/merge/info", json.dumps(info, sort_keys=True), final_step)

    with torch.no_grad():
        for i, entry in enumerate(agent.policy_pool.pool):
            vec = torch.cat([entry[k].reshape(-1) for k in _HEAD_KEYS])
            writer.add_scalar(f"analysis/pool/norm_slot_{i}", float(vec.norm()), final_step)
        if agent.alpha_mass is not None:
            raw_mass = float(agent.alpha_mass.item())
            effective_mass = float(agent.policy_pool.effective_alpha_mass().item())
            writer.add_scalar("analysis/alpha_mass_raw", raw_mass, final_step)
            writer.add_scalar("analysis/alpha_mass", effective_mass, final_step)
            writer.add_scalar("analysis/alpha_mass_effective", effective_mass, final_step)

    for name, value in agent.get_distill_metrics().items():
        if value is not None:
            writer.add_scalar(f"distillation/{name}", float(value), final_step)

    writer.add_scalar("charts/final_episodic_return", final_eval["reward"], final_step)
    if np.isfinite(final_eval["success"]):
        writer.add_scalar("charts/final_success", final_eval["success"], final_step)

    agent.save(str(run_dir))
    if args.save_analysis_snapshots:
        save_task_snapshot(
            analysis_dir / "post_finalize.pt",
            "post_finalize",
            final_step,
            args,
            agent,
            include_effective=False,
            include_critic=False,
        )

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
