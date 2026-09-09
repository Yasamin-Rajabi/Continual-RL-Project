import inspect
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
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from analysis_logging import effective_theta_vector, log_training_state, save_task_snapshot
from cka_rl import CkaRlAgent
from experiment_identity import write_manifest
from policy_utils import bound_log_std
from policy_composition import (
    sample_action, representative_action, sac_actor_objective,
    novel_sac_actor_objective, mixture_weight_sac_actor_objective,
)
from training_protocol import TaskBudget, mixture_warmup_active, bounded_buffer
from shared_arch import shared
from tasks import get_task, get_task_name


@dataclass
class Args:
    model_type: Literal["cka-rl"] = "cka-rl"
    task_suite: Literal["walker2d_dynamics", "walker2d_mixed_dynamics"] = "walker2d_dynamics"
    fusion_mode: Literal["classic_cka", "weight_delta"] = "classic_cka"
    eval_action_mode: Literal["deterministic", "stochastic"] = "deterministic"
    composition_space: Literal["parameter", "policy"] = "parameter"
    policy_student_replay: bool = False
    """Policy-space student variant: execute the full mixture, train only the
    standalone novel expert from replay, then update alpha/alpha-mass in a
    separate routing step. Historical expert heads remain frozen."""
    projection_epochs: int = 16
    projection_max_samples: int = 20_000
    distill_buffer_steps: Optional[int] = None
    """Alias for distill_extra_steps. B is INCLUDED in total_timesteps, not added."""
    save_dir: Optional[str] = None
    prev_units: Tuple[pathlib.Path, ...] = ()

    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cka-walker2d"
    wandb_entity: Optional[str] = None
    capture_video: bool = False

    task_id: int = 0
    # Unique occurrence index within the continual sequence.  task_id can
    # repeat; seq_idx is what lets buffer-lineage analysis distinguish those
    # occurrences. Scratch/single-task runs can leave this at 0.
    seq_idx: int = 0
    eval_every: int = 10_000
    num_evals: int = 5
    total_timesteps: int = 200_000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5_000
    random_actions_end: int = 5_000
    policy_lr: float = 3e-4
    alpha_lr: float = 5e-3
    alpha_warmup_steps: int = 5_000
    q_lr: float = 3e-4
    policy_frequency: int = 2
    target_network_frequency: int = 1
    # Fixed SAC entropy coefficient when autotune=False.  With autotune=True
    # the current code initializes log_alpha separately (see discussion in the
    # project notes); this value is not used as the initial temperature.
    alpha: float = 0.2
    autotune: bool = True
    autotune_init_from_alpha: bool = False
    """If True, entropy autotuning starts from --alpha. False preserves the
    legacy CleanRL-style initialization alpha_SAC=1.0. This is an ablation
    switch because changing the initial temperature changes early learning."""
    tag: str = "Debug"
    runs_root: str = "runs"

    pool_size: int = 5
    encoder_from_base: bool = True
    distillation: bool = True
    use_alpha_mass: bool = False
    use_alpha_scale: bool = False
    fix_alpha_scale: bool = False
    alpha_mass_reg: float = 0.05
    drift_reg: float = 1.0
    distill_encoder_lr_mult: float = 0.1
    """Multiplier on policy_lr for a trainable shared encoder on later
    distillation tasks. Set to 1.0 to disable the slower-encoder optimization."""
    alpha_entropy_reg: float = 0.01
    """Entropy bonus on the historical knowledge-mixture during weight-delta
    warmup. Set to 0 to disable it."""
    constrain_alpha_mass: bool = True
    """When alpha-mass is enabled, map its raw scalar through a positive
    sigmoid transform into [0,1]. Disable only for the legacy/unconstrained ablation."""
    # Was True, which contradicted both cka_rl.py's own docstring ("train_shared=False
    # (default)") and run_continual_benchmark.py, which always passes --no-train-shared.
    # Running run_sac.py directly (as the README examples do) therefore used a
    # DIFFERENT algorithm from the benchmark: the encoder was reloaded from the root
    # task every task AND left trainable, so it drifted during each task and was then
    # discarded. Pool entries trained under one encoder were being fused under another.
    train_shared: bool = False
    freeze_root_encoder: bool = False
    """Random-frozen encoder ablation. With the normal no-pretraining baseline,
    False lets task 0 learn the root encoder and freezes it on later tasks.
    A pretrained encoder is frozen from task 0 whenever train_shared=False."""
    pretrained_encoder: Optional[str] = None
    """Path to an fc.pt from tdjepa_pretrain.py. It initializes task 0 and is
    frozen by default. With --train-shared, later tasks continue from the latest
    fine-tuned encoder rather than reloading this file each task."""
    encoder_linear_out: bool = False
    """Drop the shared encoder's trailing ReLU. Must MATCH the setting the
    pretrained encoder was produced with, and changes the critic too, so baselines
    have to be re-run under the same value."""

    # Distillation modes require rollout states for behavioral KL.
    # Cosine modes do not; collect_cosine_buffers=True is available when an
    # equal retained-buffer ablation is desired for an ablation.
    distill_observation_skip: bool = False
    """Friend-method skip connection: in distillation modes concatenate raw
    observations to shared features before the policy heads. Disable for the
    pre-merge architecture ablation."""
    distill_extra_steps: int = 10_000
    """Legacy flag name: the final B steps INSIDE Delta; never extra interactions."""
    collect_cosine_buffers: bool = False
    max_distill_buffer: int = 50_000
    similarity_samples: int = 2_048
    distill_max_samples: int = 20_000
    distill_epochs: int = 16
    distill_select_best_val: bool = True
    """Restore the epoch with lowest held-out KL. Disable to reproduce the
    legacy behavior that always keeps the final distillation epoch."""
    distill_lr: float = 5e-4
    distill_batch_size: int = 256
    distill_test_frac: float = 0.2

    analysis_log_every: int = 5_000
    save_analysis_snapshots: bool = True
    analysis_root: str = "analysis_runs"


def make_env(task_id: int, task_suite: str):
    def thunk():
        return gym.wrappers.RecordEpisodeStatistics(
            get_task(task_id, task_suite=task_suite)
        )

    return thunk


def make_vector_env(task_id: int, task_suite: str):
    kwargs = {}
    # Gymnasium >=1.0 exposes autoreset_mode. Gymnasium 0.29 does not.
    if "autoreset_mode" in inspect.signature(gym.vector.SyncVectorEnv).parameters:
        kwargs["autoreset_mode"] = gym.vector.AutoresetMode.SAME_STEP
    return gym.vector.SyncVectorEnv([make_env(task_id, task_suite)], **kwargs)


class SoftQNetwork(nn.Module):
    def __init__(self, envs, linear_out=False):
        super().__init__()
        input_dim = int(np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape))
        self.fc = shared(input_dim, linear_out=linear_out)
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        return self.fc_out(self.fc(x))


class Actor(nn.Module):
    def __init__(self, envs, model):
        super().__init__()
        self.model = model
        self.register_buffer(
            "action_scale",
            torch.as_tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        mean, raw_log_std = self.model(x)
        return mean, bound_log_std(raw_log_std)

    def get_action(self, x):
        return sample_action(self.model, x, self.action_scale, self.action_bias)

    def deterministic_action(self, x):
        return representative_action(self.model, x, self.action_scale, self.action_bias)

    def actor_objective(self, obs, q1, q2, temperature):
        return sac_actor_objective(self.model, obs, q1, q2, temperature,
                                   self.action_scale, self.action_bias)

    def novel_actor_objective(self, obs, q1, q2, temperature):
        return novel_sac_actor_objective(
            self.model, obs, q1, q2, temperature, self.action_scale, self.action_bias
        )

    def mixture_weight_objective(self, obs, q1, q2, temperature):
        return mixture_weight_sac_actor_objective(
            self.model, obs, q1, q2, temperature, self.action_scale, self.action_bias
        )


@torch.no_grad()
def eval_agent(agent, test_env, num_evals, global_step, writer, device):
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(10_000)
        return _eval_agent_impl(agent, test_env, num_evals, global_step, writer, device)


def _eval_agent_impl(agent, test_env, num_evals, global_step, writer, device):
    returns, success_rates, mean_velocity_errors, mean_x_velocities = [], [], [], []
    episode_lengths, fall_flags = [], []
    for ep in range(num_evals):
        obs, _ = test_env.reset(seed=10_000 + ep)
        ep_return = 0.0
        ep_success = []
        ep_velocity_error = []
        ep_x_velocity = []
        ep_steps = 0
        ended_by_fall = False
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = (agent.get_action(obs_t)[0] if getattr(agent, "evaluation_action_mode", "deterministic") == "stochastic"
                      else agent.deterministic_action(obs_t))
            agent.evaluation_env_steps = getattr(agent, "evaluation_env_steps", 0) + 1
            obs, reward, terminated, truncated, info = test_env.step(action[0].cpu().numpy())
            ep_return += float(reward)
            ep_steps += 1
            if terminated:
                ended_by_fall = True
            if "success" in info:
                ep_success.append(float(info["success"]))
            if "velocity_error" in info:
                ep_velocity_error.append(float(info["velocity_error"]))
            if "x_velocity" in info:
                ep_x_velocity.append(float(info["x_velocity"]))
            if terminated or truncated:
                break
        returns.append(ep_return)
        success_rates.append(float(np.mean(ep_success)) if ep_success else np.nan)
        mean_velocity_errors.append(float(np.mean(ep_velocity_error)) if ep_velocity_error else np.nan)
        mean_x_velocities.append(float(np.mean(ep_x_velocity)) if ep_x_velocity else np.nan)
        episode_lengths.append(float(ep_steps))
        fall_flags.append(float(ended_by_fall))

    def finite_mean(values):
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        return float(finite.mean()) if finite.size else float("nan")

    metrics = {
        "return": float(np.mean(returns)),
        "success": finite_mean(success_rates),
        "velocity_error": finite_mean(mean_velocity_errors),
        "x_velocity": finite_mean(mean_x_velocities),
        "episode_length": finite_mean(episode_lengths),
        "fall_rate": finite_mean(fall_flags),
    }
    print(
        f"\nTEST: return={metrics['return']:.3f}, success={metrics['success']:.3f}, "
        f"velocity_error={metrics['velocity_error']:.4f}, "
        f"x_velocity={metrics['x_velocity']:.4f}, "
        f"ep_len={metrics['episode_length']:.1f}, fall_rate={metrics['fall_rate']:.3f}\n"
    )
    writer.add_scalar("charts/test_episodic_return", metrics["return"], global_step)
    writer.add_scalar("charts/test_success", metrics["success"], global_step)
    writer.add_scalar("charts/test_velocity_error", metrics["velocity_error"], global_step)
    writer.add_scalar("charts/test_x_velocity", metrics["x_velocity"], global_step)
    writer.add_scalar("charts/test_episode_length", metrics["episode_length"], global_step)
    writer.add_scalar("charts/test_fall_rate", metrics["fall_rate"], global_step)
    return metrics


def _log_finished_episodes(writer, infos, global_step):
    """Support both old and new Gymnasium vector-info layouts."""
    # Newer same-step autoreset: final_info is an object array of dictionaries.
    if "final_info" in infos and not isinstance(infos["final_info"], dict):
        final_infos = infos["final_info"]
        mask = infos.get("_final_info", np.ones(len(final_infos), dtype=bool))
        for idx, enabled in enumerate(mask):
            if not enabled or final_infos[idx] is None:
                continue
            fi = final_infos[idx]
            if "episode" in fi:
                writer.add_scalar("charts/episodic_return", float(fi["episode"]["r"]), global_step)
                writer.add_scalar("charts/episodic_length", float(fi["episode"]["l"]), global_step)
            if "success" in fi:
                writer.add_scalar("charts/success", float(fi["success"]), global_step)
            if "velocity_error" in fi:
                writer.add_scalar("charts/velocity_error", float(fi["velocity_error"]), global_step)
        return

    # Some vector wrappers expose final_info as a dict of arrays.
    if "final_info" in infos and isinstance(infos["final_info"], dict):
        fi = infos["final_info"]
        mask = infos.get("_final_info", np.ones(1, dtype=bool))
        for idx, enabled in enumerate(mask):
            if not enabled:
                continue
            if "episode" in fi:
                writer.add_scalar("charts/episodic_return", float(np.asarray(fi["episode"]["r"])[idx]), global_step)
                writer.add_scalar("charts/episodic_length", float(np.asarray(fi["episode"]["l"])[idx]), global_step)
            if "success" in fi:
                writer.add_scalar("charts/success", float(np.asarray(fi["success"])[idx]), global_step)
            if "velocity_error" in fi:
                writer.add_scalar("charts/velocity_error", float(np.asarray(fi["velocity_error"])[idx]), global_step)
        return

    # Older layouts can expose the episode record directly.
    if "episode" in infos:
        mask = infos.get("_episode", np.ones(len(np.atleast_1d(infos["episode"]["r"])), dtype=bool))
        for idx, enabled in enumerate(mask):
            if enabled:
                writer.add_scalar("charts/episodic_return", float(np.asarray(infos["episode"]["r"])[idx]), global_step)
                writer.add_scalar("charts/episodic_length", float(np.asarray(infos["episode"]["l"])[idx]), global_step)


def _replace_autoreset_observations(next_obs, terminations, truncations, infos):
    """Use the true final observation for replay when same-step autoreset is active."""
    real_next_obs = next_obs.copy()
    final_key = None
    mask_key = None
    for candidate, candidate_mask in (
        ("final_observation", "_final_observation"),
        ("final_obs", "_final_obs"),
    ):
        if candidate in infos:
            final_key, mask_key = candidate, candidate_mask
            break
    if final_key is None:
        return real_next_obs

    values = infos[final_key]
    mask = infos.get(mask_key, np.ones(len(real_next_obs), dtype=bool))
    done = np.logical_or(terminations, truncations)
    for idx in range(len(real_next_obs)):
        if done[idx] and mask[idx] and values[idx] is not None:
            real_next_obs[idx] = values[idx]
    return real_next_obs


def collect_merge_buffer(actor, envs, steps, task_id, seq_idx, device, seed):
    """Collect raw on-policy states/actions for KL similarity and distillation."""
    obs_rows, action_rows, velocity_rows, error_rows = [], [], [], []
    obs, _ = envs.reset(seed=seed)
    actor.eval()
    start = time.time()

    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, _, _ = actor.get_action(obs_t)
        actions_np = actions.cpu().numpy()
        obs_rows.append(obs.copy())
        action_rows.append(actions_np.copy())

        next_obs, _, _, _, infos = envs.step(actions_np)
        x_velocity = infos.get("x_velocity")
        velocity_error = infos.get("velocity_error")
        if x_velocity is None:
            velocity_rows.append(np.full((envs.num_envs, 1), np.nan, dtype=np.float32))
        else:
            velocity_rows.append(np.asarray(x_velocity, dtype=np.float32).reshape(envs.num_envs, 1))
        if velocity_error is None:
            error_rows.append(np.full((envs.num_envs, 1), np.nan, dtype=np.float32))
        else:
            error_rows.append(np.asarray(velocity_error, dtype=np.float32).reshape(envs.num_envs, 1))
        obs = next_obs

    buffer = {
        "obs": np.concatenate(obs_rows, axis=0).astype(np.float32, copy=False),
        "actions": np.concatenate(action_rows, axis=0).astype(np.float32, copy=False),
        "task_ids": np.full(steps * envs.num_envs, int(task_id), dtype=np.int32),
        # source_ids identify the UNIQUE occurrence that produced each row.
        # This is deliberately separate from task_ids because the continual
        # sequence revisits the same task IDs.
        "source_ids": np.full(steps * envs.num_envs, int(seq_idx), dtype=np.int32),
        "x_velocity": np.concatenate(velocity_rows, axis=0),
        "velocity_error": np.concatenate(error_rows, axis=0),
    }
    return buffer, time.time() - start


def _validate_args(args):
    if args.distill_buffer_steps is not None:
        args.distill_extra_steps = int(args.distill_buffer_steps)
    budget = TaskBudget(args.total_timesteps, args.distill_extra_steps)
    if args.learning_starts < 0 or args.random_actions_end < 0:
        raise ValueError("learning_starts and random_actions_end must be nonnegative")
    if budget.training <= args.learning_starts + 1:
        raise ValueError("Delta - B must exceed learning_starts + 1 so SAC can update")
    if args.composition_space == "policy":
        if args.distill_extra_steps < 2 or args.projection_epochs < 1 or args.projection_max_samples < 2:
            raise ValueError("Policy composition requires B >= 2 and a nonempty projection budget")
        if args.use_alpha_mass and not args.constrain_alpha_mass:
            raise ValueError("Policy mixtures require --constrain-alpha-mass")
    if args.policy_student_replay:
        if args.composition_space != "policy":
            raise ValueError("--policy-student-replay requires --composition-space=policy")
        if args.fusion_mode != "weight_delta" or not args.use_alpha_mass:
            raise ValueError("--policy-student-replay requires weight_delta with alpha-mass")
        if not args.distillation:
            raise ValueError("--policy-student-replay is the combined behavioral-distillation variant")
    if args.fusion_mode == "classic_cka" and args.use_alpha_mass:
        raise ValueError("--use-alpha-mass is only valid with --fusion-mode=weight_delta")
    if args.use_alpha_scale and args.fix_alpha_scale:
        raise ValueError("--use-alpha-scale and --fix-alpha-scale are mutually exclusive")
    if args.alpha_lr <= 0 or args.policy_lr <= 0 or args.q_lr <= 0:
        raise ValueError("policy/q/alpha learning rates must be > 0")
    if args.alpha_warmup_steps < 0:
        raise ValueError("alpha_warmup_steps must be >= 0")
    if args.alpha_mass_reg < 0 or args.drift_reg < 0 or args.alpha_entropy_reg < 0:
        raise ValueError("alpha_mass_reg, drift_reg and alpha_entropy_reg must be >= 0")
    if args.distill_encoder_lr_mult <= 0:
        raise ValueError("distill_encoder_lr_mult must be > 0")
    if args.pool_size < 2:
        raise ValueError("pool_size must be >= 2 for meaningful behavioral pair selection")
    if args.similarity_samples < 2:
        raise ValueError("similarity_samples must be >= 2")
    if (args.distillation or args.collect_cosine_buffers) and args.distill_extra_steps < 1:
        raise ValueError("distill_extra_steps must be >= 1 when a merge buffer is collected")
    if args.distill_extra_steps < 0:
        raise ValueError("distill_extra_steps must be >= 0")
    if args.max_distill_buffer < 2:
        raise ValueError("max_distill_buffer must be >= 2")
    if args.distill_max_samples < 2:
        raise ValueError("distill_max_samples must be >= 2")
    if args.distillation and args.distill_epochs < 1:
        raise ValueError("distill_epochs must be >= 1 when distillation is enabled")
    if args.distill_batch_size < 1 or args.batch_size < 1:
        raise ValueError("batch sizes must be >= 1")
    if args.total_timesteps < 1 or args.num_evals < 1:
        raise ValueError("total_timesteps and num_evals must be >= 1")
    if not 0.0 <= args.distill_test_frac < 1.0:
        raise ValueError("distill_test_frac must be in [0, 1)")
    if args.train_shared and args.freeze_root_encoder:
        raise ValueError("--train-shared and --freeze-root-encoder are contradictory")
    if args.autotune_init_from_alpha and args.alpha <= 0:
        raise ValueError("--alpha must be > 0 when --autotune-init-from-alpha is enabled")
    if args.track:
        raise NotImplementedError("--track is declared but W&B integration is not implemented")
    if args.capture_video:
        raise NotImplementedError("--capture-video is declared but video recording is not implemented")


if __name__ == "__main__":
    args = tyro.cli(Args)
    _validate_args(args)

    run_name = f"{args.task_suite}__task_{args.task_id}__{args.model_type}__{args.exp_name}__{args.seed}"
    task_name = get_task_name(args.task_id, args.task_suite)
    print(f"\n*** Run name: {run_name} | {task_name} ***\n")

    writer = SummaryWriter(str(pathlib.Path(args.runs_root) / args.tag / run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"*** Device: {device}")

    # Periodic evaluation uses its own env: stepping/resetting the training env
    # during evaluation was a real replay-buffer corruption bug in the old code.
    envs = make_vector_env(args.task_id, args.task_suite)
    eval_env = get_task(args.task_id, task_suite=args.task_suite)
    # Box.sample() owns an RNG separate from NumPy's global RNG.  The first
    # random_actions_end exploration actions come from this space, so seed it
    # explicitly for reproducibility across identical seeds/modes.
    envs.single_action_space.seed(args.seed)
    if not isinstance(envs.single_action_space, gym.spaces.Box):
        raise TypeError("SAC implementation supports continuous Box actions only")

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    base_dir = args.prev_units[0] if args.prev_units else None
    latest_dir = args.prev_units[-1] if args.prev_units else None
    model = CkaRlAgent(
        base_dir=base_dir,
        latest_dir=latest_dir,
        obs_dim=obs_dim,
        act_dim=act_dim,
        pool_size=args.pool_size,
        encoder_from_base=args.encoder_from_base,
        distillation=args.distillation,
        distill_observation_skip=args.distill_observation_skip,
        max_distill_buffer=args.max_distill_buffer,
        fusion_mode=args.fusion_mode,
        composition_space=args.composition_space,
        projection_epochs=args.projection_epochs,
        projection_max_samples=args.projection_max_samples,
        policy_student_replay=args.policy_student_replay,
        use_alpha_mass=args.use_alpha_mass,
        use_alpha_scale=args.use_alpha_scale,
        fix_alpha_scale=args.fix_alpha_scale,
        constrain_alpha_mass=args.constrain_alpha_mass,
        distill_test_frac=args.distill_test_frac,
        similarity_samples=args.similarity_samples,
        distill_max_samples=args.distill_max_samples,
        distill_epochs=args.distill_epochs,
        distill_select_best_val=args.distill_select_best_val,
        distill_lr=args.distill_lr,
        distill_batch_size=args.distill_batch_size,
        train_shared=args.train_shared,
        freeze_root_encoder=args.freeze_root_encoder,
        pretrained_encoder=args.pretrained_encoder,
        encoder_linear_out=args.encoder_linear_out,
    )

    actor = Actor(envs, model).to(device)
    actor.evaluation_action_mode = args.eval_action_mode

    # Friend-method continual encoder stabilization. When the shared encoder is
    # explicitly trainable in a distillation condition, keep a frozen copy of
    # the incoming encoder and regularize its representation on historical
    # buffer states. Frozen-encoder/default runs never enter this branch.
    old_fc = None
    past_obs_pool = None
    if args.seq_idx > 0 and args.train_shared and args.distillation:
        import copy
        old_fc = copy.deepcopy(actor.model.fc).to(device)
        old_fc.eval()
        for p in old_fc.parameters():
            p.requires_grad = False
        past_obs_list = [
            entry["buffer"]["obs"] for entry in actor.model.mean_pool.pool
            if entry.get("buffer") is not None and "obs" in entry["buffer"]
        ]
        if past_obs_list:
            past_obs_pool = np.concatenate(past_obs_list, axis=0)

    qf1 = SoftQNetwork(envs, linear_out=args.encoder_linear_out).to(device)
    qf2 = SoftQNetwork(envs, linear_out=args.encoder_linear_out).to(device)
    qf1_target = SoftQNetwork(envs, linear_out=args.encoder_linear_out).to(device)
    qf2_target = SoftQNetwork(envs, linear_out=args.encoder_linear_out).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_params = [p for p in actor.parameters() if p.requires_grad]
    if not actor_params:
        raise RuntimeError("No trainable actor parameters found")

    fc_param_ids = {id(p) for p in actor.model.fc.parameters()}
    fc_params = [p for p in actor.model.fc.parameters() if p.requires_grad]
    alpha_param_objs = []
    if actor.model.alpha is not None and actor.model.alpha.requires_grad:
        alpha_param_objs.append(actor.model.alpha)
    if actor.model.alpha_scale is not None and actor.model.alpha_scale.requires_grad:
        alpha_param_objs.append(actor.model.alpha_scale)
    if actor.model.alpha_mass is not None and actor.model.alpha_mass.requires_grad:
        alpha_param_objs.append(actor.model.alpha_mass)
    alpha_param_ids = {id(p) for p in alpha_param_objs}
    own_params = [
        p for p in actor_params
        if id(p) not in alpha_param_ids and id(p) not in fc_param_ids
    ]

    # Friend method: after task 0, a trainable shared encoder moves more slowly
    # in distillation modes; alpha parameters get their own faster learning rate.
    encoder_lr = (
        args.policy_lr * args.distill_encoder_lr_mult
        if (args.distillation and args.seq_idx > 0)
        else args.policy_lr
    )
    param_groups = []
    if own_params:
        param_groups.append({"params": own_params, "lr": args.policy_lr})
    if fc_params:
        param_groups.append({"params": fc_params, "lr": encoder_lr})
    if alpha_param_objs:
        param_groups.append({"params": alpha_param_objs, "lr": args.alpha_lr})

    actor_optimizer = None
    novel_optimizer = None
    mixture_optimizer = None
    if args.policy_student_replay:
        novel_groups = []
        if own_params:
            novel_groups.append({"params": own_params, "lr": args.policy_lr})
        if fc_params:
            novel_groups.append({"params": fc_params, "lr": encoder_lr})
        if not novel_groups:
            raise RuntimeError("Policy-student mode has no trainable novel-expert parameters")
        novel_optimizer = optim.Adam(novel_groups)
        if alpha_param_objs:
            mixture_optimizer = optim.Adam([{"params": alpha_param_objs, "lr": args.alpha_lr}])
    else:
        actor_optimizer = optim.Adam(param_groups)

    # The whole knowledge-vector formulation assumes a FIXED basis: every stored
    # pool entry was learned relative to one particular encoder. If the encoder
    # moves, those entries silently stop meaning what they meant -- and the
    # damage shows up as "forgetting" in the retention matrix and as corrupted
    # behavioural-KL merge decisions, neither of which points at the real cause.
    _encoder_frozen = (
        not args.train_shared
        and (args.pretrained_encoder is not None or latest_dir is not None or args.freeze_root_encoder)
    )
    _encoder_fingerprint = None
    if _encoder_frozen:
        assert all(not p.requires_grad for p in actor.model.fc.parameters()), \
            "train_shared=False but encoder parameters still require grad"
        _fc_ids = {id(p) for p in actor.model.fc.parameters()}
        assert not any(id(p) in _fc_ids for p in actor_params), \
            "frozen encoder parameters leaked into the actor optimizer"
        with torch.no_grad():
            _encoder_fingerprint = torch.cat(
                [p.reshape(-1) for p in actor.model.fc.parameters()]
            ).clone()

    if args.autotune:
        target_entropy = -float(np.prod(envs.single_action_space.shape))
        initial_log_alpha = np.log(args.alpha) if args.autotune_init_from_alpha else 0.0
        log_alpha = torch.tensor(
            [initial_log_alpha], dtype=torch.float32, requires_grad=True, device=device
        )
        alpha = float(log_alpha.exp().item())
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        log_alpha = None
        alpha = args.alpha
        a_optimizer = None

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    actor.model.set_mixture_warmup(mixture_warmup_active(
        0, args.learning_starts, args.alpha_warmup_steps, args.fusion_mode,
        actor.model.mean_pool.pool_length()))
    theta_task_start = effective_theta_vector(actor.model).detach().clone()
    analysis_dir = f"{args.analysis_root}/{args.tag}/{run_name}"
    if args.save_analysis_snapshots:
        save_task_snapshot(
            f"{analysis_dir}/start.pt", "start", 0, args, actor.model,
            qf1, qf2, qf1_target, qf2_target, alpha, log_alpha,
            include_effective=True, include_critics=True,
        )
    log_training_state(writer, 0, actor.model, qf1, qf2, qf1_target, qf2_target, theta_task_start)
    # Zero-shot performance before any update on this task is a useful
    # continual-transfer diagnostic and is plotted by run_continual_benchmark.py.
    actor.model.set_mixture_warmup(mixture_warmup_active(
        0, args.learning_starts, args.alpha_warmup_steps, args.fusion_mode,
        actor.model.mean_pool.pool_length()))
    eval_agent(actor, eval_env, args.num_evals, 0, writer, device)

    obs, _ = envs.reset(seed=args.seed)
    actor_loss = None
    novel_actor_loss = None
    mixture_actor_loss = None
    alpha_loss = None
    drift_loss = None
    alpha_entropy = None
    mass_loss = None
    start_time = time.time()

    budget = TaskBudget(args.total_timesteps, args.distill_extra_steps)
    for global_step in tqdm(range(budget.training)):
        mixture_warmup = mixture_warmup_active(
            global_step, args.learning_starts, args.alpha_warmup_steps,
            args.fusion_mode, actor.model.mean_pool.pool_length())
        actor.model.set_mixture_warmup(mixture_warmup)
        if global_step < args.random_actions_end:
            actions = np.asarray([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.as_tensor(obs, dtype=torch.float32, device=device))
            actions = actions.cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        _log_finished_episodes(writer, infos, global_step)
        real_next_obs = _replace_autoreset_observations(
            next_obs, terminations, truncations, infos
        )

        # Time-limit truncation is not an MDP terminal for SAC bootstrapping.
        replay_infos = [{} for _ in range(envs.num_envs)]
        rb.add(obs, real_next_obs, actions, rewards, terminations, replay_infos)
        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_actions, next_log_pi, _ = actor.get_action(data.next_observations)
                q1_next = qf1_target(data.next_observations, next_actions)
                q2_next = qf2_target(data.next_observations, next_actions)
                min_q_next = torch.min(q1_next, q2_next) - alpha * next_log_pi
                next_q_value = data.rewards.flatten() + (
                    1.0 - data.dones.flatten()
                ) * args.gamma * min_q_next.view(-1)

            qf1_values = qf1(data.observations, data.actions).view(-1)
            qf2_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    drift_loss = None
                    alpha_entropy = None
                    mass_loss = None
                    novel_actor_loss = None
                    mixture_actor_loss = None

                    if args.policy_student_replay:
                        # Alternating policy-student update. The replay buffer is
                        # generated by the execution mixture. During warmup only
                        # routing coefficients move. Afterwards: (1) update the
                        # standalone novel expert from replay states using SAC;
                        # then (2) update alpha/alpha-mass against the freshly
                        # updated execution mixture, with all expert functions
                        # detached in the routing objective.
                        if not mixture_warmup:
                            novel_actor_loss = actor.novel_actor_objective(
                                data.observations, qf1, qf2, alpha
                            )
                            if (
                                args.distillation and old_fc is not None and past_obs_pool is not None
                                and args.drift_reg > 0
                            ):
                                drift_idx = np.random.randint(0, len(past_obs_pool), size=args.batch_size)
                                s_past = torch.as_tensor(
                                    past_obs_pool[drift_idx], dtype=torch.float32, device=device
                                )
                                with torch.no_grad():
                                    phi_old = old_fc(s_past)
                                phi_curr = actor.model.fc(s_past)
                                drift_loss = args.drift_reg * F.mse_loss(phi_curr, phi_old)
                                novel_actor_loss = novel_actor_loss + drift_loss
                            novel_optimizer.zero_grad()
                            novel_actor_loss.backward()
                            novel_optimizer.step()

                        if mixture_optimizer is not None:
                            mixture_actor_loss = actor.mixture_weight_objective(
                                data.observations, qf1, qf2, alpha
                            )
                            if mixture_warmup and args.alpha_entropy_reg > 0:
                                scale = actor.model.alpha_scale if actor.model.alpha_scale is not None else 1.0
                                probs = torch.softmax(actor.model.alpha * scale, dim=-1)
                                alpha_entropy = -(probs * torch.log(probs + 1e-8)).sum()
                                mixture_actor_loss = mixture_actor_loss - args.alpha_entropy_reg * alpha_entropy
                            if (
                                not mixture_warmup and actor.model.alpha_mass is not None
                                and actor.model.alpha_mass.requires_grad and args.alpha_mass_reg > 0
                            ):
                                eff_mass = actor.model.mean_pool.effective_alpha_mass()
                                mass_loss = args.alpha_mass_reg * (eff_mass ** 2) * ((eff_mass - 1.0) ** 2)
                                mixture_actor_loss = mixture_actor_loss + mass_loss.mean()
                            mixture_optimizer.zero_grad()
                            mixture_actor_loss.backward()
                            mixture_optimizer.step()

                        actor_loss = novel_actor_loss if novel_actor_loss is not None else mixture_actor_loss
                    else:
                        actor_loss = actor.actor_objective(data.observations, qf1, qf2, alpha)

                        if (
                            args.distillation and old_fc is not None and past_obs_pool is not None
                            and args.drift_reg > 0
                        ):
                            drift_idx = np.random.randint(0, len(past_obs_pool), size=args.batch_size)
                            s_past = torch.as_tensor(
                                past_obs_pool[drift_idx], dtype=torch.float32, device=device
                            )
                            with torch.no_grad():
                                phi_old = old_fc(s_past)
                            phi_curr = actor.model.fc(s_past)
                            drift_loss = args.drift_reg * F.mse_loss(phi_curr, phi_old)
                            actor_loss = actor_loss + drift_loss

                        # mixture_warmup was set before choosing this step's action.
                        # A singleton pool still gets a historical-only phase;
                        # its alpha gradient is correctly zero (there is no choice).
                        if mixture_warmup and args.alpha_entropy_reg > 0:
                            scale = actor.model.alpha_scale if actor.model.alpha_scale is not None else 1.0
                            probs = torch.softmax(actor.model.alpha * scale, dim=-1)
                            alpha_entropy = -(probs * torch.log(probs + 1e-8)).sum()
                            actor_loss = actor_loss - args.alpha_entropy_reg * alpha_entropy

                        if (
                            not mixture_warmup and actor.model.alpha_mass is not None
                            and actor.model.alpha_mass.requires_grad and args.alpha_mass_reg > 0
                        ):
                            eff_mass = actor.model.mean_pool.effective_alpha_mass()
                            mass_loss = args.alpha_mass_reg * (eff_mass ** 2) * ((eff_mass - 1.0) ** 2)
                            actor_loss = actor_loss + mass_loss.mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()

                        # Weight-delta warmup first learns how to mix historical
                        # slots before allowing the new residual or mass to move.
                        if mixture_warmup:
                            for p in own_params + fc_params:
                                p.grad = None
                            if actor.model.alpha_mass is not None:
                                actor.model.alpha_mass.grad = None

                        actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi_alpha, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi_alpha + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = float(log_alpha.exp().item())

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", 0.5 * qf_loss.item(), global_step)
                if actor_loss is not None:
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                if novel_actor_loss is not None:
                    writer.add_scalar("losses/novel_actor_loss", novel_actor_loss.item(), global_step)
                if mixture_actor_loss is not None:
                    writer.add_scalar("losses/mixture_weight_actor_loss", mixture_actor_loss.item(), global_step)
                if drift_loss is not None:
                    writer.add_scalar("losses/encoder_drift_reg", float(drift_loss.item()), global_step)
                if alpha_entropy is not None:
                    writer.add_scalar("losses/knowledge_alpha_entropy", float(alpha_entropy.item()), global_step)
                if mass_loss is not None:
                    writer.add_scalar("losses/alpha_mass_reg", float(mass_loss.mean().item()), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                if args.autotune and alpha_loss is not None:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                writer.add_scalar(
                    "charts/SPS",
                    int((global_step + 1) / max(time.time() - start_time, 1e-9)),
                    global_step,
                )

        if args.eval_every > 0 and global_step > 0 and global_step % args.eval_every == 0:
            eval_agent(actor, eval_env, args.num_evals, global_step, writer, device)

        if args.analysis_log_every > 0 and global_step > 0 and global_step % args.analysis_log_every == 0:
            log_training_state(
                writer, global_step, actor.model, qf1, qf2, qf1_target, qf2_target,
                theta_task_start,
            )

    global_step = args.total_timesteps
    if _encoder_frozen:
        with torch.no_grad():
            _drift = float(
                (torch.cat([p.reshape(-1) for p in actor.model.fc.parameters()])
                 - _encoder_fingerprint).abs().max()
            )
        writer.add_scalar("analysis/encoder/max_drift", _drift, global_step)
        assert _drift == 0.0, f"frozen shared encoder drifted by {_drift}"
    train_loop_seconds = time.time() - start_time
    writer.add_scalar("timing/train_loop_seconds", train_loop_seconds, global_step)
    print(
        f"*** TRAIN_LOOP_SECONDS: {train_loop_seconds:.2f} for {budget.training} optimization-phase steps "
        f"({budget.training / max(train_loop_seconds, 1e-9):.2f} steps/sec) ***"
    )

    eval_agent(actor, eval_env, args.num_evals, budget.training, writer, device)

    # Every condition receives the SAME Delta-B training steps and B frozen
    # policy-controlled tail interactions. No optimizer is called in the tail.
    needs_merge_buffer = bool(args.distillation or args.collect_cosine_buffers
                              or args.composition_space == "policy")
    if budget.frozen_tail:
        print(f"*** Frozen tail: {budget.frozen_tail} steps INSIDE Delta={budget.total} ***")
        tail_buffer, buffer_seconds = collect_merge_buffer(
            actor, envs, budget.frozen_tail, args.task_id, args.seq_idx, device,
            seed=args.seed + 123_456,
        )
        merge_buffer = bounded_buffer(tail_buffer, args.max_distill_buffer) if needs_merge_buffer else None
        writer.add_scalar("timing/merge_buffer_seconds", buffer_seconds, global_step)
        writer.add_scalar("analysis/buffer/rows", 0 if merge_buffer is None else len(merge_buffer["obs"]), global_step)
    else:
        merge_buffer = None
        writer.add_scalar("timing/merge_buffer_seconds", 0.0, global_step)
    writer.add_scalar("budget/optimization_phase_env_steps", budget.training, global_step)
    writer.add_scalar("budget/frozen_tail_env_steps", budget.frozen_tail, global_step)
    writer.add_scalar("budget/total_learning_env_steps", budget.total, global_step)

    final_eval = eval_agent(actor, eval_env, args.num_evals, global_step, writer, device)
    actor.model.set_own_buffer(merge_buffer)

    if args.save_dir is not None:
        print(f"Saving trained agent in `{args.save_dir}` with name `{run_name}`")
        run_dir = f"{args.save_dir}/{run_name}"
        actor.model.save_policy_snapshot(run_dir)
        with open(pathlib.Path(run_dir) / "interaction_budget.json", "w") as f:
            json.dump({"Delta": budget.total, "optimization_phase_steps": budget.training,
                       "frozen_tail_steps": budget.frozen_tail,
                       "monitor_evaluation_steps": getattr(actor, "evaluation_env_steps", 0),
                       "evaluation_updates_policy": False}, f, indent=2)

        if args.save_analysis_snapshots:
            save_task_snapshot(
                f"{analysis_dir}/pre_finalize.pt", "pre_finalize", global_step, args,
                actor.model, qf1, qf2, qf1_target, qf2_target, alpha, log_alpha,
                include_effective=True, include_critics=True,
            )

        merge_start = time.time()
        if base_dir is None and latest_dir is None:
            actor.model.set_base()
        else:
            actor.model.finalize()
        for key, value in actor.model.last_projection_metrics.items():
            writer.add_scalar(key, value, global_step)
        with open(pathlib.Path(run_dir) / "projection_metrics.json", "w") as f:
            json.dump(actor.model.last_projection_metrics, f, indent=2)
        finalize_seconds = time.time() - merge_start
        writer.add_scalar("timing/finalize_seconds", finalize_seconds, global_step)
        print(f"*** FINALIZE_SECONDS: {finalize_seconds:.4f} ***")

        if args.save_analysis_snapshots:
            save_task_snapshot(
                f"{analysis_dir}/post_finalize.pt", "post_finalize", global_step, args,
                actor.model, qf1, qf2, qf1_target, qf2_target, alpha, log_alpha,
                include_effective=False, include_critics=False,
            )

        merge_info = actor.model.get_merge_info()
        if merge_info is not None:
            if merge_info["similarity_metric"] == "symmetric_kl":
                writer.add_scalar("analysis/merge/symmetric_kl", merge_info["symmetric_kl"], global_step)
                writer.add_scalar("analysis/merge/pairwise_kl_min", merge_info["pairwise_kl_min"], global_step)
                writer.add_scalar("analysis/merge/pairwise_kl_mean", merge_info["pairwise_kl_mean"], global_step)
                writer.add_scalar("analysis/merge/pairwise_kl_max", merge_info["pairwise_kl_max"], global_step)
                writer.add_scalar("analysis/merge/selected_state_kl_p95", merge_info["selected_state_kl_p95"], global_step)
                writer.add_scalar("analysis/merge/selected_state_kl_max", merge_info["selected_state_kl_max"], global_step)
            elif merge_info["similarity_metric"] == "cosine":
                writer.add_scalar("analysis/merge/cosine_similarity", merge_info["cosine_similarity"], global_step)
                writer.add_scalar("analysis/merge/pairwise_cosine_min", merge_info["pairwise_cosine_min"], global_step)
                writer.add_scalar("analysis/merge/pairwise_cosine_mean", merge_info["pairwise_cosine_mean"], global_step)
                writer.add_scalar("analysis/merge/pairwise_cosine_max", merge_info["pairwise_cosine_max"], global_step)
            else:
                raise RuntimeError(f"unknown merge similarity metric: {merge_info['similarity_metric']}")
            writer.add_scalar("analysis/merge/idx1", merge_info["idx1"], global_step)
            writer.add_scalar("analysis/merge/idx2", merge_info["idx2"], global_step)
            if "similarity_states" in merge_info:
                writer.add_scalar("analysis/merge/similarity_states", merge_info["similarity_states"], global_step)
            writer.add_scalar("analysis/merge/used_distillation", float(merge_info["used_distillation"]), global_step)
            writer.add_scalar("analysis/merge/pool_size_before", merge_info["pool_size_before"], global_step)
            writer.add_scalar("analysis/merge/pool_size_after", merge_info["pool_size_after"], global_step)
            lineage = {
                "task_ids": {
                    "parent_1": merge_info.get("parent_1_lineage", {}),
                    "parent_2": merge_info.get("parent_2_lineage", {}),
                    "merged": merge_info.get("merged_lineage", {}),
                },
                "source_ids": {
                    "parent_1": merge_info.get("parent_1_source_lineage", {}),
                    "parent_2": merge_info.get("parent_2_source_lineage", {}),
                    "merged": merge_info.get("merged_source_lineage", {}),
                },
            }
            writer.add_text("analysis/merge/lineage", json.dumps(lineage, sort_keys=True), global_step)
            print(f"*** MERGE_LINEAGE: {json.dumps(lineage, sort_keys=True)} ***")
        writer.add_scalar("analysis/pool/final_length", actor.model.mean_pool.pool_length(), global_step)

        # weight_delta stores V_k = own + hist, so ||V_k|| can grow along the
        # sequence. alpha_mass is parameterized through a positive transform;
        # log both its raw optimization parameter and effective positive mass.
        with torch.no_grad():
            for _i, _entry in enumerate(actor.model.mean_pool.pool):
                _v = torch.cat([_entry[_k].reshape(-1) for _k in
                                ("l0_weight", "l0_bias", "l2_weight", "l2_bias")])
                writer.add_scalar(f"analysis/pool/norm_slot_{_i}", float(_v.norm()), global_step)
            if actor.model.alpha_mass is not None:
                raw_mass = float(actor.model.alpha_mass.item())
                effective_mass = float(actor.model.mean_pool.effective_alpha_mass().item())
                writer.add_scalar("analysis/alpha_mass_raw", raw_mass, global_step)
                writer.add_scalar("analysis/alpha_mass", effective_mass, global_step)
                writer.add_scalar("analysis/alpha_mass_effective", effective_mass, global_step)

        for metric_name, value in actor.model.get_distill_metrics().items():
            if value is not None:
                writer.add_scalar(f"distillation/{metric_name}", float(value), global_step)
                print(f"*** distillation/{metric_name} = {value} ***")

        writer.add_scalar("charts/final_return", final_eval["return"], global_step)
        writer.add_scalar("charts/final_success", final_eval["success"], global_step)
        writer.add_scalar("charts/final_velocity_error", final_eval["velocity_error"], global_step)
        writer.add_scalar("charts/final_x_velocity", final_eval["x_velocity"], global_step)
        writer.add_scalar("charts/final_episode_length", final_eval["episode_length"], global_step)
        writer.add_scalar("charts/final_fall_rate", final_eval["fall_rate"], global_step)
        actor.model.save(dirname=run_dir)
        manifest = write_manifest(
            run_dir, vars(args), parent_dirs=args.prev_units,
            pretrained_encoder=args.pretrained_encoder,
        )
        print(f"*** RUN_SIGNATURE: {manifest['run_signature']} ***")

    envs.close()
    eval_env.close()
    writer.close()
