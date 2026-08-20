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
from policy_utils import bound_log_std
from shared_arch import shared
from tasks import get_task, get_task_name


@dataclass
class Args:
    model_type: Literal["cka-rl"] = "cka-rl"
    task_suite: Literal["halfcheetah_vel", "halfcheetah_wind_vel"] = "halfcheetah_vel"
    fusion_mode: Literal["classic_cka", "weight_delta"] = "classic_cka"
    save_dir: Optional[str] = None
    prev_units: Tuple[pathlib.Path, ...] = ()

    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cka-halfcheetah"
    wandb_entity: Optional[str] = None
    capture_video: bool = False

    task_id: int = 0
    eval_every: int = 10_000
    num_evals: int = 5
    total_timesteps: int = 300_000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5_000
    random_actions_end: int = 10_000
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    tag: str = "Debug"

    pool_size: int = 5
    encoder_from_base: bool = True
    distillation: bool = True
    use_alpha_mass: bool = False
    use_alpha_scale: bool = False
    train_shared: bool = False

    # Every condition stores rollout states because behavioral KL needs a
    # reference state distribution even when distillation is disabled.
    distill_extra_steps: int = 10_000
    max_distill_buffer: int = 50_000
    similarity_samples: int = 2_048
    distill_max_samples: int = 20_000
    distill_epochs: int = 8
    distill_lr: float = 3e-4
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
    def __init__(self, envs):
        super().__init__()
        input_dim = int(np.prod(envs.single_observation_space.shape) + np.prod(envs.single_action_space.shape))
        self.fc = shared(input_dim)
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
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


@torch.no_grad()
def eval_agent(agent, test_env, num_evals, global_step, writer, device):
    returns, success_rates, mean_velocity_errors = [], [], []
    for ep in range(num_evals):
        obs, _ = test_env.reset(seed=10_000 + ep)
        ep_return = 0.0
        ep_success = []
        ep_velocity_error = []
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mean, _ = agent(obs_t)
            action = torch.tanh(mean) * agent.action_scale + agent.action_bias
            obs, reward, terminated, truncated, info = test_env.step(action[0].cpu().numpy())
            ep_return += float(reward)
            if "success" in info:
                ep_success.append(float(info["success"]))
            if "velocity_error" in info:
                ep_velocity_error.append(float(info["velocity_error"]))
            if terminated or truncated:
                break
        returns.append(ep_return)
        success_rates.append(float(np.mean(ep_success)) if ep_success else np.nan)
        mean_velocity_errors.append(float(np.mean(ep_velocity_error)) if ep_velocity_error else np.nan)

    metrics = {
        "return": float(np.mean(returns)),
        "success": float(np.nanmean(success_rates)),
        "velocity_error": float(np.nanmean(mean_velocity_errors)),
    }
    print(
        f"\nTEST: return={metrics['return']:.3f}, success={metrics['success']:.3f}, "
        f"velocity_error={metrics['velocity_error']:.4f}\n"
    )
    writer.add_scalar("charts/test_episodic_return", metrics["return"], global_step)
    writer.add_scalar("charts/test_success", metrics["success"], global_step)
    writer.add_scalar("charts/test_velocity_error", metrics["velocity_error"], global_step)
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


def collect_merge_buffer(actor, envs, steps, task_id, device, seed):
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
        "x_velocity": np.concatenate(velocity_rows, axis=0),
        "velocity_error": np.concatenate(error_rows, axis=0),
    }
    return buffer, time.time() - start


def _validate_args(args):
    if args.fusion_mode == "classic_cka" and args.use_alpha_mass:
        raise ValueError("--use-alpha-mass is only valid with --fusion-mode=weight_delta")
    if args.pool_size < 2:
        raise ValueError("pool_size must be >= 2 for meaningful behavioral pair selection")
    if args.similarity_samples < 2:
        raise ValueError("similarity_samples must be >= 2")
    if args.distill_extra_steps < 1:
        raise ValueError("distill_extra_steps must be >= 1 because behavioral KL needs stored states")
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


if __name__ == "__main__":
    args = tyro.cli(Args)
    _validate_args(args)

    run_name = f"{args.task_suite}__task_{args.task_id}__{args.model_type}__{args.exp_name}__{args.seed}"
    task_name = get_task_name(args.task_id, args.task_suite)
    print(f"\n*** Run name: {run_name} | {task_name} ***\n")

    writer = SummaryWriter(f"runs/{args.tag}/{run_name}")
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
        max_distill_buffer=args.max_distill_buffer,
        fusion_mode=args.fusion_mode,
        use_alpha_mass=args.use_alpha_mass,
        use_alpha_scale=args.use_alpha_scale,
        distill_test_frac=args.distill_test_frac,
        similarity_samples=args.similarity_samples,
        distill_max_samples=args.distill_max_samples,
        distill_epochs=args.distill_epochs,
        distill_lr=args.distill_lr,
        distill_batch_size=args.distill_batch_size,
        train_shared=args.train_shared,
    )

    actor = Actor(envs, model).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_params = [p for p in actor.parameters() if p.requires_grad]
    if not actor_params:
        raise RuntimeError("No trainable actor parameters found")
    actor_optimizer = optim.Adam(actor_params, lr=args.policy_lr)

    if args.autotune:
        target_entropy = -float(np.prod(envs.single_action_space.shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
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
    eval_agent(actor, eval_env, args.num_evals, 0, writer, device)

    obs, _ = envs.reset(seed=args.seed)
    actor_loss = None
    alpha_loss = None
    start_time = time.time()

    for global_step in tqdm(range(args.total_timesteps)):
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
                    pi, log_pi, _ = actor.get_action(data.observations)
                    min_q_pi = torch.min(qf1(data.observations, pi), qf2(data.observations, pi))
                    actor_loss = (alpha * log_pi - min_q_pi).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
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
    train_loop_seconds = time.time() - start_time
    writer.add_scalar("timing/train_loop_seconds", train_loop_seconds, global_step)
    print(
        f"*** TRAIN_LOOP_SECONDS: {train_loop_seconds:.2f} for {args.total_timesteps} steps "
        f"({args.total_timesteps / max(train_loop_seconds, 1e-9):.2f} steps/sec) ***"
    )

    # Every four-way condition collects this buffer. It is needed for the new
    # output-space KL selector even when merge distillation itself is disabled.
    print(f"*** Collecting {args.distill_extra_steps} post-training states for behavioral merging ***")
    merge_buffer, buffer_seconds = collect_merge_buffer(
        actor, envs, args.distill_extra_steps, args.task_id, device,
        seed=args.seed + 123_456,
    )
    writer.add_scalar("timing/merge_buffer_seconds", buffer_seconds, global_step)
    writer.add_scalar("analysis/buffer/rows", len(merge_buffer["obs"]), global_step)
    writer.add_scalar("analysis/buffer/mean_velocity_error", float(np.nanmean(merge_buffer["velocity_error"])), global_step)

    final_eval = eval_agent(actor, eval_env, args.num_evals, global_step, writer, device)
    actor.model.set_own_buffer(merge_buffer)

    if args.save_dir is not None:
        print(f"Saving trained agent in `{args.save_dir}` with name `{run_name}`")
        run_dir = f"{args.save_dir}/{run_name}"
        actor.model.save_policy_snapshot(run_dir)

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
            writer.add_scalar("analysis/merge/symmetric_kl", merge_info["symmetric_kl"], global_step)
            writer.add_scalar("analysis/merge/pairwise_kl_min", merge_info["pairwise_kl_min"], global_step)
            writer.add_scalar("analysis/merge/pairwise_kl_mean", merge_info["pairwise_kl_mean"], global_step)
            writer.add_scalar("analysis/merge/pairwise_kl_max", merge_info["pairwise_kl_max"], global_step)
            writer.add_scalar("analysis/merge/idx1", merge_info["idx1"], global_step)
            writer.add_scalar("analysis/merge/idx2", merge_info["idx2"], global_step)
            writer.add_scalar("analysis/merge/similarity_states", merge_info["similarity_states"], global_step)
            writer.add_scalar("analysis/merge/used_distillation", float(merge_info["used_distillation"]), global_step)
            writer.add_scalar("analysis/merge/pool_size_before", merge_info["pool_size_before"], global_step)
            writer.add_scalar("analysis/merge/pool_size_after", merge_info["pool_size_after"], global_step)
            lineage = {
                "parent_1": merge_info.get("parent_1_lineage", {}),
                "parent_2": merge_info.get("parent_2_lineage", {}),
                "merged": merge_info.get("merged_lineage", {}),
            }
            writer.add_text("analysis/merge/lineage", json.dumps(lineage, sort_keys=True), global_step)
            print(f"*** MERGE_LINEAGE: {json.dumps(lineage, sort_keys=True)} ***")
        writer.add_scalar("analysis/pool/final_length", actor.model.mean_pool.pool_length(), global_step)

        for metric_name, value in actor.model.get_distill_metrics().items():
            if value is not None:
                writer.add_scalar(f"distillation/{metric_name}", float(value), global_step)
                print(f"*** distillation/{metric_name} = {value} ***")

        writer.add_scalar("charts/final_return", final_eval["return"], global_step)
        writer.add_scalar("charts/final_success", final_eval["success"], global_step)
        writer.add_scalar("charts/final_velocity_error", final_eval["velocity_error"], global_step)
        actor.model.save(dirname=run_dir)

    envs.close()
    eval_env.close()
    writer.close()
