"""PPO trainer for the legacy Atari continual-learning baselines.

This keeps each method's model-specific update rules, while aligning the shared
Atari/PPO protocol with the paper-tuned CKA-RL benchmark:
  * 8 envs, rollout 128, lr 2.5e-4, gamma .99, GAE .95,
  * PPO clip .2, 4 minibatches, 4 epochs, entropy .01, value coef .5,
  * grad clip .5, LR annealing, clipped value loss, normalized advantages,
  * corrected ALE frameskip=1 + MaxAndSkip(4), sticky-action probability .25,
  * raw full-game deterministic evaluation at step 0 and periodically,
  * exact total environment-transition budget (no floor to full rollouts),
  * sequence-indexed save/event paths supplied by run_experiments.py.

The periodic raw evaluation curves are required for FT_return/FT_success.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from loguru import logger
from tqdm import tqdm

try:
    import ale_py
except ImportError:
    ale_py = None
else:
    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


# The legacy baseline model files serialize whole nn.Module objects and call
# torch.load without weights_only=. PyTorch >=2.6 defaults that argument to
# True, so make the trusted-local-checkpoint intent explicit for this process.
_ORIGINAL_TORCH_LOAD = torch.load
def _compat_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)
torch.load = _compat_torch_load

from benchmark_protocol import canonical_env_name, canonical_method
from checkpoint_evaluation import evaluate_live_agent
from csv_summary_writer import CsvSummaryWriter
from models.cbp_modules import GnT
from models.mask_modules import set_num_tasks_learned
from utils.AdamGnT import AdamGnT
from models import (
    CnnSimpleAgent,
    CnnCompoNetAgent,
    ProgressiveNetAgent,
    PackNetAgent,
    CkaRlAgent,
    CnnMaskAgent,
    CnnCbpAgent,
    CReLUsAgent,
)


@dataclass
class Args:
    method_type: str = "FT-N"
    env_id: str = "ALE/Freeway-v5"
    mode: int = 0
    task_id: int = 0
    seq_idx: int = 0
    task_slot: Optional[int] = None
    task_seen_before: bool = False
    task_head_dir: Optional[pathlib.Path] = None
    prev_units: Tuple[pathlib.Path, ...] = ()
    total_task_num: Optional[int] = None
    num_tasks_learned: int = 0
    componet_finetune_encoder: bool = False
    prevs_to_noise: int = 0

    save_dir: Optional[pathlib.Path] = None
    event_dir: Optional[pathlib.Path] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    debug: bool = False
    track: bool = False
    capture_video: bool = False

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

    # Raw full-game learning curves used by forward-transfer metrics.
    eval_every: int = 50_000
    num_evals: int = 5
    eval_action_mode: str = "deterministic"
    success_threshold: Optional[float] = None

    # Legacy CKA-RL knobs.
    alpha_factor: float = 1e-2
    fix_alpha: bool = False
    alpha_learning_rate: float = 2.5e-4
    delta_theta_mode: str = "T"
    fuse_encoder: bool = False
    fuse_actor: bool = True
    reset_actor: bool = True
    global_alpha: bool = True
    alpha_init: str = "Randn"
    alpha_major: float = 0.6
    pool_size: int = 3


def _grayscale(env):
    cls = getattr(gym.wrappers, "GrayScaleObservation", None)
    if cls is None:
        cls = getattr(gym.wrappers, "GrayscaleObservation")
    return cls(env)


def _frame_stack(env, n=4):
    cls = getattr(gym.wrappers, "FrameStack", None)
    if cls is not None:
        return cls(env, n)
    return getattr(gym.wrappers, "FrameStackObservation")(env, stack_size=n)


def make_train_env(args: Args, idx: int, run_name: str):
    def thunk():
        kwargs = {
            "mode": int(args.mode),
            "frameskip": 1,
            "repeat_action_probability": 0.25,
        }
        if args.capture_video and idx == 0:
            kwargs["render_mode"] = "rgb_array"
        env = gym.make(args.env_id, **kwargs)
        if args.capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = _grayscale(env)
        env = _frame_stack(env, 4)
        return env
    return thunk


def _torch_load(path, map_location=None):
    kwargs = {} if map_location is None else {"map_location": map_location}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _setup_agent(args, envs, device):
    requested = canonical_method(args.method_type)
    method = "Finetune" if requested == "FT-N" else requested
    prev = [str(p) for p in args.prev_units]
    slot = int(args.mode if args.task_slot is None else args.task_slot)

    packnet_retrain_start = None
    packnet_revisit = False

    if method == "Baseline":
        agent = CnnSimpleAgent(envs)
    elif method == "Finetune":
        agent = (
            CnnSimpleAgent.load(prev[-1], envs, load_critic=False, reset_actor=False, map_location=device)
            if prev else CnnSimpleAgent(envs)
        )
    elif method == "CompoNet":
        agent = CnnCompoNetAgent(
            envs,
            prevs_paths=prev,
            finetune_encoder=args.componet_finetune_encoder,
            map_location=device,
        )
    elif method == "ProgNet":
        agent = ProgressiveNetAgent(envs, prevs_paths=prev, map_location=device)
    elif method == "PackNet":
        if args.total_task_num is None:
            raise ValueError("--total-task-num is required for PackNet")
        task_label = slot + 1
        packnet_retrain_start = args.total_timesteps - int(args.total_timesteps * 0.2)
        if not prev:
            agent = PackNetAgent(
                envs,
                task_id=task_label,
                is_first_task=True,
                total_task_num=int(args.total_task_num),
            )
        else:
            agent = PackNetAgent.load(
                prev[-1],
                task_id=task_label,
                restart_actor_critic=True,
                freeze_bias=True,
                map_location=device,
            )
            # The saved latest checkpoint is in a masked task view. Restore the
            # underlying tensor before selecting/training the next task.
            if getattr(agent.network, "view", None) is not None:
                agent.network.set_view(None)
            agent.task_id = task_label
            agent.network.task_id = task_label

            if args.task_seen_before:
                if args.task_head_dir is None:
                    raise ValueError("PackNet revisit requires --task-head-dir")
                old_task = _torch_load(pathlib.Path(args.task_head_dir) / "packnet.pt", device)
                agent.actor = old_task.actor
                agent.retrain_mode = True
                agent.network.set_view(task_label)
                packnet_revisit = True
    elif method == "CKA-RL":
        base_dir = prev[0] if prev else None
        latest_dir = prev[-1] if prev else None
        agent = CkaRlAgent(
            envs,
            base_dir=base_dir,
            latest_dir=latest_dir,
            alpha_factor=args.alpha_factor,
            fix_alpha=args.fix_alpha,
            delta_theta_mode=args.delta_theta_mode,
            fuse_encoder=args.fuse_encoder,
            fuse_actor=args.fuse_actor,
            reset_actor=args.reset_actor,
            global_alpha=args.global_alpha,
            alpha_init=args.alpha_init,
            alpha_major=args.alpha_major,
            pool_size=args.pool_size,
            map_location=device,
        )
        agent.log_alphas()
    elif method == "MaskNet":
        if args.total_task_num is None:
            raise ValueError("--total-task-num is required for MaskNet")
        if prev:
            agent = CnnMaskAgent.load(
                prev[-1],
                envs,
                num_tasks=int(args.total_task_num),
                load_critic=False,
                reset_actor=False,
                map_location=device,
            )
        else:
            agent = CnnMaskAgent(envs, num_tasks=int(args.total_task_num))
        set_num_tasks_learned(agent, int(args.num_tasks_learned), verbose=False)
        agent.set_task(slot, new_task=not args.task_seen_before)
    elif method == "CbpNet":
        if prev:
            agent = CnnCbpAgent.load(
                prev[-1], envs, load_critic=False, reset_actor=False, map_location=device
            )
            # CnnCbpAgent.save() deliberately removes hooks before serialization.
            # Re-register them before continued GnT training.
            if hasattr(agent.actor, "setup_feature_logging"):
                agent.actor.setup_feature_logging(h_dim=(512,))
        else:
            agent = CnnCbpAgent(envs)
    elif method == "CReLUs":
        agent = (
            CReLUsAgent.load(prev[-1], envs, load_critic=False, reset_actor=False, map_location=device)
            if prev else CReLUsAgent(envs)
        )
    else:
        raise ValueError(f"unsupported method {args.method_type!r}")

    return agent.to(device), requested, packnet_retrain_start, packnet_revisit


def _agent_step(agent, method, obs, action=None, writer=None, global_step=None, prevs_to_noise=0):
    if method == "CompoNet":
        return agent.get_action_and_value(
            obs,
            action,
            log_writter=writer if global_step is not None else None,
            global_step=global_step,
            prevs_to_noise=prevs_to_noise,
        )
    if method == "CKA-RL":
        return agent.get_action_and_value(
            obs, action, log_writter=writer, global_step=global_step
        )
    return agent.get_action_and_value(obs, action)


def _log_finished_episodes(writer, infos, global_step, logs):
    if "final_info" not in infos:
        return
    final_infos = infos["final_info"]
    mask = infos.get("_final_info", np.ones(len(final_infos), dtype=bool))
    for idx, enabled in enumerate(mask):
        if not enabled or final_infos[idx] is None:
            continue
        info = final_infos[idx]
        if "episode" not in info:
            continue
        ret = float(info["episode"]["r"])
        length = float(info["episode"]["l"])
        logs["global_step"].append(int(global_step))
        logs["episodic_return"].append(ret)
        writer.add_scalar("charts/episodic_return", ret, global_step)
        writer.add_scalar("charts/episodic_length", length, global_step)


def _validate(args):
    if args.total_timesteps <= 0 or args.num_envs <= 0 or args.num_steps <= 0:
        raise ValueError("timesteps/envs/steps must be positive")
    if args.total_timesteps % args.num_envs != 0:
        raise ValueError("total_timesteps must be divisible by num_envs for exact transition accounting")
    if args.num_minibatches <= 0 or args.update_epochs <= 0:
        raise ValueError("num_minibatches/update_epochs must be positive")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.clip_coef <= 0:
        raise ValueError("invalid PPO optimization setting")
    if args.eval_every < 0 or args.num_evals < 1:
        raise ValueError("eval_every must be >=0 and num_evals >=1")
    if args.eval_action_mode not in {"deterministic", "stochastic"}:
        raise ValueError("eval_action_mode must be deterministic or stochastic")


def main():
    args = tyro.cli(Args)
    _validate(args)
    if not args.debug:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    method = canonical_method(args.method_type)
    env_name = canonical_env_name(args.env_id)
    run_name = f"{env_name}_{int(args.mode)}_{method}_{int(args.seed)}"

    event_dir = pathlib.Path(args.event_dir) if args.event_dir is not None else pathlib.Path("runs") / run_name
    save_dir = pathlib.Path(args.save_dir) if args.save_dir is not None else pathlib.Path("agents") / run_name
    event_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = CsvSummaryWriter(str(event_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_train_env(args, i, run_name) for i in range(args.num_envs)]
    )
    if not isinstance(envs.single_action_space, gym.spaces.Discrete):
        raise TypeError("only discrete action spaces are supported")
    envs.single_action_space.seed(args.seed)

    agent, method, packnet_retrain_start, packnet_revisit = _setup_agent(args, envs, device)

    trainable = [
        p for name, p in agent.named_parameters()
        if p.requires_grad and "alpha" not in name
    ]
    if not trainable:
        raise RuntimeError("no trainable non-alpha parameters")

    if method == "CbpNet":
        optimizer = AdamGnT(trainable, lr=args.learning_rate, eps=1e-5)
        gnt = GnT(
            net=agent.actor.net,
            opt=optimizer,
            replacement_rate=1e-3,
            decay_rate=0.99,
            device=device,
            maturity_threshold=1000,
            util_type="contribution",
        )
    else:
        optimizer = optim.Adam(trainable, lr=args.learning_rate, eps=1e-5, betas=(0.9, 0.999))
        gnt = None

    alpha_params = [
        p for name, p in agent.named_parameters()
        if p.requires_grad and "alpha" in name
    ]
    alpha_optimizer = (
        optim.Adam(alpha_params, lr=args.alpha_learning_rate, eps=1e-5, betas=(0.9, 0.999))
        if method == "CKA-RL" and args.seq_idx > 0 and alpha_params
        else None
    )

    obs_shape = tuple(envs.single_observation_space.shape)
    obs_buf = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    logs = {"global_step": [0], "episodic_return": [0.0]}
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)

    initial_eval = evaluate_live_agent(
        agent,
        method,
        env_name,
        args.mode,
        args.num_evals,
        device,
        action_mode=args.eval_action_mode,
        success_threshold=args.success_threshold,
    )
    writer.add_scalar("charts/test_episodic_return", initial_eval["return"], 0)
    if args.success_threshold is not None:
        writer.add_scalar("charts/test_success", initial_eval["success"], 0)

    global_step = 0
    next_eval = args.eval_every if args.eval_every > 0 else None
    start_time = time.time()
    progress = tqdm(total=args.total_timesteps, desc=f"{env_name}/{method}/seq{args.seq_idx}")

    pg_loss = v_loss = entropy_loss = approx_kl = old_approx_kl = torch.tensor(float("nan"), device=device)
    clipfracs = []

    while global_step < args.total_timesteps:
        remaining = args.total_timesteps - global_step
        rollout_steps = min(args.num_steps, remaining // args.num_envs)
        if rollout_steps <= 0:
            raise RuntimeError("remaining transition budget is smaller than one vector step")

        if args.anneal_lr:
            frac = max(0.0, 1.0 - global_step / args.total_timesteps)
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            if alpha_optimizer is not None:
                alpha_optimizer.param_groups[0]["lr"] = frac * args.alpha_learning_rate

        for step in range(rollout_steps):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = _agent_step(
                    agent,
                    method,
                    next_obs / 255.0,
                    writer=writer if (method == "CKA-RL" or args.track) else None,
                    global_step=global_step if (method == "CKA-RL" or args.track) else None,
                    prevs_to_noise=args.prevs_to_noise,
                )
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            global_step += args.num_envs
            progress.update(args.num_envs)
            next_done_np = np.logical_or(terminated, truncated)
            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32, device=device)
            _log_finished_episodes(writer, infos, global_step, logs)

        with torch.no_grad():
            next_value = agent.get_value(next_obs / 255.0).reshape(-1)
            advantages = torch.zeros((rollout_steps, args.num_envs), device=device)
            lastgaelam = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones_buf[t + 1]
                    next_values = values_buf[t + 1]
                delta = rewards_buf[t] + args.gamma * next_values * next_nonterminal - values_buf[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values_buf[:rollout_steps]

        current_batch = rollout_steps * args.num_envs
        b_obs = obs_buf[:rollout_steps].reshape((-1,) + obs_shape)
        b_old_logprobs = logprobs_buf[:rollout_steps].reshape(-1)
        b_actions = actions_buf[:rollout_steps].reshape(-1).long()
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf[:rollout_steps].reshape(-1)

        inds = np.arange(current_batch)
        clipfracs = []
        stop_early = False
        for _epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for mb_inds in np.array_split(inds, min(args.num_minibatches, current_batch)):
                if not len(mb_inds):
                    continue
                _, newlogprob, entropy, newvalue = _agent_step(
                    agent,
                    method,
                    b_obs[mb_inds] / 255.0,
                    b_actions[mb_inds],
                    prevs_to_noise=args.prevs_to_noise,
                )
                logratio = newlogprob - b_old_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped_pred = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_clipped = (v_clipped_pred - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                optimizer.zero_grad(set_to_none=True)
                if alpha_optimizer is not None:
                    alpha_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

                if method == "PackNet":
                    if not packnet_revisit and packnet_retrain_start is not None and global_step >= packnet_retrain_start:
                        agent.start_retraining()
                    agent.before_update()

                optimizer.step()
                if alpha_optimizer is not None:
                    alpha_optimizer.step()
                if method == "CbpNet":
                    gnt.gen_and_test(agent.actor.get_activations())

            if args.target_kl is not None and float(approx_kl) > args.target_kl:
                stop_early = True
                break
        if stop_early:
            logger.debug("PPO epoch loop stopped at target KL")

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        if alpha_optimizer is not None:
            writer.add_scalar("charts/alpha_learning_rate", alpha_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", float(v_loss), global_step)
        writer.add_scalar("losses/policy_loss", float(pg_loss), global_step)
        writer.add_scalar("losses/entropy", float(entropy_loss), global_step)
        writer.add_scalar("losses/old_approx_kl", float(old_approx_kl), global_step)
        writer.add_scalar("losses/approx_kl", float(approx_kl), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)) if clipfracs else np.nan, global_step)
        writer.add_scalar("losses/explained_variance", float(explained_var), global_step)
        writer.add_scalar("charts/SPS", int(global_step / max(time.time() - start_time, 1e-9)), global_step)

        if next_eval is not None and global_step >= next_eval:
            result = evaluate_live_agent(
                agent,
                method,
                env_name,
                args.mode,
                args.num_evals,
                device,
                action_mode=args.eval_action_mode,
                success_threshold=args.success_threshold,
            )
            writer.add_scalar("charts/test_episodic_return", result["return"], global_step)
            if args.success_threshold is not None:
                writer.add_scalar("charts/test_success", result["success"], global_step)
            while next_eval <= global_step:
                next_eval += args.eval_every

    progress.close()

    # Always store an endpoint evaluation. Duplicate step values are handled by
    # metrics.py deterministically (the last value wins).
    final_eval = evaluate_live_agent(
        agent,
        method,
        env_name,
        args.mode,
        args.num_evals,
        device,
        action_mode=args.eval_action_mode,
        success_threshold=args.success_threshold,
    )
    writer.add_scalar("charts/test_episodic_return", final_eval["return"], global_step)
    if args.success_threshold is not None:
        writer.add_scalar("charts/test_success", final_eval["success"], global_step)
    writer.add_scalar("charts/final_return", final_eval["return"], global_step)
    if args.success_threshold is not None:
        writer.add_scalar("charts/final_success", final_eval["success"], global_step)

    if method == "MaskNet" and not args.task_seen_before:
        # Freeze the newly learned linear-combination mask into this task's
        # score tensor, then mark it as established for future revisits.
        agent.consolidate_mask()
        set_num_tasks_learned(agent, int(args.num_tasks_learned) + 1, verbose=False)

    agent.save(dirname=str(save_dir))
    pd.DataFrame(logs).to_csv(event_dir / "training_returns.csv", index=False)
    with (save_dir / "training_meta.json").open("w") as f:
        json.dump(
            {
                "env": env_name,
                "env_id": args.env_id,
                "mode": int(args.mode),
                "task_id": int(args.task_id),
                "task_slot": int(args.mode if args.task_slot is None else args.task_slot),
                "seq_idx": int(args.seq_idx),
                "method": method,
                "seed": int(args.seed),
                "total_timesteps": int(args.total_timesteps),
                "eval_every": int(args.eval_every),
                "num_evals": int(args.num_evals),
                "success_threshold": args.success_threshold,
                "eval_action_mode": args.eval_action_mode,
            },
            f,
            indent=2,
        )

    envs.close()
    writer.close()
    logger.info(f"saved {method} checkpoint to {save_dir}")


if __name__ == "__main__":
    main()
