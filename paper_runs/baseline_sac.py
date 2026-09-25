"""The SAC training loop shared by every continual-RL baseline.

Ported from the uploaded four_baselines SAC trainer; see DONOR_PROVENANCE.json.
Critics are reset at task boundaries to match the main benchmark. The seven
baseline policies use this one loop with the main project's environment and
probability-distribution helpers. This gives the following shared protocol:

* environments come from ``tasks.get_task`` through the same
  ``RecordEpisodeStatistics`` / ``SyncVectorEnv`` construction, so wrappers,
  horizons, goal freezing and info keys are identical by construction;
* the budget is the same ``training_protocol.TaskBudget`` object, so every
  condition gets exactly ``Delta - B`` optimization steps and ``B`` frozen-tail
  interactions;
* action sampling and the actor objective come from ``policy_composition``,
  the same module the method uses;
* evaluation writes the same scalar tags from the same separate eval env with
  the same forked RNG and the same episode seeds.

Suite-specific behaviour is reached ONLY through ``metrics.ERROR_KEY`` and
``metrics.EPISODIC_SUCCESS``, which metrics.py already defines per folder. This
module is shared by all target environment folders, including discrete MiniGrid.

WHAT IS DELIBERATELY DIFFERENT FROM run_sac.py
----------------------------------------------
1. No knowledge pool, no alpha/alpha-mass/alpha-scale, no distillation, no
   merge, no policy-space projection. None of the baselines have them.
2. The frozen tail collects nothing. run_sac.py uses those B steps to build a
   merge buffer; the baselines have nothing to merge, but they still spend the
   B interactions under a frozen policy so the interaction budget matches
   exactly. Skipping them would hand the baselines B fewer environment steps
   and quietly change the comparison.
3. Two lifecycle hooks are called around ``optimizer.step()`` so PackNet and
   ProgNet can enforce parameter isolation.
"""
from __future__ import annotations

import inspect
import time
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from tqdm import tqdm

from metrics import EPISODIC_SUCCESS, ERROR_KEY
from policy_composition import representative_action, sac_actor_objective, sample_action
from policy_utils import bound_log_std
from shared_arch import shared
from tasks import get_task
from training_protocol import TaskBudget


# ======================================================================
# Environment construction. Identical to run_sac.py.
# ======================================================================
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


# ======================================================================
# Networks
# ======================================================================
class SoftQNetwork(nn.Module):
    """Twin-critic network supporting both Box SAC and SAC-Discrete."""

    def __init__(self, envs, linear_out: bool = False):
        super().__init__()
        self.discrete = isinstance(envs.single_action_space, gym.spaces.Discrete)
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        if self.discrete:
            self.n_actions = int(envs.single_action_space.n)
            self.fc = shared(obs_dim, linear_out=linear_out)
            self.fc_out = nn.Linear(256, self.n_actions)
        else:
            input_dim = obs_dim + int(np.prod(envs.single_action_space.shape))
            self.fc = shared(input_dim, linear_out=linear_out)
            self.fc_out = nn.Linear(256, 1)

    def forward(self, x, a=None):
        if self.discrete:
            return self.fc_out(self.fc(x))
        if a is None:
            raise TypeError("continuous critic requires an action tensor")
        x = torch.cat([x, a], dim=1)
        return self.fc_out(self.fc(x))


class BaselineActor(nn.Module):
    """Thin actor wrapper for both continuous and categorical policies."""

    def __init__(self, envs, model):
        super().__init__()
        self.model = model
        self.discrete = isinstance(envs.single_action_space, gym.spaces.Discrete)
        if self.discrete:
            self.n_actions = int(envs.single_action_space.n)
        else:
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

    @property
    def policy(self):
        return self.model.policy

    def forward(self, x):
        head_a, head_b = self.model.policy(x)
        if self.discrete:
            return head_a + head_b
        return head_a, bound_log_std(head_b)

    def categorical_distribution(self, x):
        if not self.discrete:
            raise TypeError("categorical_distribution is only valid for Discrete actions")
        logits = self.forward(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.exp(), log_probs

    def get_action(self, x):
        if self.discrete:
            probs, log_probs = self.categorical_distribution(x)
            action = torch.distributions.Categorical(probs=probs).sample()
            selected = log_probs.gather(1, action[:, None])
            return action, selected, probs.argmax(dim=-1)
        return sample_action(self.model.policy, x, self.action_scale, self.action_bias)

    def deterministic_action(self, x):
        if self.discrete:
            probs, _ = self.categorical_distribution(x)
            return probs.argmax(dim=-1)
        return representative_action(
            self.model.policy, x, self.action_scale, self.action_bias
        )

    def actor_objective(self, obs, q1, q2, temperature):
        if self.discrete:
            probs, log_probs = self.categorical_distribution(obs)
            q = torch.minimum(q1(obs), q2(obs))
            return (probs * (temperature * log_probs - q)).sum(dim=-1).mean()
        return sac_actor_objective(
            self.model.policy, obs, q1, q2, temperature,
            self.action_scale, self.action_bias,
        )


# ======================================================================
# Evaluation. Same protocol, same seeds, same tags as run_sac.py.
# ======================================================================
@torch.no_grad()
def eval_agent(actor, test_env, num_evals, global_step, writer, device):
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(10_000)
        return _eval_agent_impl(actor, test_env, num_evals, global_step, writer, device)


def _eval_agent_impl(actor, test_env, num_evals, global_step, writer, device):
    returns, success_rates, task_errors, x_velocities = [], [], [], []
    for ep in range(num_evals):
        obs, _ = test_env.reset(seed=10_000 + ep)
        ep_return = 0.0
        ep_success, ep_error, ep_x_velocity = [], [], []
        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = (
                actor.get_action(obs_t)[0]
                if getattr(actor, "evaluation_action_mode", "deterministic") == "stochastic"
                else actor.deterministic_action(obs_t)
            )
            actor.evaluation_env_steps = getattr(actor, "evaluation_env_steps", 0) + 1
            env_action = int(action[0].item()) if actor.discrete else action[0].cpu().numpy()
            obs, reward, terminated, truncated, info = test_env.step(env_action)
            ep_return += float(reward)
            if "success" in info:
                ep_success.append(float(info["success"]))
            if ERROR_KEY in info:
                ep_error.append(float(info[ERROR_KEY]))
            if "x_velocity" in info:
                ep_x_velocity.append(float(info["x_velocity"]))
            if terminated or truncated:
                break
        returns.append(ep_return)
        # Meta-World latches success over the episode, so the episodic metric is
        # the max; HalfCheetah's per-step tolerance metric is the mean.
        if ep_success:
            success_rates.append(
                float(np.max(ep_success)) if EPISODIC_SUCCESS else float(np.mean(ep_success))
            )
        else:
            success_rates.append(np.nan)
        task_errors.append(float(np.mean(ep_error)) if ep_error else np.nan)
        x_velocities.append(float(np.mean(ep_x_velocity)) if ep_x_velocity else np.nan)

    def finite_mean(values):
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        return float(finite.mean()) if finite.size else float("nan")

    results = {
        "return": float(np.mean(returns)),
        "success": finite_mean(success_rates),
        ERROR_KEY: finite_mean(task_errors),
        "x_velocity": finite_mean(x_velocities),
    }
    print(
        f"\nTEST: return={results['return']:.3f}, success={results['success']:.3f}, "
        f"{ERROR_KEY}={results[ERROR_KEY]:.4f}\n"
    )
    writer.add_scalar("charts/test_episodic_return", results["return"], global_step)
    writer.add_scalar("charts/test_success", results["success"], global_step)
    writer.add_scalar(f"charts/test_{ERROR_KEY}", results[ERROR_KEY], global_step)
    if np.isfinite(results["x_velocity"]):
        writer.add_scalar("charts/test_x_velocity", results["x_velocity"], global_step)
    return results


def _log_finished_episodes(writer, infos, global_step):
    """Support both old and new Gymnasium vector-info layouts."""
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
            if ERROR_KEY in fi:
                writer.add_scalar(f"charts/{ERROR_KEY}", float(fi[ERROR_KEY]), global_step)
        return

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
            if ERROR_KEY in fi:
                writer.add_scalar(f"charts/{ERROR_KEY}", float(np.asarray(fi[ERROR_KEY])[idx]), global_step)
        return

    if "episode" in infos:
        mask = infos.get("_episode", np.ones(len(np.atleast_1d(infos["episode"]["r"])), dtype=bool))
        for idx, enabled in enumerate(mask):
            if enabled:
                writer.add_scalar("charts/episodic_return", float(np.asarray(infos["episode"]["r"])[idx]), global_step)
                writer.add_scalar("charts/episodic_length", float(np.asarray(infos["episode"]["l"])[idx]), global_step)


def _replace_autoreset_observations(next_obs, terminations, truncations, infos):
    """Use the true final observation for replay when same-step autoreset is active."""
    real_next_obs = next_obs.copy()
    final_key = mask_key = None
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


def run_frozen_tail(actor, envs, steps, device, seed):
    """Spend the final B interactions under a frozen policy.

    The baselines have no pool to merge, so nothing is collected. The steps are
    still taken, because run_sac.py spends them and the comparison is only fair
    if every condition consumes the same Delta total interactions. No optimizer
    is called here.
    """
    if steps <= 0:
        return 0.0
    obs, _ = envs.reset(seed=seed)
    actor.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions, _, _ = actor.get_action(obs_t)
            actions_np = actions.cpu().numpy()
            if actor.discrete:
                actions_np = np.asarray(actions_np, dtype=np.int64).reshape(envs.num_envs)
            obs, _, _, _, _ = envs.step(actions_np)
    actor.train()
    return time.time() - start


# ======================================================================
# The training loop
# ======================================================================
def train_task(agent, ctx, args, writer, device) -> Dict[str, float]:
    """Train ``agent`` on one task and return its final evaluation metrics.

    ``agent`` is a BaselineAgent, ``ctx`` a TaskContext, and ``args``
    is the immutable run configuration assembled by baseline_runner.py.
    """
    budget = TaskBudget(args.total_timesteps, args.frozen_tail_steps)

    envs = make_vector_env(ctx.task_id, ctx.suite)
    eval_env = get_task(ctx.task_id, task_suite=ctx.suite)
    # Action-space sampling owns its own RNG; seed it for reproducible random exploration.
    envs.single_action_space.seed(args.seed)
    if not isinstance(envs.single_action_space, (gym.spaces.Box, gym.spaces.Discrete)):
        raise TypeError("SAC supports Box or Discrete action spaces")

    agent.to(device)
    agent.on_task_start(ctx)

    actor = BaselineActor(envs, agent).to(device)
    actor.evaluation_action_mode = args.eval_action_mode

    qf1 = SoftQNetwork(envs, linear_out=args.encoder_linear_out).to(device)
    qf2 = SoftQNetwork(envs, linear_out=args.encoder_linear_out).to(device)
    qf1_target = SoftQNetwork(envs, linear_out=args.encoder_linear_out).to(device)
    qf2_target = SoftQNetwork(envs, linear_out=args.encoder_linear_out).to(device)

    # Match the main benchmark: reset the twin critics at each task boundary.
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_params = agent.trainable_actor_parameters()
    if not actor_params:
        raise RuntimeError(
            f"{type(agent).__name__} exposed no trainable actor parameters for "
            f"task {ctx.task_id} (seq {ctx.seq_idx})."
        )
    actor_optimizer = optim.Adam(actor_params, lr=args.policy_lr)
    agent.attach_optimizer(actor_optimizer, device)
    print(
        f"*** Trainable actor parameters: "
        f"{sum(p.numel() for p in actor_params):,} across {len(actor_params)} tensors ***"
    )

    if args.autotune:
        target_entropy = (
            0.98 * float(np.log(envs.single_action_space.n))
            if actor.discrete
            else -float(np.prod(envs.single_action_space.shape))
        )
        initial_log_alpha = np.log(args.alpha) if args.autotune_init_from_alpha else 0.0
        log_alpha = torch.tensor(
            [initial_log_alpha], dtype=torch.float32, requires_grad=True, device=device
        )
        alpha = float(log_alpha.exp().item())
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        log_alpha, a_optimizer = None, None
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    # Zero-shot performance before any update on this task: the continual
    # transfer diagnostic, logged at step 0 exactly as run_sac.py does.
    eval_agent(actor, eval_env, args.num_evals, 0, writer, device)

    obs, _ = envs.reset(seed=args.seed)
    actor_loss = alpha_loss = None
    qf1_values = qf2_values = qf1_loss = qf2_loss = qf_loss = None
    start_time = time.time()

    for global_step in tqdm(range(budget.training)):
        agent.on_phase_boundary(global_step, budget)

        if global_step < args.random_actions_end:
            actions = np.asarray(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            with torch.no_grad():
                actions, _, _ = actor.get_action(
                    torch.as_tensor(obs, dtype=torch.float32, device=device)
                )
            actions = actions.cpu().numpy()
        if actor.discrete:
            actions = np.asarray(actions, dtype=np.int64).reshape(envs.num_envs)

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
                if actor.discrete:
                    next_probs, next_log_probs = actor.categorical_distribution(data.next_observations)
                    q1_next = qf1_target(data.next_observations)
                    q2_next = qf2_target(data.next_observations)
                    min_q_next = (
                        next_probs * (torch.min(q1_next, q2_next) - alpha * next_log_probs)
                    ).sum(dim=-1)
                else:
                    next_actions, next_log_pi, _ = actor.get_action(data.next_observations)
                    q1_next = qf1_target(data.next_observations, next_actions)
                    q2_next = qf2_target(data.next_observations, next_actions)
                    min_q_next = (torch.min(q1_next, q2_next) - alpha * next_log_pi).view(-1)
                next_q_value = data.rewards.flatten() + (
                    1.0 - data.dones.flatten()
                ) * args.gamma * min_q_next.view(-1)

            if actor.discrete:
                action_idx = data.actions.long().view(-1, 1)
                qf1_values = qf1(data.observations).gather(1, action_idx).view(-1)
                qf2_values = qf2(data.observations).gather(1, action_idx).view(-1)
            else:
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
                    actor_loss = actor.actor_objective(data.observations, qf1, qf2, alpha)
                    auxiliary = agent.auxiliary_loss()
                    if auxiliary is not None:
                        actor_loss = actor_loss + auxiliary
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # Parameter isolation: mask gradients, step, then restore
                    # the protected weights bit-exactly. Masking alone is not
                    # enough under Adam, whose momentum can move a parameter
                    # whose current gradient is zero.
                    agent.before_optimizer_step()
                    actor_optimizer.step()
                    agent.after_optimizer_step()

                    if args.autotune:
                        with torch.no_grad():
                            if actor.discrete:
                                alpha_probs, alpha_log_probs = actor.categorical_distribution(data.observations)
                                policy_entropy = -(alpha_probs * alpha_log_probs).sum(dim=-1)
                            else:
                                _, log_pi_alpha, _ = actor.get_action(data.observations)
                        alpha_loss = (
                            log_alpha.exp() * (policy_entropy.detach() - target_entropy)
                            if actor.discrete
                            else -log_alpha.exp() * (log_pi_alpha + target_entropy)
                        ).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = float(log_alpha.exp().item())

            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1.0 - args.tau) * target_param.data
                    )
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1.0 - args.tau) * target_param.data
                    )

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
                for name, value in agent.scalars().items():
                    writer.add_scalar(f"baseline/{name}", float(value), global_step)

        if args.eval_every > 0 and global_step > 0 and global_step % args.eval_every == 0:
            eval_agent(actor, eval_env, args.num_evals, global_step, writer, device)

    train_loop_seconds = time.time() - start_time
    writer.add_scalar("timing/train_loop_seconds", train_loop_seconds, budget.total)
    print(
        f"*** TRAIN_LOOP_SECONDS: {train_loop_seconds:.2f} for {budget.training} "
        f"optimization-phase steps "
        f"({budget.training / max(train_loop_seconds, 1e-9):.2f} steps/sec) ***"
    )

    eval_agent(actor, eval_env, args.num_evals, budget.training, writer, device)

    # Frozen tail: B interactions INSIDE Delta, no optimizer call.
    if budget.frozen_tail:
        print(f"*** Frozen tail: {budget.frozen_tail} steps INSIDE Delta={budget.total} ***")
        tail_seconds = run_frozen_tail(
            actor, envs, budget.frozen_tail, device, seed=args.seed + 123_456
        )
        writer.add_scalar("timing/frozen_tail_seconds", tail_seconds, budget.total)

    writer.add_scalar("budget/optimization_phase_env_steps", budget.training, budget.total)
    writer.add_scalar("budget/frozen_tail_env_steps", budget.frozen_tail, budget.total)
    writer.add_scalar("budget/total_learning_env_steps", budget.total, budget.total)

    final_eval = eval_agent(actor, eval_env, args.num_evals, budget.total, writer, device)

    agent.on_task_end(ctx)

    writer.add_scalar("charts/final_return", final_eval["return"], budget.total)
    writer.add_scalar("charts/final_success", final_eval["success"], budget.total)
    writer.add_scalar(f"charts/final_{ERROR_KEY}", final_eval[ERROR_KEY], budget.total)
    for name, value in agent.scalars().items():
        writer.add_scalar(f"baseline/{name}", float(value), budget.total)

    agent.detach_optimizer()
    final_eval["monitor_evaluation_steps"] = getattr(actor, "evaluation_env_steps", 0)
    envs.close()
    eval_env.close()
    return final_eval
