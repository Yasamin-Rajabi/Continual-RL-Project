"""Continual PointMaze SAC trainer for the bounded Gaussian CKA-RL agent.

Protocol (shared with every baseline in ``baselines/``)
-------------------------------------------------------
* ``total_timesteps`` is Delta, the whole per-task interaction budget.
* The frozen tail B is INSIDE Delta.  Methods that retain replay states get no
  extra environment interaction over methods that do not.
* Evaluation never updates the policy and its interactions are counted and
  reported separately.
* Snapshots are taken at task start, pre-finalize and post-finalize; the
  pre-finalize snapshot is the exact policy that was executing, taken before
  pool topology changes make the alpha vector stale.

Only stdlib ``argparse`` is used for the CLI: Kaggle images do not reliably
carry ``tyro``, and a missing CLI parser at task 7 of a 10-task chain is an
expensive way to discover a dependency.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from analysis_logging import effective_theta_vector, log_training_state, save_task_snapshot
from cka_rl import CkaRlAgent
from csv_summary_writer import CsvSummaryWriter
from experiment_identity import write_manifest
from pointmaze_env import ACT_DIM, EPISODIC_SUCCESS, OBS_DIM
from policy_utils import SquashedGaussian, clamp_log_std
from replay_buffer import ReplayBuffer
from tasks import DEFAULT_SUITE, SUITES, get_task, get_task_name, num_tasks
from training_protocol import TaskBudget, bounded_buffer, mixture_warmup_active

_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Continual PointMaze SAC for bounded Gaussian CKA-RL",
    )
    p.add_argument("--method", default="Ours", help="label recorded in the manifest")
    p.add_argument("--suite", default=DEFAULT_SUITE, choices=sorted(SUITES))
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--seq-idx", type=int, default=0)
    p.add_argument("--prev-units", nargs="*", default=[])
    p.add_argument("--save-dir", default="agents_pointmaze/debug")
    p.add_argument("--runs-root", default="runs_pointmaze")
    p.add_argument("--tag", default="debug")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)

    # SAC
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--learning-starts", type=int, default=2_000)
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--policy-frequency", type=int, default=2)
    p.add_argument("--target-frequency", type=int, default=1)
    p.add_argument("--autotune-entropy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--entropy-coef", type=float, default=0.2)
    p.add_argument("--max-grad-norm", type=float, default=10.0)

    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--num-evals", type=int, default=10)
    p.add_argument(
        "--eval-action-mode", default="deterministic", choices=["deterministic", "stochastic"]
    )
    p.add_argument("--success-threshold", type=float, default=None)

    # Continual composition / storage
    p.add_argument("--fusion-mode", default="classic_cka", choices=["classic_cka", "weight_delta"])
    p.add_argument("--composition-space", default="parameter", choices=["parameter", "policy"])
    p.add_argument("--policy-student-replay", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--projection-epochs", type=int, default=16)
    p.add_argument("--projection-max-samples", type=int, default=4_000)
    p.add_argument("--projection-samples", type=int, default=8)

    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument("--alpha-init", default="Randn", choices=["Randn", "Major", "Uniform"])
    p.add_argument("--alpha-major", type=float, default=0.6)
    p.add_argument("--alpha-factor", type=float, default=1e-3)
    p.add_argument("--fix-alpha", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--alpha-learning-rate", type=float, default=3e-4)
    p.add_argument("--alpha-mass-learning-rate", type=float, default=None)
    p.add_argument("--alpha-warmup-steps", type=int, default=5_000)
    p.add_argument("--alpha-entropy-reg", type=float, default=0.01)
    p.add_argument("--alpha-mass-reg", type=float, default=0.05)
    p.add_argument("--use-alpha-scale", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fix-alpha-scale", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-alpha-mass", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--constrain-alpha-mass", action=argparse.BooleanOptionalAction, default=True)

    # Encoder
    p.add_argument("--encoder-from-base", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-shared", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--freeze-root-encoder", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pretrained-encoder", default=None)
    p.add_argument("--shared-dim", type=int, default=256)
    p.add_argument("--head-hidden-dim", type=int, default=256)
    p.add_argument("--distill-encoder-lr-mult", type=float, default=0.1)
    p.add_argument("--drift-reg", type=float, default=1.0)

    # Frozen tail / merge / distillation
    p.add_argument("--distillation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--collect-cosine-buffers", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--distill-extra-steps", type=int, default=4_000)
    p.add_argument("--max-distill-buffer", type=int, default=20_000)
    p.add_argument("--similarity-samples", type=int, default=1_024)
    p.add_argument("--balance-source-lineages", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--distill-max-samples", type=int, default=4_000)
    p.add_argument("--distill-epochs", type=int, default=8)
    p.add_argument("--distill-lr", type=float, default=3e-4)
    p.add_argument("--distill-batch-size", type=int, default=256)
    p.add_argument("--distill-test-frac", type=float, default=0.2)
    p.add_argument("--distill-select-best-val", action=argparse.BooleanOptionalAction, default=True)

    # Analysis
    p.add_argument("--analysis-log-every", type=int, default=5_000)
    p.add_argument("--save-analysis-snapshots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--analysis-root", default="analysis_pointmaze")
    return p


def validate_args(args) -> TaskBudget:
    if not 0 <= args.task_id < num_tasks(args.suite):
        raise ValueError(f"invalid task_id={args.task_id} for {args.suite}")
    budget = TaskBudget(args.total_timesteps, args.distill_extra_steps)
    if budget.training <= args.learning_starts:
        raise ValueError("Delta-B must exceed learning_starts")
    if args.pool_size < 2:
        raise ValueError("pool_size must be >= 2")
    if args.batch_size < 1 or args.buffer_size < args.batch_size:
        raise ValueError("buffer_size must be at least batch_size")
    if args.learning_rate <= 0 or args.alpha_learning_rate <= 0:
        raise ValueError("learning rates must be > 0")
    if not 0 < args.gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if not 0 < args.tau <= 1:
        raise ValueError("tau must be in (0, 1]")
    if args.fusion_mode == "classic_cka" and args.use_alpha_mass:
        raise ValueError("alpha_mass is only valid with fusion_mode='weight_delta'")
    if args.use_alpha_scale and args.fix_alpha_scale:
        raise ValueError("use_alpha_scale and fix_alpha_scale are mutually exclusive")
    if args.train_shared and args.freeze_root_encoder:
        raise ValueError("train_shared and freeze_root_encoder are contradictory")
    if args.composition_space == "policy" and budget.frozen_tail < 2:
        raise ValueError("policy composition needs B >= 2 for storage projection")
    if args.policy_student_replay:
        if args.composition_space != "policy":
            raise ValueError("policy_student_replay requires composition_space='policy'")
        if args.fusion_mode != "weight_delta" or not args.use_alpha_mass:
            raise ValueError("policy_student_replay requires weight_delta with alpha-mass")
    if (args.distillation or args.collect_cosine_buffers) and budget.frozen_tail < 1:
        raise ValueError("retaining reference states requires B >= 1")
    if args.eval_every < 0 or args.num_evals < 1:
        raise ValueError("eval_every must be >= 0 and num_evals >= 1")
    return budget


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluation_seed(task_id: int, episode: int) -> int:
    """Fixed per-task episode seeds, independent of the training seed.

    Continual seed 1 and scratch seed 201 must be scored on identical episode
    initializations or the forward-transfer denominator is not comparable to
    its numerator.
    """
    return 10_000 + 10_000 * int(task_id) + int(episode)


@torch.no_grad()
def evaluate(policy, args, device, global_step, writer=None, tag="charts/test"):
    env = get_task(args.task_id, args.suite)
    returns, successes = [], []
    steps = 0
    try:
        for ep in range(int(args.num_evals)):
            obs, _ = env.reset(seed=evaluation_seed(args.task_id, ep))
            total = 0.0
            success = 0.0
            while True:
                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist = policy.action_distribution(x)
                action = (
                    dist.sample()
                    if args.eval_action_mode == "stochastic"
                    else dist.deterministic_action()
                )
                obs, reward, terminated, truncated, info = env.step(
                    action.squeeze(0).cpu().numpy()
                )
                steps += 1
                total += float(reward)
                # Success latches at termination, so the episodic aggregate is
                # a max: solving fast must not score worse than solving slowly.
                if EPISODIC_SUCCESS != "max":
                    raise RuntimeError(
                        f"this evaluator assumes latched success; the environment "
                        f"declares EPISODIC_SUCCESS={EPISODIC_SUCCESS!r}"
                    )
                success = max(success, float(info.get("is_success", False)))
                if terminated or truncated:
                    break
            returns.append(total)
            successes.append(success)
    finally:
        env.close()

    result = {
        "return": float(np.mean(returns)),
        "reward": float(np.mean(returns)),
        "success": float(np.mean(successes)),
        "evaluation_interactions": int(steps),
    }
    if writer is not None:
        writer.add_scalar(f"{tag}_return", result["return"], global_step)
        writer.add_scalar(f"{tag}_success", result["success"], global_step)
    return result


# ---------------------------------------------------------------------------
# Frozen tail
# ---------------------------------------------------------------------------
def collect_frozen_tail(policy, args, device, steps, seed):
    """Collect exactly ``steps`` transitions under the frozen final policy.

    The destination arrays are preallocated: the row count is known, so there
    is no reason to build a Python list and concatenate it afterwards.
    """
    steps = int(steps)
    if steps <= 0:
        return None, 0.0
    env = get_task(args.task_id, args.suite)
    obs, _ = env.reset(seed=int(seed))
    obs_arr = np.zeros((steps, OBS_DIM), dtype=np.float32)
    act_arr = np.zeros((steps, ACT_DIM), dtype=np.float32)
    start = time.time()
    try:
        with torch.no_grad():
            for i in range(steps):
                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy.action_distribution(x).sample().squeeze(0).cpu().numpy()
                obs_arr[i] = obs
                act_arr[i] = action
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset(seed=int(seed) + i + 1)
    finally:
        env.close()

    buffer = {
        "obs": obs_arr,
        "actions": act_arr,
        "task_ids": np.full(steps, int(args.task_id), dtype=np.int32),
        "source_ids": np.full(steps, int(args.seq_idx), dtype=np.int32),
    }
    return buffer, time.time() - start


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
def build_optimizers(agent, args):
    encoder_ids = {id(p) for p in agent.fc.parameters()}
    critic_ids = {id(p) for p in agent.critic.parameters()}

    route_params = []
    if agent.alpha is not None and agent.alpha.requires_grad:
        route_params.append(agent.alpha)
    if agent.alpha_scale is not None and agent.alpha_scale.requires_grad:
        route_params.append(agent.alpha_scale)
    mass_params = []
    if agent.alpha_mass is not None and agent.alpha_mass.requires_grad:
        mass_params.append(agent.alpha_mass)
    routing_ids = {id(p) for p in route_params + mass_params}

    actor_params, encoder_params = [], []
    for p in agent.parameters():
        if not p.requires_grad or id(p) in routing_ids or id(p) in critic_ids:
            continue
        if id(p) in encoder_ids:
            encoder_params.append(p)
        else:
            actor_params.append(p)

    encoder_lr = args.learning_rate
    if args.prev_units and args.distillation and args.train_shared:
        encoder_lr *= args.distill_encoder_lr_mult

    actor_opt = torch.optim.Adam(actor_params, lr=args.learning_rate) if actor_params else None

    # In SAC the encoder is updated by the critic loss only; letting the actor
    # loss reshape the representation is a known route to feature collapse.
    critic_groups = [{"params": list(agent.critic.parameters()), "lr": args.learning_rate}]
    if encoder_params:
        critic_groups.append({"params": encoder_params, "lr": encoder_lr})
    critic_opt = torch.optim.Adam(critic_groups)

    route_opt = (
        torch.optim.Adam(route_params, lr=args.alpha_learning_rate) if route_params else None
    )
    mass_lr = (
        args.alpha_learning_rate
        if args.alpha_mass_learning_rate is None
        else args.alpha_mass_learning_rate
    )
    mass_opt = torch.optim.Adam(mass_params, lr=mass_lr) if mass_params else None

    return {
        "actor": actor_opt,
        "critic": critic_opt,
        "route": route_opt,
        "mass": mass_opt,
        "actor_params": actor_params,
        "encoder_params": encoder_params,
        "route_params": route_params,
        "mass_params": mass_params,
    }


def _rss_gib() -> float:
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except Exception:
        return float("nan")


def log_rss(writer, stage: str, step: int):
    rss = _rss_gib()
    if np.isfinite(rss):
        writer.add_scalar(f"memory/rss_gib/{stage}", rss, step)
    return rss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    args = build_parser().parse_args(argv)
    budget = validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = bool(args.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.suite}__task_{args.task_id}__{args.method}__{args.seed}"
    event_dir = pathlib.Path(args.runs_root) / args.tag / run_name
    analysis_dir = pathlib.Path(args.analysis_root) / args.tag / run_name
    writer = CsvSummaryWriter(str(event_dir))
    writer.add_text("hyperparameters", json.dumps(vars(args), default=str, sort_keys=True))
    print(
        f"[run] {run_name} | task={get_task_name(args.task_id, args.suite)} | device={device}"
    )

    env = get_task(args.task_id, args.suite)
    prev_units = [str(p) for p in (args.prev_units or [])]
    base_dir = prev_units[0] if prev_units else None
    latest_dir = prev_units[-1] if prev_units else None

    agent = CkaRlAgent(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
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
        projection_samples=args.projection_samples,
        policy_student_replay=args.policy_student_replay,
    ).to(device)

    # Historical representation stabilization when the encoder may move.
    old_fc = None
    drift_states = None
    if args.seq_idx > 0 and args.train_shared and args.distillation and args.drift_reg > 0:
        old_fc = copy.deepcopy(agent.fc).to(device).eval()
        for p in old_fc.parameters():
            p.requires_grad_(False)
        past = [
            e["buffer"]["obs"]
            for e in agent.policy_pool.pool
            if e.get("buffer") is not None and len(e["buffer"].get("obs", ())) > 0
        ]
        if past:
            drift_states = np.concatenate(past, axis=0)

    opts = build_optimizers(agent, args)

    # SAC entropy temperature.  Named ent_coef, never "alpha": in this codebase
    # alpha is the knowledge-pool mixture vector and confusing the two would be
    # a silent bug.
    target_entropy = -float(ACT_DIM)
    log_ent_coef = torch.tensor(
        float(np.log(args.entropy_coef)), device=device, requires_grad=args.autotune_entropy
    )
    ent_opt = (
        torch.optim.Adam([log_ent_coef], lr=args.learning_rate)
        if args.autotune_entropy
        else None
    )

    encoder_should_be_frozen = bool(
        not args.train_shared
        and (args.pretrained_encoder is not None or latest_dir is not None or args.freeze_root_encoder)
    )
    encoder_fingerprint = None
    if encoder_should_be_frozen:
        if any(p.requires_grad for p in agent.fc.parameters()):
            raise RuntimeError("encoder should be frozen but some parameters require grad")
        with torch.no_grad():
            encoder_fingerprint = torch.cat(
                [p.detach().reshape(-1) for p in agent.fc.parameters()]
            ).clone()

    buffer = ReplayBuffer(args.buffer_size, OBS_DIM, ACT_DIM, seed=args.seed)

    agent.set_mixture_warmup(
        mixture_warmup_active(
            0, args.learning_starts, args.alpha_warmup_steps, args.fusion_mode,
            agent.policy_pool.pool_length(),
        )
    )
    theta_task_start = effective_theta_vector(agent).detach().clone()
    if args.save_analysis_snapshots:
        save_task_snapshot(analysis_dir / "start.pt", "start", 0, args, agent)
    log_training_state(writer, 0, agent, theta_task_start)

    # Zero-shot score: what the inherited policy achieves before any update.
    zero_shot = evaluate(agent, args, device, 0, writer)
    writer.add_scalar("charts/zero_shot_return", zero_shot["return"], 0)
    writer.add_scalar("charts/zero_shot_success", zero_shot["success"], 0)
    eval_interactions = zero_shot["evaluation_interactions"]

    obs, _ = env.reset(seed=args.seed)
    episode_return, episode_len = 0.0, 0
    next_eval = args.eval_every if args.eval_every > 0 else None
    next_analysis = args.analysis_log_every if args.analysis_log_every > 0 else None
    start_time = time.time()
    last = {"critic": float("nan"), "actor": float("nan"), "ent_coef": float(np.exp(log_ent_coef.item()))}
    # Defined before the loop: the logging branch below can fire during the
    # random-action warmup, before any update has assigned it.
    drift_loss = None

    for global_step in range(1, budget.training + 1):
        warmup = mixture_warmup_active(
            global_step, args.learning_starts, args.alpha_warmup_steps,
            args.fusion_mode, agent.policy_pool.pool_length(),
        )
        agent.set_mixture_warmup(warmup)

        # ---------------- act ----------------
        if global_step <= args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                if args.policy_student_replay and not warmup:
                    # The executing behavior policy is the mixture; recording
                    # anything else would break the provenance of the data.
                    action = agent.action_distribution(x).sample()
                else:
                    action = agent.action_distribution(x).sample()
                action = action.squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        episode_len += 1
        # A time-limit truncation is not a real terminal state, so it must not
        # zero the bootstrap value.
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs

        if terminated or truncated:
            writer.add_scalar("charts/episodic_return", episode_return, global_step)
            writer.add_scalar("charts/episodic_length", episode_len, global_step)
            writer.add_scalar(
                "charts/episodic_success", float(info.get("is_success", False)), global_step
            )
            obs, _ = env.reset(seed=args.seed + global_step)
            episode_return, episode_len = 0.0, 0

        # ---------------- learn ----------------
        if global_step > args.learning_starts and global_step % args.update_every == 0:
            b_obs, b_act, b_rew, b_next, b_done = buffer.sample(args.batch_size, device)
            ent_coef = log_ent_coef.exp().detach()

            # --- critic ---
            with torch.no_grad():
                next_features = agent.encode(b_next)
                next_dist = agent._distribution_at_features(next_features)
                next_action, next_logp = next_dist.rsample_with_log_prob()
                target_q = agent.critic_target.min_q(next_features, next_action)
                backup = b_rew + args.gamma * (1.0 - b_done) * (
                    target_q - ent_coef * next_logp.unsqueeze(-1)
                )

            features = agent.encode(b_obs)
            q1, q2 = agent.critic(features, b_act)
            critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

            drift_loss = None
            if old_fc is not None and drift_states is not None:
                idx = np.random.randint(0, len(drift_states), size=min(args.batch_size, len(drift_states)))
                s_past = torch.as_tensor(drift_states[idx], dtype=torch.float32, device=device)
                with torch.no_grad():
                    phi_old = old_fc(s_past)
                drift_loss = args.drift_reg * F.mse_loss(agent.fc(s_past), phi_old)
                critic_loss = critic_loss + drift_loss

            opts["critic"].zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in opts["critic"].param_groups for p in g["params"]],
                args.max_grad_norm,
            )
            opts["critic"].step()
            last["critic"] = float(critic_loss.detach())

            # --- actor + routing ---
            if global_step % args.policy_frequency == 0:
                # Features are detached here: the encoder is the critic's to
                # shape, the actor only consumes it.
                feats = features.detach()

                if args.policy_student_replay and not warmup:
                    raw = agent.novel_policy_raw(feats)
                    mean, log_std = torch.split(raw, ACT_DIM, dim=-1)
                    novel = SquashedGaussian(mean, clamp_log_std(log_std))
                    a_new, logp = novel.rsample_with_log_prob()
                else:
                    dist = agent._distribution_at_features(feats)
                    a_new, logp = dist.rsample_with_log_prob()

                q_pi = agent.critic.min_q(feats, a_new)
                actor_loss = (ent_coef * logp - q_pi.squeeze(-1)).mean()

                alpha_entropy = None
                if warmup and args.alpha_entropy_reg > 0 and agent.alpha is not None:
                    scale = agent.alpha_scale if agent.alpha_scale is not None else 1.0
                    probs = torch.softmax(agent.alpha * scale, dim=0)
                    alpha_entropy = -(probs * torch.log(probs + 1e-8)).sum()
                    actor_loss = actor_loss - args.alpha_entropy_reg * alpha_entropy

                mass_loss = None
                if (
                    not warmup
                    and agent.alpha_mass is not None
                    and agent.alpha_mass.requires_grad
                    and args.alpha_mass_reg > 0
                ):
                    eff = agent.policy_pool.effective_alpha_mass()
                    mass_loss = args.alpha_mass_reg * (eff ** 2) * ((eff - 1.0) ** 2)
                    actor_loss = actor_loss + mass_loss.mean()

                if opts["actor"] is not None:
                    opts["actor"].zero_grad(set_to_none=True)
                if opts["route"] is not None:
                    opts["route"].zero_grad(set_to_none=True)
                if opts["mass"] is not None:
                    opts["mass"].zero_grad(set_to_none=True)
                actor_loss.backward()

                if warmup:
                    # Learn the historical mixture first; the novel residual
                    # and its mass stay put until routing has settled.
                    for p in (
                        agent.policy_pool.own_l0_weight,
                        agent.policy_pool.own_l0_bias,
                        agent.policy_pool.own_l2_weight,
                        agent.policy_pool.own_l2_bias,
                    ):
                        p.grad = None
                    if agent.alpha_mass is not None:
                        agent.alpha_mass.grad = None

                trainable = opts["actor_params"] + opts["route_params"] + opts["mass_params"]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                if opts["actor"] is not None:
                    opts["actor"].step()
                if opts["route"] is not None:
                    opts["route"].step()
                if opts["mass"] is not None:
                    opts["mass"].step()
                last["actor"] = float(actor_loss.detach())

                if args.policy_student_replay and not warmup and opts["route"] is not None:
                    # Routing is scored against the behavior actually executed,
                    # with the experts held fixed.
                    route_dist = agent.routing_distribution(feats)
                    r_action, r_logp = route_dist.rsample_with_log_prob()
                    route_q = agent.critic.min_q(feats, r_action)
                    route_loss = (ent_coef * r_logp - route_q.squeeze(-1)).mean()
                    opts["route"].zero_grad(set_to_none=True)
                    if opts["mass"] is not None:
                        opts["mass"].zero_grad(set_to_none=True)
                    route_loss.backward()
                    routing = opts["route_params"] + opts["mass_params"]
                    if routing:
                        torch.nn.utils.clip_grad_norm_(routing, args.max_grad_norm)
                    opts["route"].step()
                    if opts["mass"] is not None:
                        opts["mass"].step()

                if ent_opt is not None:
                    ent_loss = -(log_ent_coef.exp() * (logp.detach() + target_entropy)).mean()
                    ent_opt.zero_grad(set_to_none=True)
                    ent_loss.backward()
                    ent_opt.step()
                    last["ent_coef"] = float(log_ent_coef.exp().detach())

            if global_step % args.target_frequency == 0:
                with torch.no_grad():
                    for p, tp in zip(agent.critic.parameters(), agent.critic_target.parameters()):
                        tp.data.mul_(1.0 - args.tau).add_(args.tau * p.data)

        # ---------------- logging / eval ----------------
        if global_step % 1000 == 0:
            writer.add_scalar("losses/critic", last["critic"], global_step)
            writer.add_scalar("losses/actor", last["actor"], global_step)
            writer.add_scalar("losses/ent_coef", last["ent_coef"], global_step)
            writer.add_scalar(
                "charts/SPS", int(global_step / max(time.time() - start_time, 1e-9)), global_step
            )
            if drift_loss is not None:
                writer.add_scalar("losses/encoder_drift", float(drift_loss.detach()), global_step)

        if next_eval is not None and global_step >= next_eval:
            res = evaluate(agent, args, device, global_step, writer)
            eval_interactions += res["evaluation_interactions"]
            while next_eval is not None and next_eval <= global_step:
                next_eval += args.eval_every
        if next_analysis is not None and global_step >= next_analysis:
            log_training_state(writer, global_step, agent, theta_task_start)
            while next_analysis is not None and next_analysis <= global_step:
                next_analysis += args.analysis_log_every

    train_seconds = time.time() - start_time
    writer.add_scalar("timing/train_loop_seconds", train_seconds, budget.total)
    print(
        f"[run] TRAIN_LOOP_SECONDS={train_seconds:.1f} "
        f"optimization_transitions={budget.training}"
    )

    if encoder_should_be_frozen:
        with torch.no_grad():
            now = torch.cat([p.detach().reshape(-1) for p in agent.fc.parameters()])
            drift = float((now - encoder_fingerprint).abs().max())
        writer.add_scalar("analysis/encoder/max_drift", drift, budget.total)
        if drift != 0.0:
            raise RuntimeError(f"frozen encoder drifted by {drift}")

    # Release everything the merge phase does not need before it runs.
    del old_fc, drift_states, buffer
    old_fc = None
    drift_states = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log_rss(writer, "post_train_loop", budget.training)

    agent.set_mixture_warmup(
        mixture_warmup_active(
            budget.training, args.learning_starts, args.alpha_warmup_steps,
            args.fusion_mode, agent.policy_pool.pool_length(),
        )
    )
    log_training_state(writer, budget.training, agent, theta_task_start)

    # Every condition spends the same frozen B interactions; only some retain
    # them, so the interaction budget stays identical across methods.
    needs_buffer = bool(
        args.distillation or args.collect_cosine_buffers or args.composition_space == "policy"
    )
    tail_buffer, tail_seconds = collect_frozen_tail(
        agent, args, device, budget.frozen_tail, args.seed + 123_456
    )
    merge_buffer = bounded_buffer(tail_buffer, args.max_distill_buffer) if needs_buffer else None

    final_step = budget.total
    writer.add_scalar("timing/frozen_tail_seconds", tail_seconds, final_step)
    writer.add_scalar(
        "analysis/buffer/rows", 0 if merge_buffer is None else len(merge_buffer["obs"]), final_step
    )
    writer.add_scalar("budget/optimization_phase_env_steps", budget.training, final_step)
    writer.add_scalar("budget/frozen_tail_env_steps", budget.frozen_tail, final_step)
    writer.add_scalar("budget/total_learning_env_steps", budget.total, final_step)

    final_eval = evaluate(agent, args, device, final_step, writer)
    eval_interactions += final_eval["evaluation_interactions"]
    agent.set_own_buffer(merge_buffer)

    run_dir = pathlib.Path(args.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    agent.save_policy_snapshot(str(run_dir))
    with (run_dir / "interaction_budget.json").open("w") as f:
        json.dump(
            {
                "Delta": budget.total,
                "optimization_phase_steps": budget.training,
                "frozen_tail_steps": budget.frozen_tail,
                "monitor_evaluation_steps": int(eval_interactions),
                "evaluation_updates_policy": False,
            },
            f,
            indent=2,
        )

    if args.save_analysis_snapshots:
        save_task_snapshot(analysis_dir / "pre_finalize.pt", "pre_finalize", final_step, args, agent)

    log_rss(writer, "pre_finalize", final_step)
    t0 = time.time()
    if prev_units:
        agent.finalize()
    else:
        agent.set_base()
    gc.collect()
    writer.add_scalar("timing/finalize_seconds", time.time() - t0, final_step)
    writer.add_scalar("analysis/pool/final_length", agent.policy_pool.pool_length(), final_step)
    log_rss(writer, "post_finalize", final_step)

    for key, value in getattr(agent, "last_projection_metrics", {}).items():
        if value is not None:
            writer.add_scalar(key, float(value), final_step)
    with (run_dir / "projection_metrics.json").open("w") as f:
        json.dump(getattr(agent, "last_projection_metrics", {}), f, indent=2)

    info = agent.get_merge_info()
    if info:
        for key in (
            "idx1", "idx2", "similarity_states", "cosine_similarity",
            "pairwise_cosine_min", "pairwise_cosine_mean", "pairwise_cosine_max",
            "symmetric_kl", "pairwise_kl_min", "pairwise_kl_mean", "pairwise_kl_max",
            "selected_state_kl_p95", "selected_state_kl_max",
            "pool_size_before", "pool_size_after",
        ):
            value = info.get(key)
            if value is not None and np.isscalar(value):
                writer.add_scalar(f"analysis/merge/{key}", float(value), final_step)
        writer.add_scalar(
            "analysis/merge/used_distillation", float(bool(info.get("used_distillation", False))), final_step
        )
        writer.add_text("analysis/merge/info", json.dumps(info, sort_keys=True, default=str), final_step)
        with (run_dir / "merge_info.json").open("w") as f:
            json.dump(info, f, indent=2, default=str)

    for name, value in agent.get_distill_metrics().items():
        if value is not None:
            writer.add_scalar(f"distillation/{name}", float(value), final_step)

    writer.add_scalar("charts/final_return", final_eval["return"], final_step)
    writer.add_scalar("charts/final_success", final_eval["success"], final_step)

    agent.save(str(run_dir))
    if args.save_analysis_snapshots:
        save_task_snapshot(
            analysis_dir / "post_finalize.pt", "post_finalize", final_step, args, agent,
            include_effective=False,
        )

    config = dict(vars(args))
    config["num_tasks"] = num_tasks(args.suite)
    manifest = write_manifest(run_dir, config, parent_dirs=prev_units,
                              pretrained_encoder=args.pretrained_encoder)
    print(f"[run] saved {run_dir} | signature={manifest['run_signature']}")

    env.close()
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
