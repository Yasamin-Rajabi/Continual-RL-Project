"""SAC trainer for the continual PointMaze baselines.

Deliberately shares the evaluation function, the episode seeds, the interaction
budget and the frozen-tail accounting with ``run_sac_continual.py``.  If the
baselines had their own copies of those, a difference in any one of them would
show up as a difference in method quality.

Run one task of one chain:

    python -m baselines.run_baseline --method PackNet --task-id 3 --seq-idx 3 \
        --prev-units <dir0> <dir1> <dir2> --save-dir <dir3>
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Allow both "python -m baselines.run_baseline" and "python baselines/run_baseline.py".
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from csv_summary_writer import CsvSummaryWriter  # noqa: E402
from experiment_identity import write_manifest  # noqa: E402
from pointmaze_env import ACT_DIM, OBS_DIM  # noqa: E402
from replay_buffer import ReplayBuffer  # noqa: E402
from run_sac_continual import evaluate  # noqa: E402
from shared_arch import TwinCritic  # noqa: E402
from tasks import DEFAULT_SUITE, SUITES, get_task, get_task_name, num_tasks  # noqa: E402
from training_protocol import TaskBudget  # noqa: E402

from . import TASK_CONDITIONED_METHODS, canonical_method  # noqa: E402
from .agents import (  # noqa: E402
    CbpNetAgent,
    CompoNetAgent,
    CReLUsAgent,
    FtNAgent,
    MaskNetAgent,
    PackNetAgent,
    ProgNetAgent,
)
from .common import soft_update  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Continual PointMaze SAC baselines",
    )
    p.add_argument("--method", required=True)
    p.add_argument("--suite", default=DEFAULT_SUITE, choices=sorted(SUITES))
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--seq-idx", type=int, default=0)
    p.add_argument("--prev-units", nargs="*", default=[])
    p.add_argument("--save-dir", default="agents_pointmaze/baseline")
    p.add_argument("--runs-root", default="runs_pointmaze")
    p.add_argument("--tag", default="debug")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--distill-extra-steps", type=int, default=4_000)
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
    p.add_argument("--shared-dim", type=int, default=256)
    p.add_argument("--head-hidden-dim", type=int, default=256)

    p.add_argument(
        "--packnet-retrain-frac", type=float, default=0.3,
        help="Fraction of the optimization budget spent retraining inside the "
             "pruned allocation.",
    )
    p.add_argument(
        "--packnet-width", type=int, default=384,
        help="Network width for PackNet only. PackNet must fit every task into "
             "ONE network, while ProgNet grows a column per task; at equal "
             "width PackNet would carry roughly a tenth of ProgNet's "
             "parameters over a ten-task chain. Widening it is the "
             "parameter-matched comparison. Set equal to --shared-dim to "
             "disable.",
    )
    p.add_argument(
        "--packnet-capacity-mode", default="equal_share",
        choices=["equal_share", "geometric"],
        help="equal_share: every task reserves keep_frac/N of the whole "
             "network (most generous to the late tasks). geometric: textbook "
             "PackNet, keep a fraction of whatever is still free -- at N=10 "
             "this starves the tail badly.",
    )
    p.add_argument(
        "--packnet-keep-frac", type=float, default=1.0,
        help="With equal_share, the total fraction of the network handed out "
             "across all tasks (1.0 = fully used). With geometric, the "
             "fraction of the free pool each task keeps.",
    )
    p.add_argument("--cbp-replacement-rate", type=float, default=1e-4)
    p.add_argument("--cbp-maturity-threshold", type=int, default=100)
    p.add_argument("--analysis-root", default="analysis_pointmaze")
    p.add_argument("--save-analysis-snapshots", action=argparse.BooleanOptionalAction, default=False)
    return p


def _load_agent(path, map_location=None):
    agent_path = os.path.join(os.fspath(path), "agent.pt")
    if not os.path.isfile(agent_path):
        raise FileNotFoundError(f"missing baseline checkpoint: {agent_path}")
    try:
        return torch.load(agent_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(agent_path, map_location=map_location)


def effective_dims(method: str, args):
    """Network width for this method.

    Only PackNet differs, and deliberately: it is the one method that must
    pack every task into a single fixed network, so comparing it at the same
    width as methods that grow with the task count understates it.
    """
    if method == "PackNet":
        return int(args.packnet_width), int(args.packnet_width)
    return int(args.shared_dim), int(args.head_hidden_dim)


def build_agent(method: str, args, device, n_tasks: int):
    """Construct the agent for this task, continuing from prior checkpoints."""
    prev = [str(p) for p in (args.prev_units or [])]
    latest = prev[-1] if prev else None
    shared_dim, hidden_dim = effective_dims(method, args)
    common = dict(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        shared_dim=shared_dim,
        hidden_dim=hidden_dim,
    )

    if method == "FT-N":
        if latest is None:
            agent = FtNAgent(num_tasks=n_tasks, **common)
        else:
            agent = _load_agent(latest, map_location="cpu")
            # The critic is task-local: SAC's Q function is tied to this task's
            # reward, so carrying it over would poison the first updates.
            agent.critic = TwinCritic(shared_dim, ACT_DIM, hidden_dim)
            agent.critic_target = TwinCritic(shared_dim, ACT_DIM, hidden_dim)
            agent.critic_target.load_state_dict(agent.critic.state_dict())
            for p in agent.critic_target.parameters():
                p.requires_grad_(False)
        agent.set_task(args.seq_idx)
        return agent

    if method == "CReLUs":
        if latest is None:
            return CReLUsAgent(**common)
        agent = _load_agent(latest, map_location="cpu")
        agent.critic = TwinCritic(shared_dim, ACT_DIM, hidden_dim)
        agent.critic_target = TwinCritic(shared_dim, ACT_DIM, hidden_dim)
        agent.critic_target.load_state_dict(agent.critic.state_dict())
        for p in agent.critic_target.parameters():
            p.requires_grad_(False)
        return agent

    if method == "CbpNet":
        if latest is None:
            return CbpNetAgent(
                replacement_rate=args.cbp_replacement_rate,
                maturity_threshold=args.cbp_maturity_threshold,
                **common,
            )
        agent = _load_agent(latest, map_location="cpu")
        agent.critic = TwinCritic(shared_dim, ACT_DIM, hidden_dim)
        agent.critic_target = TwinCritic(shared_dim, ACT_DIM, hidden_dim)
        agent.critic_target.load_state_dict(agent.critic.state_dict())
        for p in agent.critic_target.parameters():
            p.requires_grad_(False)
        return agent

    if method == "MaskNet":
        if latest is None:
            agent = MaskNetAgent(num_tasks=n_tasks, **common)
        else:
            agent = _load_agent(latest, map_location="cpu")
            agent.critic = TwinCritic(shared_dim, ACT_DIM, hidden_dim)
            agent.critic_target = TwinCritic(shared_dim, ACT_DIM, hidden_dim)
            agent.critic_target.load_state_dict(agent.critic.state_dict())
            for p in agent.critic_target.parameters():
                p.requires_grad_(False)
        agent.set_num_tasks_learned(args.seq_idx)
        agent.set_task(args.seq_idx, new_task=True)
        return agent

    if method == "PackNet":
        if latest is None:
            return PackNetAgent(
                task_slot=args.seq_idx,
                total_tasks=n_tasks,
                capacity_mode=args.packnet_capacity_mode,
                keep_frac=args.packnet_keep_frac,
                **common,
            )
        agent = _load_agent(latest, map_location="cpu")
        agent.task_slot = int(args.seq_idx)
        agent.retrain_mode = False
        # Re-apply the capacity rule from the CLI so a resumed chain cannot
        # silently keep an older rule baked into the pickle.
        agent.capacity_mode = args.packnet_capacity_mode
        agent.keep_frac = float(args.packnet_keep_frac)
        agent.actor = type(agent.actor)(shared_dim, hidden_dim, ACT_DIM)
        agent.critic = TwinCritic(shared_dim, ACT_DIM, hidden_dim)
        agent.critic_target = TwinCritic(shared_dim, ACT_DIM, hidden_dim)
        agent.critic_target.load_state_dict(agent.critic.state_dict())
        for p in agent.critic_target.parameters():
            p.requires_grad_(False)
        # Masks now cover a freshly built actor; rebuild the mask list so the
        # shapes line up, preserving the encoder's existing allocations.
        agent.masks = [
            m if m.shape == p.shape else torch.zeros_like(p, dtype=torch.long)
            for m, p in zip(agent.masks, agent._maskable_parameters())
        ]
        for name, param in agent._named_maskable():
            if name.endswith("bias"):
                param.requires_grad = False
        return agent

    if method == "ProgNet":
        columns = []
        for path in prev:
            previous = _load_agent(path, map_location="cpu")
            columns.extend(list(getattr(previous, "previous_columns", [])))
            columns.append(previous.current_column())
        # Keep exactly one column per completed task.
        columns = columns[-len(prev):] if prev else []
        return ProgNetAgent(previous_columns=columns, **common)

    if method == "CompoNet":
        units = []
        for path in prev:
            previous = _load_agent(path, map_location="cpu")
            units.append(previous.as_previous_unit())
        return CompoNetAgent(previous_units=units, **common)

    raise ValueError(f"run_baseline does not implement method {method!r}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    method = canonical_method(args.method)
    if method in ("CKA-RL", "Ours", "Ours-parameter"):
        raise SystemExit(
            f"{method} is trained by run_sac_continual.py, not by run_baseline.py"
        )

    budget = TaskBudget(args.total_timesteps, args.distill_extra_steps)
    n_tasks = num_tasks(args.suite)
    if not 0 <= args.task_id < n_tasks:
        raise SystemExit(f"invalid task_id={args.task_id}")
    if budget.training <= args.learning_starts:
        raise SystemExit("Delta-B must exceed learning_starts")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = bool(args.torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    run_name = f"{args.suite}__task_{args.task_id}__{method}__{args.seed}"
    writer = CsvSummaryWriter(str(pathlib.Path(args.runs_root) / args.tag / run_name))
    writer.add_text("hyperparameters", json.dumps(vars(args), default=str, sort_keys=True))
    print(f"[run] {run_name} | task={get_task_name(args.task_id, args.suite)} | device={device}")

    agent = build_agent(method, args, device, n_tasks).to(device)
    if method in TASK_CONDITIONED_METHODS:
        print(f"[run] {method} receives the task index ({args.seq_idx}) by construction")
    if method == "PackNet":
        # Print the allocation the run will actually get, so a starved
        # configuration is visible at task 0 rather than inferred from a bad
        # score at task 10.
        report = agent.capacity_report()
        writer.add_text("packnet/capacity", json.dumps(report), 0)
        for key in ("total_maskable_weights", "last_task_weights"):
            writer.add_scalar(f"packnet/{key}", report[key], 0)
        writer.add_scalar("packnet/last_task_fraction", report["last_task_fraction"], 0)
        print(
            f"[run] PackNet width={agent.shared_dim} "
            f"mode={report['capacity_mode']} keep_frac={report['keep_frac']}"
        )
        print(
            f"[run] PackNet capacity: {report['total_maskable_weights']:,} maskable weights, "
            f"per task {report['per_task_weights']}"
        )
        if report["last_task_weights"] < 1000:
            print(
                f"[run] WARNING: the last task would receive only "
                f"{report['last_task_weights']:,} weights "
                f"({report['last_task_fraction']:.2%}); this configuration "
                "starves the tail of the chain."
            )

    env = get_task(args.task_id, args.suite)
    buffer = ReplayBuffer(args.buffer_size, OBS_DIM, ACT_DIM, seed=args.seed)

    actor_params = agent.actor_parameters()
    encoder_params = agent.encoder_parameters()
    actor_opt = torch.optim.Adam(actor_params, lr=args.learning_rate) if actor_params else None
    critic_groups = [{"params": list(agent.critic.parameters()), "lr": args.learning_rate}]
    if encoder_params:
        critic_groups.append({"params": encoder_params, "lr": args.learning_rate})
    critic_opt = torch.optim.Adam(critic_groups)

    if method == "CbpNet" and actor_opt is not None:
        agent.attach_gnt(actor_opt, device)

    target_entropy = -float(ACT_DIM)
    log_ent_coef = torch.tensor(
        float(np.log(args.entropy_coef)), device=device, requires_grad=args.autotune_entropy
    )
    ent_opt = (
        torch.optim.Adam([log_ent_coef], lr=args.learning_rate) if args.autotune_entropy else None
    )

    retrain_at = (
        int(budget.training * (1.0 - args.packnet_retrain_frac))
        if method == "PackNet"
        else None
    )

    zero_shot = evaluate(agent, args, device, 0, writer)
    writer.add_scalar("charts/zero_shot_return", zero_shot["return"], 0)
    writer.add_scalar("charts/zero_shot_success", zero_shot["success"], 0)
    eval_interactions = zero_shot["evaluation_interactions"]

    obs, _ = env.reset(seed=args.seed)
    episode_return, episode_len = 0.0, 0
    next_eval = args.eval_every if args.eval_every > 0 else None
    start_time = time.time()
    last = {"critic": float("nan"), "actor": float("nan")}

    for global_step in range(1, budget.training + 1):
        if retrain_at is not None and global_step == retrain_at:
            agent.start_retraining()
            print(f"[run] PackNet retraining phase begins at step {global_step}")

        if global_step <= args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = agent.action_distribution(x).sample().squeeze(0).cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        episode_len += 1
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

        if global_step > args.learning_starts and global_step % args.update_every == 0:
            b_obs, b_act, b_rew, b_next, b_done = buffer.sample(args.batch_size, device)
            ent_coef = log_ent_coef.exp().detach()

            with torch.no_grad():
                next_features = agent.encode(b_next)
                next_dist = agent.distribution_from_features(next_features) \
                    if method != "ProgNet" else agent.action_distribution(b_next)
                next_action, next_logp = next_dist.rsample_with_log_prob()
                target_q = agent.critic_target.min_q(next_features, next_action)
                backup = b_rew + args.gamma * (1.0 - b_done) * (
                    target_q - ent_coef * next_logp.unsqueeze(-1)
                )

            if method == "ProgNet":
                # ProgNet has no separate encoder: the column produces both the
                # features and the action output.  Its features are detached
                # here so the column is shaped by the actor loss only, matching
                # how every other method here keeps the actor from reshaping
                # the representation the critic sits on.
                _, features = agent.forward_actor_and_features(b_obs)
                features = features.detach()
            else:
                features = agent.encode(b_obs)
            q1, q2 = agent.critic(features, b_act)
            critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            agent.before_update()
            torch.nn.utils.clip_grad_norm_(
                [p for g in critic_opt.param_groups for p in g["params"]], args.max_grad_norm
            )
            critic_opt.step()
            last["critic"] = float(critic_loss.detach())

            if global_step % args.policy_frequency == 0 and actor_opt is not None:
                if method == "ProgNet":
                    raw, feats = agent.forward_actor_and_features(b_obs)
                    from .common import raw_to_distribution

                    dist = raw_to_distribution(raw, ACT_DIM)
                    feats = feats.detach()
                else:
                    feats = features.detach()
                    dist = agent.distribution_from_features(feats)
                a_new, logp = dist.rsample_with_log_prob()
                q_pi = agent.critic.min_q(feats, a_new)
                actor_loss = (ent_coef * logp - q_pi.squeeze(-1)).mean()

                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                agent.before_update()
                torch.nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                actor_opt.step()
                agent.after_update()
                last["actor"] = float(actor_loss.detach())

                if ent_opt is not None:
                    ent_loss = -(log_ent_coef.exp() * (logp.detach() + target_entropy)).mean()
                    ent_opt.zero_grad(set_to_none=True)
                    ent_loss.backward()
                    ent_opt.step()

            if global_step % args.target_frequency == 0:
                soft_update(agent.critic, agent.critic_target, args.tau)

        if global_step % 1000 == 0:
            writer.add_scalar("losses/critic", last["critic"], global_step)
            writer.add_scalar("losses/actor", last["actor"], global_step)
            writer.add_scalar(
                "charts/SPS", int(global_step / max(time.time() - start_time, 1e-9)), global_step
            )
        if next_eval is not None and global_step >= next_eval:
            res = evaluate(agent, args, device, global_step, writer)
            eval_interactions += res["evaluation_interactions"]
            while next_eval is not None and next_eval <= global_step:
                next_eval += args.eval_every

    train_seconds = time.time() - start_time
    writer.add_scalar("timing/train_loop_seconds", train_seconds, budget.total)
    print(f"[run] TRAIN_LOOP_SECONDS={train_seconds:.1f}")

    del buffer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Every method spends the same frozen tail B, whether or not it keeps the
    # states, so the total environment interaction is identical across methods.
    tail_env = get_task(args.task_id, args.suite)
    tail_obs, _ = tail_env.reset(seed=args.seed + 123_456)
    with torch.no_grad():
        for i in range(budget.frozen_tail):
            x = torch.as_tensor(tail_obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = agent.action_distribution(x).sample().squeeze(0).cpu().numpy()
            tail_obs, _, term, trunc, _ = tail_env.step(a)
            if term or trunc:
                tail_obs, _ = tail_env.reset(seed=args.seed + 123_456 + i + 1)
    tail_env.close()

    final_step = budget.total
    final_eval = evaluate(agent, args, device, final_step, writer)
    eval_interactions += final_eval["evaluation_interactions"]
    writer.add_scalar("charts/final_return", final_eval["return"], final_step)
    writer.add_scalar("charts/final_success", final_eval["success"], final_step)
    writer.add_scalar("budget/optimization_phase_env_steps", budget.training, final_step)
    writer.add_scalar("budget/frozen_tail_env_steps", budget.frozen_tail, final_step)
    writer.add_scalar("budget/total_learning_env_steps", budget.total, final_step)

    agent.on_task_end()

    run_dir = pathlib.Path(args.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(run_dir))
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

    config = dict(vars(args))
    config["method"] = method
    config["num_tasks"] = n_tasks
    manifest = write_manifest(run_dir, config, parent_dirs=args.prev_units)
    print(f"[run] saved {run_dir} | signature={manifest['run_signature']}")

    env.close()
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
