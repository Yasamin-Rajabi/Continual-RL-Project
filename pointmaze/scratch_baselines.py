"""From-scratch single-task references, the denominator of forward transfer.

Seeds
-----
``DEFAULT_SCRATCH_SEEDS = (201,)`` deliberately does not overlap the continual
seeds (1, 2, 3, ...).  A forward-transfer denominator that shares a seed with
its numerator is not an independent reference: the two runs would share
environment episode draws and the ratio would be biased toward whichever seed
happened to be easy.  The evaluation episode seeds are fixed per task by
``evaluation_seed`` and do not depend on the training seed, so the continual
and scratch runs are still scored on exactly the same episodes.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from baselines.agents import FtNAgent
from baselines.common import soft_update
from csv_summary_writer import CsvSummaryWriter
from pointmaze_env import ACT_DIM, OBS_DIM
from replay_buffer import ReplayBuffer
from tasks import DEFAULT_SUITE, SUITES, get_task, get_task_name, num_tasks

DEFAULT_SCRATCH_SEEDS = (201,)


def scratch_dir(root, suite: str, task_id: int, total_timesteps: int, seed: int) -> pathlib.Path:
    return (
        pathlib.Path(root)
        / "scratch"
        / str(suite)
        / f"task_{int(task_id)}"
        / f"steps_{int(total_timesteps)}"
        / f"seed_{int(seed)}"
    )


def train_scratch(args, task_id: int, seed: int, device) -> dict:
    """Plain SAC on one task, no continual machinery at all."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = get_task(task_id, args.suite)
    # num_tasks=1 makes FtNAgent a plain single-head SAC actor.
    agent = FtNAgent(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        num_tasks=1,
        shared_dim=args.shared_dim,
        hidden_dim=args.head_hidden_dim,
    ).to(device)
    agent.set_task(0)

    buffer = ReplayBuffer(args.buffer_size, OBS_DIM, ACT_DIM, seed=seed)
    actor_params = agent.actor_parameters()
    actor_opt = torch.optim.Adam(actor_params, lr=args.learning_rate)
    critic_opt = torch.optim.Adam(
        [
            {"params": list(agent.critic.parameters()), "lr": args.learning_rate},
            {"params": agent.encoder_parameters(), "lr": args.learning_rate},
        ]
    )
    target_entropy = -float(ACT_DIM)
    log_ent_coef = torch.tensor(float(np.log(0.2)), device=device, requires_grad=True)
    ent_opt = torch.optim.Adam([log_ent_coef], lr=args.learning_rate)

    run_dir = scratch_dir(args.save_root, args.suite, task_id, args.total_timesteps, seed)
    writer = CsvSummaryWriter(str(pathlib.Path(args.runs_root) / "scratch" / run_dir.name))

    obs, _ = env.reset(seed=seed)
    start = time.time()
    for step in range(1, args.total_timesteps + 1):
        if step <= args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action = agent.action_distribution(x).sample().squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + step)

        if step > args.learning_starts:
            b_obs, b_act, b_rew, b_next, b_done = buffer.sample(args.batch_size, device)
            ent_coef = log_ent_coef.exp().detach()
            with torch.no_grad():
                nf = agent.encode(b_next)
                nd = agent.distribution_from_features(nf)
                na, nlp = nd.rsample_with_log_prob()
                backup = b_rew + args.gamma * (1.0 - b_done) * (
                    agent.critic_target.min_q(nf, na) - ent_coef * nlp.unsqueeze(-1)
                )
            feats = agent.encode(b_obs)
            q1, q2 = agent.critic(feats, b_act)
            closs = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
            critic_opt.zero_grad(set_to_none=True)
            closs.backward()
            critic_opt.step()

            if step % args.policy_frequency == 0:
                fd = feats.detach()
                dist = agent.distribution_from_features(fd)
                a_new, logp = dist.rsample_with_log_prob()
                aloss = (ent_coef * logp - agent.critic.min_q(fd, a_new).squeeze(-1)).mean()
                actor_opt.zero_grad(set_to_none=True)
                aloss.backward()
                actor_opt.step()
                eloss = -(log_ent_coef.exp() * (logp.detach() + target_entropy)).mean()
                ent_opt.zero_grad(set_to_none=True)
                eloss.backward()
                ent_opt.step()

            soft_update(agent.critic, agent.critic_target, args.tau)

        if step % 5000 == 0:
            writer.add_scalar("charts/SPS", int(step / max(time.time() - start, 1e-9)), step)

    run_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(run_dir))
    with (run_dir / "manifest.json").open("w") as f:
        json.dump(
            {
                "run_signature": f"scratch-{args.suite}-{task_id}-{args.total_timesteps}-{seed}",
                "identity": {
                    "method": "scratch",
                    "suite": args.suite,
                    "task_id": int(task_id),
                    "seed": int(seed),
                    "total_timesteps": int(args.total_timesteps),
                },
            },
            f,
            indent=2,
        )
    env.close()
    writer.close()
    return {"run_dir": str(run_dir), "task_id": int(task_id), "seed": int(seed)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="From-scratch PointMaze references")
    p.add_argument("--suite", default=DEFAULT_SUITE, choices=sorted(SUITES))
    p.add_argument("--tasks", nargs="*", type=int, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SCRATCH_SEEDS))
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--learning-starts", type=int, default=2_000)
    p.add_argument("--policy-frequency", type=int, default=2)
    p.add_argument("--shared-dim", type=int, default=256)
    p.add_argument("--head-hidden-dim", type=int, default=256)
    p.add_argument("--num-evals", type=int, default=10)
    p.add_argument("--eval-action-mode", default="deterministic")
    p.add_argument("--save-root", default="agents_pointmaze")
    p.add_argument("--runs-root", default="runs_pointmaze")
    p.add_argument("--out", default="scratch_reference.json")
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    tasks = args.tasks if args.tasks else list(range(num_tasks(args.suite)))

    from checkpoint_evaluation import evaluate_checkpoint

    per_task = {}
    for task_id in tasks:
        scores = []
        for seed in args.seeds:
            run_dir = scratch_dir(
                args.save_root, args.suite, task_id, args.total_timesteps, seed
            )
            if not (run_dir / "agent.pt").is_file():
                print(f"[scratch] training task {task_id} ({get_task_name(task_id, args.suite)}) seed {seed}")
                train_scratch(args, task_id, seed, device)
            result = evaluate_checkpoint(
                run_dir, "FT-N", args.suite, task_id,
                episodes=args.num_evals, seed=seed, device=device,
                adapt_steps=0, action_mode=args.eval_action_mode,
            )
            scores.append(result["return"])
            print(f"[scratch] task {task_id} seed {seed}: return={result['return']:.2f}")
        per_task[str(int(task_id))] = float(np.mean(scores))

    payload = {
        "suite": args.suite,
        "seed": list(args.seeds),
        "total_timesteps": int(args.total_timesteps),
        "per_task": per_task,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[scratch] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
