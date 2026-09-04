"""Optional TD-JEPA pretraining for the Walker2D shared encoder.

The default Walker2D benchmark does *not* require this file; it is kept because
the mature HalfCheetah directory contains the same ablation.  Pretraining uses
dynamics settings that are deliberately different from the benchmark tasks.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from shared_arch import shared
from td_jepa import (
    LatentPredictor, RewardHead, TDJepaConfig, clone_target, ema_update,
    feature_diagnostics, orthonormality_loss, td_jepa_loss,
)
from tasks import Walker2DTask

TASK_DIM = 6

# Held-out-from-benchmark dynamics. None is exactly one of TASK_SUITES.
PRETRAIN_TASKS = (
    Walker2DTask(1.5, 1.15, 0.95, 0.90, 1.10, 0.95, "pretrain-A"),
    Walker2DTask(1.5, 0.95, 1.15, 1.10, 0.90, 1.05, "pretrain-B"),
    Walker2DTask(1.5, 1.10, 1.10, 0.85, 1.20, 1.00, "pretrain-C"),
)
HELDOUT_TASKS = (
    Walker2DTask(1.5, 1.05, 0.95, 0.88, 1.15, 0.98, "heldout-A"),
    Walker2DTask(1.5, 0.95, 1.05, 1.12, 0.95, 1.03, "heldout-B"),
)


def make_pretrain_env(task: Walker2DTask):
    import gymnasium as gym
    from walker2d_envs import Walker2dDynamicsEnv, TaskConditionedObservationWrapper

    env = Walker2dDynamicsEnv(
        target_velocity=task.target_velocity,
        right_mass_scale=task.right_mass_scale,
        left_mass_scale=task.left_mass_scale,
        foot_friction_scale=task.foot_friction_scale,
        joint_damping_scale=task.joint_damping_scale,
        actuator_strength_scale=task.actuator_strength_scale,
    )
    return gym.wrappers.TimeLimit(TaskConditionedObservationWrapper(env, task), max_episode_steps=1000)


def collect_transitions(
    task: Walker2DTask,
    steps: int,
    seed: int,
    ou_theta: float = 0.15,
    ou_sigma: float = 0.35,
    uniform_frac: float = 0.35,
) -> Dict[str, np.ndarray]:
    env = make_pretrain_env(task)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    act_dim = int(np.prod(env.action_space.shape))
    low, high = env.action_space.low, env.action_space.high
    ou = np.zeros(act_dim, dtype=np.float64)

    def sample_action():
        nonlocal ou
        if rng.random() < uniform_frac:
            ou = np.zeros(act_dim, dtype=np.float64)
            return rng.uniform(low, high)
        ou = ou - ou_theta * ou + ou_sigma * rng.standard_normal(act_dim)
        return np.clip(ou, low, high)

    S, A, R, S2, A2 = [], [], [], [], []
    action = sample_action()
    for _ in range(steps):
        next_obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
        done = terminated or truncated
        next_action = sample_action()
        if not done:
            S.append(obs); A.append(action); R.append(reward); S2.append(next_obs); A2.append(next_action)
        if done:
            obs, _ = env.reset()
            ou = np.zeros(act_dim, dtype=np.float64)
            action = sample_action()
        else:
            obs, action = next_obs, next_action
    env.close()
    return {
        "obs": np.asarray(S, dtype=np.float32),
        "act": np.asarray(A, dtype=np.float32),
        "rew": np.asarray(R, dtype=np.float32),
        "next_obs": np.asarray(S2, dtype=np.float32),
        "next_act": np.asarray(A2, dtype=np.float32),
    }


def concat_datasets(parts: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def to_tensors(data: Dict[str, np.ndarray], device, rew_mean: float, rew_std: float):
    obs = torch.as_tensor(data["obs"], device=device)
    return {
        "obs": obs,
        "act": torch.as_tensor(data["act"], device=device),
        "next_obs": torch.as_tensor(data["next_obs"], device=device),
        "next_act": torch.as_tensor(data["next_act"], device=device),
        "rew": (torch.as_tensor(data["rew"], device=device) - rew_mean) / rew_std,
        "task": obs[:, -TASK_DIM:].contiguous(),
    }


@torch.no_grad()
def evaluate(encoder, predictor, enc_t, pred_t, T, cfg, batch=4096):
    encoder.eval(); predictor.eval()
    total_sq, total_elements = 0.0, 0
    for start in range(0, T["obs"].shape[0], batch):
        sl = slice(start, start + batch)
        task = T["task"][sl] if cfg.task_conditioned else None
        z = encoder(T["obs"][sl])
        pred = predictor(z, T["act"][sl], task)
        z_next = enc_t(T["next_obs"][sl])
        tgt = z_next + cfg.gamma * pred_t(z_next, T["next_act"][sl], task)
        total_sq += float(F.mse_loss(pred, tgt, reduction="sum"))
        total_elements += int(pred.numel())
    encoder.train(); predictor.train()
    return 0.5 * total_sq / max(total_elements, 1)


def train(data, heldout, cfg, epochs, batch_size, device, linear_out, seed):
    obs_dim, act_dim = data["obs"].shape[1], data["act"].shape[1]
    task_dim = TASK_DIM if cfg.task_conditioned else 0
    rew_mean = float(data["rew"].mean())
    rew_std = float(data["rew"].std() + 1e-6)
    T = to_tensors(data, device, rew_mean, rew_std)
    H = to_tensors(heldout, device, rew_mean, rew_std)

    torch.manual_seed(seed)
    encoder = shared(input_dim=obs_dim, linear_out=linear_out).to(device)
    predictor = LatentPredictor(256, act_dim, task_dim, cfg.predictor_hidden, cfg.predictor_layers).to(device)
    reward_head = RewardHead(256, act_dim).to(device) if cfg.c_rew > 0 else None
    enc_t, pred_t = clone_target(encoder), clone_target(predictor)
    groups = [
        {"params": list(encoder.parameters()), "lr": cfg.lr},
        {"params": list(predictor.parameters()), "lr": cfg.lr * cfg.predictor_lr_mult},
    ]
    if reward_head is not None:
        groups.append({"params": list(reward_head.parameters()), "lr": cfg.lr * cfg.predictor_lr_mult})
    opt = torch.optim.Adam(groups)
    n, history = T["obs"].shape[0], []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        acc = {"td": 0.0, "reg": 0.0, "rew": 0.0}; nb = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if idx.numel() < 4: continue
            task = T["task"][idx] if cfg.task_conditioned else None
            td, _ = td_jepa_loss(
                encoder, predictor, enc_t, pred_t,
                T["obs"][idx], T["act"][idx], T["next_obs"][idx], T["next_act"][idx],
                task, cfg.gamma,
            )
            z = encoder(T["obs"][idx])
            reg = orthonormality_loss(z)
            loss = td + cfg.lam_reg * reg
            rew_l = torch.zeros((), device=device)
            if reward_head is not None:
                rew_l = F.mse_loss(reward_head(z, T["act"][idx]), T["rew"][idx])
                loss = loss + cfg.c_rew * rew_l
            opt.zero_grad(set_to_none=True); loss.backward()
            params = list(encoder.parameters()) + list(predictor.parameters())
            if reward_head is not None: params += list(reward_head.parameters())
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step(); ema_update(encoder, enc_t, cfg.tau); ema_update(predictor, pred_t, cfg.tau)
            acc["td"] += float(td.detach()); acc["reg"] += float(reg.detach()); acc["rew"] += float(rew_l.detach()); nb += 1
        with torch.no_grad(): probe = encoder(T["obs"][:4096])
        entry = {
            "epoch": epoch + 1,
            "train_td": acc["td"] / max(nb, 1),
            "train_reg": acc["reg"] / max(nb, 1),
            "train_rew": acc["rew"] / max(nb, 1),
            "heldout_td": evaluate(encoder, predictor, enc_t, pred_t, H, cfg),
        }
        entry.update(feature_diagnostics(probe)); history.append(entry); print(json.dumps(entry))
    return encoder, predictor, history, {"rew_mean": rew_mean, "rew_std": rew_std}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--steps-per-task", type=int, default=100_000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--predictor-lr-mult", type=float, default=3.0)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--lam-reg", type=float, default=1.0)
    p.add_argument("--c-rew", type=float, default=0.0)
    p.add_argument("--predictor-hidden", type=int, default=1024)
    p.add_argument("--predictor-layers", type=int, default=3)
    p.add_argument("--no-task-conditioned", action="store_true")
    p.add_argument("--relu-out", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="pretrained_encoders/walker2d_tdjepa")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args()
    if a.smoke_test:
        a.steps_per_task = 2_000; a.epochs = 2; a.predictor_hidden = 128
    return a


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    cfg = TDJepaConfig(
        gamma=args.gamma, tau=args.tau, lam_reg=args.lam_reg, c_rew=args.c_rew,
        lr=args.lr, predictor_lr_mult=args.predictor_lr_mult,
        predictor_hidden=args.predictor_hidden, predictor_layers=args.predictor_layers,
        task_conditioned=not args.no_task_conditioned,
    )
    t0 = time.time(); train_specs = PRETRAIN_TASKS[:1] if args.smoke_test else PRETRAIN_TASKS
    held_specs = HELDOUT_TASKS[:1] if args.smoke_test else HELDOUT_TASKS
    parts = []
    for i, task in enumerate(train_specs):
        print(f"collect train {task.label('pretrain')} ({args.steps_per_task} steps)")
        parts.append(collect_transitions(task, args.steps_per_task, args.seed + 1000 * i))
    data = concat_datasets(parts)
    held_parts = []
    for i, task in enumerate(held_specs):
        n = max(args.steps_per_task // 5, 1000)
        print(f"collect held-out {task.label('heldout')} ({n} steps)")
        held_parts.append(collect_transitions(task, n, args.seed + 7000 + 1000 * i))
    heldout = concat_datasets(held_parts)
    collect_seconds = time.time() - t0
    encoder, predictor, history, rew_stats = train(
        data, heldout, cfg, args.epochs, args.batch_size, device,
        linear_out=not args.relu_out, seed=args.seed,
    )
    torch.save(encoder.to("cpu"), os.path.join(args.out, "fc.pt"))
    torch.save(predictor.to("cpu").state_dict(), os.path.join(args.out, "predictor.pt"))
    final = history[-1] if history else {}
    report = {
        "method": "td_jepa_symmetric", "args": vars(args), "config": vars(cfg),
        "encoder_linear_out": not args.relu_out, "obs_dim": int(data["obs"].shape[1]),
        "act_dim": int(data["act"].shape[1]), "train_rows": int(len(data["obs"])),
        "heldout_rows": int(len(heldout["obs"])), "reward_normalisation": rew_stats,
        "collect_seconds": collect_seconds, "total_seconds": time.time() - t0,
        "history": history, "final": final,
    }
    with open(os.path.join(args.out, "report.json"), "w") as f: json.dump(report, f, indent=2)
    print(f"Saved encoder -> {os.path.join(args.out, 'fc.pt')}")
    print("Next: --pretrained-encoder", os.path.join(args.out, "fc.pt"),
          "--encoder-linear-out" if not args.relu_out else "")


if __name__ == "__main__":
    main()
