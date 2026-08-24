"""Pretrain the shared encoder with a TD latent-predictive objective, then freeze it.

Produces `fc.pt`, saved exactly the way cka_rl.py saves and loads encoders
(torch.save of the MODULE, not a state_dict), so it is a true drop-in:

    python3 run_continual_benchmark.py \
        --pretrained-encoder pretrained_encoders/hc_vel/fc.pt \
        --encoder-linear-out

See README_TDJEPA.md for the full protocol and the ablation table.

WHY THIS EXISTS
---------------
The shared encoder `fc` holds 71,168 of the actor's 138,508 parameters (51.4%)
and sits entirely outside the knowledge-vector mechanism. Under the benchmark's
current configuration it is whatever task 0 produced -- and task 0 is
_VELOCITIES[0] = 0.5 m/s, the slowest task in the suite. This script replaces
that with an encoder trained across many tasks to be predictive of long-term
latent dynamics, and then frozen so the knowledge-vector basis stops moving.

WORKS FOR BOTH ENVIRONMENT FAMILIES
-----------------------------------
`--task-suite` accepts halfcheetah_vel, halfcheetah_wind_vel, ant_vel and
ant_wind_vel. Pretraining tasks are specified as HELD-OUT velocities that do not
appear in TASK_SUITES, so no benchmark task is ever seen during pretraining.
For Ant, remember to calibrate velocities first (see ant_envs and the analysis
doc) -- the HalfCheetah numbers are wrong for Ant.

NOT EXECUTED IN THIS SANDBOX (no torch/mujoco) -- syntax-checked only.
Run `--smoke-test` before committing GPU hours.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from shared_arch import shared
from td_jepa import (
    LatentPredictor,
    RewardHead,
    TDJepaConfig,
    clone_target,
    ema_update,
    feature_diagnostics,
    orthonormality_loss,
    td_jepa_loss,
)

TASK_DIM = 3  # [target_velocity, wind_a, wind_b] appended by the env wrapper


# =========================================================================== #
# Environment construction for pretraining tasks
# =========================================================================== #
def make_pretrain_env(task_suite: str, velocity: float, wind: Tuple[float, float]):
    """Build one pretraining env WITHOUT going through TASK_SUITES.

    Deliberate: pretraining velocities are held out and therefore have no
    task_id. Everything else mirrors tasks.get_task exactly -- same env classes,
    same task-conditioning wrapper, same TimeLimit.
    """
    import gymnasium as gym
    from halfcheetah_envs import (
        HalfCheetahVelEnv,
        HalfCheetahWindVelEnv,
        TaskConditionedObservationWrapper,
    )
    from tasks import HalfCheetahTask

    task = HalfCheetahTask(target_velocity=float(velocity), wind=tuple(wind))
    kwargs = {"target_velocity": float(velocity), "render_mode": None}

    if task_suite.startswith("ant"):
        from ant_envs import AntVelEnv, AntWindVelEnv
        from tasks import _ANT_SUCCESS_TOLERANCE

        env_cls = AntWindVelEnv if task_suite == "ant_wind_vel" else AntVelEnv
        kwargs["success_tolerance"] = _ANT_SUCCESS_TOLERANCE
    else:
        env_cls = (
            HalfCheetahWindVelEnv
            if task_suite == "halfcheetah_wind_vel"
            else HalfCheetahVelEnv
        )

    if task_suite.endswith("wind_vel"):
        kwargs["wind"] = tuple(wind)

    env = TaskConditionedObservationWrapper(env_cls(**kwargs), task)
    return gym.wrappers.TimeLimit(env, max_episode_steps=1000)


def collect_transitions(
    task_suite: str,
    velocity: float,
    wind: Tuple[float, float],
    steps: int,
    seed: int,
    ou_theta: float = 0.15,
    ou_sigma: float = 0.35,
    uniform_frac: float = 0.35,
) -> Dict[str, np.ndarray]:
    """Collect (s, a, s', a') with a mixture of uniform and Ornstein-Uhlenbeck
    random actions.

    Two things matter here:

    1. Correlated noise, not just uniform. Uniform actions make a cheetah or ant
       flail in place, covering a narrow slice of state space; the resulting
       dynamics model is accurate only where a trained policy will never go. OU
       noise produces sustained pushes and therefore actual locomotion.

    2. We store a' (the NEXT action actually taken). The TD bootstrap needs an
       action at s'. Since we are not training pi_z, a' comes from the behaviour
       policy -- see the honesty note at the top of td_jepa.py. Transitions where
       the episode ended are dropped rather than bootstrapped through, because
       the stored a' would belong to a fresh episode.
    """
    env = make_pretrain_env(task_suite, velocity, wind)
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
            S.append(obs)
            A.append(action)
            R.append(reward)
            S2.append(next_obs)
            A2.append(next_action)

        if done:
            obs, _ = env.reset()
            ou = np.zeros(act_dim, dtype=np.float64)
            action = sample_action()
        else:
            obs = next_obs
            action = next_action

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


# =========================================================================== #
# Training
# =========================================================================== #
def to_tensors(data: Dict[str, np.ndarray], device, rew_mean: float, rew_std: float):
    obs = torch.as_tensor(data["obs"], device=device)
    return {
        # Observations are fed RAW. The encoder must be a drop-in for fc.pt and
        # run_sac.py feeds it raw observations -- whitening here but not there
        # would silently break the handoff.
        "obs": obs,
        "act": torch.as_tensor(data["act"], device=device),
        "next_obs": torch.as_tensor(data["next_obs"], device=device),
        "next_act": torch.as_tensor(data["next_act"], device=device),
        "rew": (torch.as_tensor(data["rew"], device=device) - rew_mean) / rew_std,
        # The task vector is the LAST TASK_DIM observation dims, appended by
        # TaskConditionedObservationWrapper.
        "task": obs[:, -TASK_DIM:].contiguous(),
    }


@torch.no_grad()
def evaluate(encoder, predictor, enc_t, pred_t, T, cfg, batch=4096):
    encoder.eval()
    predictor.eval()
    n = T["obs"].shape[0]
    total = 0.0
    for start in range(0, n, batch):
        sl = slice(start, start + batch)
        task = T["task"][sl] if cfg.task_conditioned else None
        z = encoder(T["obs"][sl])
        pred = predictor(z, T["act"][sl], task)
        z_next = enc_t(T["next_obs"][sl])
        tgt = z_next + cfg.gamma * pred_t(z_next, T["next_act"][sl], task)
        total += float(F.mse_loss(pred, tgt, reduction="sum"))
    encoder.train()
    predictor.train()
    return total / (n * T["obs"].shape[1] if False else n)


def train(data, heldout, cfg: TDJepaConfig, epochs, batch_size, device,
          linear_out, seed, log_every=1):
    obs_dim = data["obs"].shape[1]
    act_dim = data["act"].shape[1]
    task_dim = TASK_DIM if cfg.task_conditioned else 0

    rew_mean = float(data["rew"].mean())
    rew_std = float(data["rew"].std() + 1e-6)
    T = to_tensors(data, device, rew_mean, rew_std)
    H = to_tensors(heldout, device, rew_mean, rew_std) if heldout is not None else None

    torch.manual_seed(seed)
    encoder = shared(input_dim=obs_dim, linear_out=linear_out).to(device)
    predictor = LatentPredictor(
        latent_dim=256, act_dim=act_dim, task_dim=task_dim,
        hidden=cfg.predictor_hidden, n_layers=cfg.predictor_layers,
    ).to(device)
    reward_head = RewardHead(256, act_dim).to(device) if cfg.c_rew > 0 else None

    enc_t = clone_target(encoder)
    pred_t = clone_target(predictor)

    # Theorem 2 requires the predictor to be optimised faster than the
    # representation; that is what keeps the feature covariance constant and so
    # what actually prevents collapse. Two param groups, not one.
    groups = [
        {"params": list(encoder.parameters()), "lr": cfg.lr},
        {"params": list(predictor.parameters()), "lr": cfg.lr * cfg.predictor_lr_mult},
    ]
    if reward_head is not None:
        groups.append({"params": list(reward_head.parameters()),
                       "lr": cfg.lr * cfg.predictor_lr_mult})
    opt = torch.optim.Adam(groups)

    n = T["obs"].shape[0]
    history: List[dict] = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        acc = {"td": 0.0, "reg": 0.0, "rew": 0.0}
        nb = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if idx.numel() < 4:
                continue
            task = T["task"][idx] if cfg.task_conditioned else None

            td, _ = td_jepa_loss(
                encoder, predictor, enc_t, pred_t,
                T["obs"][idx], T["act"][idx], T["next_obs"][idx],
                T["next_act"][idx], task, cfg.gamma,
            )
            z = encoder(T["obs"][idx])
            reg = orthonormality_loss(z)
            loss = td + cfg.lam_reg * reg

            rew_l = torch.zeros((), device=device)
            if reward_head is not None:
                rew_l = F.mse_loss(reward_head(z, T["act"][idx]), T["rew"][idx])
                loss = loss + cfg.c_rew * rew_l

            opt.zero_grad(set_to_none=True)
            loss.backward()
            params = list(encoder.parameters()) + list(predictor.parameters())
            if reward_head is not None:
                params += list(reward_head.parameters())
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            ema_update(encoder, enc_t, cfg.tau)
            ema_update(predictor, pred_t, cfg.tau)

            acc["td"] += float(td.detach())
            acc["reg"] += float(reg.detach())
            acc["rew"] += float(rew_l.detach())
            nb += 1

        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            with torch.no_grad():
                probe = encoder(T["obs"][:4096])
            entry = {
                "epoch": epoch + 1,
                "train_td": acc["td"] / max(nb, 1),
                "train_reg": acc["reg"] / max(nb, 1),
                "train_rew": acc["rew"] / max(nb, 1),
            }
            entry.update(feature_diagnostics(probe))
            if H is not None:
                entry["heldout_td"] = evaluate(encoder, predictor, enc_t, pred_t, H, cfg)
            history.append(entry)
            print(json.dumps(entry))

    return encoder, predictor, history, {"rew_mean": rew_mean, "rew_std": rew_std}


# =========================================================================== #
# CLI
# =========================================================================== #
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task-suite", default="halfcheetah_vel",
                   choices=["halfcheetah_vel", "halfcheetah_wind_vel",
                            "ant_vel", "ant_wind_vel"])
    p.add_argument("--pretrain-velocities", nargs="+", type=float,
                   default=[0.75, 1.75, 2.75],
                   help="Held-out velocities, deliberately NOT in TASK_SUITES.")
    p.add_argument("--pretrain-winds", nargs="+", type=float, default=None,
                   help="Flat list of wind pairs (wa wb wa wb ...), one per velocity. "
                        "Required for *_wind_vel suites.")
    p.add_argument("--heldout-velocities", nargs="+", type=float, default=[0.5, 2.0, 3.0],
                   help="Evaluation-only velocities. Put REAL benchmark velocities "
                        "here: the held-out TD error is the number that shows whether "
                        "the representation generalises to unseen tasks.")
    p.add_argument("--heldout-winds", nargs="+", type=float, default=None)
    p.add_argument("--steps-per-task", type=int, default=100_000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--predictor-lr-mult", type=float, default=3.0)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--lam-reg", type=float, default=1.0)
    p.add_argument("--c-rew", type=float, default=0.0,
                   help="Auxiliary reward-prediction weight. 0 = reward-free, as in "
                        "the paper. >0 is a deliberate deviation; report it as one.")
    p.add_argument("--predictor-hidden", type=int, default=1024)
    p.add_argument("--predictor-layers", type=int, default=3)
    p.add_argument("--no-task-conditioned", action="store_true",
                   help="Ablation: unconditional predictor (closer to BYOL-gamma).")
    p.add_argument("--relu-out", action="store_true",
                   help="Keep the encoder's trailing ReLU. NOT recommended: the "
                        "orthonormality regulariser is ill-posed on non-negative "
                        "features. Kept only as an ablation.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="pretrained_encoders/tdjepa")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        args.pretrain_velocities = [0.75]
        args.heldout_velocities = [2.0]
        if args.task_suite.endswith("wind_vel"):
            args.pretrain_winds = [0.0, 0.0]
            args.heldout_winds = [0.0, 0.0]
        args.steps_per_task = 2_000
        args.epochs = 2
        args.predictor_hidden = 128
    return args


def winds_for(suite: str, flat: Optional[Sequence[float]], n: int, label: str):
    if not suite.endswith("wind_vel"):
        return [(0.0, 0.0)] * n
    if flat is None:
        raise SystemExit(f"--{label} is required for a *_wind_vel suite")
    if len(flat) != 2 * n:
        raise SystemExit(f"--{label} needs {2 * n} numbers ({n} pairs), got {len(flat)}")
    return [(float(flat[2 * i]), float(flat[2 * i + 1])) for i in range(n)]


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    overlap = set(args.pretrain_velocities) & set(args.heldout_velocities)
    if overlap:
        raise SystemExit(
            f"pretrain and heldout velocities overlap: {sorted(overlap)} -- the "
            "held-out number would be meaningless."
        )
    if args.relu_out:
        print("WARNING: --relu-out keeps non-negative features, on which the "
              "orthonormality regulariser cannot push inner products below zero. "
              "Expect the regulariser to act as a sparsity penalty instead.")

    cfg = TDJepaConfig(
        gamma=args.gamma, tau=args.tau, lam_reg=args.lam_reg, c_rew=args.c_rew,
        lr=args.lr, predictor_lr_mult=args.predictor_lr_mult,
        predictor_hidden=args.predictor_hidden, predictor_layers=args.predictor_layers,
        task_conditioned=not args.no_task_conditioned,
    )

    print(f"Device: {device} | suite: {args.task_suite} | linear_out: {not args.relu_out}")
    t0 = time.time()

    pw = winds_for(args.task_suite, args.pretrain_winds,
                   len(args.pretrain_velocities), "pretrain-winds")
    parts = []
    for i, (v, w) in enumerate(zip(args.pretrain_velocities, pw)):
        print(f"  collect train  v={v} wind={w} ({args.steps_per_task} steps)")
        parts.append(collect_transitions(args.task_suite, v, w,
                                         args.steps_per_task, seed=args.seed + 1000 * i))
    data = concat_datasets(parts)

    hw = winds_for(args.task_suite, args.heldout_winds,
                   len(args.heldout_velocities), "heldout-winds")
    held_parts = []
    for i, (v, w) in enumerate(zip(args.heldout_velocities, hw)):
        print(f"  collect HELD-OUT v={v} wind={w}")
        held_parts.append(collect_transitions(
            args.task_suite, v, w, max(args.steps_per_task // 5, 1000),
            seed=args.seed + 7000 + 1000 * i))
    heldout = concat_datasets(held_parts)

    collect_seconds = time.time() - t0
    print(f"data: train={len(data['obs'])} heldout={len(heldout['obs'])} "
          f"({collect_seconds:.1f}s)")

    encoder, predictor, history, rew_stats = train(
        data, heldout, cfg, args.epochs, args.batch_size, device,
        linear_out=not args.relu_out, seed=args.seed,
    )

    # torch.save of the MODULE, matching how cka_rl.py saves/loads encoders
    # (torch.save(self.fc, ...) / _torch_load(f"{dir}/fc.pt")). This is what makes
    # the file a true drop-in.
    torch.save(encoder.to("cpu"), os.path.join(args.out, "fc.pt"))
    torch.save(predictor.to("cpu").state_dict(), os.path.join(args.out, "predictor.pt"))

    final = history[-1] if history else {}
    report = {
        "method": "td_jepa_symmetric",
        "paper": "arXiv:2510.00739 (symmetric variant, Alg. 2)",
        "args": vars(args),
        "config": vars(cfg),
        "encoder_linear_out": not args.relu_out,
        "obs_dim": int(data["obs"].shape[1]),
        "act_dim": int(data["act"].shape[1]),
        "train_rows": int(len(data["obs"])),
        "heldout_rows": int(len(heldout["obs"])),
        "reward_normalisation": rew_stats,
        "collect_seconds": collect_seconds,
        "total_seconds": time.time() - t0,
        "history": history,
        "final": final,
    }
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved encoder -> {os.path.join(args.out, 'fc.pt')}")
    print(f"effective_rank={final.get('diag/effective_rank')} "
          f"latent_std={final.get('diag/latent_std')} "
          f"heldout_td={final.get('heldout_td')}")

    # Collapse is the failure mode that costs you a whole ablation sweep if you
    # only notice it downstream. Say it loudly here.
    eff = final.get("diag/effective_rank", 0.0)
    if final.get("diag/latent_std", 1.0) < 1e-3 or eff < 5.0:
        print("\n!!! REPRESENTATION LOOKS COLLAPSED (low std or effective rank). "
              "Raise --lam-reg, raise --predictor-lr-mult, and confirm you are NOT "
              "running with --relu-out. Do not use this fc.pt.")
    print("\nNext: pass it to the benchmark with")
    print(f"    --pretrained-encoder {os.path.join(args.out, 'fc.pt')}"
          f"{'' if args.relu_out else ' --encoder-linear-out'}")


if __name__ == "__main__":
    main()
