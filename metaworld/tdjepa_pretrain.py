"""Pretrain the shared encoder with a TD latent-predictive objective (TD-JEPA).

Meta-World port of the HalfCheetah pretrainer. The only thing that changes is
where the data comes from: instead of held-out target velocities, we collect on
held-out Meta-World TASKS (tasks.PRETRAIN_TASKS), which appear in no evaluation
suite. The loss, the architecture and the anti-collapse machinery are unchanged
-- see td_jepa.py for what was and was not taken from the paper.

Produces `fc.pt`, saved the way cka_rl.py saves and loads encoders (torch.save
of the MODULE, not a state_dict), so it is a drop-in:

    python3 run_continual_benchmark.py \\
        --pretrained-encoder pretrained_encoders/mw/fc.pt --encoder-linear-out

WHY THIS MATTERS HERE
---------------------
The shared encoder holds roughly half the actor's parameters and sits entirely
outside the knowledge-vector mechanism. Without pretraining it is whatever
task 0 produced -- on this suite, window-close -- so every later task runs on a
representation shaped by a single door-sliding skill.

USAGE
-----
    python3 tdjepa_pretrain.py --smoke-test --out /tmp/wm_smoke
    python3 tdjepa_pretrain.py --steps-per-task 60000 --epochs 30 \\
        --out pretrained_encoders/mw

NOT EXECUTED IN THIS SANDBOX (no torch / metaworld) -- syntax-checked only.
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
from tasks import PRETRAIN_TASKS, TASK_SUITES
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


# =========================================================================== #
# Data collection
# =========================================================================== #
def collect_transitions(
    task_name: str,
    task_index: int,
    steps: int,
    seed: int,
    ou_theta: float = 0.15,
    ou_sigma: float = 0.35,
    uniform_frac: float = 0.35,
) -> Dict[str, np.ndarray]:
    """Collect (s, a, r, s', a') with mixed uniform / Ornstein-Uhlenbeck noise.

    Correlated noise matters more on a manipulation arm than it did on a
    cheetah: purely uniform actions leave the gripper jittering near its start
    pose and never touching the object, so the dynamics model would only ever
    see free-space motion. OU noise produces sustained reaches.

    a' (the next action actually taken) is stored because the TD bootstrap
    needs an action at s'. Transitions where the episode ended are dropped
    rather than bootstrapped through, since the stored a' would belong to a
    fresh episode.
    """
    from metaworld_envs import make_env

    # A distinct task_index keeps the frozen goal placement of each pretraining
    # task away from the evaluation tasks' placements.
    env = make_env(task_name, task_id=1000 + task_index)
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
def to_tensors(data, device, rew_mean, rew_std, task_onehot=None):
    obs = torch.as_tensor(data["obs"], device=device)
    out = {
        # Observations are fed RAW: the encoder must be a drop-in for fc.pt and
        # run_sac.py feeds it raw observations.
        "obs": obs,
        "act": torch.as_tensor(data["act"], device=device),
        "next_obs": torch.as_tensor(data["next_obs"], device=device),
        "next_act": torch.as_tensor(data["next_act"], device=device),
        "rew": (torch.as_tensor(data["rew"], device=device) - rew_mean) / rew_std,
    }
    if task_onehot is not None:
        out["task"] = torch.as_tensor(task_onehot, device=device)
    return out


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
    return total / n


def train(data, heldout, cfg, epochs, batch_size, device, linear_out, seed,
          task_dim, log_every=1):
    obs_dim = data["obs"].shape[1]
    act_dim = data["act"].shape[1]

    rew_mean = float(data["rew"].mean())
    rew_std = float(data["rew"].std() + 1e-6)
    T = to_tensors(data, device, rew_mean, rew_std, data.get("task_onehot"))
    H = (to_tensors(heldout, device, rew_mean, rew_std, heldout.get("task_onehot"))
         if heldout is not None else None)

    torch.manual_seed(seed)
    encoder = shared(input_dim=obs_dim, linear_out=linear_out).to(device)
    predictor = LatentPredictor(
        latent_dim=256, act_dim=act_dim,
        task_dim=task_dim if cfg.task_conditioned else 0,
        hidden=cfg.predictor_hidden, n_layers=cfg.predictor_layers,
    ).to(device)
    reward_head = RewardHead(256, act_dim).to(device) if cfg.c_rew > 0 else None

    enc_t = clone_target(encoder)
    pred_t = clone_target(predictor)

    # TD-JEPA Theorem 2: the predictor must be optimised faster than the
    # representation, which is what keeps the feature covariance from
    # collapsing. Two parameter groups, not one.
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
    p.add_argument("--pretrain-tasks", nargs="+", default=list(PRETRAIN_TASKS),
                   help="Meta-World task ids to collect on. Must NOT appear in "
                        "any evaluation suite.")
    p.add_argument("--heldout-suite", default="mw_easy4",
                   help="Evaluation suite whose tasks are used ONLY to measure "
                        "generalisation. The held-out TD error is the number "
                        "that shows whether the representation transfers.")
    p.add_argument("--steps-per-task", type=int, default=60_000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--predictor-lr-mult", type=float, default=3.0)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--lam-reg", type=float, default=1.0)
    p.add_argument("--c-rew", type=float, default=0.0,
                   help="Auxiliary reward-prediction weight. 0 = reward-free, as "
                        "in the paper. >0 is a deliberate deviation.")
    p.add_argument("--predictor-hidden", type=int, default=1024)
    p.add_argument("--predictor-layers", type=int, default=3)
    p.add_argument("--no-task-conditioned", action="store_true",
                   help="Ablation: unconditional predictor (closer to BYOL-gamma).")
    p.add_argument("--relu-out", action="store_true",
                   help="Keep the encoder's trailing ReLU. NOT recommended: the "
                        "orthonormality regulariser is ill-posed on non-negative "
                        "features.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="pretrained_encoders/mw")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    if args.smoke_test:
        args.pretrain_tasks = args.pretrain_tasks[:1]
        args.steps_per_task = 2_000
        args.epochs = 2
        args.predictor_hidden = 128
    return args


def _onehot(count: int, index: int, rows: int) -> np.ndarray:
    m = np.zeros((rows, count), dtype=np.float32)
    m[:, index] = 1.0
    return m


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Hard guard: pretraining on an evaluation task silently invalidates the
    # continual protocol AND inflates every downstream number, and nothing
    # later in the pipeline would catch it.
    eval_tasks = {t.name for suite in TASK_SUITES.values() for t in suite}
    leaked = sorted(set(args.pretrain_tasks) & eval_tasks)
    if leaked:
        raise SystemExit(
            f"pretrain tasks {leaked} are EVALUATION tasks. Pretraining must "
            "only see held-out tasks -- see tasks.PRETRAIN_TASKS."
        )
    if args.relu_out:
        print("WARNING: --relu-out keeps non-negative features, on which the "
              "orthonormality regulariser cannot push inner products below zero.")

    cfg = TDJepaConfig(
        gamma=args.gamma, tau=args.tau, lam_reg=args.lam_reg, c_rew=args.c_rew,
        lr=args.lr, predictor_lr_mult=args.predictor_lr_mult,
        predictor_hidden=args.predictor_hidden, predictor_layers=args.predictor_layers,
        task_conditioned=not args.no_task_conditioned,
    )

    print(f"Device: {device} | pretrain tasks: {args.pretrain_tasks}")
    t0 = time.time()

    # Meta-World has no task-conditioning vector in the observation (unlike the
    # HalfCheetah suite, where target velocity was appended), so the predictor
    # is conditioned on a one-hot over the pretraining tasks instead. That keeps
    # the "successor features of a family of behaviour policies indexed by task"
    # reading of the TD loss intact.
    n_tasks = len(args.pretrain_tasks)
    parts = []
    for i, name in enumerate(args.pretrain_tasks):
        print(f"  collect train  {name} ({args.steps_per_task} steps)")
        d = collect_transitions(name, i, args.steps_per_task, seed=args.seed + 1000 * i)
        d["task_onehot"] = _onehot(n_tasks, i, len(d["obs"]))
        parts.append(d)
    data = concat_datasets(parts)

    held_parts = []
    held_names = [t.name for t in TASK_SUITES[args.heldout_suite]]
    for i, name in enumerate(held_names):
        print(f"  collect HELD-OUT {name}")
        d = collect_transitions(name, 500 + i, max(args.steps_per_task // 5, 1000),
                                seed=args.seed + 7000 + 1000 * i)
        # Held-out tasks have no slot in the pretraining one-hot; use a uniform
        # vector so the predictor is queried off its training conditioning
        # rather than being handed a task identity it never saw.
        d["task_onehot"] = np.full((len(d["obs"]), n_tasks), 1.0 / n_tasks, dtype=np.float32)
        held_parts.append(d)
    heldout = concat_datasets(held_parts)

    collect_seconds = time.time() - t0
    print(f"data: train={len(data['obs'])} heldout={len(heldout['obs'])} "
          f"({collect_seconds:.1f}s)")

    encoder, predictor, history, rew_stats = train(
        data, heldout, cfg, args.epochs, args.batch_size, device,
        linear_out=not args.relu_out, seed=args.seed, task_dim=n_tasks,
    )

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
