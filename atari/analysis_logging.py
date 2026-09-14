"""Lightweight analysis helpers for continual Atari PPO CKA-RL.

The active trainer is self-contained, but these helpers provide the same
categorical-policy diagnostics for notebooks/alternate trainers.  No SAC
mean/log-std heads or Q networks are assumed here.
"""
from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass

import numpy as np
import torch
import torch.nn.functional as F

_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


def _cpu_state_dict(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _lineage_counts(buffer, key):
    if buffer is None or key not in buffer:
        return {}
    values = np.asarray(buffer[key]).reshape(-1)
    ids, counts = np.unique(values, return_counts=True)
    return {str(int(i)): int(c) for i, c in zip(ids, counts)}


def _module_l2_norm(module):
    with torch.no_grad():
        params = [p.detach().reshape(-1) for p in module.parameters()]
        return 0.0 if not params else float(torch.cat(params).norm().item())


def effective_policy_vector(agent):
    """Flatten only the effective categorical policy head."""
    with torch.no_grad():
        return torch.cat([t.detach().reshape(-1) for t in agent.policy_pool._effective()])


def effective_theta_vector(agent):
    """Flatten shared CNN + current effective categorical policy head."""
    with torch.no_grad():
        pieces = [p.detach().reshape(-1) for p in agent.fc.parameters()]
        pieces.extend(t.detach().reshape(-1) for t in agent.policy_pool._effective())
        return torch.cat(pieces)


def pool_snapshot(pool, include_effective=True):
    result = {
        "head_type": pool.head_type,
        "fusion_mode": pool.fusion_mode,
        "pool_size_limit": int(pool.pool_size),
        "pool_length": int(pool.pool_length()),
        "distillation": bool(pool.distillation),
        "last_merge_info": pool.last_merge_info,
        "last_distill_train_kl": pool.last_distill_train_kl,
        "last_distill_test_kl": pool.last_distill_test_kl,
        "pool": [],
    }
    for entry in pool.pool:
        buf = entry.get("buffer")
        meta = None
        if buf is not None:
            meta = {
                "rows": int(len(buf.get("obs", []))),
                "task_lineage": _lineage_counts(buf, "task_ids"),
                "source_lineage": _lineage_counts(buf, "source_ids"),
                "arrays": {
                    k: {"shape": tuple(v.shape), "dtype": str(v.dtype)}
                    for k, v in buf.items() if isinstance(v, np.ndarray)
                },
            }
        result["pool"].append({
            **{k: entry[k].detach().cpu().clone() for k in _HEAD_KEYS},
            "buffer_meta": meta,
        })

    alpha = pool.alpha
    result["alpha_logits"] = None if alpha is None else alpha.detach().cpu().clone()
    result["alpha_scale"] = None if pool.alpha_scale is None else pool.alpha_scale.detach().cpu().clone()
    result["alpha_mass_raw"] = None if pool.alpha_mass is None else pool.alpha_mass.detach().cpu().clone()
    mass = pool.effective_alpha_mass()
    result["alpha_mass_effective"] = None if mass is None else mass.detach().cpu().clone()
    result["alpha_matches_pool_length"] = bool(alpha is None and not pool.pool) or bool(
        alpha is not None and alpha.numel() == len(pool.pool)
    )
    if alpha is not None and alpha.numel() == len(pool.pool):
        scale = 1.0 if pool.alpha_scale is None else pool.alpha_scale.detach()
        probs = F.softmax(alpha.detach() * scale, dim=0)
        weights = probs
        if pool.use_alpha_mass and pool.alpha_mass is not None:
            weights = weights * pool.effective_alpha_mass().detach()
        result["alpha_probabilities"] = probs.cpu().clone()
        result["alpha_weights"] = weights.cpu().clone()
    else:
        result["alpha_probabilities"] = None
        result["alpha_weights"] = None

    if include_effective:
        result["effective"] = {
            k: v.detach().cpu().clone()
            for k, v in zip(_HEAD_KEYS, pool._effective())
        }
    else:
        result["effective"] = None
    return result


def save_task_snapshot(
    out_path,
    phase,
    global_step,
    args,
    agent,
    *,
    include_effective=True,
    task_start_policy=None,
):
    """Save an analysis snapshot compatible with Atari merge-lineage plots."""
    os.makedirs(os.path.dirname(os.fspath(out_path)), exist_ok=True)
    args_dict = asdict(args) if is_dataclass(args) else dict(vars(args))
    payload = {
        "stage": str(phase),
        "step": int(global_step),
        "args": args_dict,
        "encoder_state_dict": _cpu_state_dict(agent.fc),
        "critic_state_dict": _cpu_state_dict(agent.critic),
        "policy_pool": pool_snapshot(agent.policy_pool, include_effective=include_effective),
        "pool_length": int(agent.policy_pool.pool_length()),
        "alpha": None if agent.alpha is None else agent.alpha.detach().cpu().clone(),
        "alpha_scale": None if agent.alpha_scale is None else agent.alpha_scale.detach().cpu().clone(),
        "alpha_mass": None if agent.alpha_mass is None else agent.alpha_mass.detach().cpu().clone(),
        "merge_info": agent.get_merge_info(),
        "distill_metrics": agent.get_distill_metrics(),
    }
    if include_effective:
        current = effective_policy_vector(agent).detach().cpu()
        payload["effective_policy"] = current
        if task_start_policy is not None and current.numel() == task_start_policy.numel():
            payload["effective_policy_delta_l2"] = float(
                (current - task_start_policy.detach().cpu()).norm().item()
            )
    torch.save(payload, out_path)


def log_training_state(writer, step, agent, task_start_policy=None):
    """Cheap Atari analysis scalars safe to call during PPO training."""
    writer.add_scalar("analysis/encoder/l2_norm", _module_l2_norm(agent.fc), step)
    writer.add_scalar("analysis/critic/l2_norm", _module_l2_norm(agent.critic), step)
    writer.add_scalar("analysis/pool/current_length", agent.policy_pool.pool_length(), step)

    current = effective_policy_vector(agent)
    writer.add_scalar("analysis/policy/effective_l2_norm", float(current.norm()), step)
    if task_start_policy is not None and current.numel() == task_start_policy.numel():
        writer.add_scalar(
            "analysis/policy/delta_from_task_start_l2",
            float((current - task_start_policy.to(current.device)).norm()),
            step,
        )

    if agent.alpha is not None and agent.alpha.numel() == agent.policy_pool.pool_length():
        scale = 1.0 if agent.alpha_scale is None else agent.alpha_scale.detach()
        probs = F.softmax(agent.alpha.detach() * scale, dim=0)
        entropy = -(probs * (probs + 1e-12).log()).sum()
        writer.add_scalar("analysis/policy/alpha_entropy", float(entropy), step)
        writer.add_scalar("analysis/policy/alpha_max", float(probs.max()), step)
        for i in range(probs.numel()):
            writer.add_scalar(f"analysis/policy/alpha_prob_{i}", float(probs[i]), step)

    if agent.alpha_mass is not None:
        writer.add_scalar("analysis/policy/alpha_mass_raw", float(agent.alpha_mass.detach()), step)
        writer.add_scalar(
            "analysis/policy/alpha_mass_effective",
            float(agent.policy_pool.effective_alpha_mass().detach()),
            step,
        )
