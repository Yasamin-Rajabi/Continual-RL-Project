"""Lightweight analysis logging for continual Atari PPO CKA-RL runs.

This is the categorical/PPO counterpart of the HalfCheetah analysis logger.
Shared semantics are kept aligned wherever they are algorithm-independent:
- TensorBoard stores cheap scalar histories during training.
- .pt snapshots store exact task-boundary tensors/metadata.
- retained behavioral buffers are described by metadata only here; the normal
  policy-pool checkpoint remains their authoritative storage.
- parameter-space composition has one effective head; policy-space composition
  does not, so analysis uses a parameter proxy there rather than pretending an
  exact policy mixture is one MLP.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F


_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


def _cpu_state_dict(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _tensor_dict_cpu(items):
    return {k: v.detach().cpu().clone() for k, v in items.items()}


def _module_vector(module):
    params = [p.detach().reshape(-1) for p in module.parameters()]
    if params:
        return torch.cat(params)
    buffers = list(module.buffers())
    device = buffers[0].device if buffers else torch.device("cpu")
    return torch.empty(0, device=device)


def _policy_parameter_proxy_vector(agent):
    """Policy-side parameter proxy for exact policy-space mixtures.

    In policy composition there is no single effective categorical MLP.  The
    proxy therefore concatenates actor/policy parameters while excluding the
    PPO critic.  It is diagnostic only and must never be interpreted as an
    inference policy.
    """
    pieces = []
    for name, param in agent.named_parameters():
        if name.startswith("critic."):
            continue
        pieces.append(param.detach().reshape(-1))
    if pieces:
        return torch.cat(pieces)
    return torch.empty(0, device=next(agent.parameters()).device)


def effective_policy_vector(agent):
    """Effective categorical head in parameter mode; proxy in policy mode."""
    if getattr(agent, "composition_space", "parameter") == "policy":
        return _policy_parameter_proxy_vector(agent)
    with torch.no_grad():
        return torch.cat(
            [t.detach().reshape(-1) for t in agent.policy_pool._effective()]
        )


def effective_theta_vector(agent):
    """Parameter-mode theta; policy-mode parameter proxy, never a fake mixture MLP."""
    if getattr(agent, "composition_space", "parameter") == "policy":
        return _policy_parameter_proxy_vector(agent)
    with torch.no_grad():
        pieces = [p.detach().reshape(-1) for p in agent.fc.parameters()]
        pieces += [t.detach().reshape(-1) for t in agent.policy_pool._effective()]
        return torch.cat(pieces)


def _lineage_counts(buffer, key):
    if buffer is None or key not in buffer:
        return {}
    values = np.asarray(buffer[key]).reshape(-1)
    ids, counts = np.unique(values, return_counts=True)
    return {str(int(i)): int(c) for i, c in zip(ids, counts)}


def _head_pool_snapshot(pool, include_effective: bool):
    result = {
        "head_type": pool.head_type,
        "fusion_mode": pool.fusion_mode,
        "composition_space": getattr(pool, "composition_space", "parameter"),
        "pool_size_limit": int(pool.pool_size),
        "pool_length": int(pool.pool_length()),
        "distillation": bool(pool.distillation),
        "base": _tensor_dict_cpu({
            "l0_weight": pool.base_l0_weight,
            "l0_bias": pool.base_l0_bias,
            "l2_weight": pool.base_l2_weight,
            "l2_bias": pool.base_l2_bias,
        }),
        "own": _tensor_dict_cpu({
            "l0_weight": pool.own_l0_weight,
            "l0_bias": pool.own_l0_bias,
            "l2_weight": pool.own_l2_weight,
            "l2_bias": pool.own_l2_bias,
        }),
        "pool": [],
        "last_merge_info": pool.last_merge_info,
        "last_distill_train_kl": getattr(pool, "last_distill_train_kl", None),
        "last_distill_test_kl": getattr(pool, "last_distill_test_kl", None),
        "last_distill_train_mse": getattr(pool, "last_distill_train_mse", None),
        "last_distill_test_mse": getattr(pool, "last_distill_test_mse", None),
    }

    for entry in pool.pool:
        buf = entry.get("buffer")
        buffer_meta = None
        if buf is not None:
            buffer_meta = {
                "rows": int(buf["obs"].shape[0]) if "obs" in buf else None,
                "task_lineage": _lineage_counts(buf, "task_ids"),
                "source_lineage": _lineage_counts(buf, "source_ids"),
                "arrays": {
                    key: {"shape": tuple(value.shape), "dtype": str(value.dtype)}
                    for key, value in buf.items()
                    if hasattr(value, "shape")
                },
            }
        result["pool"].append({
            **_tensor_dict_cpu({k: entry[k] for k in _HEAD_KEYS}),
            "buffer_meta": buffer_meta,
        })

    alpha = pool.alpha
    alpha_len = 0 if alpha is None else int(alpha.numel())
    result["alpha_logits"] = None if alpha is None else alpha.detach().cpu().clone()
    result["alpha_scale"] = (
        None if pool.alpha_scale is None else pool.alpha_scale.detach().cpu().clone()
    )
    result["alpha_mass"] = (
        None if pool.alpha_mass is None else pool.alpha_mass.detach().cpu().clone()
    )
    result["alpha_mass_raw"] = result["alpha_mass"]
    effective_mass = pool.effective_alpha_mass()
    result["alpha_mass_effective"] = (
        None if effective_mass is None else effective_mass.detach().cpu().clone()
    )
    result["alpha_matches_pool_length"] = alpha_len == len(pool.pool)

    if alpha is not None and alpha_len == len(pool.pool):
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

    # Exactly as in the new HalfCheetah logger: after finalize(), alpha refers
    # to the pre-finalize topology.  Also, policy-space composition has no single
    # effective MLP at all.  Therefore only parameter-space pre-finalize snapshots
    # receive historical/effective tensor reconstructions.
    if (
        include_effective
        and getattr(pool, "composition_space", "parameter") == "parameter"
    ):
        result["historical"] = None
        if pool.pool:
            result["historical"] = _tensor_dict_cpu(pool._historical())
        result["effective"] = _tensor_dict_cpu(
            dict(zip(_HEAD_KEYS, pool._effective()))
        )
    else:
        result["historical"] = None
        result["effective"] = None

    return result


def save_task_snapshot(
    out_path,
    phase,
    global_step,
    args,
    agent,
    *,
    include_effective: bool = True,
    include_critic: bool = True,
):
    """Save an exact, analysis-oriented Atari task-boundary snapshot."""
    directory = os.path.dirname(os.fspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {
        "meta": {
            "phase": str(phase),
            "global_step": int(global_step),
            "task_id": int(args.task_id),
            "seq_idx": int(getattr(args, "seq_idx", 0)),
            "task_suite": str(args.task_suite),
            "seed": int(args.seed),
            "tag": str(args.tag),
            "fusion_mode": str(args.fusion_mode),
            "composition_space": str(
                getattr(args, "composition_space", "parameter")
            ),
            "policy_student_replay": bool(
                getattr(args, "policy_student_replay", False)
            ),
            "distillation": bool(args.distillation),
            "pool_size": int(args.pool_size),
            "similarity_samples": int(args.similarity_samples),
            "distill_extra_steps": int(args.distill_extra_steps),
            "distill_max_samples": int(args.distill_max_samples),
            "distill_epochs": int(args.distill_epochs),
            "balance_source_lineages": bool(
                getattr(args, "balance_source_lineages", False)
            ),
            "projection_epochs": int(getattr(args, "projection_epochs", 0)),
            "projection_max_samples": int(
                getattr(args, "projection_max_samples", 0)
            ),
        },
        "actor": {
            "encoder": _cpu_state_dict(agent.fc),
            "policy_headpool": _head_pool_snapshot(
                agent.policy_pool, include_effective
            ),
            # export_effective_policy() is expected to dispatch to an exact
            # policy-ensemble snapshot when composition_space == "policy",
            # matching the new HalfCheetah semantics.
            "effective_policy": (
                agent.export_effective_policy() if include_effective else None
            ),
            "last_distill_metrics": agent.get_distill_metrics(),
            "last_projection_metrics": dict(
                getattr(agent, "last_projection_metrics", {})
            ),
        },
        # PPO has one task-local value function rather than SAC's Q1/Q2 pair.
        "critic": (
            _cpu_state_dict(agent.critic)
            if include_critic and getattr(agent, "critic", None) is not None
            else None
        ),
    }

    torch.save(payload, out_path)


def _param_norm(module):
    if module is None:
        return 0.0
    with torch.no_grad():
        params = list(module.parameters())
        if not params:
            return 0.0
        total = torch.zeros((), device=params[0].device)
        for p in params:
            total = total + p.detach().pow(2).sum()
        return total.sqrt().item()


def _head_tensor_norm(tensors):
    with torch.no_grad():
        total = sum(t.detach().pow(2).sum() for t in tensors)
        return total.sqrt().item()


def _log_head(writer, prefix, pool, step):
    writer.add_scalar(f"analysis/{prefix}/pool_length", len(pool.pool), step)
    writer.add_scalar(
        f"analysis/{prefix}/own_norm",
        _head_tensor_norm(
            (
                pool.own_l0_weight,
                pool.own_l0_bias,
                pool.own_l2_weight,
                pool.own_l2_bias,
            )
        ),
        step,
    )
    writer.add_scalar(
        f"analysis/{prefix}/base_norm",
        _head_tensor_norm(
            (
                pool.base_l0_weight,
                pool.base_l0_bias,
                pool.base_l2_weight,
                pool.base_l2_bias,
            )
        ),
        step,
    )
    if getattr(pool, "composition_space", "parameter") == "parameter":
        writer.add_scalar(
            f"analysis/{prefix}/effective_norm",
            _head_tensor_norm(pool._effective()),
            step,
        )

    for i, entry in enumerate(pool.pool):
        writer.add_scalar(
            f"analysis/{prefix}/pool_entry_{i}_norm",
            _head_tensor_norm(tuple(entry[k] for k in _HEAD_KEYS)),
            step,
        )

    if pool.alpha is not None and pool.alpha.numel() == len(pool.pool):
        logits = pool.alpha.detach()
        scale = 1.0 if pool.alpha_scale is None else pool.alpha_scale.detach()
        probs = F.softmax(logits * scale, dim=0)
        weights = probs
        if pool.use_alpha_mass and pool.alpha_mass is not None:
            weights = weights * pool.effective_alpha_mass().detach()
        entropy = -(probs * (probs + 1e-12).log()).sum().item()
        writer.add_scalar(f"analysis/{prefix}/alpha_entropy", entropy, step)
        if pool.alpha_scale is not None:
            writer.add_scalar(
                f"analysis/{prefix}/alpha_scale",
                pool.alpha_scale.detach().item(),
                step,
            )
        if pool.alpha_mass is not None:
            writer.add_scalar(
                f"analysis/{prefix}/alpha_mass_raw",
                pool.alpha_mass.detach().item(),
                step,
            )
            writer.add_scalar(
                f"analysis/{prefix}/alpha_mass",
                pool.effective_alpha_mass().detach().item(),
                step,
            )
            writer.add_scalar(
                f"analysis/{prefix}/alpha_mass_effective",
                pool.effective_alpha_mass().detach().item(),
                step,
            )
        for i in range(logits.numel()):
            writer.add_scalar(
                f"analysis/{prefix}/alpha_logit_{i}", logits[i].item(), step
            )
            writer.add_scalar(
                f"analysis/{prefix}/alpha_weight_{i}", weights[i].item(), step
            )


def log_training_state(writer, step, agent, theta_task_start=None):
    """Cheap Atari histories to call every few thousand environment steps."""
    with torch.no_grad():
        theta = effective_theta_vector(agent)
        theta_label = (
            "parameter_proxy"
            if getattr(agent, "composition_space", "parameter") == "policy"
            else "theta"
        )
        writer.add_scalar(f"analysis/{theta_label}/l2_norm", theta.norm().item(), step)

        if theta_task_start is not None and theta.numel() == theta_task_start.numel():
            start = theta_task_start.to(theta.device)
            writer.add_scalar(
                f"analysis/{theta_label}/drift_from_task_start_l2",
                (theta - start).norm().item(),
                step,
            )
            writer.add_scalar(
                f"analysis/{theta_label}/cosine_to_task_start",
                F.cosine_similarity(theta, start, dim=0).item(),
                step,
            )

        writer.add_scalar("analysis/encoder/l2_norm", _param_norm(agent.fc), step)
        _log_head(writer, "policy", agent.policy_pool, step)
        writer.add_scalar(
            "analysis/critic/param_norm",
            _param_norm(getattr(agent, "critic", None)),
            step,
        )
