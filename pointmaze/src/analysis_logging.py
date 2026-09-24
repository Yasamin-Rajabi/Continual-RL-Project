"""Task-boundary snapshots and cheap training-state histories.

Two kinds of record:

* scalar histories written every few thousand steps (cheap, TensorBoard/CSV);
* ``.pt`` snapshots at task start, pre-finalize and post-finalize, holding the
  exact tensors needed to reconstruct what the agent was doing at a boundary.

Snapshots store replay buffers as *metadata only* (row counts and lineage
histograms).  The policy-pool checkpoint is their authoritative home, and
copying multi-thousand-row state arrays into three snapshots per task would
multiply checkpoint size for no analytical gain.
"""
from __future__ import annotations

import os

import numpy as np
import torch

_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


def _cpu_state_dict(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _tensor_dict_cpu(items):
    return {k: v.detach().cpu().clone() for k, v in items.items()}


def _policy_parameter_proxy(agent):
    """Concatenated actor parameters, excluding the task-local critic.

    In policy-space composition there is no single effective policy MLP, so a
    proxy is used for drift diagnostics.  It is explicitly *not* an inference
    policy and must never be treated as one.
    """
    pieces = [
        p.detach().reshape(-1)
        for name, p in agent.named_parameters()
        if not name.startswith("critic")
    ]
    if pieces:
        return torch.cat(pieces)
    return torch.empty(0)


def effective_theta_vector(agent):
    if getattr(agent, "composition_space", "parameter") == "policy":
        return _policy_parameter_proxy(agent)
    with torch.no_grad():
        pieces = [p.detach().reshape(-1) for p in agent.fc.parameters()]
        pieces += [t.detach().reshape(-1) for t in agent.policy_pool._effective()]
        return torch.cat(pieces)


def _lineage_counts(buffer, key):
    if buffer is None or key not in buffer:
        return {}
    ids, counts = np.unique(np.asarray(buffer[key]).reshape(-1), return_counts=True)
    return {str(int(i)): int(c) for i, c in zip(ids, counts)}


def _head_pool_snapshot(pool, include_effective: bool):
    result = {
        "head_type": pool.head_type,
        "fusion_mode": pool.fusion_mode,
        "composition_space": getattr(pool, "composition_space", "parameter"),
        "pool_size_limit": int(pool.pool_size),
        "pool_length": int(pool.pool_length()),
        "distillation": bool(pool.distillation),
        "base": _tensor_dict_cpu(
            {k: getattr(pool, "base_" + k) for k in _HEAD_KEYS}
        ),
        "own": _tensor_dict_cpu({k: getattr(pool, "own_" + k) for k in _HEAD_KEYS}),
        "pool": [],
        "last_merge_info": pool.last_merge_info,
    }
    for entry in pool.pool:
        buf = entry.get("buffer")
        meta = None
        if buf is not None:
            meta = {
                "rows": int(len(buf["obs"])) if "obs" in buf else None,
                "task_lineage": _lineage_counts(buf, "task_ids"),
                "source_lineage": _lineage_counts(buf, "source_ids"),
            }
        result["pool"].append(
            {**_tensor_dict_cpu({k: entry[k] for k in _HEAD_KEYS}), "buffer_meta": meta}
        )

    alpha = pool.alpha
    result["alpha_logits"] = None if alpha is None else alpha.detach().cpu().clone()
    result["alpha_scale"] = (
        None if pool.alpha_scale is None else pool.alpha_scale.detach().cpu().clone()
    )
    result["alpha_mass_raw"] = (
        None if pool.alpha_mass is None else pool.alpha_mass.detach().cpu().clone()
    )
    eff = pool.effective_alpha_mass()
    result["alpha_mass_effective"] = None if eff is None else eff.detach().cpu().clone()
    result["alpha_matches_pool_length"] = (
        alpha is not None and int(alpha.numel()) == len(pool.pool)
    )

    if result["alpha_matches_pool_length"]:
        scale = 1.0 if pool.alpha_scale is None else pool.alpha_scale.detach()
        probs = torch.softmax(alpha.detach() * scale, dim=0)
        weights = probs * eff.detach() if (pool.use_alpha_mass and eff is not None) else probs
        result["alpha_probabilities"] = probs.cpu().clone()
        result["alpha_weights"] = weights.cpu().clone()
    else:
        result["alpha_probabilities"] = None
        result["alpha_weights"] = None

    # After finalize the alpha vector refers to the pre-finalize topology, and
    # policy-space composition has no single effective head at all, so an
    # "effective" reconstruction is only meaningful in the parameter case.
    if include_effective and result["composition_space"] == "parameter":
        result["historical"] = _tensor_dict_cpu(pool._historical()) if pool.pool else None
        result["effective"] = _tensor_dict_cpu(dict(zip(_HEAD_KEYS, pool._effective())))
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
    directory = os.path.dirname(os.fspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload = {
        "meta": {
            "phase": str(phase),
            "global_step": int(global_step),
            "task_id": int(getattr(args, "task_id", -1)),
            "seq_idx": int(getattr(args, "seq_idx", 0)),
            "suite": str(getattr(args, "suite", "")),
            "seed": int(getattr(args, "seed", 0)),
            "method": str(getattr(args, "method", "")),
            "fusion_mode": str(getattr(args, "fusion_mode", "")),
            "composition_space": str(getattr(args, "composition_space", "parameter")),
            "policy_student_replay": bool(getattr(args, "policy_student_replay", False)),
            "distillation": bool(getattr(args, "distillation", False)),
            "pool_size": int(getattr(args, "pool_size", 0)),
            "balance_source_lineages": bool(getattr(args, "balance_source_lineages", False)),
        },
        "actor": {
            "encoder": _cpu_state_dict(agent.fc),
            "policy_headpool": _head_pool_snapshot(agent.policy_pool, include_effective),
            "effective_policy": (
                agent.export_effective_policy() if include_effective else None
            ),
            "last_distill_metrics": agent.get_distill_metrics(),
            "last_projection_metrics": dict(getattr(agent, "last_projection_metrics", {})),
        },
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
        return float(torch.cat([p.detach().reshape(-1) for p in params]).norm())


def _head_tensor_norm(tensors):
    with torch.no_grad():
        return float(torch.sqrt(sum(t.detach().pow(2).sum() for t in tensors)))


def log_training_state(writer, step, agent, theta_task_start=None):
    """Cheap scalar histories; safe to call every few thousand steps."""
    with torch.no_grad():
        theta = effective_theta_vector(agent)
        label = (
            "parameter_proxy"
            if getattr(agent, "composition_space", "parameter") == "policy"
            else "theta"
        )
        writer.add_scalar(f"analysis/{label}/l2_norm", float(theta.norm()), step)
        if theta_task_start is not None and theta.numel() == theta_task_start.numel():
            start = theta_task_start.to(theta.device)
            writer.add_scalar(
                f"analysis/{label}/drift_from_task_start_l2", float((theta - start).norm()), step
            )

        writer.add_scalar("analysis/encoder/l2_norm", _param_norm(agent.fc), step)
        writer.add_scalar("analysis/critic/param_norm", _param_norm(agent.critic), step)

        pool = agent.policy_pool
        writer.add_scalar("analysis/policy/pool_length", len(pool.pool), step)
        writer.add_scalar(
            "analysis/policy/own_norm",
            _head_tensor_norm(tuple(getattr(pool, "own_" + k) for k in _HEAD_KEYS)),
            step,
        )
        for i, entry in enumerate(pool.pool):
            writer.add_scalar(
                f"analysis/policy/pool_entry_{i}_norm",
                _head_tensor_norm(tuple(entry[k] for k in _HEAD_KEYS)),
                step,
            )

        if pool.alpha is not None and pool.alpha.numel() == len(pool.pool):
            scale = 1.0 if pool.alpha_scale is None else pool.alpha_scale.detach()
            probs = torch.softmax(pool.alpha.detach() * scale, dim=0)
            entropy = float(-(probs * (probs + 1e-12).log()).sum())
            writer.add_scalar("analysis/policy/alpha_entropy", entropy, step)
            writer.add_scalar("analysis/policy/alpha_max", float(probs.max()), step)
            for i in range(probs.numel()):
                writer.add_scalar(f"analysis/policy/alpha_weight_{i}", float(probs[i]), step)
        if pool.alpha_mass is not None:
            writer.add_scalar(
                "analysis/policy/alpha_mass_raw", float(pool.alpha_mass.detach()), step
            )
            eff = pool.effective_alpha_mass()
            if eff is not None:
                writer.add_scalar("analysis/policy/alpha_mass", float(eff.detach()), step)
