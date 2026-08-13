"""Lightweight analysis logging for continual SAC runs.

Design:
- TensorBoard: cheap scalar histories during training (alphas, parameter norms,
  theta drift, critic norms).  This is suitable for frequent logging.
- .pt snapshots: exact tensors only at task boundaries.  Distillation buffers
  are intentionally NOT duplicated here because the normal HeadPool checkpoint
  already stores them and they can be very large.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F


_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


def _cpu_state_dict(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _tensor_dict_cpu(items):
    return {k: v.detach().cpu().clone() for k, v in items.items()}


def _module_vector(module):
    tensors = [p.detach().reshape(-1) for p in module.parameters()]
    if not tensors:
        return torch.empty(0, device=next(module.buffers()).device)
    return torch.cat(tensors)


def effective_theta_vector(model):
    """Flatten the actual actor theta used by forward(): encoder + both heads."""
    with torch.no_grad():
        mean_eff = model.mean_pool._effective()
        log_eff = model.logstd_pool._effective()
        pieces = [p.detach().reshape(-1) for p in model.fc.parameters()]
        pieces += [t.detach().reshape(-1) for t in mean_eff]
        pieces += [t.detach().reshape(-1) for t in log_eff]
        return torch.cat(pieces)


def _head_pool_snapshot(pool, include_effective: bool):
    result = {
        "head_type": pool.head_type,
        "fusion_mode": pool.fusion_mode,
        "pool_size_limit": int(pool.pool_size),
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
        "last_distill_train_mse": pool.last_distill_train_mse,
        "last_distill_test_mse": pool.last_distill_test_mse,
    }

    for entry in pool.pool:
        buf = entry.get("buffer")
        buffer_meta = None
        if buf is not None:
            buffer_meta = {
                "rows": int(buf["obs"].shape[0]) if "obs" in buf else None,
                "obs_shape": tuple(buf["obs"].shape) if "obs" in buf else None,
                "shared_shape": tuple(buf["shared"].shape) if "shared" in buf else None,
                "targets_shape": tuple(buf["targets"].shape) if "targets" in buf else None,
            }
        result["pool"].append({
            **_tensor_dict_cpu({k: entry[k] for k in _HEAD_KEYS}),
            "buffer_meta": buffer_meta,
        })

    alpha = pool.alpha
    alpha_len = 0 if alpha is None else int(alpha.numel())
    result["alpha_logits"] = None if alpha is None else alpha.detach().cpu().clone()
    result["alpha_scale"] = None if pool.alpha_scale is None else pool.alpha_scale.detach().cpu().clone()
    result["alpha_mass"] = None if pool.alpha_mass is None else pool.alpha_mass.detach().cpu().clone()
    result["alpha_matches_pool_length"] = (alpha_len == len(pool.pool))

    if alpha is not None and alpha_len == len(pool.pool):
        weights = F.softmax(alpha.detach() * pool.alpha_scale.detach(), dim=0)
        if pool.use_alpha_mass and pool.alpha_mass is not None:
            weights = weights * pool.alpha_mass.detach()
        result["alpha_weights"] = weights.cpu().clone()
    else:
        result["alpha_weights"] = None

    # After finalize(), alpha refers to the pre-finalize composition and can no
    # longer be interpreted as the exact current policy.  Therefore callers set
    # include_effective=False for post-finalize snapshots.
    if include_effective:
        w0, b0, w2, b2 = pool._effective()
        result["historical"] = None
        if pool.pool:
            hist = pool._historical()
            result["historical"] = _tensor_dict_cpu(hist)
        result["effective"] = _tensor_dict_cpu({
            "l0_weight": w0,
            "l0_bias": b0,
            "l2_weight": w2,
            "l2_bias": b2,
        })
    else:
        result["historical"] = None
        result["effective"] = None

    return result


def save_task_snapshot(
    out_path: str,
    phase: str,
    global_step: int,
    args,
    actor_model,
    qf1=None,
    qf2=None,
    qf1_target=None,
    qf2_target=None,
    entropy_alpha: Optional[float] = None,
    log_alpha=None,
    include_effective: bool = True,
    include_critics: bool = True,
):
    """Save an exact, analysis-oriented task-boundary snapshot."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    payload = {
        "meta": {
            "phase": phase,
            "global_step": int(global_step),
            "task_id": int(args.task_id),
            "seed": int(args.seed),
            "tag": str(args.tag),
            "fusion_mode": str(args.fusion_mode),
            "distillation": bool(args.distillation),
            "pool_size": int(args.pool_size),
        },
        "actor": {
            "encoder": _cpu_state_dict(actor_model.fc),
            "mean_headpool": _head_pool_snapshot(actor_model.mean_pool, include_effective),
            "logstd_headpool": _head_pool_snapshot(actor_model.logstd_pool, include_effective),
            # This is theta itself, in structured form, when it is meaningful.
            "effective_policy": actor_model.export_effective_policy() if include_effective else None,
        },
        "sac_entropy": {
            "alpha": None if entropy_alpha is None else float(entropy_alpha),
            "log_alpha": None if log_alpha is None else log_alpha.detach().cpu().clone(),
        },
    }

    if include_critics:
        payload["critics"] = {
            "qf1": _cpu_state_dict(qf1),
            "qf2": _cpu_state_dict(qf2),
            "qf1_target": _cpu_state_dict(qf1_target),
            "qf2_target": _cpu_state_dict(qf2_target),
        }
    else:
        payload["critics"] = None

    torch.save(payload, out_path)


def _param_norm(module):
    with torch.no_grad():
        total = torch.zeros((), device=next(module.parameters()).device)
        for p in module.parameters():
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
        _head_tensor_norm((pool.own_l0_weight, pool.own_l0_bias, pool.own_l2_weight, pool.own_l2_bias)),
        step,
    )
    writer.add_scalar(
        f"analysis/{prefix}/base_norm",
        _head_tensor_norm((pool.base_l0_weight, pool.base_l0_bias, pool.base_l2_weight, pool.base_l2_bias)),
        step,
    )
    writer.add_scalar(f"analysis/{prefix}/effective_norm", _head_tensor_norm(pool._effective()), step)

    for i, entry in enumerate(pool.pool):
        writer.add_scalar(
            f"analysis/{prefix}/pool_entry_{i}_norm",
            _head_tensor_norm(tuple(entry[k] for k in _HEAD_KEYS)),
            step,
        )

    if pool.alpha is not None and pool.alpha.numel() == len(pool.pool):
        logits = pool.alpha.detach()
        weights = F.softmax(logits * pool.alpha_scale.detach(), dim=0)
        if pool.use_alpha_mass and pool.alpha_mass is not None:
            weights = weights * pool.alpha_mass.detach()
        probs = F.softmax(logits * pool.alpha_scale.detach(), dim=0)
        entropy = -(probs * (probs + 1e-12).log()).sum().item()
        writer.add_scalar(f"analysis/{prefix}/alpha_entropy", entropy, step)
        writer.add_scalar(f"analysis/{prefix}/alpha_scale", pool.alpha_scale.detach().item(), step)
        if pool.alpha_mass is not None:
            writer.add_scalar(f"analysis/{prefix}/alpha_mass", pool.alpha_mass.detach().item(), step)
        for i in range(logits.numel()):
            writer.add_scalar(f"analysis/{prefix}/alpha_logit_{i}", logits[i].item(), step)
            writer.add_scalar(f"analysis/{prefix}/alpha_weight_{i}", weights[i].item(), step)


def log_training_state(writer, step, actor_model, qf1, qf2, qf1_target, qf2_target, theta_task_start):
    """Cheap histories to call every few thousand environment steps."""
    with torch.no_grad():
        theta = effective_theta_vector(actor_model)
        start = theta_task_start.to(theta.device)
        writer.add_scalar("analysis/theta/l2_norm", theta.norm().item(), step)
        writer.add_scalar("analysis/theta/drift_from_task_start_l2", (theta - start).norm().item(), step)
        writer.add_scalar(
            "analysis/theta/cosine_to_task_start",
            F.cosine_similarity(theta, start, dim=0).item(),
            step,
        )
        writer.add_scalar("analysis/encoder/l2_norm", _param_norm(actor_model.fc), step)

        _log_head(writer, "mean", actor_model.mean_pool, step)
        _log_head(writer, "logstd", actor_model.logstd_pool, step)

        writer.add_scalar("analysis/critic/qf1_param_norm", _param_norm(qf1), step)
        writer.add_scalar("analysis/critic/qf2_param_norm", _param_norm(qf2), step)
        writer.add_scalar("analysis/critic/qf1_target_param_norm", _param_norm(qf1_target), step)
        writer.add_scalar("analysis/critic/qf2_target_param_norm", _param_norm(qf2_target), step)
