"""Categorical-policy distribution helpers shared by Atari PPO and pool merging."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def categorical_kl(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
) -> torch.Tensor:
    """KL[Categorical(p) || Categorical(q)], one scalar per state."""
    if logits_p.shape != logits_q.shape:
        raise ValueError(
            f"categorical KL requires matching shapes, got "
            f"{tuple(logits_p.shape)} and {tuple(logits_q.shape)}"
        )
    if logits_p.ndim < 2:
        raise ValueError(
            f"categorical KL expects [..., num_actions] logits, got {tuple(logits_p.shape)}"
        )
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return torch.nan_to_num(kl, nan=1e12, posinf=1e12, neginf=0.0).clamp_min(0.0)


def categorical_kl_from_probs(
    probs_p: torch.Tensor,
    logits_q: torch.Tensor,
) -> torch.Tensor:
    """KL[p || Categorical(logits_q)] when the teacher is already probabilities."""
    if probs_p.shape != logits_q.shape:
        raise ValueError(
            f"categorical KL requires matching shapes, got "
            f"{tuple(probs_p.shape)} and {tuple(logits_q.shape)}"
        )
    tiny = torch.finfo(probs_p.dtype).tiny
    p = probs_p.clamp_min(tiny)
    p = p / p.sum(dim=-1, keepdim=True)
    log_p = p.log()
    log_q = F.log_softmax(logits_q, dim=-1)
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return torch.nan_to_num(kl, nan=1e12, posinf=1e12, neginf=0.0).clamp_min(0.0)


def symmetric_categorical_kl(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
) -> torch.Tensor:
    """0.5 * [KL(a||b) + KL(b||a)], one scalar per state."""
    return 0.5 * (
        categorical_kl(logits_a, logits_b)
        + categorical_kl(logits_b, logits_a)
    )
