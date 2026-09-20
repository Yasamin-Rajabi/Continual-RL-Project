"""Exact mixtures of categorical Atari policies.

For categorical policies, a mixture of categorical distributions is itself a
categorical distribution whose action probabilities are the weighted sum of the
component probabilities. No approximation is needed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def stacked_head_forward(z, head):
    """Evaluate K complete two-layer heads.

    z: [B,D]
    head tensors: first dimension K
    returns logits [B,K,A]
    """
    h = torch.einsum("bd,khd->bkh", z, head["l0_weight"]) + head["l0_bias"][None, :, :]
    return torch.einsum("bkh,kah->bka", F.relu(h), head["l2_weight"]) + head["l2_bias"][None, :, :]


def components(model, obs):
    """Return component logits [B,K,A] and simplex weights [K]."""
    if hasattr(model, "policy_components"):
        return model.policy_components(obs)
    logits = model(obs)
    return logits[:, None, :], logits.new_ones(1)


def categorical_mixture_probs(logits, weights):
    """Exact mixture probabilities from component logits and global weights."""
    if logits.ndim != 3:
        raise ValueError(f"expected [B,K,A] logits, got {tuple(logits.shape)}")
    if weights.ndim != 1 or weights.numel() != logits.shape[1]:
        raise ValueError(
            f"weights shape {tuple(weights.shape)} does not match K={logits.shape[1]}"
        )
    probs = F.softmax(logits, dim=-1)
    mixed = (probs * weights[None, :, None]).sum(dim=1)
    mixed = mixed.clamp_min(torch.finfo(mixed.dtype).tiny)
    return mixed / mixed.sum(dim=-1, keepdim=True)


def categorical_mixture_logits(logits, weights):
    """Canonical logits for the exact mixture distribution."""
    return categorical_mixture_probs(logits, weights).log()


def categorical_mixture_distribution(logits, weights):
    return Categorical(probs=categorical_mixture_probs(logits, weights))


def distribution(model, obs):
    logits, weights = components(model, obs)
    return categorical_mixture_distribution(logits, weights)


def deterministic_action(model, obs):
    logits, weights = components(model, obs)
    return categorical_mixture_probs(logits, weights).argmax(dim=-1)


def sample_action(model, obs):
    logits, weights = components(model, obs)
    dist = categorical_mixture_distribution(logits, weights)
    action = dist.sample()
    return action, dist.log_prob(action), dist.entropy()
