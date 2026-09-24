"""Exact mixtures of squashed-Gaussian PointMaze policies.

The Atari version of this module could lean on a convenient fact: a mixture of
categoricals *is* a categorical, so composing in policy space cost nothing.
For continuous actions that shortcut does not exist -- a mixture of Gaussians
is a genuine mixture.  It is still represented exactly (see
``policy_utils.SquashedGaussianMixture``); what changes is that the two places
needing a divergence handle it explicitly rather than in closed form:

* pairwise merge selection and pair distillation compare *single* stored heads,
  so they use the exact closed-form diagonal-Gaussian KL;
* policy-space projection fits one head to a whole mixture, so it minimizes a
  Monte-Carlo cross-entropy, which differs from the true KL only by the
  mixture's own entropy and therefore has the same minimizer.
"""
from __future__ import annotations

import torch

from policy_utils import SquashedGaussianMixture, clamp_log_std


def stacked_head_forward(z: torch.Tensor, head: dict) -> torch.Tensor:
    """Evaluate K complete two-layer heads at once.

    z: [B, D]; head tensors carry K as their first dimension.
    Returns raw head output [B, K, 2*A].
    """
    h = torch.einsum("bd,khd->bkh", z, head["l0_weight"]) + head["l0_bias"][None, :, :]
    return (
        torch.einsum("bkh,koh->bko", torch.relu(h), head["l2_weight"])
        + head["l2_bias"][None, :, :]
    )


def split_stacked(raw: torch.Tensor, act_dim: int):
    """Split stacked head output [B, K, 2A] into means and clamped log-stds."""
    act_dim = int(act_dim)
    if raw.shape[-1] != 2 * act_dim:
        raise ValueError(
            f"stacked head output has {raw.shape[-1]} units, expected {2 * act_dim}"
        )
    means, log_stds = torch.split(raw, act_dim, dim=-1)
    return means, clamp_log_std(log_stds)


def gaussian_mixture_distribution(raw: torch.Tensor, weights: torch.Tensor, act_dim: int):
    means, log_stds = split_stacked(raw, act_dim)
    return SquashedGaussianMixture(means, log_stds, weights)


def components(model, obs):
    """Return stacked component output [B, K, 2A] and simplex weights [K]."""
    if hasattr(model, "policy_components"):
        return model.policy_components(obs)
    raw = model(obs)
    return raw[:, None, :], raw.new_ones(1)


def distribution(model, obs, act_dim: int):
    raw, weights = components(model, obs)
    return gaussian_mixture_distribution(raw, weights, act_dim)
