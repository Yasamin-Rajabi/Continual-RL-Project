"""Squashed-Gaussian policy math for continuous PointMaze control.

A head emits ``[mean, log_std]`` for the pre-squash Gaussian; the executed
action is ``tanh(u)``.  Everything the continual machinery needs is defined
here so that ``cka_rl.py`` and the baselines agree exactly on what a policy
divergence means.

Why divergences are measured BEFORE the squash
----------------------------------------------
``tanh`` is a fixed, deterministic, invertible map, and KL divergence is
invariant under invertible transformations:

    KL( tanh#p || tanh#q ) == KL( p || q )

So the closed-form diagonal-Gaussian KL computed on the pre-squash
distributions *is* the behavioral KL between the executed policies.  There is
no approximation being hidden here, and it avoids the sampling noise a
post-squash estimate would carry into merge-pair selection.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def clamp_log_std(log_std: torch.Tensor) -> torch.Tensor:
    """Bound log-sigma smoothly, as in the standard SAC implementation."""
    return LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1.0)


def split_head_output(raw: torch.Tensor, act_dim: int):
    """Split a head's ``2*act_dim`` output into (mean, clamped log_std)."""
    if raw.shape[-1] != 2 * int(act_dim):
        raise ValueError(
            f"head output has {raw.shape[-1]} units, expected {2 * int(act_dim)}"
        )
    mean, log_std = torch.split(raw, int(act_dim), dim=-1)
    return mean, clamp_log_std(log_std)


def gaussian_kl(mean_p, log_std_p, mean_q, log_std_q) -> torch.Tensor:
    """KL( N(p) || N(q) ) for diagonal Gaussians, summed over action dims.

    Returns one value per state.  Non-finite results are mapped to a large
    finite number so that a degenerate pair loses a minimum-KL selection
    instead of silently winning it via NaN comparison.
    """
    var_p = torch.exp(2.0 * log_std_p)
    var_q = torch.exp(2.0 * log_std_q)
    kl = (
        log_std_q
        - log_std_p
        + (var_p + (mean_p - mean_q) ** 2) / (2.0 * var_q)
        - 0.5
    ).sum(dim=-1)
    return torch.nan_to_num(kl, nan=1e12, posinf=1e12, neginf=0.0).clamp_min(0.0)


def symmetric_gaussian_kl(mean_a, log_std_a, mean_b, log_std_b) -> torch.Tensor:
    """0.5 * [ KL(a||b) + KL(b||a) ], one value per state."""
    return 0.5 * (
        gaussian_kl(mean_a, log_std_a, mean_b, log_std_b)
        + gaussian_kl(mean_b, log_std_b, mean_a, log_std_a)
    )


def gaussian_log_prob(u: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """log N(u; mean, exp(log_std)), summed over action dims."""
    var = torch.exp(2.0 * log_std)
    return (
        -0.5 * ((u - mean) ** 2) / var
        - log_std
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)


def squash_correction(u: torch.Tensor) -> torch.Tensor:
    """log |det d tanh(u) / du|, summed over action dims.

    Uses the numerically stable identity
    ``log(1 - tanh(u)^2) = 2*(log 2 - u - softplus(-2u))``.
    """
    return (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(dim=-1)


class SquashedGaussian:
    """A single tanh-squashed diagonal Gaussian policy."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = log_std
        self.std = torch.exp(log_std)

    def rsample_with_log_prob(self):
        """Reparameterized sample plus its log density, as SAC requires."""
        noise = torch.randn_like(self.mean)
        u = self.mean + self.std * noise
        action = torch.tanh(u)
        log_prob = gaussian_log_prob(u, self.mean, self.log_std) - squash_correction(u)
        return action, log_prob

    def sample(self):
        with torch.no_grad():
            noise = torch.randn_like(self.mean)
            return torch.tanh(self.mean + self.std * noise)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        u = atanh(action)
        return gaussian_log_prob(u, self.mean, self.log_std) - squash_correction(u)

    def deterministic_action(self) -> torch.Tensor:
        return torch.tanh(self.mean)

    def entropy_estimate(self) -> torch.Tensor:
        """Analytic pre-squash entropy; diagnostics only."""
        return (self.log_std + 0.5 * math.log(2.0 * math.pi * math.e)).sum(dim=-1)


def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(-1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class SquashedGaussianMixture:
    """A tanh-squashed mixture of diagonal Gaussians.

    Unlike the categorical Atari case -- where a mixture of categoricals is
    itself a categorical and nothing is lost -- a mixture of Gaussians is
    genuinely a mixture.  It is still *exactly* representable: sampling picks a
    component and then samples it, and the density is the weighted sum of the
    component densities.  Nothing here is approximated; only the KL between two
    mixtures has no closed form, and the one place that is needed
    (policy-space projection) uses an explicit Monte-Carlo cross-entropy.

    means/log_stds: [B, K, A].  weights: [K], on the simplex.
    """

    def __init__(self, means: torch.Tensor, log_stds: torch.Tensor, weights: torch.Tensor):
        if means.ndim != 3 or log_stds.shape != means.shape:
            raise ValueError(
                f"expected [B, K, A] means/log_stds, got {tuple(means.shape)} "
                f"and {tuple(log_stds.shape)}"
            )
        if weights.ndim != 1 or weights.numel() != means.shape[1]:
            raise ValueError(
                f"weights {tuple(weights.shape)} do not match K={means.shape[1]}"
            )
        self.means = means
        self.log_stds = log_stds
        self.stds = torch.exp(log_stds)
        self.weights = weights
        self.log_weights = torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))

    # -- densities ------------------------------------------------------
    def _component_log_prob(self, u: torch.Tensor) -> torch.Tensor:
        """log N(u; component k) for every k.  u: [B, A] -> [B, K]."""
        u = u.unsqueeze(1)
        var = self.stds ** 2
        return (
            -0.5 * ((u - self.means) ** 2) / var
            - self.log_stds
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)

    def log_prob_pre_squash(self, u: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self._component_log_prob(u) + self.log_weights, dim=-1)

    def log_prob_from_u(self, u: torch.Tensor) -> torch.Tensor:
        return self.log_prob_pre_squash(u) - squash_correction(u)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.log_prob_from_u(atanh(action))

    # -- sampling -------------------------------------------------------
    def _sample_u(self, reparameterized: bool) -> torch.Tensor:
        b, k, a = self.means.shape
        idx = torch.multinomial(
            self.weights.expand(b, k) if self.weights.ndim == 1 else self.weights,
            num_samples=1,
        )
        gather = idx.unsqueeze(-1).expand(b, 1, a)
        mean = self.means.gather(1, gather).squeeze(1)
        std = self.stds.gather(1, gather).squeeze(1)
        noise = torch.randn_like(mean)
        if not reparameterized:
            noise = noise.detach()
        return mean + std * noise

    def rsample_with_log_prob(self):
        """Sample and score.

        The component index is a discrete draw and is therefore not
        differentiable; the pathwise gradient flows through the selected
        component's mean and std, and the log-density is evaluated against the
        full mixture.  This is the standard pathwise estimator for a mixture
        and is what makes the SAC actor loss well defined here.
        """
        u = self._sample_u(reparameterized=True)
        action = torch.tanh(u)
        return action, self.log_prob_from_u(u)

    def sample(self):
        with torch.no_grad():
            return torch.tanh(self._sample_u(reparameterized=False))

    def sample_pre_squash(self, n: int = 1) -> torch.Tensor:
        """Draw ``n`` pre-squash samples per state: [n, B, A]."""
        with torch.no_grad():
            return torch.stack([self._sample_u(False) for _ in range(int(n))], dim=0)

    def deterministic_action(self) -> torch.Tensor:
        """Highest-weight component's mean, squashed.

        A mixture has no closed-form mode; the dominant component's mean is
        the standard deterministic proxy and is stable to evaluate.
        """
        best = int(torch.argmax(self.weights).item())
        return torch.tanh(self.means[:, best, :])

    def entropy_estimate(self) -> torch.Tensor:
        """One-sample Monte-Carlo entropy of the squashed mixture."""
        u = self._sample_u(reparameterized=False)
        return -self.log_prob_from_u(u)


def mixture_cross_entropy(
    samples_u: torch.Tensor, mean_q: torch.Tensor, log_std_q: torch.Tensor
) -> torch.Tensor:
    """E_{u~p}[ -log q(u) ] for a single Gaussian q.  samples_u: [n, B, A].

    Minimizing this over ``q`` minimizes ``KL(p || q)`` exactly: the two differ
    only by ``H(p)``, which does not depend on ``q``.  Drawing the samples once
    and reusing them makes the objective a fixed, stable dataset rather than a
    fresh stochastic target at every step.
    """
    n, b, a = samples_u.shape
    mean = mean_q.unsqueeze(0).expand(n, b, a)
    log_std = log_std_q.unsqueeze(0).expand(n, b, a)
    return -gaussian_log_prob(samples_u, mean, log_std).mean(dim=0)
