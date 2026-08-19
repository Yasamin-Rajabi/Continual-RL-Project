"""Policy-distribution helpers shared by SAC and behavioral pool merging."""
from __future__ import annotations

import torch

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def bound_log_std(raw_log_std: torch.Tensor) -> torch.Tensor:
    """Map an unconstrained log-std head output to SAC's bounded range."""
    x = torch.tanh(raw_log_std)
    return LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (x + 1.0)


def diagonal_gaussian_kl(
    mean_p: torch.Tensor,
    log_std_p: torch.Tensor,
    mean_q: torch.Tensor,
    log_std_q: torch.Tensor,
) -> torch.Tensor:
    """KL[N_p || N_q] for diagonal Gaussians, summed over action dims.

    Returns one scalar per batch element. Inputs are already-bounded log stds.
    """
    # Work in log-variance space instead of dividing two tiny variances.
    # SAC permits log_std down to -20, where exp(2*log_std) is ~4e-18;
    # explicit variance division with an epsilon would badly bias the KL.
    log_std_p = torch.clamp(log_std_p, LOG_STD_MIN, LOG_STD_MAX)
    log_std_q = torch.clamp(log_std_q, LOG_STD_MIN, LOG_STD_MAX)
    log_ratio = torch.clamp(2.0 * (log_std_p - log_std_q), -60.0, 60.0)
    inv_var_q = torch.exp(torch.clamp(-2.0 * log_std_q, -60.0, 60.0))
    kl_per_dim = (
        log_std_q
        - log_std_p
        + 0.5 * (torch.exp(log_ratio) + (mean_p - mean_q).pow(2) * inv_var_q - 1.0)
    )
    return torch.nan_to_num(kl_per_dim, nan=1e12, posinf=1e12, neginf=0.0).sum(dim=-1)


def symmetric_diagonal_gaussian_kl(
    mean_a: torch.Tensor,
    raw_log_std_a: torch.Tensor,
    mean_b: torch.Tensor,
    raw_log_std_b: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL between the two SAC pre-tanh Gaussian policies.

    The same tanh bijection is applied to both policies at action sampling time,
    so comparing the pre-tanh distributions is the stable/cheap way to compare
    their behavior. Returns one scalar per state.
    """
    log_std_a = bound_log_std(raw_log_std_a)
    log_std_b = bound_log_std(raw_log_std_b)
    return 0.5 * (
        diagonal_gaussian_kl(mean_a, log_std_a, mean_b, log_std_b)
        + diagonal_gaussian_kl(mean_b, log_std_b, mean_a, log_std_a)
    )
