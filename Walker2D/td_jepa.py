"""TD latent-predictive representation learning for the shared encoder.

Adapted from TD-JEPA (Bagatella et al., arXiv:2510.00739, ICLR 2026 oral),
specifically its SYMMETRIC variant (their Algorithm 2), which uses a single
encoder as both state and task encoder. Their own ablation (Sec. 5.3) reports
the symmetric variant "performs comparatively rather well, while relying on a
single predictor-encoder pair" -- and a single encoder is what this codebase
needs, because `fc.pt` must stay a drop-in for CkaRlAgent.

WHAT WE TAKE FROM THE PAPER
---------------------------
1. The TD latent-predictive loss (their Eq. 7 / Alg. 2), which makes the encoder
   predictive of LONG-TERM latent dynamics rather than one-step transitions:

       L = || T(phi(s), a, z) - sg(phi-(s')) - gamma * sg(T-(phi-(s'), a', z)) ||^2

   The bootstrap term is what turns a one-step dynamics model into an estimate
   of successor features. This is the single most important upgrade over a plain
   next-state predictor.

2. The orthonormality regulariser (Alg. 2), which is how the paper prevents
   collapse without any reconstruction term:

       L_REG = 1/(2B(B-1)) * sum_{i!=j} (phi(s_i)^T phi(s_j))^2
               - 1/B * sum_i phi(s_i)^T phi(s_i)

   Their Theorem 2 gives the reason it works: if the predictor is optimised
   faster than the representation, the feature covariance is preserved, so a
   non-degenerate initialisation cannot collapse. Hence `predictor_lr_mult`.

3. EMA target networks for both encoder and predictor.

4. Their architectural finding for proprioceptive control (Table 4, App. D.4):
   a SHALLOW state encoder with a DEEP predictor. DMC used a 0-hidden-layer
   encoder with d_phi = 256 and a 3-hidden-layer predictor of width 1024. This
   codebase's encoder is already 2x256 with d_phi = 256, so the encoder needs no
   change at all -- the capacity goes in the predictor, which is thrown away.

WHAT WE DELIBERATELY DO NOT TAKE
--------------------------------
- The zero-shot machinery: task encoder psi, policy-conditioned latent policies
  pi_z, and reward-inference by linear regression onto psi. TD-JEPA is an
  unsupervised zero-shot RL algorithm; this project is continual RL where every
  task has an explicit reward and SAC trains a policy per task. Importing the
  zero-shot stack would add a second algorithm, not a better encoder.
- Their 2M gradient steps. Not affordable here (see the README's budget table).

HONEST NOTE ON "POLICY-CONDITIONED"
-----------------------------------
TD-JEPA's headline claim over BYOL-gamma is that it predicts the successor
features of the LEARNED policies pi_z. We do not train pi_z. Instead z is the
task vector already present in the observation (target velocity plus Walker2D dynamics scales)
and a' is the action the BEHAVIOUR policy actually took in the pretraining data.
So the predictor learns successor features of a family of exploratory behaviour
policies indexed by task, not of optimal policies. That is a legitimate
instantiation of the TD loss and it is genuinely multi-policy, but it is closer
to a task-conditioned BYOL-gamma than to full TD-JEPA. Describe it that way in
the paper; do not claim the zero-shot result.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class LatentPredictor(nn.Module):
    """T(phi(s), a, z) -> R^{d_phi}.

    Deep by design: the paper puts capacity here rather than in the encoder for
    proprioceptive domains, and this network is discarded after pretraining, so
    its size costs nothing at continual-RL time.
    """

    def __init__(
        self,
        latent_dim: int,
        act_dim: int,
        task_dim: int = 0,
        hidden: int = 1024,
        n_layers: int = 3,
    ):
        super().__init__()
        self.task_dim = int(task_dim)
        in_dim = latent_dim + act_dim + self.task_dim
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.Mish()]
            d = hidden
        layers += [nn.Linear(d, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z_state: torch.Tensor, action: torch.Tensor,
                task: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [z_state, action]
        if self.task_dim > 0:
            if task is None:
                raise ValueError("predictor was built task-conditioned but got task=None")
            parts.append(task)
        return self.net(torch.cat(parts, dim=-1))


class RewardHead(nn.Module):
    """Optional auxiliary reward head.

    TD-JEPA is reward-free by design, and its Fig. 4 shows frozen reward-free
    representations already support fast downstream adaptation. We keep this head
    available as an ABLATION AXIS (--c-rew), not as part of the default method,
    because we happen to know the reward family here and it is worth measuring
    whether that knowledge helps. Report it as a deviation from the paper.
    """

    def __init__(self, latent_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_state, action):
        return self.net(torch.cat([z_state, action], dim=-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def td_jepa_loss(
    encoder: nn.Module,
    predictor: LatentPredictor,
    encoder_target: nn.Module,
    predictor_target: LatentPredictor,
    obs: torch.Tensor,
    act: torch.Tensor,
    next_obs: torch.Tensor,
    next_act: torch.Tensor,
    task: Optional[torch.Tensor],
    gamma: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """TD-JEPA loss, symmetric variant (paper Alg. 2, blue branch).

    Both the phi(s') term and the bootstrapped T(phi(s'), a', z) term come from
    TARGET networks and carry no gradient -- that is the stop-gradient in the
    paper's equations. Letting gradients through either one is the classic way to
    get silent collapse.
    """
    z = encoder(obs)
    pred = predictor(z, act, task)

    with torch.no_grad():
        z_next_t = encoder_target(next_obs)
        boot = predictor_target(z_next_t, next_act, task)
        target = z_next_t + gamma * boot

    loss = 0.5 * F.mse_loss(pred, target)
    stats = {
        "loss/td_jepa": float(loss.detach()),
        "diag/target_norm": float(target.detach().norm(dim=-1).mean()),
        "diag/pred_norm": float(pred.detach().norm(dim=-1).mean()),
    }
    return loss, stats


def orthonormality_loss(z: torch.Tensor) -> torch.Tensor:
    """Paper Alg. 2:

        1/(2B(B-1)) * sum_{i != j} (z_i^T z_j)^2  -  1/B * sum_i z_i^T z_i

    The first term decorrelates features across states; the second is negative,
    so it pushes feature norms up and is what actually stops the trivial z = 0
    solution.

    WARNING: if the encoder ends in ReLU, every z_i is non-negative and z_i^T z_j
    can never go below zero, so the off-diagonal term can only be satisfied by
    driving features to disjoint sparse supports. Use shared(..., linear_out=True)
    -- the pretrainer defaults to it and warns otherwise.
    """
    B = z.shape[0]
    gram = z @ z.T
    diag = torch.diagonal(gram)
    off_sq = (gram.pow(2).sum() - diag.pow(2).sum())
    return off_sq / (2.0 * B * (B - 1)) - diag.sum() / B


@torch.no_grad()
def feature_diagnostics(z: torch.Tensor) -> Dict[str, float]:
    """Collapse and rank diagnostics. Log these every epoch; they are how you
    find out the run is dead long before the downstream RL numbers tell you."""
    B, d = z.shape
    zc = z - z.mean(0, keepdim=True)
    cov = (zc.T @ zc) / max(B - 1, 1)
    eig = torch.linalg.eigvalsh(cov.double()).clamp_min(0)
    total = eig.sum()
    if total <= 0:
        return {"diag/latent_std": 0.0, "diag/effective_rank": 0.0,
                "diag/dead_units": 1.0, "diag/mean_abs_cosine": 0.0}
    p = (eig / total).clamp_min(1e-12)
    eff_rank = float(torch.exp(-(p * p.log()).sum()))

    zn = F.normalize(z, dim=-1)
    cos = zn @ zn.T
    off = cos - torch.diag(torch.diagonal(cos))
    mean_abs_cos = float(off.abs().sum() / max(B * (B - 1), 1))

    return {
        "diag/latent_std": float(z.std()),
        "diag/effective_rank": eff_rank,
        "diag/dead_units": float((z.abs().mean(0) < 1e-6).float().mean()),
        "diag/mean_abs_cosine": mean_abs_cos,
    }


# --------------------------------------------------------------------------- #
# EMA targets
# --------------------------------------------------------------------------- #
def clone_target(module: nn.Module) -> nn.Module:
    tgt = copy.deepcopy(module)
    for p in tgt.parameters():
        p.requires_grad_(False)
    tgt.eval()
    return tgt


@torch.no_grad()
def ema_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    for p, pt in zip(online.parameters(), target.parameters()):
        pt.mul_(1.0 - tau).add_(p, alpha=tau)
    for b, bt in zip(online.buffers(), target.buffers()):
        bt.copy_(b)


@dataclass
class TDJepaConfig:
    gamma: float = 0.98            # paper's DMC value; SAC here uses 0.99
    tau: float = 0.01              # EMA rate for both targets
    lam_reg: float = 1.0           # weight on the orthonormality regulariser
    c_rew: float = 0.0             # >0 enables the (non-paper) reward head
    lr: float = 3e-4               # encoder LR
    predictor_lr_mult: float = 3.0 # Theorem 2: predictor must move FASTER than phi
    grad_clip: float = 10.0
    predictor_hidden: int = 1024
    predictor_layers: int = 3
    task_conditioned: bool = True
