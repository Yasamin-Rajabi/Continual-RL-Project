"""Continual Backprop: generate-and-test on hidden units (CbpNet).

Ported from the project's Atari implementation.  The mechanism is already
MLP-based, so only the surrounding network changes.

Each hidden unit accumulates a utility estimate; units older than a maturity
threshold and in the lowest utility are periodically reinitialized, with their
outgoing weights zeroed and the next layer's bias corrected so the reset does
not perturb the function.  The point is to keep plasticity from decaying over
a long task sequence.
"""
from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn

from shared_arch import layer_init


class CbpGaussianActor(nn.Module):
    """One hidden layer with generate-and-test tracking, then a Gaussian head."""

    def __init__(self, in_dim: int, hidden_dim: int, act_dim: int):
        super().__init__()
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = layer_init(nn.Linear(int(in_dim), self.hidden_dim))
        self.act = nn.ReLU()
        self.fc2 = layer_init(nn.Linear(self.hidden_dim, 2 * int(act_dim)), std=0.01)
        self._last_features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        # Keep the activation for the generate-and-test step; detached so it
        # never extends the autograd graph past the update.
        self._last_features = h.detach()
        return self.fc2(h)

    def get_activations(self):
        return [] if self._last_features is None else [self._last_features]


class GnT:
    """Generate-and-test over one hidden layer."""

    def __init__(
        self,
        actor: CbpGaussianActor,
        optimizer,
        decay_rate: float = 0.99,
        replacement_rate: float = 1e-4,
        maturity_threshold: int = 100,
        util_type: str = "contribution",
        device="cpu",
        accumulate: bool = False,
    ):
        self.actor = actor
        self.opt = optimizer
        self.decay_rate = float(decay_rate)
        self.replacement_rate = float(replacement_rate)
        self.maturity_threshold = int(maturity_threshold)
        self.util_type = str(util_type)
        self.device = device
        self.accumulate = bool(accumulate)

        n = actor.hidden_dim
        self.util = torch.zeros(n, device=device)
        self.bias_corrected_util = torch.zeros(n, device=device)
        self.ages = torch.zeros(n, device=device)
        self.mean_feature_act = torch.zeros(n, device=device)
        self.accumulated = 0.0
        gain = nn.init.calculate_gain("relu")
        self.bound = gain * sqrt(3.0 / actor.fc1.in_features)

    @torch.no_grad()
    def _update_utility(self, features: torch.Tensor):
        self.util *= self.decay_rate
        bias_correction = 1.0 - self.decay_rate ** self.ages.clamp_min(1.0)

        self.mean_feature_act *= self.decay_rate
        self.mean_feature_act += (1.0 - self.decay_rate) * features.mean(dim=0)
        corrected_act = self.mean_feature_act / bias_correction

        out_mag = self.actor.fc2.weight.data.abs().mean(dim=0)
        in_mag = self.actor.fc1.weight.data.abs().mean(dim=1)

        if self.util_type == "weight":
            new_util = out_mag
        elif self.util_type == "contribution":
            new_util = out_mag * features.abs().mean(dim=0)
        elif self.util_type == "adaptable_contribution":
            new_util = out_mag * (features - corrected_act).abs().mean(dim=0) / in_mag.clamp_min(1e-8)
        else:
            new_util = out_mag * features.abs().mean(dim=0)

        self.util += (1.0 - self.decay_rate) * new_util
        self.bias_corrected_util = self.util / bias_correction

    @torch.no_grad()
    def _select(self):
        self.ages += 1
        eligible = torch.where(self.ages > self.maturity_threshold)[0]
        if eligible.numel() == 0 or self.replacement_rate <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device), 0

        n_new = self.replacement_rate * float(eligible.numel())
        if self.accumulate:
            self.accumulated += n_new
            n_replace = int(self.accumulated)
            self.accumulated -= n_replace
        else:
            if n_new < 1.0:
                n_replace = 1 if float(torch.rand(1)) <= n_new else 0
            else:
                n_replace = int(n_new)
        if n_replace == 0:
            return torch.empty(0, dtype=torch.long, device=self.device), 0

        n_replace = min(n_replace, eligible.numel())
        worst = torch.topk(-self.bias_corrected_util[eligible], n_replace)[1]
        return eligible[worst], n_replace

    @torch.no_grad()
    def _regenerate(self, idx: torch.Tensor):
        if idx.numel() == 0:
            return
        fc1, fc2 = self.actor.fc1, self.actor.fc2
        fc1.weight.data[idx, :] = torch.empty(
            idx.numel(), fc1.in_features, device=fc1.weight.device
        ).uniform_(-self.bound, self.bound)
        if fc1.bias is not None:
            fc1.bias.data[idx] = 0.0

        # Fold each removed unit's mean contribution into the next layer's
        # bias so the reset does not shift the function, then zero its
        # outgoing weights and its age.
        correction = 1.0 - self.decay_rate ** self.ages[idx].clamp_min(1.0)
        fc2.bias.data += (
            fc2.weight.data[:, idx] * (self.mean_feature_act[idx] / correction)
        ).sum(dim=1)
        fc2.weight.data[:, idx] = 0.0

        self.util[idx] = 0.0
        self.mean_feature_act[idx] = 0.0
        self.ages[idx] = 0

    @torch.no_grad()
    def _reset_optimizer_state(self, idx: torch.Tensor):
        """Clear Adam moments for regenerated units.

        A fresh unit inheriting the momentum of the unit it replaced would be
        dragged straight back toward the function that was just removed.
        """
        if idx.numel() == 0 or self.opt is None:
            return
        for param, dim in ((self.actor.fc1.weight, 0), (self.actor.fc1.bias, 0),
                           (self.actor.fc2.weight, 1)):
            if param is None:
                continue
            state = self.opt.state.get(param)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                buf = state.get(key)
                if buf is None:
                    continue
                if dim == 0:
                    buf[idx] = 0.0
                else:
                    buf[:, idx] = 0.0

    @torch.no_grad()
    def step(self):
        features = self.actor.get_activations()
        if not features:
            return 0
        self._update_utility(features[0])
        idx, n = self._select()
        self._regenerate(idx)
        self._reset_optimizer_state(idx)
        return int(n)
