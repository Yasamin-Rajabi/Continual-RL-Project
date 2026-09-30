"""Shared plumbing for the continual PointMaze baselines.

Every baseline exposes the same small surface so that one SAC trainer can
drive all of them:

    agent.encode(obs)              -> features
    agent.actor_raw(features)      -> [B, 2*act_dim]  (mean and log-std)
    agent.action_distribution(obs) -> SquashedGaussian
    agent.critic / agent.critic_target
    agent.save(dir) / Class.load(dir, ...)

Optional hooks, used only by the methods that need them:

    set_task(task_slot, new_task)  MaskNet
    before_update()                PackNet gradient masking
    start_retraining()             PackNet second phase
    after_update()                 CbpNet generate-and-test

Task input
----------
FT-N, PackNet, MaskNet, ProgNet and CompoNet all receive the task index, by
construction -- they select a head, a mask, or a column with it.  That is how
those methods are defined and they are given it here.  Our method does not
receive it: it must infer which stored knowledge is relevant from behavior
alone.  This asymmetry favours the baselines and is stated rather than hidden.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from policy_utils import SquashedGaussian, clamp_log_std
from shared_arch import SharedEncoder, TwinCritic, layer_init


def make_encoder(obs_dim: int, shared_dim: int, hidden_dim: int = 256) -> SharedEncoder:
    return SharedEncoder(input_dim=obs_dim, output_dim=shared_dim, hidden_dim=hidden_dim)


class GaussianHead(nn.Module):
    """One hidden layer producing ``[mean, log_std]``."""

    def __init__(self, in_dim: int, hidden_dim: int, act_dim: int):
        super().__init__()
        self.act_dim = int(act_dim)
        self.net = nn.Sequential(
            layer_init(nn.Linear(int(in_dim), int(hidden_dim))),
            nn.ReLU(),
            layer_init(nn.Linear(int(hidden_dim), 2 * int(act_dim)), std=0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def raw_to_distribution(raw: torch.Tensor, act_dim: int) -> SquashedGaussian:
    mean, log_std = torch.split(raw, int(act_dim), dim=-1)
    return SquashedGaussian(mean, clamp_log_std(log_std))


class BaselineAgent(nn.Module):
    """Common SAC scaffolding; subclasses supply ``actor_raw``."""

    def __init__(self, obs_dim: int, act_dim: int, shared_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.shared_dim = int(shared_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = make_encoder(self.obs_dim, self.shared_dim, self.hidden_dim)
        self.critic = TwinCritic(self.shared_dim, self.act_dim, self.hidden_dim)
        self.critic_target = TwinCritic(self.shared_dim, self.act_dim, self.hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    # -- interface ------------------------------------------------------
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def actor_raw(self, features: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def actor_parameters(self):
        """Parameters updated by the SAC actor loss."""
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        critic_ids = {id(p) for p in self.critic.parameters()}
        critic_ids |= {id(p) for p in self.critic_target.parameters()}
        return [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in encoder_ids and id(p) not in critic_ids
        ]

    def encoder_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def action_distribution(self, obs: torch.Tensor) -> SquashedGaussian:
        return raw_to_distribution(self.actor_raw(self.encode(obs)), self.act_dim)

    def distribution_from_features(self, features: torch.Tensor) -> SquashedGaussian:
        return raw_to_distribution(self.actor_raw(features), self.act_dim)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.action_distribution(obs)
        return dist.deterministic_action() if deterministic else dist.sample()

    # -- optional hooks -------------------------------------------------
    def set_task(self, task_slot: int, new_task: bool = False):
        return None

    def before_update(self):
        return None

    def after_update(self):
        return None

    def start_retraining(self):
        return None

    def on_task_end(self):
        return None

    # -- persistence ----------------------------------------------------
    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, os.path.join(dirname, "agent.pt"))

    @classmethod
    def load(cls, dirname, map_location=None, **_ignored):
        path = os.path.join(os.fspath(dirname), "agent.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing baseline checkpoint: {path}")
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)


def count_trainable(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def soft_update(source: nn.Module, target: nn.Module, tau: float):
    with torch.no_grad():
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)
