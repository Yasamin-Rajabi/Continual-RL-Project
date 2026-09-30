"""Shared MLP encoder for PointMaze, plus checkpoint-compatibility checks.

This is the continuous-control counterpart of the Atari CNN encoder.  The
encoder is the part of the agent that is genuinely task independent: the eight
ray sensors report local maze geometry, which is identical in every task of a
suite, while the goal-specific behavior lives entirely in the policy heads.

``ENCODER_FORMAT_VERSION`` is stamped into every checkpoint.  A chain that was
started under one architecture can then never be silently continued under a
different one.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

ENCODER_FORMAT_VERSION = 1


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SharedEncoder(nn.Module):
    """Two hidden layers, ReLU, feeding a fixed-width feature vector."""

    def __init__(self, input_dim: int, output_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.format_version = ENCODER_FORMAT_VERSION
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, self.hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(self.hidden_dim, self.output_dim)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def shared(input_shape, output_dim: int = 256, hidden_dim: int = 256) -> SharedEncoder:
    """Build the encoder from an observation shape, mirroring the Atari API."""
    shape = tuple(int(x) for x in np.atleast_1d(input_shape))
    if len(shape) != 1:
        raise ValueError(f"PointMaze expects flat observations, got shape {shape}")
    return SharedEncoder(input_dim=shape[0], output_dim=output_dim, hidden_dim=hidden_dim)


def validate_shared_encoder(encoder, input_shape, output_dim: int, source: str) -> None:
    """Fail loudly when a loaded encoder does not match the current run."""
    shape = tuple(int(x) for x in np.atleast_1d(input_shape))
    version = int(getattr(encoder, "format_version", -1))
    if version != ENCODER_FORMAT_VERSION:
        raise RuntimeError(
            f"{source}: encoder format_version={version}, expected "
            f"{ENCODER_FORMAT_VERSION}; start a fresh continual chain"
        )
    if int(getattr(encoder, "input_dim", -1)) != shape[0]:
        raise RuntimeError(
            f"{source}: encoder input_dim={getattr(encoder, 'input_dim', None)} "
            f"does not match observation dim {shape[0]}"
        )
    if int(getattr(encoder, "output_dim", -1)) != int(output_dim):
        raise RuntimeError(
            f"{source}: encoder output_dim={getattr(encoder, 'output_dim', None)} "
            f"does not match requested shared_dim {output_dim}"
        )

    with torch.no_grad():
        probe = torch.zeros(2, shape[0], dtype=torch.float32)
        device = next(encoder.parameters()).device
        out = encoder(probe.to(device))
    if tuple(out.shape) != (2, int(output_dim)):
        raise RuntimeError(
            f"{source}: encoder produced {tuple(out.shape)}, expected (2, {output_dim})"
        )


class Critic(nn.Module):
    """One SAC Q network: Q(s, a) from encoder features and an action."""

    def __init__(self, shared_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(int(shared_dim) + int(act_dim), hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([features, action], dim=-1))


class TwinCritic(nn.Module):
    """SAC's clipped double-Q pair, kept task-local and out of the pool."""

    def __init__(self, shared_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = Critic(shared_dim, act_dim, hidden_dim)
        self.q2 = Critic(shared_dim, act_dim, hidden_dim)

    def forward(self, features: torch.Tensor, action: torch.Tensor):
        return self.q1(features, action), self.q2(features, action)

    def min_q(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(features, action)
        return torch.min(q1, q2)
