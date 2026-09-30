"""Fixed-capacity SAC replay buffer.

Preallocated NumPy arrays, no Python lists, no per-step concatenation.  One
PointMaze transition is about 120 bytes, so a full 10^5 buffer costs ~12 MB --
but the same discipline is what keeps a pixel version from doubling its own
memory every time it samples.
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, seed: int = 0):
        capacity = int(capacity)
        if capacity < 1:
            raise ValueError("replay capacity must be >= 1")
        self.capacity = capacity
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.obs = np.zeros((capacity, self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, self.act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self._rng = np.random.default_rng(int(seed))

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        idx = self._rng.integers(0, self.size, size=int(batch_size))
        to = lambda a: torch.as_tensor(a[idx], device=device)  # noqa: E731
        return (
            to(self.obs),
            to(self.actions),
            torch.as_tensor(self.rewards[idx], device=device).unsqueeze(-1),
            to(self.next_obs),
            torch.as_tensor(self.dones[idx], device=device).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return self.size
