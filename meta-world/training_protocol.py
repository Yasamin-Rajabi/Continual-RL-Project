"""Pure, simulator-independent protocol helpers used by all SAC runners."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TaskBudget:
    total: int
    frozen_tail: int

    def __post_init__(self):
        if self.total <= 0 or not 0 <= self.frozen_tail < self.total:
            raise ValueError("Require 0 <= B < Delta: buffer steps must be INSIDE total_timesteps")

    @property
    def training(self):
        return self.total - self.frozen_tail


def mixture_warmup_active(step, learning_starts, warmup_steps, fusion_mode, pool_length):
    return bool(fusion_mode == "weight_delta" and pool_length > 0 and warmup_steps > 0
                and step < learning_starts + warmup_steps)


def bounded_buffer(buffer, max_rows):
    """Apply one set of sample indices to every row-aligned memory field."""
    if buffer is None or len(buffer["obs"]) <= max_rows:
        return buffer
    n = len(buffer["obs"])
    indices = np.random.choice(n, max_rows, replace=False)
    return {key: value[indices] if isinstance(value, np.ndarray) and len(value) == n else value
            for key, value in buffer.items()}
