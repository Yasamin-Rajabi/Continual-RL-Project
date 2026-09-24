"""Pure, simulator-independent protocol helpers shared by every runner.

The interaction budget is the part of a continual-RL comparison that is
easiest to get quietly wrong, so it is defined once, here, and every method
imports it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskBudget:
    """Delta total environment transitions per task, with B of them frozen.

    The frozen tail B is *inside* Delta, never additional.  If it were extra,
    methods that retain replay states would quietly receive more environment
    interaction than methods that do not, and every comparison against them
    would be invalid.
    """

    total: int
    frozen_tail: int

    def __post_init__(self):
        if self.total <= 0 or not 0 <= self.frozen_tail < self.total:
            raise ValueError(
                "require 0 <= B < Delta: the frozen tail must be inside total_timesteps"
            )

    @property
    def training(self) -> int:
        return self.total - self.frozen_tail


def mixture_warmup_active(step, learning_starts, warmup_steps, fusion_mode, pool_length) -> bool:
    """True while only routing weights should move, before the novel expert does."""
    return bool(
        fusion_mode == "weight_delta"
        and pool_length > 0
        and warmup_steps > 0
        and step < learning_starts + warmup_steps
    )


def bounded_buffer(buffer, max_rows):
    """Apply one set of sampled indices to every row-aligned field."""
    if buffer is None or len(buffer["obs"]) <= max_rows:
        return buffer
    n = len(buffer["obs"])
    indices = np.random.choice(n, int(max_rows), replace=False)
    return {
        key: value[indices]
        if isinstance(value, np.ndarray) and len(value) == n
        else value
        for key, value in buffer.items()
    }
