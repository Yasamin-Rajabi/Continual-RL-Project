"""Evaluation package for Continual RL Atari benchmarks."""

from .benchmark_protocol import (
    METHODS,
    canonical_env_name,
    canonical_method,
    checkpoint_dir,
    event_dir,
    success_threshold,
)
from .checkpoint_evaluation import (
    evaluate_checkpoint,
    evaluate_live_agent,
    load_policy_for_task,
)

__all__ = [
    "METHODS",
    "canonical_env_name",
    "canonical_method",
    "checkpoint_dir",
    "event_dir",
    "success_threshold",
    "evaluate_checkpoint",
    "evaluate_live_agent",
    "load_policy_for_task",
]