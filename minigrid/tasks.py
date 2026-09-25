"""Continual MiniGrid task suites centred on DoorKey.

The default ``doorkey4`` suite is intentionally small enough for a short paper
run while still containing both related and structurally different navigation
tasks.  The 60k interaction budget used by ``job.sh`` is an experimental
budget, not a claim that every task is guaranteed to converge by that point;
run ``pilot_check.py`` (or a one-seed paper run) before spending the full grid.

Default sequence::

    0, 1, 2, 3, 0, 2, 1, 3

where the four task ids are DoorKey-5x5, Empty-Random-5x5, DoorKey-6x6 and
Unlock.  Every task appears twice and the default pool capacity is four, so a
full eight-occurrence run exercises bounded compression four times.  All tasks
use MiniGrid's common egocentric image observation and seven-action interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class MiniGridTask:
    """One task plus the imposed horizon and progress stages used for diagnostics."""

    env_id: str
    max_episode_steps: int
    note: str = ""
    progress_stages: Tuple[str, ...] = ("goal",)

    def label(self, suite: str = "") -> str:
        return self.env_id.replace("MiniGrid-", "").replace("-v0", "")


_DOORKEY4: Tuple[MiniGridTask, ...] = (
    MiniGridTask(
        "MiniGrid-DoorKey-5x5-v0", 100,
        "full key->door->goal chain",
        ("key", "door", "goal"),
    ),
    MiniGridTask(
        "MiniGrid-Empty-Random-5x5-v0", 60,
        "goal navigation without key or door",
        ("goal",),
    ),
    MiniGridTask(
        "MiniGrid-DoorKey-6x6-v0", 160,
        "same DoorKey structure on a larger board",
        ("key", "door", "goal"),
    ),
    MiniGridTask(
        "MiniGrid-Unlock-v0", 120,
        "key and door objective",
        ("key", "door"),
    ),
)

# Two-task suite for the smoke stage. Not a scientific suite: it exists to
# exercise the whole pipeline (chain, merge, retention, metrics) in minutes.
_SMOKE2: Tuple[MiniGridTask, ...] = (
    MiniGridTask("MiniGrid-Empty-Random-5x5-v0", 60, "smoke only", ("goal",)),
    MiniGridTask("MiniGrid-DoorKey-5x5-v0", 100, "smoke only", ("key", "door", "goal")),
)

# Optional wider suite if four tasks turn out to saturate too early.
_DOORKEY6: Tuple[MiniGridTask, ...] = _DOORKEY4 + (
    MiniGridTask("MiniGrid-LavaGapS5-v0", 80, "hazard-navigation task", ("goal",)),
    MiniGridTask("MiniGrid-DoorKey-5x5-v0", 100, "second DoorKey-5x5 instance", ("key", "door", "goal")),
)

TASK_SUITES: Dict[str, List[MiniGridTask]] = {
    "doorkey4": list(_DOORKEY4),
    "doorkey6": list(_DOORKEY6),
    "mg_smoke2": list(_SMOKE2),
}

DEFAULT_CONTINUAL_SEQUENCE = (0, 1, 2, 3, 0, 2, 1, 3)
DOORKEY6_CONTINUAL_SEQUENCE = (0, 1, 2, 3, 4, 5, 0, 2, 1, 5, 3, 4)
# 3 positions with pool_size 2 forces exactly one merge, which is the part of
# the pipeline most likely to fail silently.
SMOKE_CONTINUAL_SEQUENCE = (0, 1, 0)

SEQUENCES = {
    "doorkey4": DEFAULT_CONTINUAL_SEQUENCE,
    "doorkey6": DOORKEY6_CONTINUAL_SEQUENCE,
    "mg_smoke2": SMOKE_CONTINUAL_SEQUENCE,
}


def available_task_suites():
    return tuple(TASK_SUITES.keys())


def default_sequence(task_suite: str = "doorkey4"):
    return SEQUENCES.get(task_suite, DEFAULT_CONTINUAL_SEQUENCE)


def get_task_spec(task_id: int, task_suite: str = "doorkey4") -> MiniGridTask:
    return TASK_SUITES[task_suite][task_id]


def get_task_name(task_id: int, task_suite: str = "doorkey4") -> str:
    return get_task_spec(task_id, task_suite).label(task_suite)


def get_task(task_id: int, task_suite: str = "doorkey4", render: bool = False):
    """Build the environment for one task. Signature matches the HalfCheetah
    suite's get_task, so nothing downstream changes."""
    from minigrid_envs import make_env

    spec = get_task_spec(task_id, task_suite)
    return make_env(spec.env_id, max_episode_steps=spec.max_episode_steps, render=render, progress_stages=spec.progress_stages)


if __name__ == "__main__":
    import sys

    for suite in available_task_suites():
        print(f"{suite}  (sequence: {default_sequence(suite)})")
        for idx, task in enumerate(TASK_SUITES[suite]):
            print(f"  {idx}: {task.env_id:32s} H={task.max_episode_steps:4d}  {task.note}")
        print()

    if "--check" in sys.argv:
        import numpy as np

        check_pos = sys.argv.index("--check")
        requested = sys.argv[check_pos + 1] if len(sys.argv) > check_pos + 1 else "all"
        suites = available_task_suites() if requested == "all" else (requested,)
        if requested != "all" and requested not in TASK_SUITES:
            raise SystemExit(f"unknown suite for --check: {requested}")
        for suite in suites:
            shapes = set()
            for idx in range(len(TASK_SUITES[suite])):
                env = get_task(idx, task_suite=suite)
                env.reset(seed=0)
                _, _, _, _, info = env.step(env.action_space.sample())
                assert "success" in info, f"{suite}/{idx}: missing success"
                assert "task_error" in info, f"{suite}/{idx}: missing task_error"
                shapes.add((int(np.prod(env.observation_space.shape)), int(env.action_space.n)))
                env.close()
            assert len(shapes) == 1, f"{suite}: inconsistent spaces {shapes}"
            obs_dim, n_act = shapes.pop()
            print(f"[ok] {suite}: obs={obs_dim}, actions={n_act}, info keys present")
