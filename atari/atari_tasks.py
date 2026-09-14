"""Deterministic continual Atari task suites.

Two suites are provided:
- ``space_invaders``: ALE/SpaceInvaders-v5 modes 0..9
- ``freeway``: ALE/Freeway-v5 modes 0..7

These mode sets come from the Atari setup already used by this project.  Every
task in a suite has the same image preprocessing and discrete action interface;
only the ALE mode changes.

Unlike ``halfcheetah_envs.py``, task identity is NOT concatenated to the image
observation, because the existing Atari agents expect a 4-frame CNN input.
Task metadata is available through the environment attributes and ``info``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class AtariTask:
    env_id: str
    mode: int

    @property
    def game(self) -> str:
        return self.env_id.split("/")[-1].split("-")[0]

    def label(self, suite: str | None = None) -> str:
        del suite
        return f"{self.game} mode {self.mode}"


SPACE_INVADERS_ENV_ID = "ALE/SpaceInvaders-v5"
FREEWAY_ENV_ID = "ALE/Freeway-v5"

SPACE_INVADERS_MODES: Tuple[int, ...] = tuple(range(10))
FREEWAY_MODES: Tuple[int, ...] = tuple(range(8))


TASK_SUITES: Dict[str, List[AtariTask]] = {
    "space_invaders": [
        AtariTask(SPACE_INVADERS_ENV_ID, mode) for mode in SPACE_INVADERS_MODES
    ],
    "freeway": [
        AtariTask(FREEWAY_ENV_ID, mode) for mode in FREEWAY_MODES
    ],
}

# Backward-compatible form used by the existing task_utils.py/run_experiments.py.
TASKS = {
    SPACE_INVADERS_ENV_ID: list(SPACE_INVADERS_MODES),
    FREEWAY_ENV_ID: list(FREEWAY_MODES),
}

# One pass is the current Atari experiment protocol.  Use
# get_continual_sequence(..., repeats=2) if you want the same second-pass
# retention/relearning stress test used by the HalfCheetah benchmark.
DEFAULT_CONTINUAL_SEQUENCES = {
    suite: tuple(range(len(tasks))) for suite, tasks in TASK_SUITES.items()
}


_SUITE_ALIASES = {
    "space_invaders": "space_invaders",
    "spaceinvaders": "space_invaders",
    "SpaceInvaders": "space_invaders",
    SPACE_INVADERS_ENV_ID: "space_invaders",
    "freeway": "freeway",
    "Freeway": "freeway",
    FREEWAY_ENV_ID: "freeway",
}


def _canonical_suite(task_suite: str) -> str:
    try:
        return _SUITE_ALIASES[task_suite]
    except KeyError as exc:
        valid = ", ".join(TASK_SUITES)
        raise ValueError(
            f"Unknown Atari task suite {task_suite!r}. "
            f"Canonical suites are: {valid}."
        ) from exc


def available_task_suites() -> Tuple[str, ...]:
    return tuple(TASK_SUITES.keys())


def num_tasks(task_suite: str) -> int:
    suite = _canonical_suite(task_suite)
    return len(TASK_SUITES[suite])


def get_continual_sequence(
    task_suite: str,
    *,
    repeats: int = 1,
) -> Tuple[int, ...]:
    """Return task IDs in deterministic mode order.

    ``repeats=1`` reproduces the current Atari protocol.
    ``repeats=2`` gives a second pass through the same tasks, analogous to the
    HalfCheetah continual sequence used for retention/relearning evaluation.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    suite = _canonical_suite(task_suite)
    return DEFAULT_CONTINUAL_SEQUENCES[suite] * int(repeats)


def get_task_name(
    task_id: int,
    task_suite: str = "freeway",
) -> str:
    return get_task_spec(task_id, task_suite).label()


def get_task_spec(
    task_id: int,
    task_suite: str = "freeway",
) -> AtariTask:
    suite = _canonical_suite(task_suite)
    tasks = TASK_SUITES[suite]

    if task_id < 0 or task_id >= len(tasks):
        raise IndexError(
            f"task_id={task_id} is invalid for {suite!r}; "
            f"expected 0..{len(tasks) - 1}."
        )
    return tasks[task_id]


def get_task(
    task_id: int,
    task_suite: str = "freeway",
    render: bool = False,
    *,
    capture_video: bool = False,
    video_folder: str | None = None,
    **env_kwargs,
):
    """Construct one fully preprocessed Atari task."""
    from atari_envs import make_atari_env

    task = get_task_spec(task_id, task_suite)
    return make_atari_env(
        task.env_id,
        task.mode,
        task_id=task_id,
        render=render,
        capture_video=capture_video,
        video_folder=video_folder,
        **env_kwargs,
    )


def make_task_thunk(
    task_id: int,
    task_suite: str = "freeway",
    *,
    capture_video: bool = False,
    video_folder: str | None = None,
    **env_kwargs,
):
    """Create a thunk suitable for ``gym.vector.SyncVectorEnv``."""
    from atari_envs import make_atari_env_thunk

    task = get_task_spec(task_id, task_suite)
    return make_atari_env_thunk(
        task.env_id,
        task.mode,
        task_id=task_id,
        capture_video=capture_video,
        video_folder=video_folder,
        **env_kwargs,
    )


if __name__ == "__main__":
    import sys
    import numpy as np

    for suite in available_task_suites():
        print(suite)
        for idx, task in enumerate(TASK_SUITES[suite]):
            print(f"  {idx}: {task.label()}")

    if "--check" in sys.argv:
        for suite in available_task_suites():
            observation_shapes = set()
            action_sizes = set()

            for idx in range(num_tasks(suite)):
                env = get_task(idx, suite)
                obs, info = env.reset(seed=0)

                assert tuple(obs.shape) == tuple(env.observation_space.shape)
                assert info["task_id"] == idx
                assert info["mode"] == get_task_spec(idx, suite).mode

                env.action_space.seed(0)
                obs2, _, _, _, info2 = env.step(env.action_space.sample())
                assert tuple(obs2.shape) == tuple(env.observation_space.shape)
                assert info2["task_id"] == idx

                observation_shapes.add(tuple(env.observation_space.shape))
                action_sizes.add(int(env.action_space.n))
                env.close()

            assert len(observation_shapes) == 1, (
                f"{suite}: inconsistent observation shapes "
                f"{observation_shapes}"
            )
            assert len(action_sizes) == 1, (
                f"{suite}: inconsistent action sizes {action_sizes}"
            )

            print(
                f"[ok] {suite}: obs={observation_shapes.pop()}, "
                f"actions={action_sizes.pop()}"
            )
