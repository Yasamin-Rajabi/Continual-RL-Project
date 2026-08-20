"""Deterministic continual HalfCheetah task suites.

Two suites are provided:
- halfcheetah_vel: target velocity changes across tasks.
- halfcheetah_wind_vel: both target velocity and a hidden fixed wind change.

get_task() wraps every environment with
halfcheetah_envs.TaskConditionedObservationWrapper, which appends
[target_velocity, wind_x, wind_z] to every observation (wind is (0,0) for
halfcheetah_vel tasks). So all tasks share the same 20-D observation (17
raw MuJoCo dims + 3 task-conditioning dims) and 6-D continuous action space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class HalfCheetahTask:
    target_velocity: float
    wind: Tuple[float, float] = (0.0, 0.0)

    def label(self, suite: str) -> str:
        if suite == "halfcheetah_vel":
            return f"HC-Vel {self.target_velocity:g}m/s"
        return (
            f"HC-WindVel v={self.target_velocity:g}, "
            f"wind=({self.wind[0]:g},{self.wind[1]:g})"
        )


# Eight distinct tasks is enough to force several merges with the recommended
# pool_size=5, while keeping a 2-pass continual benchmark computationally sane.
_VELOCITIES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 1.25, 2.25)
_WIND_PAIRS = (
    (-2.5, 0.0),
    (2.5, 0.0),
    (0.0, -5.0),
    (0.0, 5.0),
    (-1.25, -2.5),
    (1.25, 2.5),
    (-2.5, 5.0),
    (2.5, -5.0),
)

TASK_SUITES: Dict[str, List[HalfCheetahTask]] = {
    "halfcheetah_vel": [HalfCheetahTask(v) for v in _VELOCITIES],
    "halfcheetah_wind_vel": [
        HalfCheetahTask(v, wind=w) for v, w in zip(_VELOCITIES, _WIND_PAIRS)
    ],
}

# Paper-style second pass through the same tasks to expose retention/relearning.
DEFAULT_CONTINUAL_SEQUENCE = tuple(range(6)) + tuple(range(6))


def available_task_suites():
    return tuple(TASK_SUITES.keys())


def get_task_name(task_id: int, task_suite: str = "halfcheetah_vel") -> str:
    task = TASK_SUITES[task_suite][task_id]
    return task.label(task_suite)


def get_task_spec(task_id: int, task_suite: str = "halfcheetah_vel") -> HalfCheetahTask:
    return TASK_SUITES[task_suite][task_id]


def get_task(task_id: int, task_suite: str = "halfcheetah_vel", render: bool = False):
    import gymnasium as gym
    from halfcheetah_envs import HalfCheetahVelEnv, HalfCheetahWindVelEnv, TaskConditionedObservationWrapper

    task = get_task_spec(task_id, task_suite)
    env_cls = HalfCheetahVelEnv if task_suite == "halfcheetah_vel" else HalfCheetahWindVelEnv
    kwargs = {
        "target_velocity": task.target_velocity,
        "render_mode": "human" if render else None,
    }
    if task_suite == "halfcheetah_wind_vel":
        kwargs["wind"] = task.wind
    env = env_cls(**kwargs)
    # Every observation from this point on (reset AND step) carries
    # [target_velocity, wind_x, wind_z] appended -- see
    # halfcheetah_envs.make_task_specific_observation for why the critic
    # needs this too, not just the actor. Applied once, here, at
    # construction time -- everything downstream (replay buffer, Actor,
    # SoftQNetwork, SyncVectorEnv batching, eval loops,
    # metrics.evaluate_checkpoint) reads obs_dim dynamically from
    # observation_space, so no other file needs to change.
    env = TaskConditionedObservationWrapper(env, task)
    # Directly instantiating a MuJoCo class bypasses gym.make's TimeLimit.
    return gym.wrappers.TimeLimit(env, max_episode_steps=1000)


if __name__ == "__main__":
    for suite in available_task_suites():
        print(suite)
        for idx, task in enumerate(TASK_SUITES[suite]):
            print(f"  {idx}: {task.label(suite)}")
