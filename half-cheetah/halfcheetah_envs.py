"""HalfCheetah target-velocity tasks, with an optional fixed external wind.

The velocity reward follows the widely used HalfCheetahVel meta-RL benchmark:
    reward = -|v_x - target_velocity| - 0.05 * ||action||^2

The wind implementation follows the standard MuJoCo mechanism of applying an
external body force through ``data.xfrc_applied`` during every simulation frame.

The raw environment classes below do NOT include target_velocity/wind in the
observation themselves. tasks.get_task() wraps every env it builds with
TaskConditionedObservationWrapper (see the bottom of this file), which
appends [target_velocity, wind_x, wind_z] to every observation -- both the
actor and critic need this (see make_task_specific_observation's docstring
for why). Constructing HalfCheetahVelEnv/HalfCheetahWindVelEnv directly,
bypassing tasks.get_task(), skips this and gets the raw, task-blind
observation instead.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    import mujoco
    try:
        from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
    except ImportError:
        from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
except ImportError as exc:  # Make import errors actionable on non-MuJoCo machines.
    gym = None
    mujoco = None
    HalfCheetahEnv = object
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class HalfCheetahVelocityEnv(HalfCheetahEnv):
    """HalfCheetah whose objective is to track a fixed target x velocity."""

    def __init__(
        self,
        target_velocity: float,
        wind: Tuple[float, float] = (0.0, 0.0),
        success_tolerance: float = 0.2,
        ctrl_cost_weight: float = 0.05,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "HalfCheetah tasks require gymnasium[mujoco] and mujoco. "
                "Install the repository requirements first."
            ) from _IMPORT_ERROR
        super().__init__(render_mode=render_mode, ctrl_cost_weight=ctrl_cost_weight, **kwargs)
        self.target_velocity = float(target_velocity)
        self.wind_x = float(wind[0])
        self.wind_z = float(wind[1])
        self.success_tolerance = float(success_tolerance)
        self.velocity_ctrl_cost_weight = float(ctrl_cost_weight)

    @property
    def task_name(self) -> str:
        if abs(self.wind_x) < 1e-12 and abs(self.wind_z) < 1e-12:
            return f"HalfCheetahVel(v={self.target_velocity:g})"
        return (
            f"HalfCheetahWindVel(v={self.target_velocity:g},"
            f"wx={self.wind_x:g},wz={self.wind_z:g})"
        )

    def _simulate(self, action: np.ndarray) -> None:
        """Run MuJoCo frames, applying the task's fixed wind each frame."""
        if abs(self.wind_x) < 1e-12 and abs(self.wind_z) < 1e-12:
            self.do_simulation(action, self.frame_skip)
            return

        ctrlrange = self.model.actuator_ctrlrange
        if ctrlrange is not None:
            action = np.clip(action, ctrlrange[:, 0], ctrlrange[:, 1])
        self.data.ctrl[:] = action
        wind_force = np.asarray(
            [self.wind_x, 0.0, self.wind_z, 0.0, 0.0, 0.0], dtype=np.float64
        )
        for _ in range(self.frame_skip):
            # Broadcast the same environmental force to every body, matching
            # the common WindHalfCheetah construction.
            self.data.xfrc_applied[:] = wind_force
            mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def reset_model(self):
        obs = super().reset_model()
        self.data.xfrc_applied[:] = 0.0
        return obs

    def step(self, action):
        x_before = float(self.data.qpos[0])
        self._simulate(action)
        x_after = float(self.data.qpos[0])
        x_velocity = (x_after - x_before) / self.dt

        velocity_error = abs(x_velocity - self.target_velocity)
        reward_velocity = -velocity_error
        ctrl_cost = self.velocity_ctrl_cost_weight * float(np.square(action).sum())
        reward = reward_velocity - ctrl_cost
        observation = self._get_obs()

        info = {
            "x_position": x_after,
            "x_velocity": x_velocity,
            "target_velocity": self.target_velocity,
            "velocity_error": velocity_error,
            "reward_velocity": reward_velocity,
            "reward_ctrl": -ctrl_cost,
            "wind_x": self.wind_x,
            "wind_z": self.wind_z,
            "success": float(velocity_error <= self.success_tolerance),
        }

        if self.render_mode == "human":
            self.render()

        # HalfCheetah itself has no terminal state. The TimeLimit wrapper in
        # tasks.py supplies the 1000-step truncation used by Gymnasium.
        return observation, reward, False, False, info


class HalfCheetahVelEnv(HalfCheetahVelocityEnv):
    """Named convenience wrapper for the target-velocity-only suite."""

    def __init__(self, target_velocity: float, **kwargs):
        super().__init__(target_velocity=target_velocity, wind=(0.0, 0.0), **kwargs)


class HalfCheetahWindVelEnv(HalfCheetahVelocityEnv):
    """Named convenience wrapper for joint target-velocity + hidden-wind tasks."""

    def __init__(self, target_velocity: float, wind: Tuple[float, float], **kwargs):
        super().__init__(target_velocity=target_velocity, wind=wind, **kwargs)


def make_task_specific_observation(task, observation: np.ndarray) -> np.ndarray:
    """Concatenates [target_velocity, wind_x, wind_z] onto a raw observation,
    so both the actor AND critic receive an explicit task identifier.

    This matters beyond evaluation convenience: reward = -|v_x -
    target_velocity| - ctrl_cost depends directly on target_velocity, so two
    DIFFERENT tasks can reach an IDENTICAL (obs, action) pair with genuinely
    DIFFERENT true Q-values. Without this signal, the critic is forced to fit
    one value to two different targets for the same input, which corrupts
    the TD-error signal driving the actor's gradient -- not just an
    evaluation-fairness issue.

    `task` is any object exposing `.target_velocity` (float) and `.wind`
    (a 2-tuple) -- e.g. tasks.HalfCheetahTask. wind is (0.0, 0.0) for
    halfcheetah_vel tasks (HalfCheetahTask's own default), so this adds
    exactly 3 dims uniformly across both suites. Works for a single
    observation (1-D array) or a batch (N-D array; the task vector
    broadcasts across every leading dim).
    """
    observation = np.asarray(observation)
    task_vec = np.asarray(
        [task.target_velocity, task.wind[0], task.wind[1]], dtype=observation.dtype
    )
    if observation.ndim == 1:
        return np.concatenate([observation, task_vec], axis=-1)
    task_vec = np.broadcast_to(task_vec, observation.shape[:-1] + (3,))
    return np.concatenate([observation, task_vec], axis=-1)


# Mirrors the module-level gym/mujoco import guard above: if gymnasium isn't
# installed, gym is None and HalfCheetahEnv falls back to `object` -- do the
# same here so importing this module doesn't crash on a machine that only
# needs, say, make_task_specific_observation and not the live env classes.
_ObservationWrapperBase = gym.ObservationWrapper if gym is not None else object


class TaskConditionedObservationWrapper(_ObservationWrapperBase):
    """Applies make_task_specific_observation to every observation this env
    returns.

    gym.ObservationWrapper calls self.observation(...) automatically on
    BOTH reset() and step(), so wrapping the env ONCE here (see
    tasks.get_task) is enough -- no other file needs to call
    make_task_specific_observation directly. Everything downstream (the
    replay buffer, Actor, SoftQNetwork, gym.vector.SyncVectorEnv batching,
    eval loops, metrics.evaluate_checkpoint) already reads obs_dim
    dynamically from observation_space, so the extra 3 dims propagate
    automatically with no further code changes.
    """

    def __init__(self, env, task):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "HalfCheetah tasks require gymnasium[mujoco] and mujoco. "
                "Install the repository requirements first."
            ) from _IMPORT_ERROR
        super().__init__(env)
        self.task = task
        low = np.concatenate(
            [self.observation_space.low, np.full(3, -np.inf, dtype=np.float32)]
        )
        high = np.concatenate(
            [self.observation_space.high, np.full(3, np.inf, dtype=np.float32)]
        )
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        return make_task_specific_observation(self.task, observation).astype(np.float32)
