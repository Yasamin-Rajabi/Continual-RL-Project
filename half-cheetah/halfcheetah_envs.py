"""HalfCheetah target-velocity tasks, with an optional fixed external wind.

The velocity reward follows the widely used HalfCheetahVel meta-RL benchmark:
    reward = -|v_x - target_velocity| - 0.05 * ||action||^2

The wind implementation follows the standard MuJoCo mechanism of applying an
external body force through ``data.xfrc_applied`` during every simulation frame.
Wind is *not* included in the observation; each continual task has a fixed wind.
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
