"""Ant target-velocity tasks, with an optional fixed external wind.

Deliberately mirrors halfcheetah_envs.py one-for-one so that nothing
downstream has to learn a second pattern. Same reward, same info keys, same
task-conditioning wrapper (imported from halfcheetah_envs, not duplicated).

    reward = -|v_x - target_velocity| - ctrl_cost_weight * ||action||^2

TWO CONSTRUCTOR CHOICES THAT ARE NOT COSMETIC
---------------------------------------------
1. ``terminate_when_unhealthy=False``.
   The reward above is always <= 0. If the episode can end early when the
   robot falls, the optimal policy is to fall over immediately and stop
   accumulating negative reward. HalfCheetah has no terminal state at all,
   which is why the existing suite never hit this. Ant does, so we switch it
   off. Keeping it on would silently invert the objective.

   This also preserves r_max = 0, which metrics.py's FT_return derivation
   depends on ("reward = -|v_error| - ctrl_cost is always <= 0"). Adding a
   healthy_reward instead of disabling termination would break that
   derivation and require implementing the full survey formula with r_min.

2. ``include_cfrc_ext_in_observation=False``.
   Drops Ant-v5's 78 contact-force dimensions (105 -> 27). They are almost
   always zero and cost training throughput for no benefit here.

Final shapes: obs = 27 + 3 task-conditioning = 30, act = 8.

WIND. HalfCheetah is planar, so its wind is applied in (x, z). Ant moves in
3D, so the two wind components are applied in the horizontal plane (x, y):
a crosswind is the meaningful perturbation for a quadruped. The task vector
stays 2-D either way, so TaskConditionedObservationWrapper is unchanged.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    import mujoco
    try:
        from gymnasium.envs.mujoco.ant_v5 import AntEnv
    except ImportError:
        from gymnasium.envs.mujoco.ant_v4 import AntEnv
except ImportError as exc:
    gym = None
    mujoco = None
    AntEnv = object
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class AntVelocityEnv(AntEnv):
    """Ant whose objective is to track a fixed target x velocity."""

    def __init__(
        self,
        target_velocity: float,
        wind: Tuple[float, float] = (0.0, 0.0),
        success_tolerance: float = 0.1,
        ctrl_cost_weight: float = 0.05,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Ant tasks require gymnasium[mujoco] and mujoco. "
                "Install the repository requirements first."
            ) from _IMPORT_ERROR

        # v5 exposes these; v4 does not know include_cfrc_ext_in_observation
        # (it used use_contact_forces instead) so fall back gracefully.
        base_kwargs = dict(
            render_mode=render_mode,
            ctrl_cost_weight=ctrl_cost_weight,
            terminate_when_unhealthy=False,
            **kwargs,
        )
        try:
            super().__init__(include_cfrc_ext_in_observation=False, **base_kwargs)
        except TypeError:
            super().__init__(**base_kwargs)

        self.target_velocity = float(target_velocity)
        self.wind_x = float(wind[0])
        self.wind_y = float(wind[1])
        self.success_tolerance = float(success_tolerance)
        self.velocity_ctrl_cost_weight = float(ctrl_cost_weight)

    @property
    def task_name(self) -> str:
        if abs(self.wind_x) < 1e-12 and abs(self.wind_y) < 1e-12:
            return f"AntVel(v={self.target_velocity:g})"
        return (
            f"AntWindVel(v={self.target_velocity:g},"
            f"wx={self.wind_x:g},wy={self.wind_y:g})"
        )

    def _simulate(self, action: np.ndarray) -> None:
        """Run MuJoCo frames, applying the task's fixed wind each frame."""
        if abs(self.wind_x) < 1e-12 and abs(self.wind_y) < 1e-12:
            self.do_simulation(action, self.frame_skip)
            return

        ctrlrange = self.model.actuator_ctrlrange
        if ctrlrange is not None:
            action = np.clip(action, ctrlrange[:, 0], ctrlrange[:, 1])
        self.data.ctrl[:] = action
        wind_force = np.asarray(
            [self.wind_x, self.wind_y, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        for _ in range(self.frame_skip):
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
            # Same three keys as the HalfCheetah suite so nothing downstream
            # needs a branch; for Ant the second component is a y-wind.
            "wind_x": self.wind_x,
            "wind_z": self.wind_y,
            "success": float(velocity_error <= self.success_tolerance),
        }

        if self.render_mode == "human":
            self.render()

        # terminate_when_unhealthy=False, so there is no terminal state; the
        # TimeLimit wrapper in tasks.py supplies the 1000-step truncation.
        return observation, reward, False, False, info


class AntVelEnv(AntVelocityEnv):
    """Named convenience wrapper for the target-velocity-only suite."""

    def __init__(self, target_velocity: float, **kwargs):
        super().__init__(target_velocity=target_velocity, wind=(0.0, 0.0), **kwargs)


class AntWindVelEnv(AntVelocityEnv):
    """Named convenience wrapper for joint target-velocity + hidden-wind tasks."""

    def __init__(self, target_velocity: float, wind: Tuple[float, float], **kwargs):
        super().__init__(target_velocity=target_velocity, wind=wind, **kwargs)


# ---------------------------------------------------------------------------
# Velocity calibration helper -- READ THIS BEFORE THE FIRST REAL RUN
# ---------------------------------------------------------------------------
def measure_reachable_velocity(total_timesteps: int = 150_000, seed: int = 0) -> float:
    """Train plain forward-reward SAC on Ant and report the velocity reached.

    The HalfCheetah suite's targets (0.5 .. 3.0 m/s) were chosen for a robot
    that reaches roughly 5-8 m/s. Ant is substantially slower. Copying those
    numbers across would make the fast tasks unreachable, collapsing several
    of them onto the same behaviour ("run flat out") and destroying the
    benchmark's ability to distinguish tasks at all.

    This is a stub on purpose: run it in your own environment, read off the
    achievable v_max, then set _ANT_VELOCITIES in tasks.py to roughly
    {0.15, 0.3, 0.45, 0.6, 0.75, 0.9} * v_max and success_tolerance to
    0.1 * v_max.
    """
    raise NotImplementedError(
        "Run a plain SAC baseline on Ant-v5 (standard forward reward) and read "
        "off info['x_velocity'] at convergence. See section 4.4 of the analysis doc."
    )
