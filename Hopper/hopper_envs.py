"""New Hopper-v5 velocity-tracking dynamics suite; no oracle context inputs.

This is a new benchmark extension, not a reconstruction of a Hopper experiment
in the original archive. Gymnasium's health/termination rules are preserved.
"""
from __future__ import annotations
import numpy as np
try:
    import mujoco
    from gymnasium.envs.mujoco.hopper_v5 import HopperEnv
except ImportError as exc:
    HopperEnv = object
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class HopperDynamicsEnv(HopperEnv):
    def __init__(self, target_velocity=1.5, mass_scale=1.0,
                 foot_friction_scale=1.0, joint_damping_scale=1.0,
                 actuator_strength_scale=1.0, success_tolerance=0.30,
                 render_mode=None, **kwargs):
        if _IMPORT_ERROR is not None:
            raise ImportError("Install gymnasium[mujoco] and mujoco for Hopper") from _IMPORT_ERROR
        if not np.isfinite(target_velocity):
            raise ValueError("target_velocity must be finite")
        for key, value in (("mass_scale", mass_scale), ("foot_friction_scale", foot_friction_scale),
                           ("joint_damping_scale", joint_damping_scale),
                           ("actuator_strength_scale", actuator_strength_scale),
                           ("success_tolerance", success_tolerance)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        super().__init__(render_mode=render_mode, **kwargs)
        self.target_velocity = float(target_velocity)
        self.success_tolerance = float(success_tolerance)
        # Scale mass AND inertia, keeping each body's radius of gyration fixed.
        self.model.body_mass[1:] *= mass_scale
        self.model.body_inertia[1:] *= mass_scale
        for name in ("thigh_joint", "leg_joint", "foot_joint"):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"Canonical Hopper-v5 joint {name} not found")
            self.model.dof_damping[self.model.jnt_dofadr[joint_id]] *= joint_damping_scale
            if name == "foot_joint":
                body = self.model.jnt_bodyid[joint_id]
                adr, count = self.model.body_geomadr[body], self.model.body_geomnum[body]
                self.model.geom_friction[adr:adr + count, 0] *= foot_friction_scale
        self.model.actuator_gear[:] *= actuator_strength_scale
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        velocity = float(info["x_velocity"])
        error = abs(velocity - self.target_velocity)
        healthy = bool(self.is_healthy)
        reward_survive = float(self.healthy_reward)
        reward_ctrl = -float(self.control_cost(action))
        reward = reward_survive - error + reward_ctrl
        info.update(velocity_error=error, reward_velocity=-error,
                    reward_survive=reward_survive, reward_ctrl=reward_ctrl,
                    healthy=float(healthy),
                    success=float(healthy and error <= self.success_tolerance))
        return obs, reward, terminated, truncated, info
