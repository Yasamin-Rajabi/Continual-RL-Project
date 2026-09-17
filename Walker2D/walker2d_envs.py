"""Walker2D dynamics environments without appended task/context observations.

The legacy task-observation wrapper remains as an identity adapter for import
compatibility. get_task and encoder pretraining do not apply task conditioning.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    import mujoco
    from gymnasium.envs.mujoco.walker2d_v5 import Walker2dEnv
except ImportError as exc:  # Keep pure-Python imports actionable without MuJoCo.
    gym = None
    mujoco = None
    Walker2dEnv = object
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


TASK_VECTOR_DIM = 6
_RIGHT_JOINTS = ("thigh_joint", "leg_joint", "foot_joint")
_LEFT_JOINTS = ("thigh_left_joint", "leg_left_joint", "foot_left_joint")
_ACTUATED_JOINTS = _RIGHT_JOINTS + _LEFT_JOINTS
_FOOT_JOINTS = ("foot_joint", "foot_left_joint")


def _name_id(model, obj_type, name: str) -> int:
    idx = int(mujoco.mj_name2id(model, obj_type, name))
    if idx < 0:
        raise RuntimeError(
            f"Walker2d-v5 model is missing expected MuJoCo object {name!r}. "
            "This benchmark targets Gymnasium Walker2d-v5; check the installed "
            "Gymnasium version or custom XML."
        )
    return idx


class Walker2dDynamicsEnv(Walker2dEnv):
    """Walker2d-v5 with fixed task-specific dynamics and velocity tracking.

    The health/termination semantics intentionally remain those of Gymnasium's
    Walker2d-v5.  Only the model parameters listed in the module docstring and
    the forward reward are changed.
    """

    def __init__(
        self,
        target_velocity: float = 1.5,
        right_mass_scale: float = 1.0,
        left_mass_scale: float = 1.0,
        foot_friction_scale: float = 1.0,
        joint_damping_scale: float = 1.0,
        actuator_strength_scale: float = 1.0,
        success_tolerance: float = 0.30,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Walker2D tasks require gymnasium[mujoco] and mujoco. "
                "Install requirements.txt first."
            ) from _IMPORT_ERROR

        if not np.isfinite(target_velocity):
            raise ValueError(f"target_velocity must be finite, got {target_velocity}")
        if not np.isfinite(success_tolerance) or float(success_tolerance) <= 0.0:
            raise ValueError(
                f"success_tolerance must be finite and > 0, got {success_tolerance}"
            )

        scales = {
            "right_mass_scale": right_mass_scale,
            "left_mass_scale": left_mass_scale,
            "foot_friction_scale": foot_friction_scale,
            "joint_damping_scale": joint_damping_scale,
            "actuator_strength_scale": actuator_strength_scale,
        }
        for key, value in scales.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{key} must be finite and > 0, got {value}")

        # Keep Gymnasium's normal Walker2D health termination.  The standard
        # forward-reward coefficient is irrelevant because step() below uses a
        # target-velocity tracking reward instead.
        super().__init__(
            render_mode=render_mode,
            ctrl_cost_weight=ctrl_cost_weight,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=True,
            **kwargs,
        )

        self.target_velocity = float(target_velocity)
        self.right_mass_scale = float(right_mass_scale)
        self.left_mass_scale = float(left_mass_scale)
        self.foot_friction_scale = float(foot_friction_scale)
        self.joint_damping_scale = float(joint_damping_scale)
        self.actuator_strength_scale = float(actuator_strength_scale)
        self.success_tolerance = float(success_tolerance)

        # Apply perturbations exactly once to this freshly-created model.
        self._apply_dynamics_perturbation()

    @property
    def task_name(self) -> str:
        return (
            "Walker2dDynamics("
            f"v={self.target_velocity:g}, "
            f"mR={self.right_mass_scale:g}, mL={self.left_mass_scale:g}, "
            f"mu={self.foot_friction_scale:g}, "
            f"damp={self.joint_damping_scale:g}, "
            f"motor={self.actuator_strength_scale:g})"
        )

    def _body_id_for_joint(self, joint_name: str) -> int:
        joint_id = _name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_bodyid[joint_id])

    def _scale_body_masses_for_joints(self, joint_names, scale: float) -> None:
        for joint_name in joint_names:
            body_id = self._body_id_for_joint(joint_name)
            self.model.body_mass[body_id] *= scale
            # Inertias must scale with mass to preserve each body's shape and
            # radius of gyration. Scaling mass alone gives an unphysical task.
            self.model.body_inertia[body_id] *= scale

    def _apply_dynamics_perturbation(self) -> None:
        # The joint names are part of Walker2d's documented action interface.
        # Derive the connected body ids from those joints instead of relying on
        # less-stable body/geom names in the XML.
        self._scale_body_masses_for_joints(_RIGHT_JOINTS, self.right_mass_scale)
        self._scale_body_masses_for_joints(_LEFT_JOINTS, self.left_mass_scale)

        # Scale all geoms attached to each foot-joint body.
        for foot_joint in _FOOT_JOINTS:
            body_id = self._body_id_for_joint(foot_joint)
            geom_adr = int(self.model.body_geomadr[body_id])
            geom_num = int(self.model.body_geomnum[body_id])
            if geom_num <= 0:
                raise RuntimeError(f"Walker2d body for {foot_joint!r} has no geoms")
            geom_ids = np.arange(geom_adr, geom_adr + geom_num, dtype=np.int64)
            # MuJoCo geom_friction columns are sliding, torsional, rolling.
            # Change sliding friction only; this is the dominant walking-contact
            # term and avoids silently changing unrelated contact mechanics.
            self.model.geom_friction[geom_ids, 0] *= self.foot_friction_scale

        for name in _ACTUATED_JOINTS:
            joint_id = _name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            self.model.dof_damping[dof_adr] *= self.joint_damping_scale

        # Walker2D has exactly six motor actuators. Scaling the full gear row
        # works for the canonical model and keeps action bounds unchanged.
        self.model.actuator_gear[:] *= self.actuator_strength_scale

        # Recompute model constants after mass/inertia edits, then synchronize
        # derived data.  This makes the perturbation robust rather than relying
        # on stale subtree/inertial quantities from XML compilation.
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        x_before = float(self.data.qpos[0])
        self.do_simulation(action, self.frame_skip)
        x_after = float(self.data.qpos[0])
        x_velocity = (x_after - x_before) / self.dt

        healthy = bool(self.is_healthy)
        healthy_reward = float(self.healthy_reward)
        velocity_error = abs(x_velocity - self.target_velocity)
        reward_velocity = -velocity_error
        ctrl_cost = float(self.control_cost(action))
        reward = healthy_reward + reward_velocity - ctrl_cost
        terminated = (not healthy) and self._terminate_when_unhealthy
        observation = self._get_obs()

        info = {
            "x_position": x_after,
            "z_distance_from_origin": float(self.data.qpos[1] - self.init_qpos[1]),
            "x_velocity": x_velocity,
            "target_velocity": self.target_velocity,
            "velocity_error": velocity_error,
            "reward_velocity": reward_velocity,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "healthy": float(healthy),
            # Success is intentionally stricter than just surviving: the gait
            # must remain healthy and be close to the common target speed.
            "success": float(healthy and velocity_error <= self.success_tolerance),
            "right_mass_scale": self.right_mass_scale,
            "left_mass_scale": self.left_mass_scale,
            "foot_friction_scale": self.foot_friction_scale,
            "joint_damping_scale": self.joint_damping_scale,
            "actuator_strength_scale": self.actuator_strength_scale,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info



def make_task_specific_observation(task, observation: np.ndarray) -> np.ndarray:
    """Legacy compatibility helper. Task concatenation is disabled."""
    del task
    # Previously: np.concatenate([observation, task_vec], axis=-1).
    return np.asarray(observation)


_ObservationWrapperBase = gym.ObservationWrapper if gym is not None else object


class TaskConditionedObservationWrapper(_ObservationWrapperBase):
    """Deprecated identity wrapper; it never appends task parameters."""

    def __init__(self, env, task):
        if _IMPORT_ERROR is not None:
            raise ImportError("Install gymnasium[mujoco] and mujoco") from _IMPORT_ERROR
        super().__init__(env)
        # Do not store or expose task context. Observation bounds stay raw.
        del task

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32)
