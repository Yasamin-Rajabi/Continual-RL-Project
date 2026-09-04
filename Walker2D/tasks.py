"""Deterministic continual Walker2D dynamics benchmark.

The primary suite, ``walker2d_dynamics``, keeps one target velocity and changes
only the robot/contact dynamics.  This is intentional: each specialist solves
the same semantic task, while different valid gaits make parameter averaging a
meaningful consolidation stress test.

Every task is wrapped with ``TaskConditionedObservationWrapper``.  Walker2d-v5
has a 17-D raw observation; six task-conditioning values are appended, giving a
23-D observation and 6-D continuous action space for every task.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Walker2DTask:
    target_velocity: float = 1.5
    right_mass_scale: float = 1.0
    left_mass_scale: float = 1.0
    foot_friction_scale: float = 1.0
    joint_damping_scale: float = 1.0
    actuator_strength_scale: float = 1.0
    short_name: str = "nominal"

    def label(self, suite: str) -> str:
        del suite
        return (
            f"W2D-{self.short_name}: v={self.target_velocity:g}, "
            f"mR={self.right_mass_scale:g}, mL={self.left_mass_scale:g}, "
            f"mu={self.foot_friction_scale:g}, "
            f"damp={self.joint_damping_scale:g}, "
            f"motor={self.actuator_strength_scale:g}"
        )


# Moderate perturbations: strong enough to demand different gait compensation,
# but intentionally far from pathological values that would turn the benchmark
# into another Ant-like long-convergence problem.
_DYNAMICS_TASKS = [
    Walker2DTask(short_name="nominal"),
    Walker2DTask(right_mass_scale=1.25, left_mass_scale=0.90, short_name="right-heavy"),
    Walker2DTask(right_mass_scale=0.90, left_mass_scale=1.25, short_name="left-heavy"),
    Walker2DTask(foot_friction_scale=0.70, short_name="slippery-feet"),
    Walker2DTask(joint_damping_scale=1.50, short_name="high-damping"),
    Walker2DTask(actuator_strength_scale=0.80, short_name="weak-motors"),
]

# Optional stronger consolidation stress suite.  It remains in the directory so
# you can deepen the merge gap without changing code after the moderate suite is
# validated.  It is NOT run by the default Kaggle/recommended commands.
_MIXED_TASKS = [
    Walker2DTask(short_name="nominal"),
    Walker2DTask(
        right_mass_scale=1.20, left_mass_scale=0.90,
        foot_friction_scale=0.78, actuator_strength_scale=0.90,
        short_name="right-heavy-slip-weak",
    ),
    Walker2DTask(
        right_mass_scale=0.90, left_mass_scale=1.20,
        foot_friction_scale=0.78, actuator_strength_scale=0.90,
        short_name="left-heavy-slip-weak",
    ),
    Walker2DTask(
        right_mass_scale=1.10, left_mass_scale=1.10,
        joint_damping_scale=1.35, actuator_strength_scale=0.88,
        short_name="heavy-damped-weak",
    ),
    Walker2DTask(
        right_mass_scale=0.90, left_mass_scale=0.90,
        foot_friction_scale=1.20, actuator_strength_scale=1.10,
        short_name="light-grippy-strong",
    ),
    Walker2DTask(
        foot_friction_scale=0.80, joint_damping_scale=1.30,
        actuator_strength_scale=0.90,
        short_name="slip-damped-weak",
    ),
]

TASK_SUITES: Dict[str, List[Walker2DTask]] = {
    "walker2d_dynamics": _DYNAMICS_TASKS,
    "walker2d_mixed_dynamics": _MIXED_TASKS,
}

# Repeat the six tasks.  With the default pool_size=5 this guarantees pool
# pressure/merging and also measures retention/relearning on a second pass.
DEFAULT_CONTINUAL_SEQUENCE = tuple(range(len(_DYNAMICS_TASKS))) * 2


def available_task_suites():
    return tuple(TASK_SUITES.keys())


def get_task_name(task_id: int, task_suite: str = "walker2d_dynamics") -> str:
    return TASK_SUITES[task_suite][task_id].label(task_suite)


def get_task_spec(task_id: int, task_suite: str = "walker2d_dynamics") -> Walker2DTask:
    return TASK_SUITES[task_suite][task_id]


def get_task(task_id: int, task_suite: str = "walker2d_dynamics", render: bool = False):
    import gymnasium as gym
    from walker2d_envs import Walker2dDynamicsEnv, TaskConditionedObservationWrapper

    task = get_task_spec(task_id, task_suite)
    env = Walker2dDynamicsEnv(
        target_velocity=task.target_velocity,
        right_mass_scale=task.right_mass_scale,
        left_mass_scale=task.left_mass_scale,
        foot_friction_scale=task.foot_friction_scale,
        joint_damping_scale=task.joint_damping_scale,
        actuator_strength_scale=task.actuator_strength_scale,
        render_mode="human" if render else None,
    )
    env = TaskConditionedObservationWrapper(env, task)

    # Direct class construction bypasses gym.make's registry TimeLimit.
    return gym.wrappers.TimeLimit(env, max_episode_steps=1000)


if __name__ == "__main__":
    import sys
    import numpy as np

    for suite in available_task_suites():
        print(suite)
        for idx, task in enumerate(TASK_SUITES[suite]):
            print(f"  {idx}: {task.label(suite)}")

    if "--check" in sys.argv:
        for suite in available_task_suites():
            shapes = set()
            for idx, spec in enumerate(TASK_SUITES[suite]):
                env = get_task(idx, task_suite=suite)
                obs, info = env.reset(seed=0)
                env.action_space.seed(0)
                obs2, reward, terminated, truncated, step_info = env.step(env.action_space.sample())
                assert obs.shape == env.observation_space.shape
                assert obs2.shape == env.observation_space.shape
                assert np.all(np.isfinite(obs)), (suite, idx, "reset obs")
                assert np.all(np.isfinite(obs2)), (suite, idx, "step obs")
                assert np.isfinite(reward), (suite, idx, reward)
                assert "velocity_error" in step_info and "success" in step_info
                tail = obs[-6:]
                expected = np.asarray([
                    spec.target_velocity, spec.right_mass_scale, spec.left_mass_scale,
                    spec.foot_friction_scale, spec.joint_damping_scale,
                    spec.actuator_strength_scale,
                ], dtype=np.float32)
                assert np.allclose(tail, expected), (suite, idx, tail, expected)
                shapes.add((
                    int(np.prod(env.observation_space.shape)),
                    int(np.prod(env.action_space.shape)),
                ))
                env.close()
            assert len(shapes) == 1, f"{suite}: inconsistent obs/action shapes {shapes}"
            obs_dim, act_dim = shapes.pop()
            assert (obs_dim, act_dim) == (23, 6), (suite, obs_dim, act_dim)
            print(f"[ok] {suite}: obs={obs_dim}, act={act_dim}")
