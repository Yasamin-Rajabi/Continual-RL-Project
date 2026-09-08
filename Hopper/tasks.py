"""Fixed-objective Hopper dynamics tasks with native 11-D observations."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class HopperTask:
    target_velocity: float = 1.5
    mass_scale: float = 1.0
    foot_friction_scale: float = 1.0
    joint_damping_scale: float = 1.0
    actuator_strength_scale: float = 1.0
    short_name: str = "nominal"

    def label(self, suite=""):
        return f"Hopper-{self.short_name}"


TASK_SUITES = {"hopper_dynamics": [
    HopperTask(),
    HopperTask(mass_scale=1.25, short_name="heavy"),
    HopperTask(mass_scale=0.80, short_name="light"),
    HopperTask(foot_friction_scale=0.70, short_name="slippery-foot"),
    HopperTask(joint_damping_scale=1.50, short_name="high-damping"),
    HopperTask(actuator_strength_scale=0.80, short_name="weak-motors"),
]}
DEFAULT_CONTINUAL_SEQUENCE = tuple(range(6)) * 2


def available_task_suites():
    return tuple(TASK_SUITES)


def get_task_spec(task_id, task_suite="hopper_dynamics"):
    if task_suite not in TASK_SUITES or not 0 <= task_id < len(TASK_SUITES[task_suite]):
        raise ValueError(f"Unknown task {task_suite}/{task_id}")
    return TASK_SUITES[task_suite][task_id]


def get_task_name(task_id, task_suite="hopper_dynamics"):
    return get_task_spec(task_id, task_suite).label()


def get_task(task_id, task_suite="hopper_dynamics", render=False):
    import gymnasium as gym
    from hopper_envs import HopperDynamicsEnv
    spec = get_task_spec(task_id, task_suite)
    args = {k: v for k, v in vars(spec).items() if k != "short_name"}
    env = HopperDynamicsEnv(**args, render_mode="human" if render else None)
    return gym.wrappers.TimeLimit(env, max_episode_steps=1000)


if __name__ == "__main__":
    import sys
    for idx, spec in enumerate(TASK_SUITES["hopper_dynamics"]):
        print(idx, spec)
        if "--check" in sys.argv:
            import numpy as np
            env = get_task(idx)
            obs, _ = env.reset(seed=0)
            assert obs.shape == (11,) and env.action_space.shape == (3,)
            obs, reward, _, _, info = env.step(env.action_space.sample())
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert "success" in info and "velocity_error" in info
            env.close()
    print("Hopper tasks checked" if "--check" in sys.argv else "Use --check for MuJoCo smoke tests")
