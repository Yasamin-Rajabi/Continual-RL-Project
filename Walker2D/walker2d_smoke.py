"""Fast live checks for Walker2D task construction and dynamics perturbations."""
from __future__ import annotations

import numpy as np
import mujoco

from tasks import TASK_SUITES, get_task
from walker2d_envs import _ACTUATED_JOINTS, _FOOT_JOINTS, _LEFT_JOINTS, _RIGHT_JOINTS


def obj_id(model, obj_type, name):
    idx = int(mujoco.mj_name2id(model, obj_type, name))
    assert idx >= 0, f"missing MuJoCo object {name}"
    return idx


def body_id_for_joint(model, joint_name):
    jid = obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_bodyid[jid])


def snapshot(env):
    u = env.unwrapped
    right = {
        n: float(u.model.body_mass[body_id_for_joint(u.model, n)])
        for n in _RIGHT_JOINTS
    }
    left = {
        n: float(u.model.body_mass[body_id_for_joint(u.model, n)])
        for n in _LEFT_JOINTS
    }
    friction = {}
    for foot_joint in _FOOT_JOINTS:
        bid = body_id_for_joint(u.model, foot_joint)
        adr = int(u.model.body_geomadr[bid])
        num = int(u.model.body_geomnum[bid])
        assert num > 0, f"body for {foot_joint} has no geoms"
        for local_i, gid in enumerate(range(adr, adr + num)):
            friction[f"{foot_joint}:{local_i}"] = float(u.model.geom_friction[gid, 0])
    damping = {}
    for n in _ACTUATED_JOINTS:
        jid = obj_id(u.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        damping[n] = float(u.model.dof_damping[int(u.model.jnt_dofadr[jid])])
    gear = np.asarray(u.model.actuator_gear, dtype=np.float64).copy()
    return right, left, friction, damping, gear


def assert_scaled(actual, nominal, scale, label):
    for key, base in nominal.items():
        got = actual[key]
        assert np.isclose(got, base * scale, rtol=2e-5, atol=1e-9), (
            label, key, got, base, scale
        )


def main():
    suite = "walker2d_dynamics"
    specs = TASK_SUITES[suite]

    nominal_env = get_task(0, suite)
    obs0, _ = nominal_env.reset(seed=123)
    nominal = snapshot(nominal_env)
    assert nominal_env.observation_space.shape == (17,)
    assert nominal_env.action_space.shape == (6,)
    assert obs0.shape == (17,)  # native observation, no appended context
    nominal_env.close()

    for task_id, spec in enumerate(specs):
        env = get_task(task_id, suite)
        obs, _ = env.reset(seed=123)
        snap = snapshot(env)
        assert_scaled(snap[0], nominal[0], spec.right_mass_scale, f"task {task_id} right mass")
        assert_scaled(snap[1], nominal[1], spec.left_mass_scale, f"task {task_id} left mass")
        assert_scaled(snap[2], nominal[2], spec.foot_friction_scale, f"task {task_id} friction")
        assert_scaled(snap[3], nominal[3], spec.joint_damping_scale, f"task {task_id} damping")
        assert np.allclose(
            snap[4], nominal[4] * spec.actuator_strength_scale, rtol=2e-5, atol=1e-9
        ), f"task {task_id} actuator gear scale mismatch"

        assert obs.shape == (17,)

        # A short stochastic rollout catches NaNs, invalid contacts, bad action
        # shapes, and termination/reset problems without asking SAC to learn.
        env.action_space.seed(1000 + task_id)
        terminations = 0
        rewards = []
        for _ in range(250):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            assert np.isfinite(info["x_velocity"])
            assert np.isfinite(info["velocity_error"])
            rewards.append(float(reward))
            if terminated or truncated:
                terminations += int(terminated)
                obs, _ = env.reset()
        env.close()
        print(
            f"[ok] task {task_id}: {spec.short_name:>16s} | "
            f"random-reward mean={np.mean(rewards): .3f} | falls={terminations}"
        )

    print("\nWalker2D live environment checks passed.")


if __name__ == "__main__":
    main()
