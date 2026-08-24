"""Deterministic continual locomotion task suites.

Four suites are provided:
- halfcheetah_vel:      target velocity changes across tasks.
- halfcheetah_wind_vel: target velocity + a hidden fixed wind change.
- ant_vel:              same template on Ant (3D quadruped).
- ant_wind_vel:         same, plus a hidden horizontal crosswind.

get_task() wraps every environment with
halfcheetah_envs.TaskConditionedObservationWrapper, which appends
[target_velocity, wind_a, wind_b] to every observation. So all tasks WITHIN
a suite share the same observation and action space:

    halfcheetah_* : 17 raw + 3 task dims = 20-D obs, 6-D action
    ant_*         : 27 raw + 3 task dims = 30-D obs, 8-D action

Shapes must be constant within a suite (knowledge vectors are added
element-wise to the head parameters), but NOT across suites -- each suite is
its own independent continual chain, exactly as halfcheetah_vel and
halfcheetah_wind_vel already are.

DROP-IN: this file keeps every public symbol and every existing behaviour of
the original tasks.py. The HalfCheetah labels are byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class HalfCheetahTask:
    """Name kept for backward compatibility; used by all suites.

    Any object with .target_velocity (float) and .wind (2-tuple) satisfies
    halfcheetah_envs.make_task_specific_observation, which is why the Ant
    suites reuse this class rather than defining a parallel one.
    """

    target_velocity: float
    wind: Tuple[float, float] = (0.0, 0.0)

    def label(self, suite: str) -> str:
        prefix = "Ant" if suite.startswith("ant") else "HC"
        if not suite.endswith("wind_vel"):
            return f"{prefix}-Vel {self.target_velocity:g}m/s"
        return (
            f"{prefix}-WindVel v={self.target_velocity:g}, "
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

# --------------------------------------------------------------------------
# ANT TARGETS
#
# Ant is much slower than HalfCheetah (which reaches ~5-8 m/s under SAC), so
# targets are defined as FRACTIONS of the velocity Ant can actually reach, not
# as absolute numbers copied across robots. Copying HalfCheetah's absolute
# targets would put the fast tasks out of reach, several of them would collapse
# onto the single behaviour "run flat out", and they would stop being distinct
# tasks at all.
#
# CALIBRATE _ANT_V_MAX BEFORE THE FIRST REAL RUN:
#     bash run_kaggle.sh calibrate ant
# (or: python3 calibrate_ant.py). It trains plain forward-reward SAC on Ant and
# prints the value to paste here.
#
# WHY THE RANGE GOES UP TO 1.1x AND DOWN TO 0.1x
# The spread across tasks is what makes methods distinguishable. Two of the
# paper's claims -- weight_delta vs classic_cka, and a task-agnostic shared
# encoder vs one fit to task 0 -- both predict effects that GROW with the shift
# between consecutive tasks. A narrow band of similar velocities would show all
# methods performing the same and measure nothing. So the band deliberately
# spans from a near-crawl to slightly past the calibrated ceiling.
#
# The 1.1x task is intentionally at or just past what SAC reached during
# calibration; that is where methods separate. But watch for the floor effect:
# if EVERY method scores ~0 on task 5, it is measuring "impossible" rather than
# "hard" -- drop the top fraction to 1.0 and re-run. Check this on the first
# seed before spending the rest of the budget.
# --------------------------------------------------------------------------
_ANT_V_MAX = 3.3                      # PROVISIONAL -- replace with your calibration

# Indices 0-5 are the continual sequence; 6-7 are intermediates that only enter
# the pool, mirroring how _VELOCITIES is structured for HalfCheetah.
_ANT_VELOCITY_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90, 1.10, 0.40, 0.80)
_ANT_VELOCITIES = tuple(round(f * _ANT_V_MAX, 3) for f in _ANT_VELOCITY_FRACTIONS)

# HalfCheetah uses a fixed 0.2 over a 0.5-3.0 target band, i.e. ~8% of the band.
# Keeping the same relative tolerance means "success" is equally strict on both
# robots, which is what makes the success-rate-based metrics comparable across
# suites. An absolute 0.2 on Ant would be a much harsher criterion.
_ANT_SUCCESS_TOLERANCE = round(0.08 * _ANT_V_MAX, 3)

# Ant is lighter than HalfCheetah, so the same force is a much larger
# perturbation; these are ~40% of the HalfCheetah magnitudes. Wind is kept as a
# SECONDARY axis: unlike target velocity, the wind vector is handed to the
# policy through the observation and mainly induces a conditional correction
# rather than a genuinely different gait. Widen the velocity band first if you
# need more separation between methods; only then touch these.
_ANT_WIND_PAIRS = (
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, -2.0),
    (0.0, 2.0),
    (-0.5, -1.0),
    (0.5, 1.0),
    (-1.0, 2.0),
    (1.0, -2.0),
)

# Single throwaway task used only by calibrate_ant.py. With a target this far
# above anything Ant can do, reward = -|v - 1000| - ctrl_cost reduces to
# v - 1000 - ctrl_cost, i.e. plain forward-reward SAC -- so calibration needs no
# new training code, just this task and the existing run_sac.py.
_ANT_CALIBRATION_VELOCITY = 1000.0

TASK_SUITES: Dict[str, List[HalfCheetahTask]] = {
    "halfcheetah_vel": [HalfCheetahTask(v) for v in _VELOCITIES],
    "halfcheetah_wind_vel": [
        HalfCheetahTask(v, wind=w) for v, w in zip(_VELOCITIES, _WIND_PAIRS)
    ],
    "ant_vel": [HalfCheetahTask(v) for v in _ANT_VELOCITIES],
    "ant_wind_vel": [
        HalfCheetahTask(v, wind=w) for v, w in zip(_ANT_VELOCITIES, _ANT_WIND_PAIRS)
    ],
    "ant_calibrate": [HalfCheetahTask(_ANT_CALIBRATION_VELOCITY)],
}

# Paper-style second pass through the same tasks to expose retention/relearning.
DEFAULT_CONTINUAL_SEQUENCE = tuple(range(6)) + tuple(range(6))

# Reversed-order variant of the first pass. Running this changes ONLY which
# task the frozen shared encoder is trained on (task 0 is the fastest instead
# of the slowest). It is the cheapest possible test of whether the root-task
# encoder bias described in the analysis doc is real -- no code changes, one
# CLI flag.
REVERSED_CONTINUAL_SEQUENCE = tuple(range(5, -1, -1)) + tuple(range(5, -1, -1))


def available_task_suites():
    return tuple(TASK_SUITES.keys())


def get_task_name(task_id: int, task_suite: str = "halfcheetah_vel") -> str:
    task = TASK_SUITES[task_suite][task_id]
    return task.label(task_suite)


def get_task_spec(task_id: int, task_suite: str = "halfcheetah_vel") -> HalfCheetahTask:
    return TASK_SUITES[task_suite][task_id]


def get_task(task_id: int, task_suite: str = "halfcheetah_vel", render: bool = False):
    import gymnasium as gym
    from halfcheetah_envs import (
        HalfCheetahVelEnv,
        HalfCheetahWindVelEnv,
        TaskConditionedObservationWrapper,
    )

    task = get_task_spec(task_id, task_suite)
    kwargs = {
        "target_velocity": task.target_velocity,
        "render_mode": "human" if render else None,
    }

    if task_suite.startswith("ant"):
        from ant_envs import AntVelEnv, AntWindVelEnv

        env_cls = AntWindVelEnv if task_suite == "ant_wind_vel" else AntVelEnv
        kwargs["success_tolerance"] = _ANT_SUCCESS_TOLERANCE
    else:
        env_cls = (
            HalfCheetahWindVelEnv
            if task_suite == "halfcheetah_wind_vel"
            else HalfCheetahVelEnv
        )

    if task_suite.endswith("wind_vel"):
        kwargs["wind"] = task.wind

    env = env_cls(**kwargs)
    # Every observation from this point on (reset AND step) carries
    # [target_velocity, wind_a, wind_b] appended -- see
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
    import sys

    for suite in available_task_suites():
        print(suite)
        for idx, task in enumerate(TASK_SUITES[suite]):
            print(f"  {idx}: {task.label(suite)}")

    if "--check" in sys.argv:
        # Verifies the hard constraint: constant obs/act shape within a suite.
        import numpy as np

        for suite in available_task_suites():
            shapes = set()
            for idx in range(len(TASK_SUITES[suite])):
                env = get_task(idx, task_suite=suite)
                env.reset(seed=0)
                env.step(env.action_space.sample())
                shapes.add(
                    (
                        int(np.prod(env.observation_space.shape)),
                        int(np.prod(env.action_space.shape)),
                    )
                )
                env.close()
            assert len(shapes) == 1, f"{suite}: inconsistent shapes {shapes}"
            obs_dim, act_dim = shapes.pop()
            print(f"[ok] {suite}: obs={obs_dim}, act={act_dim}")
