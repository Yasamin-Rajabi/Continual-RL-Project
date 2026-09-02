"""Deterministic continual locomotion task suites used by the Ant project.

The Ant directory intentionally keeps the reusable HalfCheetah environment
helpers because TD-JEPA/pretraining utilities share the same target-velocity
interface.  Final Ant experiments use ``ant_vel`` and ``ant_wind_vel``.

Every benchmark task is wrapped with ``TaskConditionedObservationWrapper``,
which appends [target_velocity, wind_a, wind_b] so actor and critic receive the
reward context.  The *calibration* environment is the one exception: it uses a
plain forward reward and raw Ant observations, with no fake target value.
"""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class LocomotionTask:
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


# Backward-compatible public name used by tdjepa_pretrain.py and older scripts.
HalfCheetahTask = LocomotionTask


# Reusable HalfCheetah task specs kept only because the Ant-side TD-JEPA tools
# can build either robot family from the same module.
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


# ---------------------------------------------------------------------------
# Ant velocity calibration
# ---------------------------------------------------------------------------
# A calibration run writes this JSON next to tasks.py by default.  Final Ant
# benchmark/pretraining scripts refuse the provisional fallback unless an
# explicit --allow-provisional-ant-calibration flag is supplied.
ANT_CALIBRATION_FILENAME = "ant_calibration.json"
_ANT_V_MAX_FALLBACK = 3.3


def ant_calibration_path() -> pathlib.Path:
    override = os.environ.get("CKA_ANT_CALIBRATION_FILE")
    if override:
        return pathlib.Path(override).expanduser().resolve()
    return pathlib.Path(__file__).resolve().with_name(ANT_CALIBRATION_FILENAME)


def _read_ant_calibration():
    path = ant_calibration_path()
    if not path.exists():
        return {
            "schema_version": 1,
            "v_max": _ANT_V_MAX_FALLBACK,
            "provisional": True,
            "source": "built-in fallback",
        }
    try:
        with path.open() as f:
            data = json.load(f)
        v_max = float(data["v_max"])
    except Exception as exc:
        raise RuntimeError(f"invalid Ant calibration file {path}: {exc}") from exc
    if not (v_max > 0.0):
        raise RuntimeError(f"invalid Ant calibration v_max={v_max!r} in {path}")
    data = dict(data)
    data["v_max"] = v_max
    data["provisional"] = False
    data["source"] = str(path)
    return data


ANT_CALIBRATION = _read_ant_calibration()
ANT_CALIBRATION_IS_PROVISIONAL = bool(ANT_CALIBRATION["provisional"])
ANT_CALIBRATION_PATH = ant_calibration_path()
_ANT_V_MAX = float(ANT_CALIBRATION["v_max"])


def require_ant_calibration():
    if ANT_CALIBRATION_IS_PROVISIONAL:
        raise RuntimeError(
            "Ant velocity calibration is still provisional. Run `python3 calibrate_ant.py` "
            f"to create {ANT_CALIBRATION_PATH}, then restart the Python process. "
            "For smoke tests only, pass --allow-provisional-ant-calibration."
        )


# Eight distinct task targets.  Fractions are deliberately spread from a crawl
# to slightly beyond the sustainable calibration speed so the continual methods
# face meaningful shifts.  If every method floors on 1.10x, reduce the top
# fraction in a *new* experiment rather than silently changing an existing run.
_ANT_VELOCITY_FRACTIONS = (0.10, 0.30, 0.50, 0.70, 0.90, 1.10, 0.40, 0.80)
_ANT_VELOCITIES = tuple(round(f * _ANT_V_MAX, 3) for f in _ANT_VELOCITY_FRACTIONS)
_ANT_SUCCESS_TOLERANCE_FRAC = float(ANT_CALIBRATION.get("success_tolerance_frac", 0.08))
if not (0.0 < _ANT_SUCCESS_TOLERANCE_FRAC < 1.0):
    raise RuntimeError(
        f"invalid Ant success_tolerance_frac={_ANT_SUCCESS_TOLERANCE_FRAC!r}; expected (0, 1)"
    )
_ANT_SUCCESS_TOLERANCE = round(_ANT_SUCCESS_TOLERANCE_FRAC * _ANT_V_MAX, 3)

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


def ant_pretrain_velocities():
    """Held-out TD-JEPA train velocities between benchmark targets."""
    return tuple(round(f * _ANT_V_MAX, 3) for f in (0.20, 0.60, 1.00))


def ant_heldout_velocities():
    """Real benchmark velocities reserved for TD-JEPA representation checks."""
    return tuple(round(f * _ANT_V_MAX, 3) for f in (0.10, 0.50, 0.90))


TASK_SUITES: Dict[str, List[LocomotionTask]] = {
    "halfcheetah_vel": [LocomotionTask(v) for v in _VELOCITIES],
    "halfcheetah_wind_vel": [
        LocomotionTask(v, wind=w) for v, w in zip(_VELOCITIES, _WIND_PAIRS)
    ],
    "ant_vel": [LocomotionTask(v) for v in _ANT_VELOCITIES],
    "ant_wind_vel": [
        LocomotionTask(v, wind=w) for v, w in zip(_ANT_VELOCITIES, _ANT_WIND_PAIRS)
    ],
}

CALIBRATION_SUITE = "ant_calibrate"

# Two passes through all eight tasks.  source_ids/seq_idx distinguish repeated
# occurrences for the upcoming buffer-decay analysis.
DEFAULT_CONTINUAL_SEQUENCE = tuple(range(len(_ANT_VELOCITIES))) * 2
REVERSED_CONTINUAL_SEQUENCE = tuple(reversed(range(len(_ANT_VELOCITIES)))) * 2


def available_task_suites():
    return tuple(TASK_SUITES.keys())


def get_task_name(task_id: int, task_suite: str = "ant_vel") -> str:
    if task_suite == CALIBRATION_SUITE:
        if task_id != 0:
            raise IndexError("ant_calibrate contains exactly one task (task_id=0)")
        return "Ant forward-reward calibration"
    task = TASK_SUITES[task_suite][task_id]
    return task.label(task_suite)


def get_task_spec(task_id: int, task_suite: str = "ant_vel") -> LocomotionTask:
    if task_suite == CALIBRATION_SUITE:
        raise ValueError("ant_calibrate has no target-velocity task spec")
    return TASK_SUITES[task_suite][task_id]


def get_task(task_id: int, task_suite: str = "ant_vel", render: bool = False):
    import gymnasium as gym
    from locomotion_envs import (
        HalfCheetahVelEnv,
        HalfCheetahWindVelEnv,
        TaskConditionedObservationWrapper,
    )

    if task_suite == CALIBRATION_SUITE:
        if task_id != 0:
            raise IndexError("ant_calibrate contains exactly one task (task_id=0)")
        from ant_envs import AntForwardCalibrationEnv

        env = AntForwardCalibrationEnv(render_mode="human" if render else None)
        # No TaskConditionedObservationWrapper here: calibration is forward
        # reward with no target variable, so appending an artificial target is
        # both unnecessary and numerically harmful.
        return gym.wrappers.TimeLimit(env, max_episode_steps=1000)

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
    env = TaskConditionedObservationWrapper(env, task)
    return gym.wrappers.TimeLimit(env, max_episode_steps=1000)


if __name__ == "__main__":
    import sys

    print(
        f"Ant calibration: v_max={_ANT_V_MAX:g} m/s | "
        f"source={ANT_CALIBRATION['source']} | provisional={ANT_CALIBRATION_IS_PROVISIONAL}"
    )
    for suite in available_task_suites():
        print(suite)
        for idx, task in enumerate(TASK_SUITES[suite]):
            print(f"  {idx}: {task.label(suite)}")

    if "--check" in sys.argv:
        import numpy as np

        for suite in available_task_suites():
            shapes = set()
            for idx in range(len(TASK_SUITES[suite])):
                env = get_task(idx, task_suite=suite)
                env.reset(seed=0)
                env.action_space.seed(0)
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
