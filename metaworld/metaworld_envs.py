"""Meta-World environment construction for the continual benchmark.

This module is the ONLY place that knows about Meta-World. Everything
downstream (run_sac, metrics, plots, cka_rl) is domain-agnostic and reads
obs_dim / act_dim from the spaces and two info keys from the wrapper below.

WHY META-WORLD FITS THIS CODEBASE BETTER THAN THE MUJOCO LOCOMOTION SUITES
-------------------------------------------------------------------------
1. All 50 tasks share a 39-D observation and a 4-D action space. The knowledge
   vectors are added element-wise to the head parameters, so a constant shape
   across the sequence is a hard requirement -- here it holds by construction,
   with no task-conditioning wrapper needed.
2. Meta-World emits a native binary `success` flag. The survey metrics need
   p_i(t) in [0,1]; on locomotion that required inventing a normalized score
   against per-task reference runs. Here it is free and standard.
3. Episodes never terminate early, only truncate at the horizon, so the
   "falling over ends the episode" pathology that plagues Hopper/Walker/Ant
   under a negative reward cannot occur.

TWO CHOICES THAT ARE NOT COSMETIC
---------------------------------
1. HORIZON = 200, not Meta-World's default 500.
   This is the Continual World convention. At a 150k-step budget a horizon of
   500 yields only 300 episodes; 200 yields 750. Since `success` is an episodic
   quantity, more episodes is directly more signal per environment step.

2. FIXED goal/object positions (`_freeze_rand_vec = True`) by default.
   This is Meta-World's own MT10/MT50 protocol: "the positions of objects and
   goal positions are fixed in all tasks in this evaluation, so as to focus on
   acquiring the distinct skills, rather than generalization and robustness."
   The goal-observable randomized variant is explicitly described in the
   TD-MPC paper as harder than the single-goal variant used in most related
   work, and at 150k steps that difference decides whether tasks are learnable
   at all.

   CRITICAL: with frozen goals, the goal is determined by the seed passed at
   construction. Train and eval envs MUST get the same seed or they are
   different tasks. `_TASK_SEED_BASE + task_id` guarantees that. (The previous
   Meta-World code in this project used `seed=np.random.randint(0, 1024)`,
   which is harmless with randomization on and silently fatal with it off.)

INFO KEYS EXPOSED
-----------------
- `success`    : Meta-World's native binary flag, latched for the episode.
- `task_error` : Meta-World's `obj_to_target` distance. Plays the role that
                 `velocity_error` played in the HalfCheetah suite -- a dense,
                 lower-is-better progress signal -- so metrics.py and plots.py
                 work unchanged.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Continual World's horizon, not Meta-World's default 500. See module docstring.
HORIZON = 200

# Fixed base so a given task_id always yields the same goal placement, in both
# the training env and every evaluation env.
_TASK_SEED_BASE = 12345


class MetaWorldInfoWrapper:
    """Normalizes Meta-World's info dict and latches success over the episode.

    Not a gym.Wrapper subclass by import order convenience -- see make_env()
    below, which wraps this in the gymnasium wrappers the rest of the code
    expects.
    """

    def __init__(self, env):
        self.env = env
        self._succeeded = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, **kwargs):
        self._succeeded = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Meta-World reports success on the step it happens, not afterwards.
        # Continual World's metric is "did the agent succeed at any point in
        # the episode", so latch it -- otherwise a policy that reaches the goal
        # and then drifts is scored as a failure and the success curve becomes
        # far noisier than the underlying behaviour.
        step_success = float(info.get("success", 0.0))
        self._succeeded = self._succeeded or bool(step_success)

        info = dict(info)
        info["success"] = float(self._succeeded)
        info["step_success"] = step_success
        # Dense lower-is-better progress signal; fills the slot that
        # velocity_error occupied in the HalfCheetah suite.
        info["task_error"] = float(info.get("obj_to_target", np.nan))
        return obs, reward, terminated, truncated, info


def make_env(
    task_name: str,
    task_id: int,
    seed: Optional[int] = None,
    freeze_goal: bool = True,
    render: bool = False,
):
    """Build one Meta-World task, wrapped for this codebase.

    task_name is a Meta-World v2 id such as "window-close-v2".
    """
    import gymnasium as gym

    try:
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Meta-World is required. Install with:\n"
            "  pip install git+https://github.com/Farama-Foundation/Metaworld.git@master"
        ) from exc

    key = f"{task_name}-goal-observable"
    if key not in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE:
        raise KeyError(
            f"unknown Meta-World task '{task_name}'. "
            f"Available example: 'window-close-v2'."
        )

    env_seed = _TASK_SEED_BASE + int(task_id) if seed is None else int(seed)
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[key](seed=env_seed)

    # See module docstring: fixed placements are the MT10/MT50 protocol and are
    # what makes a 150k-step budget viable.
    env._freeze_rand_vec = bool(freeze_goal)
    if render:
        env.render_mode = "human"

    env = MetaWorldInfoWrapper(env)
    return gym.wrappers.TimeLimit(env, max_episode_steps=HORIZON)


# Meta-World v2 shapes each task's reward into roughly [0, 10] per step. Used
# by metrics.py to put episodic return on a [0,1] scale so the survey's
# forward-transfer formula applies unchanged -- the HalfCheetah derivation
# assumed r_max = 0, which is false here.
MAX_REWARD_PER_STEP = 10.0
MAX_EPISODE_RETURN = MAX_REWARD_PER_STEP * HORIZON
