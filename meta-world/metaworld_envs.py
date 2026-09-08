"""Meta-World environment construction for the continual benchmark.

The ONLY module that knows about Meta-World. Everything downstream (run_sac,
metrics, plots, cka_rl) reads obs_dim/act_dim from the spaces and two info keys
from the wrapper below.

WHY META-WORLD FITS THIS CODEBASE BETTER THAN MUJOCO LOCOMOTION
---------------------------------------------------------------
1. All 50 tasks share a 39-D observation and a 4-D action space. Knowledge
   vectors are added element-wise to the head parameters, so a constant shape
   across the sequence is a hard requirement -- here it holds by construction,
   with no task-conditioning wrapper needed.
2. Native binary `success`. The survey metrics need p_i(t) in [0,1]; on
   locomotion that required inventing a normalized score against per-task
   reference runs. Here it is free and standard.
3. Episodes never terminate early, only truncate at the horizon.

TWO CHOICES THAT ARE NOT COSMETIC
---------------------------------
1. HORIZON = 200, not Meta-World's default 500.
   The Continual World convention. At a 150k-step budget a horizon of 500
   yields 300 episodes; 200 yields 750. `success` is episodic, so more episodes
   is directly more signal per environment step.

2. FIXED goal/object placements (`_freeze_rand_vec = True`).
   Meta-World's own MT10/MT50 protocol: positions are fixed "so as to focus on
   acquiring the distinct skills, rather than generalization and robustness".
   The randomized goal-observable variant is described in the TD-MPC paper as
   harder than the single-goal variant used in most related work; at 150k steps
   that difference decides whether the tasks are learnable at all.

   CRITICAL: with frozen goals the goal IS the seed, so training and evaluation
   envs must be built with the same one. `_TASK_SEED_BASE + task_id` guarantees
   that.

INFO KEYS EXPOSED
-----------------
- `success`    : Meta-World's binary flag, latched over the episode.
- `task_error` : Meta-World's `obj_to_target` distance -- a dense,
                 lower-is-better progress signal filling the slot that
                 `velocity_error` occupied in the HalfCheetah suite, so
                 metrics.py and plots.py work unchanged.
"""
from __future__ import annotations

from typing import Optional

import gymnasium
import numpy as np

# Continual World's horizon, not Meta-World's default 500.
HORIZON = 200

# Fixed base so a task_id always yields the same goal placement everywhere.
_TASK_SEED_BASE = 12345

# Meta-World v2 shapes each task's reward into roughly [0, 10] per step. Used by
# metrics.py to put episodic return on a [0,1] scale so the survey's
# forward-transfer formula applies unchanged -- the HalfCheetah derivation
# assumed r_max = 0, which is false here.
MAX_REWARD_PER_STEP = 10.0
MAX_EPISODE_RETURN = MAX_REWARD_PER_STEP * HORIZON


class MetaWorldInfoWrapper(gymnasium.Env):
    """Normalizes Meta-World's step API and info dict, and latches success.

    Deliberately subclasses gymnasium.Env and delegates by hand rather than
    subclassing gymnasium.Wrapper. Two reasons, both load-bearing:

    1. gymnasium.wrappers.TimeLimit (applied immediately after this) asserts
       isinstance(env, gymnasium.Env). A plain __getattr__ delegate fails that
       assertion.
    2. gymnasium.Wrapper.__init__ runs the SAME assertion on the env it is
       given. Depending on which Meta-World release is installed, the
       underlying env may be an old `gym.Env` rather than a `gymnasium.Env`, in
       which case Wrapper would fail too. Subclassing Env and holding the inner
       env as a plain attribute is immune to that.

    It also normalizes the old-gym 4-tuple step / obs-only reset returns to the
    gymnasium 5-tuple and (obs, info) forms, so the rest of the codebase sees
    exactly one API regardless of the installed version.
    """

    def __init__(self, env):
        self.env = env
        def native_box(space):
            if isinstance(space, gymnasium.spaces.Box):
                return space
            # Older MetaWorld versions expose gym.spaces.Box, which fails
            # Gymnasium/SB3 isinstance checks even though the vectors agree.
            return gymnasium.spaces.Box(low=np.asarray(space.low),
                high=np.asarray(space.high), dtype=space.dtype)
        self.observation_space = native_box(env.observation_space)
        self.action_space = native_box(env.action_space)
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self.reward_range = getattr(env, "reward_range", (-float("inf"), float("inf")))
        self.spec = getattr(env, "spec", None)
        self._succeeded = False

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    def reset(self, *, seed=None, options=None):
        self._succeeded = False
        try:
            out = self.env.reset(seed=seed, options=options)
        except TypeError:
            # old gym: reset() takes no seed/options kwargs
            if seed is not None and hasattr(self.env, "seed"):
                self.env.seed(seed)
            out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:  # old gym 4-tuple
            obs, reward, done, info = out
            terminated, truncated = bool(done), False

        # Meta-World reports success on the step it happens, not afterwards.
        # Continual World's metric is "did the agent succeed at any point in the
        # episode", so latch it -- otherwise a policy that reaches the goal and
        # then drifts is scored as a failure and the success curve is far
        # noisier than the behaviour underneath it.
        step_success = float(info.get("success", 0.0))
        self._succeeded = self._succeeded or bool(step_success)

        info = dict(info)
        info["success"] = float(self._succeeded)
        info["step_success"] = step_success
        info["task_error"] = float(info.get("obj_to_target", np.nan))
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


def _resolve_env_factory(task_name: str):
    """Return (factory, api_name), tolerating Meta-World API generations.

    Meta-World's public API has changed more than once. Rather than pinning a
    commit in code -- which silently rots -- try the shapes that exist in the
    wild. The pinned commit in run_kaggle.sh is what we actually install; this
    is the safety net if that pin is ever changed.
    """
    import metaworld

    # (1) v2 goal-observable dict -- what the pinned commit provides.
    try:
        from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as REG
        key = f"{task_name}-goal-observable"
        if key in REG:
            return (lambda seed: REG[key](seed=seed)), "v2-goal-observable"
    except Exception:
        pass

    # (2) MT1 benchmark API, stable across versions. set_task fixes the goal,
    # which is the MT10/MT50 protocol we want anyway.
    for candidate in (task_name, task_name.replace("-v2", "-v3"), task_name.replace("-v2", "")):
        try:
            def _factory(seed, _name=candidate):
                bench = metaworld.MT1(_name, seed=seed)
                env = bench.train_classes[_name]()
                env.set_task(bench.train_tasks[0])
                return env
            probe = _factory(0)
            probe.close()
            return _factory, f"MT1({candidate})"
        except Exception:
            continue

    # (3) gymnasium registration, newest releases.
    try:
        import gymnasium as gym
        for candidate in (f"Meta-World/{task_name.replace('-v2', '-v3')}",
                          f"Meta-World/{task_name}"):
            try:
                probe = gym.make(candidate)
                probe.close()
                return (lambda seed, _id=candidate: gym.make(_id)), f"gym.make({candidate})"
            except Exception:
                continue
    except Exception:
        pass

    available = [a for a in dir(metaworld) if not a.startswith("_")]
    raise RuntimeError(
        f"could not construct Meta-World task '{task_name}' with any known API.\n"
        f"Installed metaworld exposes: {available}\n"
        "Install the pinned commit that provides ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE "
        "(see run_kaggle.sh step_setup)."
    )


def make_env(task_name: str, task_id: int, seed: Optional[int] = None,
             freeze_goal: bool = True, render: bool = False):
    """Build one Meta-World task, wrapped for this codebase."""
    try:
        import metaworld  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Meta-World is required. See run_kaggle.sh step_setup for the "
            "pinned install command."
        ) from exc

    factory, api = _resolve_env_factory(task_name)
    env_seed = _TASK_SEED_BASE + int(task_id) if seed is None else int(seed)
    env = factory(env_seed)

    if hasattr(env, "_freeze_rand_vec"):
        env._freeze_rand_vec = bool(freeze_goal)
    if render:
        env.render_mode = "human"

    wrapped = MetaWorldInfoWrapper(env)
    wrapped.metaworld_api = api
    return gymnasium.wrappers.TimeLimit(wrapped, max_episode_steps=HORIZON)
