"""MiniGrid environment construction for the continual benchmark.

This is the only module that knows about MiniGrid. Everything downstream reads
obs_dim / n_actions from the spaces and two info keys from the wrapper below.

WHY MINIGRID FITS THIS PROJECT
------------------------------
1. Every MiniGrid task exposes the same 7x7x3 egocentric view and the same
   7-action discrete space, whatever the underlying grid size or layout. The
   knowledge vectors are added element-wise to the head parameters, so a
   constant shape across the sequence is a hard requirement; here it holds by
   construction, with no task-conditioning wrapper needed.
2. The reward is already normalized: MiniGrid returns
   ``1 - 0.9 * (steps / max_steps)`` on success and 0 otherwise, so episodic
   return lies in [0, 1] and the survey's forward-transfer formula applies with
   no per-task reference scaling.
3. MiniGrid is computationally lightweight relative to the MuJoCo suites used
   elsewhere in this project, making it suitable for a short-budget additional
   benchmark.
4. The action space is discrete, so the behavioural similarity between two pool
   entries is an exact categorical KL rather than the diagonal-Gaussian
   approximation used on continuous control. The merge criterion is therefore
   better founded here than in the original setting, not merely cheaper.

OBSERVATIONS
------------
We take the ``image`` field only (``ImgObsWrapper``) and flatten it to 147
dimensions, scaling each of the three channels by its own maximum. The three
channels are (object index, colour index, state index) with different ranges,
so dividing everything by a single constant would silently compress two of
them. Per-channel scaling keeps all three in [0, 1].

The ``direction`` field is deliberately dropped. The 7x7 view is egocentric and
always oriented "in front of" the agent, so direction is redundant given the
image, and including it would add a dimension whose meaning differs from every
other input.

INFO KEYS EXPOSED
-----------------
- ``success``   : 1 if the goal was reached at any point in the episode.
- ``task_error``: ``1 - progress``, where progress counts the sub-goals reached
                  (holding the key, door opened, goal reached). Lower is better,
                  matching the semantics the plotting and metric code expects.

  This second key exists because MiniGrid's reward is completely sparse. In a
  previous port to a sparse-reward suite we could not distinguish "the agent is
  learning but has not yet succeeded" from "the agent is doing nothing at all",
  and only found out after spending the compute. Sub-goal progress makes that
  distinction visible from the first evaluation.
"""
from __future__ import annotations

from typing import Optional, Sequence

import gymnasium
import numpy as np

# MiniGrid channel maxima: (OBJECT_TO_IDX, COLOR_TO_IDX, STATE) upper bounds.
# Used to scale each channel independently; see module docstring.
_CHANNEL_SCALE = np.array([10.0, 5.0, 2.0], dtype=np.float32)

# Episodic return is already in [0, 1], so no per-task reference run is needed
# to normalize it for the forward-transfer metric.
MAX_EPISODE_RETURN = 1.0


class MiniGridFlatWrapper(gymnasium.Env):
    """Flattens the egocentric view and adds success / progress info.

    Subclasses gymnasium.Env and delegates by hand rather than subclassing
    gymnasium.Wrapper. Two reasons, both load-bearing:

    1. gymnasium.wrappers.TimeLimit asserts isinstance(env, gymnasium.Env), and
       a plain __getattr__ delegate fails that assertion.
    2. gymnasium.Wrapper.__init__ runs the same assertion on the env it is
       handed, so if an installed MiniGrid release returns an old ``gym.Env``
       the Wrapper path would fail too. Holding the inner env as an attribute
       is immune to that.

    It also normalizes old-gym 4-tuple step returns and obs-only resets to the
    gymnasium forms, so the rest of the codebase sees exactly one API.
    """

    def __init__(self, env, progress_stages: Sequence[str] = ("goal",)):
        self.env = env
        self.progress_stages = tuple(progress_stages)
        unknown = set(self.progress_stages) - {"key", "door", "goal"}
        if unknown:
            raise ValueError(f"Unknown MiniGrid progress stages: {sorted(unknown)}")
        if not self.progress_stages:
            raise ValueError("progress_stages must contain at least one stage")
        inner = env.observation_space
        image_space = inner["image"] if hasattr(inner, "spaces") else inner
        self.obs_dim = int(np.prod(image_space.shape))
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self.spec = getattr(env, "spec", None)
        self._succeeded = False
        self._progress = 0.0

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    # -- observation ---------------------------------------------------- #
    def _flatten(self, obs) -> np.ndarray:
        image = obs["image"] if isinstance(obs, dict) else obs
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 3 and image.shape[-1] == 3:
            image = image / _CHANNEL_SCALE
        else:  # already one-hot or otherwise pre-processed
            image = image / max(float(image.max()), 1.0)
        return image.reshape(-1).astype(np.float32)

    # -- sub-goal progress ---------------------------------------------- #
    def _sub_goal_progress(self) -> float:
        """Fraction of the DoorKey sub-goals currently satisfied, in [0, 1].

        Reads the grid state directly rather than inferring it from rewards,
        because the reward is zero until the very last step. Tasks without a
        key or a door simply never light up those terms, which is correct: for
        them the only sub-goal is reaching the goal.
        """
        env = self.unwrapped
        reached = 1.0 if self._succeeded else 0.0

        carrying = getattr(env, "carrying", None)
        holds_key = 1.0 if (carrying is not None and getattr(carrying, "type", "") == "key") else 0.0

        door_open = 0.0
        grid = getattr(env, "grid", None)
        if grid is not None:
            for cell in getattr(grid, "grid", []) or []:
                if cell is not None and getattr(cell, "type", "") == "door":
                    if getattr(cell, "is_open", False):
                        door_open = 1.0
                    break

        values = {"key": holds_key, "door": door_open, "goal": reached}
        return float(sum(values[name] for name in self.progress_stages)) / float(len(self.progress_stages))

    # -- gym API --------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        self._succeeded = False
        self._progress = 0.0
        try:
            out = self.env.reset(seed=seed, options=options)
        except TypeError:
            if seed is not None and hasattr(self.env, "seed"):
                self.env.seed(seed)
            out = self.env.reset()
        obs, info = out if isinstance(out, tuple) and len(out) == 2 else (out, {})
        return self._flatten(obs), dict(info)

    def step(self, action):
        out = self.env.step(int(action))
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False

        # MiniGrid only pays out on the terminating step, so latch it: the
        # metric of interest is "did the agent solve the task in this episode".
        if reward > 0.0:
            self._succeeded = True
        self._progress = max(self._progress, self._sub_goal_progress())

        info = dict(info)
        info["success"] = float(self._succeeded)
        info["progress"] = self._progress
        info["task_error"] = 1.0 - self._progress
        info["aux_metric"] = float(reward)
        return self._flatten(obs), float(reward), bool(terminated), bool(truncated), info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()


def make_env(env_id: str, max_episode_steps: Optional[int] = None, render: bool = False,
             progress_stages: Sequence[str] = ("goal",)):
    """Build one MiniGrid task, wrapped for this codebase.

    max_episode_steps overrides MiniGrid's default horizon. The defaults scale
    as 10*n^2 (250 for a 5x5 grid, 640 for 8x8), which is generous; shortening
    it raises the density of terminal signals per environment step and is the
    single cheapest way to speed up learning in a sparse-reward grid world.
    """
    import gymnasium as gym
    try:
        import minigrid  # noqa: F401  (registers the environments)
        from minigrid.wrappers import ImgObsWrapper
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "MiniGrid is required. Install with:  pip install minigrid"
        ) from exc

    kwargs = {"render_mode": "human"} if render else {}
    if max_episode_steps is not None:
        kwargs["max_steps"] = int(max_episode_steps)

    try:
        env = gym.make(env_id, **kwargs)
    except TypeError:
        # Older releases do not accept max_steps through gym.make.
        kwargs.pop("max_steps", None)
        env = gym.make(env_id, **kwargs)

    env = ImgObsWrapper(env)
    wrapped = MiniGridFlatWrapper(env, progress_stages=progress_stages)
    if max_episode_steps is not None:
        wrapped = gymnasium.wrappers.TimeLimit(wrapped, max_episode_steps=int(max_episode_steps))
    return wrapped
