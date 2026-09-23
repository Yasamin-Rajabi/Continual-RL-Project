"""Atari task environments for continual-RL experiments.

Each task is an ALE game mode.  This module intentionally keeps the observation
format identical to the Atari PPO pipeline used by this project:

    raw ALE RGB observation
      -> NoopResetEnv
      -> MaxAndSkipEnv
      -> EpisodicLifeEnv
      -> optional FireResetEnv
      -> ClipRewardEnv
      -> ResizeObservation(84, 84)
      -> GrayScaleObservation
      -> FrameStack(4)

Unlike the HalfCheetah benchmark, no scalar task identifier is appended to the
observation.  Doing so would change the CNN input shape used by the existing
Atari agents.  The task identity is instead exposed through ``env.task_id``,
``env.mode``, ``env.env_id`` and the ``info`` dictionary.

The wrapper order follows the project's PPO preprocessing.  ALE v5 is created
with base ``frameskip=1`` because MaxAndSkipEnv performs the intended skip=4.
This corrects the accidental double-frame-skip that occurs when the v5 default
frameskip=4 is combined with MaxAndSkipEnv(skip=4); old checkpoints trained
under that double-skip setup must not be mixed with this corrected protocol.
"""
from __future__ import annotations

from typing import Callable, Optional

try:
    import gymnasium as gym
    from stable_baselines3.common.atari_wrappers import (
        ClipRewardEnv,
        EpisodicLifeEnv,
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
    )
except ImportError as exc:
    gym = None
    ClipRewardEnv = None
    EpisodicLifeEnv = None
    FireResetEnv = None
    MaxAndSkipEnv = None
    NoopResetEnv = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None
    
import ale_py
gym.register_envs(ale_py)


_WrapperBase = gym.Wrapper if gym is not None else object


class AtariTaskInfoWrapper(_WrapperBase):
    """Attach task metadata without changing observations or rewards."""

    def __init__(self, env, *, env_id: str, mode: int, task_id: Optional[int] = None):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Atari tasks require gymnasium, stable-baselines3, and the ALE "
                "dependencies. Install the Atari requirements before creating "
                "an environment."
            ) from _IMPORT_ERROR

        super().__init__(env)
        self.env_id = str(env_id)
        self.mode = int(mode)
        self.task_id = int(mode if task_id is None else task_id)

    @property
    def task_name(self) -> str:
        game = self.env_id.split("/")[-1].split("-")[0]
        return f"{game}(mode={self.mode})"

    def _add_task_info(self, info):
        info = dict(info)
        info.setdefault("task_id", self.task_id)
        info.setdefault("mode", self.mode)
        info.setdefault("env_id", self.env_id)
        info.setdefault("task_name", self.task_name)
        return info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation, self._add_task_info(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._add_task_info(info),
        )


def make_atari_env(
    env_id: str,
    mode: int,
    *,
    task_id: Optional[int] = None,
    render: bool = False,
    capture_video: bool = False,
    video_folder: Optional[str] = None,
    noop_max: int = 30,
    skip: int = 4,
    frame_stack: int = 4,
    screen_size: int = 84,
    clip_reward: bool = True,
    episodic_life: bool = True,
    ale_frameskip: int = 1,
    repeat_action_probability: float = 0.25,
):
    """Create one preprocessed Atari task.

    Parameters
    ----------
    env_id:
        Gymnasium ALE id, e.g. ``"ALE/Freeway-v5"``.
    mode:
        ALE mode used as the continual-learning task.
    task_id:
        Position/identifier of the task in the continual suite.  If omitted,
        ``mode`` is used.
    render:
        If True, construct the base ALE environment with ``render_mode="human"``.
    capture_video:
        Record video using Gymnasium's RecordVideo wrapper.
    video_folder:
        Output folder used when ``capture_video=True``.
    noop_max, skip, frame_stack, screen_size:
        Atari preprocessing parameters matching ``run_ppo.py``.
    clip_reward:
        Apply sign reward clipping, as in the existing PPO training code.
    episodic_life:
        Treat loss of life as an episode boundary for learning, while the
        inner RecordEpisodeStatistics wrapper continues to record full-game
        returns.

    Notes
    -----
    ALE v5 defaults to ``frameskip=4``.  Since this wrapper also applies
    ``MaxAndSkipEnv(skip=4)``, the base ALE environment is explicitly created
    with ``ale_frameskip=1`` so frame skipping is applied exactly once.
    Training and evaluation both use this same constructor, so preprocessing
    remains identical across them.
    """
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "Atari tasks require gymnasium, stable-baselines3, and the ALE "
            "dependencies. Install the Atari requirements before creating "
            "an environment."
        ) from _IMPORT_ERROR

    if render and capture_video:
        raise ValueError("render=True and capture_video=True cannot be used together")
    if int(noop_max) < 0:
        raise ValueError("noop_max must be >= 0")
    if int(skip) < 1 or int(frame_stack) < 1 or int(screen_size) < 1:
        raise ValueError("skip, frame_stack, and screen_size must be >= 1")
    if int(ale_frameskip) < 1:
        raise ValueError("ale_frameskip must be >= 1")
    if not 0.0 <= float(repeat_action_probability) <= 1.0:
        raise ValueError("repeat_action_probability must be in [0, 1]")

    # ALE/*-v5 already defaults to frameskip=4.  Because this project also uses
    # MaxAndSkipEnv(skip=4), leaving the v5 default would repeat each chosen
    # action roughly 16 emulator frames.  Use a no-frameskip base environment
    # and let MaxAndSkipEnv own the standard 4-frame skip/max-pool operation.
    # Keep v5 sticky actions (0.25) explicit for reproducibility.
    gym_kwargs = {
        "mode": int(mode),
        "frameskip": int(ale_frameskip),
        "repeat_action_probability": float(repeat_action_probability),
    }
    if render:
        gym_kwargs["render_mode"] = "human"
    elif capture_video:
        gym_kwargs["render_mode"] = "rgb_array"

    env = gym.make(env_id, **gym_kwargs)

    # Keep RecordVideo and RecordEpisodeStatistics in the same positions used
    # by run_ppo.py.  RecordEpisodeStatistics is intentionally inside
    # EpisodicLifeEnv, so its "episode" return corresponds to the full game.
    if capture_video:
        if not video_folder:
            game = env_id.split("/")[-1].split("-")[0]
            video_folder = f"videos/{game}_mode_{mode}"
        env = gym.wrappers.RecordVideo(env, video_folder)

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=skip)

    if episodic_life:
        env = EpisodicLifeEnv(env)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    if clip_reward:
        env = ClipRewardEnv(env)

    env = gym.wrappers.ResizeObservation(env, (screen_size, screen_size))

    # Keep compatibility across Gymnasium releases.  The project currently
    # uses GrayScaleObservation; newer releases may expose GrayscaleObservation.
    grayscale_cls = getattr(gym.wrappers, "GrayScaleObservation", None)
    if grayscale_cls is None:
        grayscale_cls = getattr(gym.wrappers, "GrayscaleObservation")
    env = grayscale_cls(env)

    # Same compatibility treatment for FrameStack / FrameStackObservation.
    frame_stack_cls = getattr(gym.wrappers, "FrameStack", None)
    if frame_stack_cls is not None:
        env = frame_stack_cls(env, frame_stack)
    else:
        frame_stack_cls = getattr(gym.wrappers, "FrameStackObservation")
        env = frame_stack_cls(env, stack_size=frame_stack)

    env = AtariTaskInfoWrapper(
        env,
        env_id=env_id,
        mode=mode,
        task_id=task_id,
    )
    return env


def make_atari_env_thunk(
    env_id: str,
    mode: int,
    *,
    task_id: Optional[int] = None,
    capture_video: bool = False,
    video_folder: Optional[str] = None,
    **kwargs,
) -> Callable[[], object]:
    """Return a zero-argument constructor for ``gym.vector.SyncVectorEnv``."""

    def thunk():
        return make_atari_env(
            env_id,
            mode,
            task_id=task_id,
            capture_video=capture_video,
            video_folder=video_folder,
            **kwargs,
        )

    return thunk
