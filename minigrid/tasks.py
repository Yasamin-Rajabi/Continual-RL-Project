"""Continual MiniGrid task suites, centred on DoorKey.

TASK SELECTION
==============
The requirement was a sequence that (a) converges in very few steps and
(b) separates our method from the CKA-RL baseline as sharply as possible.
Those two pull in opposite directions, so the design is explicit about how it
satisfies both.

**Convergence.** DoorKey-5x5 is the cheapest member of the family that still
requires the full key -> door -> goal chain: published PPO runs reach ~0.95
return on it within roughly 60k environment steps. Everything in the suite is
at or below that difficulty. DoorKey-8x8 and 16x16 are deliberately excluded:
they are standard testbeds for *intrinsic motivation* research precisely
because plain RL does not solve them, and a task that never leaves zero
contributes only noise to forgetting and transfer.

**Separation.** Grid-size variation alone would be a weak benchmark here. Going
from DoorKey-5x5 to DoorKey-6x6 asks for the same skill on a larger board, so
every method transfers well and the conditions collapse onto each other. What
distinguishes a good continual learner is *interference*: tasks that pull the
policy in incompatible directions. So the suite pairs DoorKey with two
siblings that are structurally the same environment minus one component, and
that reward contradictory behaviour:

  - ``Empty-Random-5x5``: no key, no door. Picking anything up is wasted time
    and the optimal policy beelines to the goal. This directly contradicts the
    DoorKey policy, which must detour to the key first.
  - ``Unlock``: key and door, but the episode ENDS when the door opens. A
    DoorKey policy that walks on through gains nothing; a policy trained here
    learns to stop at exactly the point where DoorKey requires it to continue.

Both are cheap, both share the 7x7x3 view and the 7 discrete actions, and both
create genuine conflict rather than benign variation.

SEQUENCE
========
    0, 1, 2, 3, 0, 2, 1, 3
    DoorKey-5x5, Empty-Random-5x5, DoorKey-6x6, Unlock,
    DoorKey-5x5, DoorKey-6x6, Empty-Random-5x5, Unlock

- Positions 0-2 form a transfer-with-distractor triplet in the style of
  Continual World: DoorKey-5x5 transfers positively to DoorKey-6x6, and the
  Empty distractor sits between them and actively rewards the opposite
  behaviour. This is the cleanest test of the central claim. Weight-delta
  fusion starts each task from a learned combination of *complete* previous
  policies, so on reaching DoorKey-6x6 it can weight the DoorKey-5x5 entry
  highly and the distractor near zero. Classic CKA fusion has to rebuild the
  same behaviour out of residual deltas.
- Every task appears exactly twice, with repeat gaps 3, 4, 4 and 5. Varied gaps
  let the retention matrix describe forgetting as a function of distance rather
  than as a single number.
- Position 4 is where the gap between the two parameterizations should be
  widest: DoorKey-5x5 reappears while a complete, already-good policy for that
  exact task sits in the pool, so the fusion weights only have to find it. The
  CKA-RL paper itself notes that a task representable from the existing pool
  drives its residual vector toward zero, which is exactly the case that our
  formulation is meant to handle.
- No task is repeated back to back.

POOL SIZE
=========
With 8 positions and 4 distinct tasks the pool would hold 8 entries, and the
*correct* compression is to merge each duplicate pair down to 4. Setting
pool_size = 4 therefore gives the merge step a verifiable ground truth: a good
selector should pair entries derived from the same task. That turns "does
behavioural KL pick same-task pairs more often than parameter cosine?" into a
measurable claim, and produces 4 merge events per run instead of 3.

HORIZON
=======
MiniGrid's default horizon is 10*n^2 (250 steps on a 5x5 grid). We shorten it,
per task, to roughly the length of a competent trajectory plus slack. In a
sparse-reward grid world nothing is learned until an episode terminates, so a
shorter horizon raises the density of terminal signals per environment step.
It also caps how much a failing episode costs, which matters most early in
training when almost every episode fails.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class MiniGridTask:
    """One task: a MiniGrid id plus the horizon we impose on it."""

    env_id: str
    max_episode_steps: int
    note: str = ""

    def label(self, suite: str = "") -> str:
        return self.env_id.replace("MiniGrid-", "").replace("-v0", "")


_DOORKEY4: Tuple[MiniGridTask, ...] = (
    MiniGridTask(
        "MiniGrid-DoorKey-5x5-v0", 100,
        "full key->door->goal chain; cheapest task that still requires all three steps",
    ),
    MiniGridTask(
        "MiniGrid-Empty-Random-5x5-v0", 60,
        "distractor: no key, no door; picking anything up is wasted time",
    ),
    MiniGridTask(
        "MiniGrid-DoorKey-6x6-v0", 160,
        "same skill as task 0 on a larger board; receives positive transfer from it",
    ),
    MiniGridTask(
        "MiniGrid-Unlock-v0", 120,
        "key->door only; episode ends at the door, where DoorKey must continue",
    ),
)

# Two-task suite for the smoke stage. Not a scientific suite: it exists to
# exercise the whole pipeline (chain, merge, retention, metrics) in minutes.
_SMOKE2: Tuple[MiniGridTask, ...] = (
    MiniGridTask("MiniGrid-Empty-Random-5x5-v0", 60, "smoke only"),
    MiniGridTask("MiniGrid-DoorKey-5x5-v0", 100, "smoke only"),
)

# Optional wider suite if four tasks turn out to saturate too early.
_DOORKEY6: Tuple[MiniGridTask, ...] = _DOORKEY4 + (
    MiniGridTask("MiniGrid-LavaGapS5-v0", 80, "no key; a hazard to avoid instead"),
    MiniGridTask("MiniGrid-DoorKey-5x5-v0", 100, "second DoorKey-5x5 instance"),
)

TASK_SUITES: Dict[str, List[MiniGridTask]] = {
    "doorkey4": list(_DOORKEY4),
    "doorkey6": list(_DOORKEY6),
    "mg_smoke2": list(_SMOKE2),
}

DEFAULT_CONTINUAL_SEQUENCE = (0, 1, 2, 3, 0, 2, 1, 3)
DOORKEY6_CONTINUAL_SEQUENCE = (0, 1, 2, 3, 4, 5, 0, 2, 1, 5, 3, 4)
# 3 positions with pool_size 2 forces exactly one merge, which is the part of
# the pipeline most likely to fail silently.
SMOKE_CONTINUAL_SEQUENCE = (0, 1, 0)

SEQUENCES = {
    "doorkey4": DEFAULT_CONTINUAL_SEQUENCE,
    "doorkey6": DOORKEY6_CONTINUAL_SEQUENCE,
    "mg_smoke2": SMOKE_CONTINUAL_SEQUENCE,
}


def available_task_suites():
    return tuple(TASK_SUITES.keys())


def default_sequence(task_suite: str = "doorkey4"):
    return SEQUENCES.get(task_suite, DEFAULT_CONTINUAL_SEQUENCE)


def get_task_spec(task_id: int, task_suite: str = "doorkey4") -> MiniGridTask:
    return TASK_SUITES[task_suite][task_id]


def get_task_name(task_id: int, task_suite: str = "doorkey4") -> str:
    return get_task_spec(task_id, task_suite).label(task_suite)


def get_task(task_id: int, task_suite: str = "doorkey4", render: bool = False):
    """Build the environment for one task. Signature matches the HalfCheetah
    suite's get_task, so nothing downstream changes."""
    from minigrid_envs import make_env

    spec = get_task_spec(task_id, task_suite)
    return make_env(spec.env_id, max_episode_steps=spec.max_episode_steps, render=render)


if __name__ == "__main__":
    import sys

    for suite in available_task_suites():
        print(f"{suite}  (sequence: {default_sequence(suite)})")
        for idx, task in enumerate(TASK_SUITES[suite]):
            print(f"  {idx}: {task.env_id:32s} H={task.max_episode_steps:4d}  {task.note}")
        print()

    if "--check" in sys.argv:
        import numpy as np

        for suite in available_task_suites():
            shapes = set()
            for idx in range(len(TASK_SUITES[suite])):
                env = get_task(idx, task_suite=suite)
                env.reset(seed=0)
                _, _, _, _, info = env.step(env.action_space.sample())
                assert "success" in info, f"{suite}/{idx}: missing success"
                assert "task_error" in info, f"{suite}/{idx}: missing task_error"
                shapes.add((int(np.prod(env.observation_space.shape)), int(env.action_space.n)))
                env.close()
            assert len(shapes) == 1, f"{suite}: inconsistent spaces {shapes}"
            obs_dim, n_act = shapes.pop()
            print(f"[ok] {suite}: obs={obs_dim}, actions={n_act}, info keys present")
