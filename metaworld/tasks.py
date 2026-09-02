"""Continual Meta-World task suites.

TASK SELECTION -- WHY THESE FOUR
================================
Two published constraints drove the choice.

1. BUDGET. Continual World picked its CW10 tasks as those "not too easy or too
   hard in the assumed sample budget of 1M steps". Our budget is 150k, ~7x
   smaller, so CW10's criterion is calibrated for a budget we do not have.
   CW10 contains stick-pull and shelf-place, which sit in the *Very Hard* tier
   of the published MT50 difficulty partition; at 150k those never leave zero
   and contribute pure noise to every metric.

   So we take the intersection: tasks that are in CW10 (established, with
   documented transfer behaviour) AND in the *Easy* tier of the MT50 partition.
   Exactly four tasks satisfy both:

       window-close-v2, faucet-close-v2, handle-press-side-v2, peg-unplug-side-v2

2. TRANSFER STRUCTURE. Continual World also defines 8 three-task "triplets",
   built so that "there is a positive forward transfer from task 1 to task 3,
   but task 2 is used as a distraction that interferes with these learning
   dynamics". Of the eight, exactly ONE has all three tasks in the Easy tier:

       window-close -> handle-press-side -> peg-unplug-side

   That triplet is embedded verbatim at positions 0-2 of our sequence below.
   It is the cleanest available test of the central claim: weight-delta fusion
   starts each task from a learned combination of complete previous policies,
   so it should be able to weight window-close highly and the distractor
   handle-press-side near zero when it reaches peg-unplug-side. Classic CKA
   fusion has to rebuild the same behaviour out of residual deltas.

   The fourth task, faucet-close, is included because CSP's subspace analysis
   of CW10 reports that policies good at faucet-close are also good at
   peg-unplug-side (they occupy the same region of the policy subspace), and
   that window-close shows compositional transfer from a subspace spanned in
   part by faucet-close. So it adds a second documented transfer edge rather
   than an arbitrary fourth task.

SEQUENCE ORDERING
=================
    0, 2, 3, 1, 0, 3, 2, 1
    window-close, handle-press-side, peg-unplug-side, faucet-close,
    window-close, peg-unplug-side, handle-press-side, faucet-close

- Positions 0-2 are CW triplet #3 verbatim (transfer with a distractor).
- Every task appears exactly twice, with repeat gaps of 3, 4, 4 and 5. Varied
  gaps are what let the retention matrix say something about forgetting *as a
  function of distance*, rather than a single number.
- Position 5 places peg-unplug-side one step after window-close: the same
  documented transfer pair as the triplet, but now at minimum distance. The
  contrast between position 2 (distractor in between) and position 5 (none) is
  a direct read of how much the distractor costs each method.
- The second occurrence of a task is where weight-delta fusion should win
  outright: a complete, already-good policy for that exact task sits in the
  pool, so the fusion weights only have to find it.

POOL SIZE
=========
With 8 positions and 4 distinct tasks, the pool would hold 8 entries and the
*correct* compression is to merge each duplicate pair down to 4. Setting
pool_size = 4 therefore gives the merge step a verifiable ground truth: a good
merge selector should pair entries derived from the SAME task. That makes
"does behavioural KL pick same-task pairs more often than parameter cosine?"
a directly measurable claim rather than a qualitative one. It also yields 4
merge events in a single run instead of 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class MetaWorldTask:
    """One Meta-World task. `name` is a v2 environment id."""

    name: str
    tier: str = "easy"          # MT50 difficulty partition tier
    note: str = ""              # why this task is in the suite

    def label(self, suite: str = "") -> str:
        return self.name.replace("-v2", "")


# --------------------------------------------------------------------------
# The four-task suite. Order here defines task_id 0..3.
# --------------------------------------------------------------------------
_EASY4: Tuple[MetaWorldTask, ...] = (
    MetaWorldTask("window-close-v2", "easy",
                  "CW triplet #3 source; documented positive transfer to peg-unplug-side"),
    MetaWorldTask("faucet-close-v2", "easy",
                  "shares subspace region with peg-unplug-side (CSP subspace analysis)"),
    MetaWorldTask("handle-press-side-v2", "easy",
                  "CW triplet #3 distractor -- deliberately interferes"),
    MetaWorldTask("peg-unplug-side-v2", "easy",
                  "CW triplet #3 target; receives transfer from tasks 0 and 1"),
)

# A slightly longer variant, for the case where 4 tasks turn out to saturate.
# The two extra tasks are also Easy-tier and involve the same object families
# in the opposite direction, which maximises interference at minimum cost.
_EASY6: Tuple[MetaWorldTask, ...] = _EASY4 + (
    MetaWorldTask("door-close-v2", "easy", "revolute-joint close; interferes with window-close"),
    MetaWorldTask("drawer-close-v2", "easy", "sliding close; interferes with window-close"),
)

TASK_SUITES: Dict[str, List[MetaWorldTask]] = {
    "mw_easy4": list(_EASY4),
    "mw_easy6": list(_EASY6),
}

# See the SEQUENCE ORDERING section of the module docstring.
DEFAULT_CONTINUAL_SEQUENCE = (0, 2, 3, 1, 0, 3, 2, 1)

# For mw_easy6: 6 tasks x 2 passes, same design rules (triplet first, varied
# repeat gaps, no task repeated back to back).
EASY6_CONTINUAL_SEQUENCE = (0, 2, 3, 1, 4, 5, 0, 3, 2, 5, 1, 4)

# Held-out tasks for TD-JEPA encoder pretraining. Deliberately NOT in any
# suite above, so pretraining never sees an evaluation task, but from the same
# Easy tier and the same robot so the dynamics are representative.
PRETRAIN_TASKS = ("button-press-topdown-v2", "plate-slide-v2", "door-open-v2")


def available_task_suites():
    return tuple(TASK_SUITES.keys())


def default_sequence(task_suite: str = "mw_easy4"):
    return EASY6_CONTINUAL_SEQUENCE if task_suite == "mw_easy6" else DEFAULT_CONTINUAL_SEQUENCE


def get_task_spec(task_id: int, task_suite: str = "mw_easy4") -> MetaWorldTask:
    return TASK_SUITES[task_suite][task_id]


def get_task_name(task_id: int, task_suite: str = "mw_easy4") -> str:
    return get_task_spec(task_id, task_suite).label(task_suite)


def get_task(task_id: int, task_suite: str = "mw_easy4", render: bool = False):
    """Build the environment for one task. Signature matches the HalfCheetah
    suite's get_task so nothing downstream changes."""
    from metaworld_envs import make_env

    spec = get_task_spec(task_id, task_suite)
    # task_id, not a random seed: with frozen goal placements the seed IS the
    # task, so training and evaluation envs must agree. See metaworld_envs.
    return make_env(spec.name, task_id=task_id, render=render)


if __name__ == "__main__":
    import sys

    for suite in available_task_suites():
        print(f"{suite}  (sequence: {default_sequence(suite)})")
        for idx, task in enumerate(TASK_SUITES[suite]):
            print(f"  {idx}: {task.name:26s} [{task.tier}]  {task.note}")
        print()

    if "--check" in sys.argv:
        import numpy as np

        for suite in available_task_suites():
            shapes = set()
            for idx in range(len(TASK_SUITES[suite])):
                env = get_task(idx, task_suite=suite)
                obs, _ = env.reset(seed=0)
                obs2, r, term, trunc, info = env.step(env.action_space.sample())
                assert "success" in info, f"{suite}/{idx}: missing success key"
                assert "task_error" in info, f"{suite}/{idx}: missing task_error key"
                shapes.add((int(np.prod(env.observation_space.shape)),
                            int(np.prod(env.action_space.shape))))
                env.close()
            assert len(shapes) == 1, f"{suite}: inconsistent shapes {shapes}"
            obs_dim, act_dim = shapes.pop()
            print(f"[ok] {suite}: obs={obs_dim}, act={act_dim}, info keys present")
