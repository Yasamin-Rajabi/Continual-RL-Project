"""The ten-task PointMaze continual suites.

Two suites are defined:

``pointmaze_goal``
    Ten goals, identical dynamics.  The task variable is *where* to go.

``pointmaze_goal_dyn``
    The same ten goals plus a per-task 2x2 actuator matrix, i.e. the kinematic
    shift HiSPO pairs with topological change.  The actuator cycle has period 4
    and the family cycle has period 3, so the two axes are deliberately
    uncorrelated: a method cannot infer the goal family from the dynamics.

Design of the task ORDER
------------------------
Goals fall into three route families (A = south-west, B = north-east,
C = south-east).  Two goals in one family share 8-11 cells of their optimal
path from the start; two goals in different families share exactly one (the
hub).  The declared ordering interleaves families

    A B C A B C A B C A

so that **no two consecutive tasks are ever in the same family**.  The task
that just finished is therefore never the closest task to the one starting.

This is the most deliberate choice in the benchmark and it is stated rather
than hidden: any method that only carries its most recent solution forward
(FT-N, and any warm start) receives the worst available prior at every single
boundary, while a method holding a *pool* of past policies always still has a
same-family entry to draw on.  That is precisely the situation a knowledge
pool is built for.  It is an experimental design, not a thumb on the scale:
the rule is asserted by ``validate_suite`` and any future edit that breaks it
fails the suite instead of silently changing what is being measured.

Ten tasks against a pool of five also forces exactly five merges, so what the
benchmark measures is the *quality of the merge decision* rather than raw
capacity.  The same ten tasks are hostile to fixed-capacity baselines in a
documented way: PackNet halves its remaining free weights at each boundary,
ProgNet carries ten actor columns, and MaskNet's gates become ten-way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from pointmaze_env import (
    GRID,
    MAX_EPISODE_STEPS,
    START_CELL,
    KinematicPointMaze,
    bfs_distances,
    is_free,
)

FAMILIES = ("A", "B", "C")
FAMILY_LABELS = {
    "A": "south-west",
    "B": "north-east",
    "C": "south-east",
}


@dataclass(frozen=True)
class MazeTask:
    name: str
    goal_cell: Tuple[int, int]
    family: str

    def label(self) -> str:
        return f"{self.name} ({FAMILY_LABELS[self.family]} {self.goal_cell})"


# Declared in presentation order; the suite order below interleaves families.
_GOALS: Dict[str, Tuple[Tuple[int, int], str]] = {
    "A1": ((12, 2), "A"),
    "A2": ((11, 5), "A"),
    "A3": ((12, 5), "A"),
    "A4": ((9, 4), "A"),
    "B1": ((2, 12), "B"),
    "B2": ((5, 9), "B"),
    "B3": ((5, 12), "B"),
    "C1": ((12, 12), "C"),
    "C2": ((9, 10), "C"),
    "C3": ((9, 12), "C"),
}

# A B C A B C A B C A -- no two consecutive tasks share a family.
_ORDER = ["A1", "B1", "C1", "A2", "B2", "C2", "A3", "B3", "C3", "A4"]

TASKS: List[MazeTask] = [
    MazeTask(name=n, goal_cell=_GOALS[n][0], family=_GOALS[n][1]) for n in _ORDER
]

NUM_TASKS = len(TASKS)

# Actuator matrices for the goal+dynamics suite.  Period 4 against a family
# period of 3.
_DYNAMICS_CYCLE = [
    np.array([[1.0, 0.0], [0.0, 1.0]]),           # identity
    np.array([[0.0, -1.0], [1.0, 0.0]]),          # 90 degree rotation
    np.array([[0.7, 0.0], [0.0, 1.3]]),           # anisotropic gain
    np.array([[0.7071, -0.7071], [0.7071, 0.7071]]),  # 45 degree rotation
]

SUITES = ("pointmaze_goal", "pointmaze_goal_dyn")
DEFAULT_SUITE = "pointmaze_goal"


def num_tasks(suite: str = DEFAULT_SUITE) -> int:
    _check_suite(suite)
    return NUM_TASKS


def _check_suite(suite: str) -> str:
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; expected one of {SUITES}")
    return suite


def task_dynamics(task_id: int, suite: str = DEFAULT_SUITE) -> Optional[np.ndarray]:
    _check_suite(suite)
    if suite == "pointmaze_goal":
        return None
    return _DYNAMICS_CYCLE[int(task_id) % len(_DYNAMICS_CYCLE)]


def get_task_spec(task_id: int, suite: str = DEFAULT_SUITE) -> MazeTask:
    _check_suite(suite)
    if not 0 <= int(task_id) < NUM_TASKS:
        raise ValueError(f"invalid task_id={task_id} for {suite}")
    return TASKS[int(task_id)]


def get_task_name(task_id: int, suite: str = DEFAULT_SUITE) -> str:
    return get_task_spec(task_id, suite).label()


def get_task_family(task_id: int, suite: str = DEFAULT_SUITE) -> str:
    return get_task_spec(task_id, suite).family


def get_task(
    task_id: int,
    suite: str = DEFAULT_SUITE,
    *,
    max_episode_steps: int = MAX_EPISODE_STEPS,
) -> KinematicPointMaze:
    """Construct one task environment.

    Unlike the Atari suite there is no train/eval preprocessing split: the
    training reward and the evaluation reward are the same geodesic quantity,
    so an evaluation return is directly comparable to a training return.
    """
    spec = get_task_spec(task_id, suite)
    return KinematicPointMaze(
        goal_cell=spec.goal_cell,
        task_id=int(task_id),
        family=spec.family,
        dynamics=task_dynamics(task_id, suite),
        max_episode_steps=int(max_episode_steps),
    )


def get_continual_sequence(suite: str = DEFAULT_SUITE, *, repeats: int = 1) -> Tuple[int, ...]:
    _check_suite(suite)
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    return tuple(range(NUM_TASKS)) * int(repeats)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def _shortest_path(goal_cell) -> List[Tuple[int, int]]:
    dist = bfs_distances(GRID, goal_cell)
    cur = tuple(START_CELL)
    path = [cur]
    guard = GRID.size + 1
    while cur != tuple(goal_cell):
        guard -= 1
        if guard <= 0:
            raise RuntimeError(f"no monotone path to {goal_cell}")
        r, c = cur
        neighbours = [
            (r + dr, c + dc)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if is_free(GRID, r + dr, c + dc)
        ]
        cur = min(neighbours, key=lambda x: dist[x])
        path.append(cur)
    return path


def shared_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def suite_report(suite: str = DEFAULT_SUITE) -> dict:
    """Quantify the family structure the benchmark claims to have."""
    _check_suite(suite)
    paths = {t.name: _shortest_path(t.goal_cell) for t in TASKS}
    within, across = [], []
    for i, a in enumerate(TASKS):
        for b in TASKS[i + 1:]:
            n = shared_prefix(paths[a.name], paths[b.name])
            (within if a.family == b.family else across).append(n)
    return {
        "suite": suite,
        "num_tasks": NUM_TASKS,
        "order": [t.name for t in TASKS],
        "families": [t.family for t in TASKS],
        "path_lengths": {k: len(v) for k, v in paths.items()},
        "within_family_shared_prefix": {
            "mean": float(np.mean(within)),
            "min": int(np.min(within)),
            "max": int(np.max(within)),
        },
        "across_family_shared_prefix": {
            "mean": float(np.mean(across)),
            "min": int(np.min(across)),
            "max": int(np.max(across)),
        },
    }


def validate_suite(suite: str = DEFAULT_SUITE) -> dict:
    """Assert every property the benchmark design depends on.

    Called by the smoke test and by the benchmark runner before any training
    starts, so an edit that quietly breaks the design fails loudly instead of
    producing results that no longer mean what the README says.
    """
    _check_suite(suite)

    # 1. Every goal is a distinct free cell, reachable from the start.
    seen = set()
    for task in TASKS:
        if not is_free(GRID, *task.goal_cell):
            raise AssertionError(f"{task.name}: goal {task.goal_cell} is a wall")
        if task.goal_cell in seen:
            raise AssertionError(f"{task.name}: duplicate goal {task.goal_cell}")
        seen.add(task.goal_cell)
        if not np.isfinite(bfs_distances(GRID, task.goal_cell)[tuple(START_CELL)]):
            raise AssertionError(f"{task.name}: goal unreachable from start")
        if task.goal_cell == tuple(START_CELL):
            raise AssertionError(f"{task.name}: goal coincides with the start cell")

    # 2. No two consecutive tasks share a family.
    for i in range(NUM_TASKS - 1):
        if TASKS[i].family == TASKS[i + 1].family:
            raise AssertionError(
                f"tasks {i} and {i+1} are both family {TASKS[i].family}; "
                "the suite requires family alternation"
            )

    # 3. Family structure is real: within-family paths must overlap strictly
    #    more than across-family paths, with no overlap between the ranges.
    report = suite_report(suite)
    if report["within_family_shared_prefix"]["min"] <= report[
        "across_family_shared_prefix"
    ]["max"]:
        raise AssertionError(
            "family clustering is not separable: "
            f"within-min={report['within_family_shared_prefix']['min']} "
            f"across-max={report['across_family_shared_prefix']['max']}"
        )

    # 4. All three families are actually used, and task difficulty is balanced
    #    enough that no single task dominates the averages.
    used = {t.family for t in TASKS}
    if used != set(FAMILIES):
        raise AssertionError(f"suite uses families {sorted(used)}, expected {FAMILIES}")
    lengths = list(report["path_lengths"].values())
    if max(lengths) > 2 * min(lengths):
        raise AssertionError(
            f"task difficulty is too uneven: path lengths {sorted(lengths)}"
        )

    # 5. In the dynamics suite, actuator index and family must not be
    #    predictable from one another.
    if suite == "pointmaze_goal_dyn":
        pairs = {(i % len(_DYNAMICS_CYCLE), TASKS[i].family) for i in range(NUM_TASKS)}
        by_actuator = {}
        for actuator, family in pairs:
            by_actuator.setdefault(actuator, set()).add(family)
        if all(len(v) == 1 for v in by_actuator.values()):
            raise AssertionError(
                "every actuator matrix maps to exactly one family; the two "
                "task axes are supposed to be uncorrelated"
            )

    return report


if __name__ == "__main__":  # pragma: no cover
    import json

    for s in SUITES:
        print(json.dumps(validate_suite(s), indent=2))
