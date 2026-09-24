"""Kinematic PointMaze for continual RL.

Pure NumPy physics on top of a Gymnasium ``Env``.  No MuJoCo, no
gymnasium-robotics, no JAX.  One ``step`` costs tens of microseconds, so wall
clock is dominated by SAC updates rather than by simulation.

Observation (12 floats, goal is NOT included)
---------------------------------------------
    [x, y, vx, vy] + 8 ray distances to the nearest wall

The ray sensors are identical across every task in a suite.  They are what
makes a *shared* encoder meaningful: the encoder has a genuinely task
independent quantity to learn (local maze geometry) while the policy heads
carry goal specific behavior.  With a bare ``[x, y, vx, vy]`` observation,
"freeze the encoder after task 0" would be an empty statement.

Reward
------
    r(s) = -d_geo(s) / REWARD_SCALE

``d_geo`` is the shortest path length *through the maze* (BFS on free cells,
refined inside the current cell), not Euclidean distance.

Both standard PointMaze rewards are wrong for this study:

* ``sparse`` (reward 1 only at the goal) leaves SAC near chance on a maze this
  branchy, so every method scores the same and the benchmark measures nothing.
* ``dense`` (``exp(-d)``) is positive at every step, so an agent is *punished*
  for ending the episode by reaching the goal, and its reward gradient points
  straight into walls.

The geodesic form has no local optimum on a wall, improves monotonically with
reaching the goal sooner, and is ``<= 0`` everywhere.  That last property is
what keeps ``RETURN_UPPER_BOUND = 0`` exact, which the forward transfer
algebra in ``metrics.py`` depends on.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly by the test stub
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    gym = None
    spaces = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# ----------------------------------------------------------------------------
# Physics / protocol constants.  These are part of the benchmark definition:
# changing one invalidates comparisons against previously produced results.
# ----------------------------------------------------------------------------
CELL_SIZE = 1.0
DT = 0.1
ACCEL = 6.0
DAMPING = 0.92
MAX_SPEED = 3.0
AGENT_RADIUS = 0.18
GOAL_RADIUS = 0.35
MAX_EPISODE_STEPS = 300
REWARD_SCALE = 10.0
N_RAYS = 8
RAY_MAX = 4.0
OBS_DIM = 4 + N_RAYS
ACT_DIM = 2

# Every reward is <= 0, so an undiscounted return can never exceed zero.
# metrics.py reads this to normalize forward transfer.
RETURN_UPPER_BOUND = 0.0

# Success latches at termination, so the episodic aggregate must be a max and
# never a mean: with a mean, solving in 20 steps would score 20x worse than
# solving in 400.  This constant is read by the trainers and by metrics.py.
EPISODIC_SUCCESS = "max"

GRID_H = 15
GRID_W = 15


def _carve(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    """Carve an axis-aligned corridor of free cells, inclusive of both ends."""
    if r0 != r1 and c0 != c1:
        raise ValueError("corridors must be axis aligned")
    rr = range(min(r0, r1), max(r0, r1) + 1)
    cc = range(min(c0, c1), max(c0, c1) + 1)
    for r in rr:
        for c in cc:
            grid[r, c] = 0


def build_grid() -> np.ndarray:
    """Return the occupancy grid (1 = wall, 0 = free).

    The maze is three trunks leaving one hub in three different directions,
    each trunk fanning out into its own cluster of goal chambers.  Because the
    trunks separate at the very first cell out of the hub, two goals in
    different families share almost none of their optimal path, while two goals
    in the same family share nearly all of it.  That is the structure the
    behavioral (symmetric-KL) merge selector is supposed to be able to see and
    that a parameter-cosine selector may not.
    """
    grid = np.ones((GRID_H, GRID_W), dtype=np.int8)

    # Hub.
    _carve(grid, 7, 7, 7, 7)

    # --- Family A (south-west): west trunk, then down the western wall ------
    _carve(grid, 7, 2, 7, 7)      # hub -> west
    _carve(grid, 7, 2, 12, 2)     # west -> south
    _carve(grid, 9, 2, 9, 4)      # branch A4
    _carve(grid, 11, 2, 11, 5)    # branch A2
    _carve(grid, 12, 2, 12, 5)    # branch A3

    # --- Family B (north-east): north trunk, then east along the top -------
    _carve(grid, 2, 7, 7, 7)      # hub -> north
    _carve(grid, 2, 7, 2, 12)     # north -> east
    _carve(grid, 2, 9, 5, 9)      # branch B2
    _carve(grid, 2, 12, 5, 12)    # branch B3

    # --- Family C (south-east): south trunk, then east along the bottom ----
    _carve(grid, 7, 7, 12, 7)     # hub -> south
    _carve(grid, 12, 7, 12, 12)   # south -> east
    _carve(grid, 9, 10, 12, 10)   # branch C2
    _carve(grid, 9, 12, 12, 12)   # branch C3

    # Border is always wall.
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    return grid


GRID = build_grid()
START_CELL = (7, 7)

# ----------------------------------------------------------------------------
# Precomputed geometry.
#
# The maze is identical for every task in a suite, so ray distances and
# collision tests depend only on position and can be tabulated once at import
# and then read in O(1).  Doing them with Python loops costs ~1.3 ms per step,
# which would make the simulator, not the SAC update, the bottleneck; the
# tables bring a step to a few tens of microseconds.  Tables are built with
# vectorized NumPy, are fully deterministic, and add a few MB of RAM.
# ----------------------------------------------------------------------------
TABLE_RES = 0.025
_NX = int(round(GRID_W * CELL_SIZE / TABLE_RES))
_NY = int(round(GRID_H * CELL_SIZE / TABLE_RES))


def _table_coords():
    xs = (np.arange(_NX) + 0.5) * TABLE_RES
    ys = (np.arange(_NY) + 0.5) * TABLE_RES
    return np.meshgrid(xs, ys, indexing="ij")


def _cells_of(px, py):
    cc = np.clip((px / CELL_SIZE).astype(np.int64), 0, GRID_W - 1)
    rr = np.clip((py / CELL_SIZE).astype(np.int64), 0, GRID_H - 1)
    return rr, cc


def _build_blocked_table() -> np.ndarray:
    """True where a disc of AGENT_RADIUS overlaps a wall.

    The radius is inflated by half the table cell diagonal so that nearest
    neighbour lookup is *conservative*: any position whose true disc touches a
    wall is guaranteed to be marked blocked.  The agent can therefore never be
    placed inside a wall, where the sensors and the geodesic reward would both
    be meaningless.

    Each wall cell only affects a window of radius (CELL_SIZE + AGENT_RADIUS)
    around itself, so the update is restricted to that window instead of
    touching the whole arena once per wall.
    """
    margin = TABLE_RES * np.sqrt(2.0) / 2.0
    radius = AGENT_RADIUS + margin
    blocked = np.zeros((_NX, _NY), dtype=bool)
    pad = int(np.ceil((CELL_SIZE + radius) / TABLE_RES)) + 1

    for r, c in np.argwhere(GRID == 1):
        x0, y0 = c * CELL_SIZE, r * CELL_SIZE
        ix0 = max(int(x0 / TABLE_RES) - pad, 0)
        ix1 = min(int((x0 + CELL_SIZE) / TABLE_RES) + pad, _NX)
        iy0 = max(int(y0 / TABLE_RES) - pad, 0)
        iy1 = min(int((y0 + CELL_SIZE) / TABLE_RES) + pad, _NY)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        xs = (np.arange(ix0, ix1) + 0.5) * TABLE_RES
        ys = (np.arange(iy0, iy1) + 0.5) * TABLE_RES
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        nx = np.clip(X, x0, x0 + CELL_SIZE)
        ny = np.clip(Y, y0, y0 + CELL_SIZE)
        blocked[ix0:ix1, iy0:iy1] |= ((X - nx) ** 2 + (Y - ny) ** 2) < radius * radius
    return blocked


# Ray distances are a sensor, not a collision test, so they are tabulated on a
# coarser grid; this is the dominant build cost and halving the resolution
# quarters it.
RAY_RES = 0.05
_RX = int(round(GRID_W * CELL_SIZE / RAY_RES))
_RY = int(round(GRID_H * CELL_SIZE / RAY_RES))


def _build_ray_table() -> np.ndarray:
    """Distance to the first wall along each of the eight fixed directions."""
    xs = (np.arange(_RX) + 0.5) * RAY_RES
    ys = (np.arange(_RY) + 0.5) * RAY_RES
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    probe = 0.05
    n_steps = int(RAY_MAX / probe)
    dirs = np.stack(
        [
            np.array([np.cos(a), np.sin(a)])
            for a in np.linspace(0.0, 2.0 * np.pi, N_RAYS, endpoint=False)
        ]
    )
    table = np.full((_RX, _RY, N_RAYS), RAY_MAX, dtype=np.float32)
    for i, (dx, dy) in enumerate(dirs):
        hit = np.zeros((_RX, _RY), dtype=bool)
        layer = table[..., i]
        for k in range(1, n_steps + 1):
            t = k * probe
            rr, cc = _cells_of(X + dx * t, Y + dy * t)
            fresh = (GRID[rr, cc] == 1) & ~hit
            if fresh.any():
                layer[fresh] = t
                hit |= fresh
                if hit.all():
                    break
        table[..., i] = layer
    return table


_BLOCKED_TABLE = _build_blocked_table()
_RAY_TABLE = _build_ray_table()
_GEO_TABLE_CACHE: dict = {}


def _table_index(pos: np.ndarray):
    ix = int(pos[0] / TABLE_RES)
    iy = int(pos[1] / TABLE_RES)
    if ix < 0:
        ix = 0
    elif ix >= _NX:
        ix = _NX - 1
    if iy < 0:
        iy = 0
    elif iy >= _NY:
        iy = _NY - 1
    return ix, iy


def cell_center(cell) -> np.ndarray:
    r, c = cell
    return np.array([(c + 0.5) * CELL_SIZE, (r + 0.5) * CELL_SIZE], dtype=np.float64)


def is_free(grid: np.ndarray, r: int, c: int) -> bool:
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == 0


def bfs_distances(grid: np.ndarray, goal_cell) -> np.ndarray:
    """4-connected BFS distance in cells from every free cell to ``goal_cell``."""
    dist = np.full(grid.shape, np.inf, dtype=np.float64)
    gr, gc = goal_cell
    if not is_free(grid, gr, gc):
        raise ValueError(f"goal cell {goal_cell} is a wall")
    dist[gr, gc] = 0.0
    queue = deque([(gr, gc)])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if is_free(grid, nr, nc) and not np.isfinite(dist[nr, nc]):
                dist[nr, nc] = dist[r, c] + 1.0
                queue.append((nr, nc))
    return dist


def pos_to_cell(pos: np.ndarray) -> Tuple[int, int]:
    c = int(np.clip(np.floor(pos[0] / CELL_SIZE), 0, GRID_W - 1))
    r = int(np.clip(np.floor(pos[1] / CELL_SIZE), 0, GRID_H - 1))
    return r, c


class KinematicPointMaze(gym.Env if gym is not None else object):
    """One PointMaze task: a fixed maze with one fixed goal.

    Parameters
    ----------
    goal_cell:
        Cell holding the goal.
    task_id, family:
        Benchmark metadata; exposed on the instance and in ``info``.
    dynamics:
        2x2 matrix applied to the commanded action before it becomes
        acceleration.  Identity for the goal-only suite; the goal+dynamics
        suite rotates/scales it per task, which is the kinematic shift that
        HiSPO pairs with topological change.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        goal_cell,
        *,
        task_id: int = 0,
        family: str = "A",
        dynamics: Optional[np.ndarray] = None,
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ):
        if _IMPORT_ERROR is not None:  # pragma: no cover
            raise ImportError("PointMaze requires gymnasium") from _IMPORT_ERROR
        super().__init__()

        self.grid = GRID
        self.goal_cell = (int(goal_cell[0]), int(goal_cell[1]))
        self.task_id = int(task_id)
        self.family = str(family)
        self.max_episode_steps = int(max_episode_steps)

        self.dynamics = (
            np.eye(2, dtype=np.float64)
            if dynamics is None
            else np.asarray(dynamics, dtype=np.float64).reshape(2, 2)
        )

        self.goal_pos = cell_center(self.goal_cell)
        self._bfs = bfs_distances(self.grid, self.goal_cell)
        if not np.isfinite(self._bfs[START_CELL]):
            raise ValueError(f"goal {self.goal_cell} is unreachable from the start")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        # Goal-conditioned table, shared between every env that has the same
        # goal so that repeatedly constructing evaluation envs is free.
        cached = _GEO_TABLE_CACHE.get(self.goal_cell)
        if cached is None:
            cached = self._build_geo_table()
            _GEO_TABLE_CACHE[self.goal_cell] = cached
        self._geo_table = cached

        self.pos = cell_center(START_CELL)
        self.vel = np.zeros(2, dtype=np.float64)
        self._steps = 0
        self._succeeded = False

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _build_geo_table(self) -> np.ndarray:
        """Tabulate the in-maze distance to the goal over the whole arena.

        The value at a position is the minimum over the containing cell and
        its free 4-neighbours of (Euclidean reach to that cell's centre + that
        cell's BFS distance), plus the direct distance inside the goal cell.
        A minimum of continuous functions is continuous, so the reward has no
        step at cell boundaries -- which matters for SAC.
        """
        X, Y = _table_coords()
        best = np.full((_NX, _NY), np.inf, dtype=np.float64)

        for r in range(GRID_H):
            for c in range(GRID_W):
                base = self._bfs[r, c]
                if not np.isfinite(base):
                    continue
                cx, cy = cell_center((r, c))
                reach = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
                np.minimum(best, reach + float(base) * CELL_SIZE, out=best)

        gx, gy = self.goal_pos
        in_goal_cell = (
            (X >= self.goal_cell[1] * CELL_SIZE)
            & (X < (self.goal_cell[1] + 1) * CELL_SIZE)
            & (Y >= self.goal_cell[0] * CELL_SIZE)
            & (Y < (self.goal_cell[0] + 1) * CELL_SIZE)
        )
        direct = np.sqrt((X - gx) ** 2 + (Y - gy) ** 2)
        best = np.where(in_goal_cell, np.minimum(best, direct), best)

        if not np.all(np.isfinite(best)):
            raise RuntimeError("geodesic table has unreachable positions")
        return best.astype(np.float32)

    def _blocked(self, pos: np.ndarray) -> bool:
        ix, iy = _table_index(pos)
        return bool(_BLOCKED_TABLE[ix, iy])

    def _ray_cast(self, pos: np.ndarray) -> np.ndarray:
        ix = int(pos[0] / RAY_RES)
        iy = int(pos[1] / RAY_RES)
        ix = 0 if ix < 0 else (_RX - 1 if ix >= _RX else ix)
        iy = 0 if iy < 0 else (_RY - 1 if iy >= _RY else iy)
        return _RAY_TABLE[ix, iy]

    def _geodesic(self, pos: np.ndarray) -> float:
        """Bilinearly interpolated in-maze distance from ``pos`` to the goal.

        Interpolating rather than snapping keeps the reward strictly monotone
        along a corridor, so the policy gradient does not see a staircase.
        """
        fx = pos[0] / TABLE_RES - 0.5
        fy = pos[1] / TABLE_RES - 0.5
        x0 = int(np.floor(fx))
        y0 = int(np.floor(fy))
        tx = fx - x0
        ty = fy - y0
        x0 = min(max(x0, 0), _NX - 2)
        y0 = min(max(y0, 0), _NY - 2)
        tx = min(max(tx, 0.0), 1.0)
        ty = min(max(ty, 0.0), 1.0)

        g = self._geo_table
        v00 = g[x0, y0]
        v10 = g[x0 + 1, y0]
        v01 = g[x0, y0 + 1]
        v11 = g[x0 + 1, y0 + 1]
        return float(
            v00 * (1 - tx) * (1 - ty)
            + v10 * tx * (1 - ty)
            + v01 * (1 - tx) * ty
            + v11 * tx * ty
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        return np.concatenate(
            [self.pos, self.vel, self._ray_cast(self.pos)]
        ).astype(np.float32)

    def _info(self, **extra) -> dict:
        info = {
            "task_id": self.task_id,
            "family": self.family,
            "goal_cell": self.goal_cell,
            "is_success": bool(self._succeeded),
            "geodesic": float(self._geodesic(self.pos)),
        }
        info.update(extra)
        return info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            self._rng = np.random.default_rng(seed)
        elif not hasattr(self, "_rng"):
            self._rng = np.random.default_rng()

        # A small start jitter keeps the task from being solvable by one
        # memorized open-loop action sequence, without changing which cell the
        # agent starts in.
        jitter = self._rng.uniform(-0.15, 0.15, size=2)
        self.pos = cell_center(START_CELL) + jitter
        self.vel = np.zeros(2, dtype=np.float64)
        self._steps = 0
        self._succeeded = False
        return self._obs(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)
        accel = self.dynamics @ action * ACCEL

        self.vel = self.vel * DAMPING + accel * DT
        speed = float(np.linalg.norm(self.vel))
        if speed > MAX_SPEED:
            self.vel *= MAX_SPEED / speed

        proposed = self.pos + self.vel * DT
        if not self._blocked(proposed):
            self.pos = proposed
        else:
            # Try the axes one at a time and keep the FIRST that is free.
            #
            # Applying both independently-free axis moves can place the agent
            # diagonally through a blocked corner and leave it inside a wall,
            # where the ray sensors and the geodesic reward are both
            # meaningless.  Taking only one axis cannot do that.
            moved = False
            for axis in (0, 1):
                candidate = self.pos.copy()
                candidate[axis] += self.vel[axis] * DT
                if not self._blocked(candidate):
                    self.pos = candidate
                    self.vel[1 - axis] = 0.0
                    moved = True
                    break
            if not moved:
                self.vel[:] = 0.0

        self._steps += 1
        distance = self._geodesic(self.pos)
        reward = -distance / REWARD_SCALE

        reached = float(np.linalg.norm(self.pos - self.goal_pos)) <= GOAL_RADIUS
        if reached:
            self._succeeded = True
        terminated = bool(reached)
        truncated = bool(self._steps >= self.max_episode_steps)
        return self._obs(), float(reward), terminated, truncated, self._info()

    def render(self):  # pragma: no cover - not used by the benchmark
        raise NotImplementedError("PointMaze has no renderer")

    def close(self):
        return None


def ascii_map(goal_cell=None) -> str:
    """Human-readable maze, for README figures and debugging."""
    rows = []
    for r in range(GRID_H):
        line = []
        for c in range(GRID_W):
            if (r, c) == START_CELL:
                line.append("S")
            elif goal_cell is not None and (r, c) == tuple(goal_cell):
                line.append("G")
            else:
                line.append("#" if GRID[r, c] else ".")
        rows.append("".join(line))
    return "\n".join(rows)
