"""Fast environment smoke test.  Needs only NumPy and Gymnasium -- no torch.

Run this first on a new machine; it takes a few seconds and catches every
environment-level problem before a GPU session is spent.

    python pointmaze_smoke.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

import pointmaze_env as pm
import tasks


def scripted_controller(env, max_steps=None, seed=0):
    """Follow the BFS gradient, inverting the task's actuator matrix."""
    max_steps = max_steps or pm.MAX_EPISODE_STEPS
    env.reset(seed=seed)
    dist = pm.bfs_distances(pm.GRID, env.goal_cell)
    inv = np.linalg.inv(env.dynamics)
    total = 0.0
    for step in range(max_steps):
        r, c = pm.pos_to_cell(env.pos)
        best, best_d = None, dist[r, c]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if pm.is_free(pm.GRID, rr, cc) and dist[rr, cc] < best_d:
                best_d, best = dist[rr, cc], (rr, cc)
        target = env.goal_pos if best is None else pm.cell_center(best)
        delta = target - env.pos
        norm = float(np.linalg.norm(delta))
        want = (delta / norm if norm > 1e-9 else np.zeros(2)) - 0.35 * env.vel
        action = np.clip(inv @ want, -1.0, 1.0)
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            return step + 1, total, bool(info["is_success"])
    return max_steps, total, False


def main() -> int:
    failures = []

    def check(label, condition, detail=""):
        status = "ok  " if condition else "FAIL"
        print(f"  [{status}] {label}{(' -- ' + detail) if detail else ''}")
        if not condition:
            failures.append(label)

    print("suite validation")
    for suite in tasks.SUITES:
        report = tasks.validate_suite(suite)
        within = report["within_family_shared_prefix"]
        across = report["across_family_shared_prefix"]
        check(
            f"{suite}: families alternate and cluster",
            within["min"] > across["max"],
            f"within>={within['min']} across<={across['max']} order={''.join(report['families'])}",
        )

    print("\nsolvability (scripted controller, 5 seeds x 10 tasks)")
    for suite in tasks.SUITES:
        lengths, returns, successes = [], [], []
        for task_id in range(tasks.NUM_TASKS):
            for seed in range(5):
                n, ret, ok = scripted_controller(tasks.get_task(task_id, suite), seed=seed)
                lengths.append(n)
                returns.append(ret)
                successes.append(ok)
        check(
            f"{suite}: every task solved",
            all(successes),
            f"steps {min(lengths)}-{max(lengths)} of {pm.MAX_EPISODE_STEPS}, "
            f"mean return {np.mean(returns):.1f}",
        )

    print("\nlearning signal")
    random_returns = []
    for task_id in range(tasks.NUM_TASKS):
        env = tasks.get_task(task_id)
        env.reset(seed=7)
        rng = np.random.default_rng(task_id)
        total = 0.0
        for _ in range(pm.MAX_EPISODE_STEPS):
            _, reward, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 2))
            total += reward
            if terminated or truncated:
                break
        random_returns.append(total)
    scripted = np.mean(
        [scripted_controller(tasks.get_task(t), seed=0)[1] for t in range(tasks.NUM_TASKS)]
    )
    check(
        "scripted policy clearly beats random",
        scripted > np.mean(random_returns) + 50,
        f"random {np.mean(random_returns):.1f} vs scripted {scripted:.1f}",
    )

    print("\ninvariants")
    positive, in_wall, steps = 0, 0, 0
    for task_id in range(tasks.NUM_TASKS):
        for suite in tasks.SUITES:
            env = tasks.get_task(task_id, suite)
            env.reset(seed=3)
            rng = np.random.default_rng(100 + task_id)
            for _ in range(pm.MAX_EPISODE_STEPS):
                _, reward, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 2))
                steps += 1
                positive += int(reward > 0)
                in_wall += int(env._blocked(env.pos))
                if terminated or truncated:
                    break
    check("reward never positive (RETURN_UPPER_BOUND=0 holds)", positive == 0, f"{steps} steps")
    check("agent never inside a wall", in_wall == 0)

    env_a, env_b = tasks.get_task(4), tasks.get_task(4)
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    same = np.allclose(obs_a, obs_b)
    for action in np.random.default_rng(0).uniform(-1, 1, (50, 2)):
        ra, rb = env_a.step(action), env_b.step(action)
        same &= np.allclose(ra[0], rb[0]) and ra[1] == rb[1]
    check("reset(seed) + 50 steps reproduce exactly", bool(same))

    env = tasks.get_task(0)
    env.reset(seed=0)
    values = [env._geodesic(np.array([x, 7.5])) for x in np.linspace(2.6, 7.4, 300)]
    deltas = np.diff(values)
    check(
        "geodesic reward is monotone along a corridor",
        bool(np.all(deltas < 0) or np.all(deltas > 0)),
        f"max step {np.max(np.abs(deltas)):.4f}",
    )

    print("\nspeed")
    env = tasks.get_task(0)
    env.reset(seed=0)
    action = np.array([0.3, 0.4])
    t0 = time.time()
    n = 20_000
    for _ in range(n):
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=0)
    per_step = (time.time() - t0) / n * 1e6
    check("simulation is not the bottleneck", per_step < 200, f"{per_step:.1f} us/step")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("environment smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
