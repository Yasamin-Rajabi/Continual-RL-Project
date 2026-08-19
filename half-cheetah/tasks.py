import numpy as np

from utils import MujocoMetaBenchmark, MujocoTaskSamplerEnv

# ==========================================================================
# HalfCheetah-Vel continual setup: each "task" is a fixed target forward
# velocity for the cheetah to run at (reward = -|actual_velocity -
# target_velocity| - control_cost, from MujocoTaskSamplerEnv). Unlike
# MetaWorld, there's no natural notion of a discrete "success" here -- see
# the note at the bottom of this file.
#
# The task list is built ONCE, deterministically, from BENCHMARK_SEED, so
# task_id -> target velocity is stable across every process/run (each task
# is trained in its own subprocess -- this must NOT be re-randomized per
# call, or task_id=3 would mean a different velocity every time it's used).
# ==========================================================================
BENCHMARK_SEED = 1
MIN_VELOCITY = 0.0
MAX_VELOCITY = 3.0
NUM_TASKS = 7  # change this to however many distinct velocities you want

_benchmark = MujocoMetaBenchmark(
    task_set="HalfCheetahVel",
    seed=BENCHMARK_SEED,
    num_train_tasks=NUM_TASKS,
    min_velocity=MIN_VELOCITY,
    max_velocity=MAX_VELOCITY,
)
_tasks_list = _benchmark.train_tasks  # length == NUM_TASKS, fixed order (seeded)

tasks = [t.env_name for t in _tasks_list]


def get_task_name(task_id):
    return tasks[task_id]


def get_target_velocity(task_id):
    """Not part of the original tasks.py interface -- convenience accessor
    if you want to print/log which velocity a task_id actually corresponds
    to, e.g. in TASK_SEQUENCE docstrings or plot titles."""
    return _tasks_list[task_id].target_velocity


def get_task(task_id, render=False):
    task = _tasks_list[task_id]
    render_kwargs = {"render_mode": "human"} if render else None
    # Restricting `tasks` to just this one task means sample_new_task()
    # (called internally by __init__ with no arguments) has only one
    # possible choice -- deterministically this exact velocity, every time.
    env = MujocoTaskSamplerEnv(
        classes={task.env_name: None},
        tasks=[task],
        seed=int(np.random.randint(0, 1024)),
        render_kwargs=render_kwargs,
    )
    return env


if __name__ == "__main__":
    env = get_task(0, render=True)

    for _ in range(200):
        obs, _ = env.reset()
        a = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(a)
        if terminated:
            break

# ==========================================================================
# NOTE on "success": HalfCheetah-Vel has no discrete success signal --
# MujocoTaskSamplerEnv's info dict always includes "success": False
# (unconditionally, never True). This means run_sac.py/run_continual_
# benchmark.py's success-rate logging (charts/success, charts/test_success)
# will still run without crashing (the key exists), but will always read
# exactly 0 -- not "no data", just a constant zero line. episodic_return is
# the metric that actually reflects performance for this environment; treat
# any "success" plots for this setup as meaningless, not as "0% success".
# ==========================================================================
