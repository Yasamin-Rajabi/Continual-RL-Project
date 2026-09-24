"""Fixed, reproducible eight-direction AntDir continual benchmark.

Eight directions at pi/4 intervals, visited twice. This is a declared fixed
suite, not a claim to reproduce the randomly sampled PEARL meta-test split.
Success is a custom per-step heading/speed diagnostic; report raw return too.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class AntDirTask:
    direction: float
    def label(self, suite=''):
        return f'direction-{round(math.degrees(self.direction)):03d}'

TASK_SUITES = {'ant_dir': [AntDirTask(k * math.pi / 4) for k in range(8)]}
DEFAULT_CONTINUAL_SEQUENCE = tuple(range(8)) * 2
SEQUENCES = {'ant_dir': DEFAULT_CONTINUAL_SEQUENCE}

def available_task_suites():
    return tuple(TASK_SUITES)

def default_sequence(task_suite='ant_dir'):
    if task_suite not in SEQUENCES:
        raise ValueError(f'Unknown AntDir suite: {task_suite}')
    return SEQUENCES[task_suite]

def get_task_spec(task_id, task_suite='ant_dir'):
    return TASK_SUITES[task_suite][task_id]

def get_task_name(task_id, task_suite='ant_dir'):
    return get_task_spec(task_id, task_suite).label()

def get_task(task_id, task_suite='ant_dir', render=False):
    from antdir_envs import make_env
    return make_env(get_task_spec(task_id, task_suite).direction, render=render)

if __name__ == '__main__':
    print('ant_dir:', [(i,t.label()) for i,t in enumerate(TASK_SUITES['ant_dir'])])
    print('sequence:', DEFAULT_CONTINUAL_SEQUENCE)
