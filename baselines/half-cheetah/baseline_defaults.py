"""Suite-specific defaults for the baseline runner. HALFCHEETAH COPY.

This is the ONLY file in the baseline stack that differs between
half-cheetah/ and metaworld/. Everything else -- run_baseline.py,
baseline_identity.py, baseline_smoke.py and the whole baselines/ package -- is
byte-identical across the two folders, exactly as cka_rl.py and
knowledge_pools.py already are.

The values below MIRROR run_sac.py's dataclass defaults for this suite. They
are duplicated rather than imported because importing run_sac.py executes its
tyro entry point. Keep them in sync: a baseline trained at a different budget
than the method is not a comparison.

    run_sac.Args.task_suite       = "halfcheetah_vel"
    run_sac.Args.total_timesteps  = 300_000
    run_sac.Args.distill_extra_steps = 10_000     (B, the frozen tail)
    run_sac.Args.learning_starts  = 5_000
    run_sac.Args.random_actions_end = 5_000
    run_sac.Args.eval_every       = 10_000
    run_sac.Args.num_evals        = 5

The sequence is 12 positions over 6 unique tasks: range(6) twice, so the second
pass measures retention and relearning rather than first-time acquisition.
"""
from __future__ import annotations

import tasks

DEFAULT_TASK_SUITE = "halfcheetah_vel"
DEFAULT_TOTAL_TIMESTEPS = 300_000
DEFAULT_FROZEN_TAIL_STEPS = 10_000
DEFAULT_LEARNING_STARTS = 5_000
DEFAULT_RANDOM_ACTIONS_END = 5_000
DEFAULT_EVAL_EVERY = 10_000
DEFAULT_NUM_EVALS = 5

#: Seeds for the continual chains, matching run_kaggle.sh's HalfCheetah stage.
DEFAULT_SEEDS = (101, 102)
#: Disjoint from DEFAULT_SEEDS, as scratch_baselines.py requires.
DEFAULT_SCRATCH_SEEDS = (201,)


def default_sequence(task_suite: str = DEFAULT_TASK_SUITE):
    """Return the continual sequence for a suite.

    Mirrors run_continual_benchmark.py's resolution: Meta-World's tasks.py
    exposes a per-suite ``default_sequence``; HalfCheetah's exposes a single
    ``DEFAULT_CONTINUAL_SEQUENCE``.
    """
    if hasattr(tasks, "default_sequence"):
        return tuple(tasks.default_sequence(task_suite))
    return tuple(tasks.DEFAULT_CONTINUAL_SEQUENCE)
