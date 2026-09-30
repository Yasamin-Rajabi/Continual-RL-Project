"""Quick 2-task smoke test of the alpha/alpha_mass gradient fix.

Trains "Ours" on only the FIRST 2 TASKS of the sequence (not all 10), into
its own separate output folders, so it never touches or overwrites your
real 10-task run's checkpoints. This is meant to answer one question fast
(roughly 1-1.5h instead of 7-8h): does the novel/just-trained component's
mixture weight now move in response to the task, instead of collapsing to
~0 every time regardless of fit?

Usage (from a Kaggle cell, in the folder containing the pointmaze package):

    !python run_alpha_fix_smoke.py

Then run check_alpha_fix_smoke.py to see the verdict.
"""
import sys

import tasks
import run_continual_benchmark as bench

_ORIG_SEQUENCE = tasks.get_continual_sequence


def _short_sequence(suite=tasks.DEFAULT_SUITE, *, repeats=1):
    return _ORIG_SEQUENCE(suite, repeats=repeats)[:2]


# Patch the sequence length everywhere it's looked up from.
tasks.get_continual_sequence = _short_sequence
bench.get_continual_sequence = _short_sequence

SMOKE_ARGS = [
    "--methods", "Ours",
    "--seeds", "1",
    "--tag", "alpha_fix_smoke",
    "--save-root", "agents_pointmaze_smoke",
    "--runs-root", "runs_pointmaze_smoke",
    "--results-root", "results_pointmaze_smoke",
    "--analysis-root", "analysis_pointmaze_smoke",
    # Same per-task budget as the real run, locked so nothing silently
    # recomputes a different value and retrains from scratch later.
    "--total-timesteps", "260000",
    "--distill-extra-steps", "10400",
    "--skip-eval",  # we only care about the checkpoints/logs here, not scoring
]

if __name__ == "__main__":
    raise SystemExit(bench.main(SMOKE_ARGS))
