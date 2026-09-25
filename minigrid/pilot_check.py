"""Cheap single-task pilot for the 60k MiniGrid paper budget.

This is a screening tool, not part of metric collection. It trains the same
categorical SAC implementation used by the continual benchmark on each task and
reports its observed success curve. The canonical FT scratch references are
created by `job_paper.sh --phase scratch` / the normal paper launcher.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

import numpy as np

from tasks import TASK_SUITES, get_task_name

TEST_LINE = re.compile(r"TEST:\s*return=(?P<ret>[-0-9.]+),\s*success=(?P<succ>[0-9.]+)")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task-suite", default="doorkey4", choices=sorted(TASK_SUITES))
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=2_500)
    p.add_argument("--tasks", nargs="+", type=int, default=None)
    p.add_argument("--runs-root", default="runs_pilot")
    p.add_argument("--save-dir", default="agents_pilot")
    p.add_argument("--analysis-root", default="analysis_pilot")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def run_one(args, task_id: int):
    cmd = [
        sys.executable, "run_sac.py",
        "--model-type=cka-rl",
        f"--task-suite={args.task_suite}",
        f"--task-id={task_id}",
        f"--seed={args.seed}",
        "--tag=pilot",
        f"--runs-root={args.runs_root}",
        f"--save-dir={args.save_dir}",
        f"--analysis-root={args.analysis_root}",
        f"--total-timesteps={args.steps}",
        f"--eval-every={args.eval_every}",
        "--learning-starts=1000",
        "--random-actions-end=2000",
        "--fusion-mode=classic_cka",
        "--composition-space=parameter",
        "--no-use-alpha-scale",
        "--no-distillation",
        "--no-use-alpha-mass",
        "--no-save-analysis-snapshots",
    ]
    if args.cpu:
        cmd.append("--no-cuda")

    print(f"\n>>> pilot task {task_id} ({get_task_name(task_id, args.task_suite)})")
    print("   ", " ".join(cmd), flush=True)
    successes = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        match = TEST_LINE.search(line)
        if match:
            successes.append(float(match.group("succ")))
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"pilot run for task {task_id} failed ({proc.returncode})")
    return successes


def main():
    args = parse_args()
    if not pathlib.Path("run_sac.py").exists():
        raise SystemExit("run this from the minigrid directory")
    task_ids = args.tasks if args.tasks is not None else list(range(len(TASK_SUITES[args.task_suite])))
    results = {task_id: run_one(args, task_id) for task_id in task_ids}

    print("\n" + "=" * 78)
    print(f"MINIGRID PILOT ({args.steps} steps, seed {args.seed}; screening only)")
    print("=" * 78)
    print(f"{'task':30s} {'final':>7s} {'best':>7s} {'first >=0.5':>12s}")
    for task_id, curve in results.items():
        name = get_task_name(task_id, args.task_suite)
        if not curve:
            print(f"{name:30s} {'--':>7s} {'--':>7s} {'--':>12s}")
            continue
        values = np.asarray(curve, dtype=np.float64)
        above = np.flatnonzero(values >= 0.5)
        first = int((above[0] + 1) * args.eval_every) if above.size else None
        first_text = str(first) if first is not None else "never"
        print(f"{name:30s} {values[-1]:7.3f} {values.max():7.3f} {first_text:>12s}")

    print("\nInterpret this as an empirical budget check. If a task remains near-zero, "
          "do not launch the full grid blindly; inspect its curve/horizon or increase "
          "the budget. Use job_paper.sh for the canonical scratch references and paper metrics.")


if __name__ == "__main__":
    main()
