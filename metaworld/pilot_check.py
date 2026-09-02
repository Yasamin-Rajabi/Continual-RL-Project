"""Pilot check: is a 150k-step budget actually enough for these four tasks?

RUN THIS BEFORE ANYTHING ELSE. It costs about one GPU-hour and decides whether
the whole experimental plan is viable.

WHY IT IS NOT OPTIONAL
----------------------
No published source reports single-task SAC learning curves for Meta-World at
the 150k scale -- the community convention is 1M steps per task (Continual
World, TD-MPC). Our task selection is an argued extrapolation from the MT50
difficulty partition, not a measured fact. If the four chosen tasks do not
reach a meaningful success rate in 150k steps, every downstream comparison
between fusion modes is comparing noise, and no amount of seeds will fix it.

WHAT IT DOES
------------
Trains from-scratch SAC on each task in the suite for --steps steps, one seed,
and reports the success curve. Reuses run_sac.py exactly, so what it measures
is what the benchmark will do.

HOW TO READ THE RESULT
----------------------
For each task the script prints final success and the step at which success
first exceeds 0.5.

  - final success >= 0.6 on all four         -> proceed as planned.
  - final success >= 0.6 but reached before
    ~30k steps on most tasks                 -> tasks saturate too early; the
                                                comparison window is tiny. Move
                                                to the mw_easy6 suite or lower
                                                --steps so the curve, not the
                                                plateau, dominates the AUC.
  - any task stuck near 0                    -> drop that task. Easy-tier
                                                alternatives that keep the
                                                interference structure:
                                                door-close-v2, drawer-close-v2,
                                                button-press-topdown-v2,
                                                plate-slide-v2.

The from-scratch runs this produces are ALSO the FT baselines the benchmark
needs, so nothing here is wasted work -- point scratch_baselines at the same
--runs-root and it will reuse them.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

import numpy as np

from tasks import TASK_SUITES, get_task_name

TEST_LINE = re.compile(
    r"TEST:\s*return=(?P<ret>[-0-9.]+),\s*success=(?P<succ>[0-9.]+)"
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task-suite", default="mw_easy4", choices=sorted(TASK_SUITES))
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--tasks", nargs="+", type=int, default=None,
                   help="Task ids to check. Default: all in the suite.")
    p.add_argument("--runs-root", default="runs")
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
        "--fusion-mode=classic_cka",
        "--no-use-alpha-scale",
        "--no-distillation",
        "--no-use-alpha-mass",
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
        m = TEST_LINE.search(line)
        if m:
            successes.append(float(m.group("succ")))
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"pilot run for task {task_id} failed ({proc.returncode})")
    return successes


def main():
    args = parse_args()
    if not pathlib.Path("run_sac.py").exists():
        raise SystemExit("run this from the project directory containing run_sac.py")

    task_ids = args.tasks if args.tasks is not None else list(range(len(TASK_SUITES[args.task_suite])))

    results = {}
    for tid in task_ids:
        results[tid] = run_one(args, tid)

    print("\n" + "=" * 74)
    print(f"PILOT RESULT  ({args.steps} steps, seed {args.seed}, 1 seed only)")
    print("=" * 74)
    print(f"{'task':28s} {'final':>7s} {'best':>7s} {'step@0.5':>10s}  verdict")

    verdicts = []
    for tid, curve in results.items():
        name = get_task_name(tid, args.task_suite)
        if not curve:
            print(f"{name:28s} {'--':>7s} {'--':>7s} {'--':>10s}  NO EVAL DATA")
            verdicts.append("bad")
            continue
        arr = np.asarray(curve)
        final, best = float(arr[-1]), float(arr.max())
        over = np.nonzero(arr >= 0.5)[0]
        step_half = int((over[0] + 1) * args.eval_every) if over.size else -1

        if final >= 0.6 and 0 < step_half <= 0.2 * args.steps:
            verdict, tag = "saturates early", "early"
        elif final >= 0.6:
            verdict, tag = "good", "good"
        elif best >= 0.3:
            verdict, tag = "marginal - needs more steps", "marginal"
        else:
            verdict, tag = "NOT LEARNED - replace this task", "bad"
        verdicts.append(tag)
        shown = f"{step_half}" if step_half > 0 else "never"
        print(f"{name:28s} {final:7.2f} {best:7.2f} {shown:>10s}  {verdict}")

    print("-" * 74)
    if "bad" in verdicts:
        print("ACTION: at least one task never learned. Replace it before running the "
              "benchmark -- it would contribute pure noise to FG, BWT and FT.\n"
              "        Easy-tier substitutes: door-close-v2, drawer-close-v2,\n"
              "        button-press-topdown-v2, plate-slide-v2.")
    elif verdicts.count("early") >= len(verdicts) // 2:
        print("ACTION: most tasks saturate in the first fifth of the budget, so the "
              "AUC is dominated by the plateau and methods will look identical.\n"
              "        Either lower --total-timesteps, or switch to the mw_easy6 suite.")
    elif "marginal" in verdicts:
        print("ACTION: some tasks are marginal. Either raise the budget for those, or "
              "accept that they mostly measure forgetting rather than transfer.")
    else:
        print("ACTION: all four tasks learn inside the budget. Proceed with the "
              "full benchmark.")
    print("\nThese from-scratch runs double as the FT baselines -- point "
          "scratch_baselines.py at the same --runs-root to reuse them.")


if __name__ == "__main__":
    main()
