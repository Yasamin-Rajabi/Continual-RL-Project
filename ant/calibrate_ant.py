"""Measure the forward velocity Ant can actually reach, then print the task
targets to paste into tasks.py.

WHY THIS IS MANDATORY BEFORE ANY REAL ANT RUN
---------------------------------------------
The HalfCheetah suite's targets (0.5 .. 3.0 m/s) were chosen for a robot that
reaches roughly 5-8 m/s under SAC. Ant is substantially slower. If you keep
absolute targets across robots, the fast tasks become unreachable, several of
them collapse onto the single behaviour "run flat out", and they stop being
distinct tasks -- at which point the benchmark cannot distinguish any method
from any other and the whole Ant sweep measures nothing.

HOW IT WORKS
------------
No new training code. tasks.py defines a throwaway `ant_calibrate` suite with a
single task whose target velocity is 1000 m/s. Since Ant can never approach
that, the reward

    -|v_x - 1000| - ctrl_cost   ==   v_x - 1000 - ctrl_cost

reduces to plain forward-reward SAC with the same control penalty the real
tasks use. We run the existing run_sac.py on it, read the evaluation
velocity_error off stdout, and recover v_max = 1000 - velocity_error.

Using the same ctrl_cost_weight as the real tasks matters: a velocity ceiling
measured without the control penalty would be optimistic, and the top task
would then be unreachable in exactly the way this script exists to prevent.

USAGE
-----
    python3 calibrate_ant.py --total-timesteps 150000
    # then paste the printed _ANT_V_MAX into tasks.py

NOT EXECUTED IN THIS SANDBOX (no torch/mujoco) -- syntax-checked only.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

CALIBRATION_TARGET = 1000.0
VELOCITY_ERROR_RE = re.compile(r"velocity_error=([0-9]*\.?[0-9]+)")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--total-timesteps", type=int, default=150_000,
                   help="Long enough for the gait to stabilise. Too short and you "
                        "will underestimate v_max, making every task too easy.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-dir", default="agents_ant_calibrate")
    p.add_argument("--analysis-root", default="logs_ant_calibrate")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--fractions", nargs="+", type=float,
                   default=[0.10, 0.30, 0.50, 0.70, 0.90, 1.10, 0.40, 0.80],
                   help="Must match _ANT_VELOCITY_FRACTIONS in tasks.py.")
    p.add_argument("--tolerance-frac", type=float, default=0.08,
                   help="Success tolerance as a fraction of v_max. HalfCheetah's "
                        "fixed 0.2 over a 0.5-3.0 band is about this much.")
    return p.parse_args()


def run_calibration(args) -> float:
    cmd = [
        sys.executable, "run_sac.py",
        "--model-type=cka-rl",
        "--task-suite=ant_calibrate",
        "--task-id=0",
        f"--seed={args.seed}",
        "--tag=ant_calibrate",
        f"--save-dir={args.save_dir}",
        f"--analysis-root={args.analysis_root}",
        f"--total-timesteps={args.total_timesteps}",
        "--fusion-mode=classic_cka",
        "--no-use-alpha-scale",
        "--no-distillation",
        "--no-use-alpha-mass",
    ]
    if args.cpu:
        cmd.append("--no-cuda")

    print(">>> calibration run:", " ".join(cmd), flush=True)

    errors = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        m = VELOCITY_ERROR_RE.search(line)
        if m:
            errors.append(float(m.group(1)))
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"calibration run failed with code {proc.returncode}")
    if not errors:
        raise SystemExit(
            "no velocity_error found in the run output. Check that the "
            "ant_calibrate suite exists in tasks.py and that evaluation ran."
        )

    # Take the best (lowest) evaluation error rather than the last one: we want
    # the ceiling the policy demonstrably reached, and the final eval can dip
    # below its own peak on a noisy seed.
    best_error = min(errors)
    return CALIBRATION_TARGET - best_error


def main():
    args = parse_args()
    if not pathlib.Path("run_sac.py").exists():
        raise SystemExit("run this from the project directory containing run_sac.py")

    v_max = run_calibration(args)
    velocities = tuple(round(f * v_max, 3) for f in args.fractions)
    tolerance = round(args.tolerance_frac * v_max, 3)

    print("\n" + "=" * 68)
    print(f"Measured reachable velocity: v_max = {v_max:.3f} m/s")
    print("=" * 68)
    print("\nPaste into tasks.py:\n")
    print(f"_ANT_V_MAX = {round(v_max, 3)}")
    print(f"_ANT_VELOCITY_FRACTIONS = {tuple(args.fractions)}")
    print("\nWhich yields:")
    print(f"  _ANT_VELOCITIES        = {velocities}")
    print(f"  _ANT_SUCCESS_TOLERANCE = {tolerance}")
    print(f"  continual sequence (tasks 0-5): {velocities[:6]}")

    if v_max < 0.5:
        print("\nWARNING: v_max is very low. Either the run was too short, or Ant "
              "never learned to move. Do NOT use these numbers -- re-run with more "
              "timesteps first.")
    print("\nAfter the first Ant seed finishes, check task 5 (the 1.1x task). If "
          "EVERY method scores ~0 there, it is measuring 'impossible' rather than "
          "'hard' -- lower the top fraction to 1.0 and re-run.")


if __name__ == "__main__":
    main()
