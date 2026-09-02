"""Calibrate Ant's sustainable forward velocity with a dedicated reward.

The old calibration encoded "run forward" as a target velocity of 1000 m/s.
Although subtracting a constant does not change an exact fixed-horizon optimum,
it forced SAC to learn Q-values around -100,000 and injected a 1000-valued task
coordinate into the network input.  This script instead trains on
AntForwardCalibrationEnv, whose reward is simply

    x_velocity - ctrl_cost_weight * ||action||^2

and whose observation is the raw Ant state.  The environment otherwise uses the
same dynamics, control penalty and 1000-step horizon as the benchmark.

The script runs several seeds, takes a robust sustainable velocity estimate,
and writes ant_calibration.json.  tasks.py automatically consumes that file on
the next Python process, so no manual source-code editing is required.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from tasks import ANT_CALIBRATION_PATH, _ANT_VELOCITY_FRACTIONS

X_VELOCITY_RE = re.compile(r"x_velocity=(-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--total-timesteps", type=int, default=150_000)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--num-evals", type=int, default=5)
    p.add_argument("--save-root", default="agents_ant_calibrate")
    p.add_argument("--runs-root", default="runs_ant_calibrate")
    p.add_argument("--analysis-root", default="analysis_ant_calibrate")
    p.add_argument("--output", default=str(ANT_CALIBRATION_PATH))
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--tail-evals", type=int, default=3,
        help="Per seed, use the median mean velocity among the final N evaluations. "
             "This avoids both a single noisy last evaluation and optimistic max-selection.",
    )
    p.add_argument("--tolerance-frac", type=float, default=0.08)
    return p.parse_args()


def _run_one_seed(args, seed: int) -> tuple[float, list[float]]:
    tag = f"ant_calibrate/seed_{seed}"
    save_parent = pathlib.Path(args.save_root) / f"seed_{seed}"
    event_dir = pathlib.Path(args.runs_root) / tag
    analysis_dir = pathlib.Path(args.analysis_root) / tag
    for path in (save_parent, event_dir, analysis_dir):
        if path.exists():
            shutil.rmtree(path)

    cmd = [
        sys.executable,
        "run_sac.py",
        "--model-type=cka-rl",
        "--task-suite=ant_calibrate",
        "--task-id=0",
        "--seq-idx=0",
        f"--seed={seed}",
        f"--tag={tag}",
        f"--save-dir={save_parent}",
        f"--runs-root={args.runs_root}",
        f"--analysis-root={args.analysis_root}",
        f"--total-timesteps={args.total_timesteps}",
        f"--eval-every={args.eval_every}",
        f"--num-evals={args.num_evals}",
        "--fusion-mode=classic_cka",
        "--no-use-alpha-scale",
        "--no-distillation",
        "--no-use-alpha-mass",
        "--no-collect-cosine-buffers",
        "--no-train-shared",
        "--no-freeze-root-encoder",
        "--no-encoder-linear-out",
    ]
    if args.cpu:
        cmd.append("--no-cuda")

    print("\n>>> calibration seed", seed, flush=True)
    print(" ".join(cmd), flush=True)

    velocities: list[float] = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        match = X_VELOCITY_RE.search(line)
        if match:
            velocities.append(float(match.group(1)))
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit(f"calibration seed {seed} failed with code {proc.returncode}")
    if not velocities:
        raise SystemExit(
            f"calibration seed {seed} produced no x_velocity evaluations; "
            "check run_sac.py evaluation logging"
        )

    tail = velocities[-max(1, int(args.tail_evals)):]
    sustainable = float(statistics.median(tail))
    return sustainable, velocities


def main():
    args = parse_args()
    if not pathlib.Path("run_sac.py").exists():
        raise SystemExit("run this from the ant directory containing run_sac.py")
    if args.total_timesteps < args.eval_every:
        raise SystemExit("total_timesteps must be >= eval_every so calibration is evaluated")
    if args.tail_evals < 1:
        raise SystemExit("tail_evals must be >= 1")

    per_seed = {}
    all_eval_traces = {}
    for seed in args.seeds:
        sustainable, trace = _run_one_seed(args, seed)
        per_seed[str(seed)] = float(sustainable)
        all_eval_traces[str(seed)] = [float(x) for x in trace]

    v_max = float(statistics.median(per_seed.values()))
    if v_max <= 0.0:
        raise SystemExit(
            f"calibration produced non-positive sustainable velocity {v_max:.3f}; "
            "increase training time or debug Ant learning before defining tasks"
        )

    output = pathlib.Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "method": "AntForwardCalibrationEnv",
        "reward": "x_velocity - 0.05 * sum(action^2)",
        "v_max": v_max,
        "seeds": [int(x) for x in args.seeds],
        "per_seed_sustainable_velocity": per_seed,
        "evaluation_traces": all_eval_traces,
        "total_timesteps": int(args.total_timesteps),
        "eval_every": int(args.eval_every),
        "num_evals": int(args.num_evals),
        "tail_evals": int(args.tail_evals),
        "success_tolerance_frac": float(args.tolerance_frac),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with output.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    velocities = tuple(round(f * v_max, 3) for f in _ANT_VELOCITY_FRACTIONS)
    tolerance = round(args.tolerance_frac * v_max, 3)

    print("\n" + "=" * 72)
    print(f"Calibrated sustainable Ant velocity: v_max = {v_max:.3f} m/s")
    print(f"Per-seed estimates: {per_seed}")
    print(f"Wrote: {output}")
    print("=" * 72)
    print(f"Benchmark target velocities: {velocities}")
    print(f"Success tolerance: {tolerance}")
    print(
        "Restart/re-run the next Python command so tasks.py reloads the new JSON. "
        "No source edit is needed."
    )
    print(
        "After the first full seed, inspect the 1.10x task. If every method floors, "
        "change the task fractions only in a new explicitly-versioned experiment."
    )


if __name__ == "__main__":
    main()
