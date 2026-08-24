"""Summarize timing scalars from completed HalfCheetah benchmark runs."""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plots-root", default="plots_halfcheetah_continual")
    return p.parse_args()


def last_scalar(directory, tag):
    try:
        ea = event_accumulator.EventAccumulator(
            str(directory), size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
    except Exception:
        return None
    if tag not in ea.Tags().get("scalars", []):
        return None
    vals = ea.Scalars(tag)
    return vals[-1].value if vals else None


def main():
    args = parse_args()
    config_path = pathlib.Path(args.plots_root) / "benchmark_config.json"
    if not config_path.exists():
        raise SystemExit(
            f"Missing {config_path}. Run run_continual_benchmark.py first (a --quick-test is enough)."
        )
    config = json.load(open(config_path))
    runs_root = pathlib.Path(config["runs_root"])
    event_dirs = sorted({p.parent for p in runs_root.rglob("events.out.tfevents.*")})
    if not event_dirs:
        raise SystemExit(f"No TensorBoard event files under {runs_root}")

    train, buffer, finalize = [], [], []
    for directory in event_dirs:
        t = last_scalar(directory, "timing/train_loop_seconds")
        b = last_scalar(directory, "timing/merge_buffer_seconds")
        f = last_scalar(directory, "timing/finalize_seconds")
        if t is not None:
            train.append(t)
        if b is not None:
            buffer.append(b)
        if f is not None:
            finalize.append(f)

    total_steps = float(config["total_timesteps"])
    buffer_steps = float(config["distill_extra_steps"])
    print(f"Runs with training timing: {len(train)}")
    if train:
        ms = 1000.0 * np.asarray(train) / total_steps
        print(f"Training: median {np.median(ms):.4f} ms/env-step, mean {np.mean(ms):.4f}")
    if buffer:
        ms = 1000.0 * np.asarray(buffer) / buffer_steps
        print(f"Merge-buffer rollout: median {np.median(ms):.4f} ms/env-step, mean {np.mean(ms):.4f}")
    if finalize:
        print(
            f"Finalize/merge: median {np.median(finalize):.4f} s, "
            f"mean {np.mean(finalize):.4f} s, max {np.max(finalize):.4f} s"
        )

    if train:
        task_runs = (
            len(config["task_suites"])
            * 4
            * len(config["seeds"])
            * len(config["task_sequence"])
        )
        projected_train = np.median(train) * task_runs
        print(
            f"Rough training-loop projection for the configured benchmark: "
            f"{projected_train / 3600.0:.2f} GPU-hours (serial), excluding retention evaluation."
        )


if __name__ == "__main__":
    main()
