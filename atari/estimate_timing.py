"""Summarize timing scalars from one continual Atari PPO benchmark configuration.

Unlike the older helper, this script only scans the exact benchmark event
folders described by ``benchmark_config.json``.  Scratch runs and unrelated
TensorBoard runs under the same ``runs_root`` are therefore not mixed into the
reported medians or the projection.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from tensorboard.backend.event_processing import event_accumulator


CONDITION_NAMES = ("baseline", "distil_only", "weight_only", "combined")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plots-root", default="plots_atari_continual")
    return p.parse_args()


def last_scalar(directory, tag):
    directory = pathlib.Path(directory)
    if not directory.exists():
        return None
    try:
        ea = event_accumulator.EventAccumulator(
            str(directory), size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
    except Exception:
        return None
    if tag not in ea.Tags().get("scalars", []):
        return None
    values = ea.Scalars(tag)
    return float(values[-1].value) if values else None


def _selected_conditions(config):
    condition_index = config.get("condition_index", [0])
    if isinstance(condition_index, int):
        condition_index = [condition_index]
    if 0 in condition_index:
        return list(CONDITION_NAMES)
    out = []
    for idx in condition_index:
        idx = int(idx)
        if not 1 <= idx <= len(CONDITION_NAMES):
            raise ValueError(f"invalid condition index in benchmark config: {idx}")
        name = CONDITION_NAMES[idx - 1]
        if name not in out:
            out.append(name)
    return out


def _resolved_sequences(config):
    suites = list(config.get("task_suites", []))
    resolved = config.get("resolved_sequences") or {}
    result = {}
    for suite in suites:
        if suite in resolved:
            result[suite] = [int(x) for x in resolved[suite]]
            continue
        sequence = config.get("task_sequence")
        if sequence is None:
            raise ValueError(
                "benchmark_config.json has neither resolved_sequences nor a "
                f"task_sequence for suite {suite!r}"
            )
        result[suite] = [int(x) for x in sequence]
    return result


def _run_name(suite, task_id, seed):
    return f"{suite}__task_{task_id}__cka-rl__run_ppo__{seed}"


def _expected_event_dirs(config):
    runs_root = pathlib.Path(config["runs_root"])
    conditions = _selected_conditions(config)
    sequences = _resolved_sequences(config)
    seeds = [int(x) for x in config.get("seeds", [])]
    rows = []
    for suite, sequence in sequences.items():
        for condition in conditions:
            for seed in seeds:
                for seq_idx, task_id in enumerate(sequence):
                    tag = pathlib.Path(suite) / condition / f"seed_{seed}" / f"seq_{seq_idx}"
                    rows.append(
                        runs_root / tag / _run_name(suite, task_id, seed)
                    )
    return rows


def main():
    args = parse_args()
    config_path = pathlib.Path(args.plots_root) / "benchmark_config.json"
    if not config_path.exists():
        raise SystemExit(
            f"Missing {config_path}. Run run_continual_benchmark.py first."
        )
    with config_path.open() as f:
        config = json.load(f)

    expected_dirs = _expected_event_dirs(config)
    existing_dirs = [d for d in expected_dirs if d.exists()]
    if not existing_dirs:
        raise SystemExit(
            "No configured benchmark TensorBoard directories were found under "
            f"{config.get('runs_root')}."
        )

    train_seconds = []
    train_ms_per_transition = []
    buffer_ms_per_transition = []
    finalize_seconds = []

    fallback_steps = float(config.get("total_timesteps", 0))
    fallback_buffer_rows = float(config.get("distill_extra_steps", 0))

    for directory in existing_dirs:
        train = last_scalar(directory, "timing/train_loop_seconds")
        actual_steps = last_scalar(directory, "timing/actual_total_timesteps")
        buffer = last_scalar(directory, "timing/merge_buffer_seconds")
        buffer_rows = last_scalar(directory, "analysis/buffer/rows")
        finalize = last_scalar(directory, "timing/finalize_seconds")

        if train is not None:
            train_seconds.append(train)
            denom = actual_steps if actual_steps is not None and actual_steps > 0 else fallback_steps
            if denom > 0:
                train_ms_per_transition.append(1000.0 * train / denom)

        if buffer is not None and buffer > 0:
            denom = buffer_rows if buffer_rows is not None and buffer_rows > 0 else fallback_buffer_rows
            if denom > 0:
                buffer_ms_per_transition.append(1000.0 * buffer / denom)

        if finalize is not None:
            finalize_seconds.append(finalize)

    print(
        f"Configured task runs: {len(expected_dirs)} | "
        f"event directories found: {len(existing_dirs)} | "
        f"runs with training timing: {len(train_seconds)}"
    )

    if train_ms_per_transition:
        arr = np.asarray(train_ms_per_transition, dtype=np.float64)
        print(
            f"PPO training: median {np.median(arr):.4f} ms/transition, "
            f"mean {np.mean(arr):.4f}"
        )
    if buffer_ms_per_transition:
        arr = np.asarray(buffer_ms_per_transition, dtype=np.float64)
        print(
            f"Merge-buffer rollout: median {np.median(arr):.4f} ms/stored transition, "
            f"mean {np.mean(arr):.4f}"
        )
    if finalize_seconds:
        arr = np.asarray(finalize_seconds, dtype=np.float64)
        print(
            f"Finalize/merge: median {np.median(arr):.4f} s, "
            f"mean {np.mean(arr):.4f} s, max {np.max(arr):.4f} s"
        )

    if train_seconds:
        projected = float(np.median(train_seconds)) * len(expected_dirs)
        print(
            "Rough training-loop projection for the configured benchmark: "
            f"{projected / 3600.0:.2f} GPU-hours (serial), excluding retention "
            "evaluation, scratch baselines, and missing/failed runs."
        )


if __name__ == "__main__":
    main()
