"""Summarize timing scalars from one continual Atari PPO benchmark configuration.

Only the exact benchmark event folders described by benchmark_config.json are
scanned, so scratch runs and unrelated TensorBoard logs are not mixed into the
reported timing statistics.

The task budget follows the current protocol:
    Delta = total_timesteps
    B     = distill_extra_steps, inside Delta
    optimization phase = Delta - B
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


def _base_conditions(config):
    condition_index = config.get("condition_index", [0])
    if isinstance(condition_index, int):
        condition_index = [condition_index]

    if 0 in condition_index:
        return list(CONDITION_NAMES)

    result = []
    for idx in condition_index:
        idx = int(idx)
        if not 1 <= idx <= len(CONDITION_NAMES):
            raise ValueError(f"invalid condition index in benchmark config: {idx}")
        name = CONDITION_NAMES[idx - 1]
        if name not in result:
            result.append(name)
    return result


def _selected_conditions(config):
    """Reproduce run_continual_benchmark's condition-label construction."""
    base = _base_conditions(config)
    spaces = list(dict.fromkeys(config.get("composition_spaces", ["parameter"])))
    policy_student = bool(config.get("policy_student_replay", False))

    result = []
    for name in base:
        for space in spaces:
            if space == "parameter":
                label = name
            elif space == "policy":
                label = (
                    name + "_policy_student"
                    if policy_student
                    else name + "_policy"
                )
            else:
                raise ValueError(
                    f"unknown composition space in benchmark config: {space!r}"
                )
            if label not in result:
                result.append(label)
    return result


def _resolved_sequences(config):
    """Resolve exactly the sequence used by each Atari suite."""
    suites = list(config.get("task_suites", []))
    resolved = config.get("resolved_sequences") or {}
    requested = config.get("task_sequence")

    result = {}
    for suite in suites:
        if suite in resolved:
            result[suite] = [int(x) for x in resolved[suite]]
            continue

        if requested is not None:
            result[suite] = [int(x) for x in requested]
            continue

        # Current benchmark configs can be written before suite-specific defaults
        # are resolved, so recover the default from atari_tasks.
        import atari_tasks

        if hasattr(atari_tasks, "default_sequence"):
            sequence = atari_tasks.default_sequence(suite)
        elif hasattr(atari_tasks, "DEFAULT_CONTINUAL_SEQUENCE"):
            sequence = atari_tasks.DEFAULT_CONTINUAL_SEQUENCE
        else:
            sequence = range(len(atari_tasks.TASK_SUITES[suite]))
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
                    tag = (
                        pathlib.Path(suite)
                        / condition
                        / f"seed_{seed}"
                        / f"seq_{seq_idx}"
                    )
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
    existing_dirs = [directory for directory in expected_dirs if directory.exists()]
    if not existing_dirs:
        raise SystemExit(
            "No configured benchmark TensorBoard directories were found under "
            f"{config.get('runs_root')}."
        )

    train_seconds = []
    train_ms_per_transition = []
    buffer_ms_per_transition = []
    finalize_seconds = []

    delta = float(config.get("total_timesteps", 0))
    frozen_tail = float(config.get("distill_extra_steps", 0))
    fallback_training_steps = max(delta - frozen_tail, 0.0)

    for directory in existing_dirs:
        train = last_scalar(directory, "timing/train_loop_seconds")
        optimization_steps = last_scalar(
            directory, "budget/optimization_phase_env_steps"
        )
        buffer = last_scalar(directory, "timing/merge_buffer_seconds")
        tail_steps = last_scalar(directory, "budget/frozen_tail_env_steps")
        finalize = last_scalar(directory, "timing/finalize_seconds")

        if train is not None:
            train_seconds.append(train)
            denom = (
                optimization_steps
                if optimization_steps is not None and optimization_steps > 0
                else fallback_training_steps
            )
            if denom > 0:
                train_ms_per_transition.append(1000.0 * train / denom)

        if buffer is not None and buffer > 0:
            denom = (
                tail_steps
                if tail_steps is not None and tail_steps > 0
                else frozen_tail
            )
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
            f"PPO optimization phase: median {np.median(arr):.4f} ms/transition, "
            f"mean {np.mean(arr):.4f}"
        )

    if buffer_ms_per_transition:
        arr = np.asarray(buffer_ms_per_transition, dtype=np.float64)
        print(
            f"Frozen-tail rollout: median {np.median(arr):.4f} ms/transition, "
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
            "Rough optimization-loop projection for the configured benchmark: "
            f"{projected / 3600.0:.2f} GPU-hours (serial), excluding frozen-tail "
            "collection, finalization, retention evaluation, scratch baselines, "
            "and missing/failed runs."
        )


if __name__ == "__main__":
    main()
