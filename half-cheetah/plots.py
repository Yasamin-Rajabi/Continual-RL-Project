"""
Pure plotting functions for the continual-RL benchmark.

These read TensorBoard scalar data (or take an already-computed retention
`results` dict) and write PNG files. No training, no environment rollouts,
no subprocess calls, no model loading -- see run_continual_benchmark.py for
those. Every function takes everything it needs as an explicit parameter
(no module-global config), so it behaves identically whether called from
run_continual_benchmark.py or from a standalone script.
"""
import os
import glob
import json

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_scalar(tag_dir_pattern, scalar_tag):
    run_dirs = glob.glob(tag_dir_pattern)
    if not run_dirs:
        return None, None
    event_files = glob.glob(f"{run_dirs[0]}/*events.out*")
    if not event_files:
        return None, None
    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()
    if scalar_tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(scalar_tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def rolling_mean(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_during_training(seq_idx, task_id, task_name, *,
                          conditions, runs_root, seed, metrics, rolling_window, plots_root):
    """One PNG per metric, for this one position in the task sequence, with
    every condition in `conditions` overlaid. Silently skips any condition
    that has no TensorBoard data yet (partial runs are fine)."""
    os.makedirs(plots_root, exist_ok=True)
    run_name = f"task_{task_id}__cka-rl__run_sac__{seed}"

    for scalar_tag, (title, fname) in metrics.items():
        plt.figure(figsize=(10, 5))
        plotted_anything = False
        for cond_name, cfg in conditions.items():
            position_tag = f"{cfg['tag']}_seq{seq_idx}"
            pattern = f"{runs_root}/{position_tag}/{run_name}"
            steps, values = load_scalar(pattern, scalar_tag)
            if steps is None:
                print(f"  (no data for {cond_name} / {scalar_tag} / seq{seq_idx} task {task_id})")
                continue
            plotted_anything = True
            if "reward" in fname or "success" in fname or "velocity_error" in fname:
                smoothed = rolling_mean(values, rolling_window)
                smoothed_steps = steps[: len(smoothed)] if len(smoothed) == len(steps) \
                    else steps[len(steps) - len(smoothed):]
                plt.plot(smoothed_steps, smoothed, label=cond_name, linewidth=1.8, alpha=0.9)
            else:
                plt.plot(steps, values, label=cond_name, linewidth=1.2, alpha=0.75)

        if not plotted_anything:
            plt.close()
            continue

        plt.title(f"{title} during training - seq{seq_idx}: Task {task_id} ({task_name})")
        plt.xlabel("Timesteps")
        plt.ylabel(title)
        plt.legend(loc="best")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        out_path = f"{plots_root}/{fname}_seq{seq_idx}_task_{task_id}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  saved {out_path}")


def plot_retention(results, *, conditions, task_sequence, get_task_name, plots_root):
    """Draws and saves retention_reward.png / retention_success.png (plus a
    retention_results.json dump) from an ALREADY-COMPUTED `results` dict --
    shaped like {condition_name: {"reward": [...], "success": [...]}}, one
    value per entry in task_sequence. Does not evaluate anything itself; see
    run_continual_benchmark.collect_retention_results for that part."""
    os.makedirs(plots_root, exist_ok=True)

    with open(f"{plots_root}/retention_results.json", "w") as f:
        json.dump(results, f, indent=2)

    x = np.arange(len(task_sequence))
    labels = [f"T{t}\n{get_task_name(t)}" for t in task_sequence]

    for metric in ["reward", "success"]:
        plt.figure(figsize=(10, 5.5))
        for cond_name in conditions:
            plt.plot(x, results[cond_name][metric], marker="o", linewidth=2, label=cond_name)
        plt.xticks(x, labels)
        plt.title(f"Retention across the task chain (evaluated with the FINAL merged model) - {metric}")
        plt.xlabel("Task (in training order)")
        plt.ylabel(metric)
        plt.legend(loc="best")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        out_path = f"{plots_root}/retention_{metric}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"saved {out_path}")
