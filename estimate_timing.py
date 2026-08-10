"""
Timing estimator: NOT for measuring learning quality -- for estimating
wall-clock budget before committing to a real run. Runs a SMALL, quick chain
(few tasks, few timesteps) for each of the 4 modes, and reports:
  - average training-step time (ms/step)
  - average distillation-buffer-collection step time (ms/step)
  - average merge time (seconds per merge), measured separately

Requires the 3 small timing additions in run_sac.py (see the instructions
you were given) -- this script reads them back from TensorBoard
(timing/train_loop_seconds, timing/distill_buffer_seconds,
timing/finalize_seconds), it doesn't re-measure anything itself.

POOL_SIZE is deliberately set to 1 here so a merge is forced almost every
task -- otherwise a short chain might never actually trigger one, and you'd
get no merge-timing data at all.

Run: `python3 estimate_timing.py`
"""
import glob
import time
import subprocess
import pathlib

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from tasks import get_task_name

# ==========================================================================
SEED = 42
POOL_SIZE = 1
# LEARNING_STARTS must be well below TOTAL_TIMESTEPS_PER_TASK -- otherwise
# every step in this tiny run would be a random-action step (env stepping
# only, no gradient update), which is NOT representative of what most of a
# real run's steps actually cost.
LEARNING_STARTS = 20
TOTAL_TIMESTEPS_PER_TASK = 200
DISTILL_EXTRA_STEPS = 256
TASK_SEQUENCE = [0, 1, 2]

SAVE_ROOT = "agents_timing"
RUNS_ROOT = "runs_timing"

CONDITIONS = {
    "Mode1_Baseline":     {"distillation": False, "fusion_mode": "classic_cka"},
    "Mode2_DistillOnly":  {"distillation": True,  "fusion_mode": "classic_cka"},
    "Mode3_WeightOnly":   {"distillation": False, "fusion_mode": "weight_delta"},
    "Mode4_OursCombined": {"distillation": True,  "fusion_mode": "weight_delta"},
}


def load_scalar_last(run_dir_pattern, scalar_tag):
    run_dirs = glob.glob(run_dir_pattern)
    if not run_dirs:
        return None
    event_files = glob.glob(f"{run_dirs[0]}/*events.out*")
    if not event_files:
        return None
    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()
    if scalar_tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(scalar_tag)
    return events[-1].value if events else None


def run_and_time_task(tag, task_id, prev_units, cfg):
    run_name = f"task_{task_id}__cka-rl__run_sac__{SEED}"
    save_dir = f"{SAVE_ROOT}/{tag}"
    run_dir = pathlib.Path(f"{save_dir}/{run_name}")

    cmd = [
        "python3", "run_sac.py",
        "--model-type=cka-rl",
        f"--task-id={task_id}",
        f"--seed={SEED}",
        f"--tag={tag}",
        f"--total-timesteps={TOTAL_TIMESTEPS_PER_TASK}",
        f"--learning-starts={LEARNING_STARTS}",
        f"--distill-extra-steps={DISTILL_EXTRA_STEPS}",
        f"--eval-every={TOTAL_TIMESTEPS_PER_TASK + 1}",  # skip mid-run eval, not what we're timing
        f"--pool-size={POOL_SIZE}",
        f"--save-dir={save_dir}",
        f"--fusion-mode={cfg['fusion_mode']}",
        "--distillation" if cfg["distillation"] else "--no-distillation",
    ]
    if prev_units:
        cmd.append("--prev-units")
        cmd.extend(str(p) for p in prev_units)

    wall_start = time.time()
    subprocess.run(cmd, check=True)
    wall_total = time.time() - wall_start

    pattern = f"{RUNS_ROOT}/{tag}/{run_name}"
    return {
        "wall_total_seconds": wall_total,
        "train_loop_seconds": load_scalar_last(pattern, "timing/train_loop_seconds"),
        "distill_buffer_seconds": load_scalar_last(pattern, "timing/distill_buffer_seconds"),
        "merge_seconds": load_scalar_last(pattern, "timing/finalize_seconds"),
        "run_dir": run_dir,
    }


if __name__ == "__main__":
    print(f"*** Timing estimate: {len(CONDITIONS)} modes x {len(TASK_SEQUENCE)} tasks, "
          f"total_timesteps={TOTAL_TIMESTEPS_PER_TASK}, distill_extra_steps={DISTILL_EXTRA_STEPS}, "
          f"pool_size={POOL_SIZE} (forces merges) ***")
    print(f"*** Task sequence: {[(t, get_task_name(t)) for t in TASK_SEQUENCE]} ***\n")

    all_results = {}
    for cond_name, cfg in CONDITIONS.items():
        print(f"\n========== {cond_name} (distillation={cfg['distillation']}, "
              f"fusion_mode={cfg['fusion_mode']}) ==========")
        prev_units = []
        cond_results = []
        for task_id in TASK_SEQUENCE:
            print(f"-- task {task_id} ({get_task_name(task_id)}) --")
            r = run_and_time_task(cond_name, task_id, prev_units, cfg)
            cond_results.append(r)
            prev_units.append(r["run_dir"])
            merge_str = f"{r['merge_seconds']:.4f}s" if r["merge_seconds"] is not None else "no merge"
            train_str = f"{r['train_loop_seconds']:.2f}s" if r["train_loop_seconds"] is not None else "n/a"
            print(f"   wall_total={r['wall_total_seconds']:.2f}s, "
                  f"train_loop={train_str}, merge_time={merge_str}")
        all_results[cond_name] = cond_results

    # ---- Summary ----
    print("\n\n================ SUMMARY ================")
    for cond_name, results in all_results.items():
        train_times = [r["train_loop_seconds"] for r in results if r["train_loop_seconds"] is not None]
        distill_times = [r["distill_buffer_seconds"] for r in results if r["distill_buffer_seconds"] is not None]
        merge_times = [r["merge_seconds"] for r in results if r["merge_seconds"] is not None]

        print(f"\n[{cond_name}]")
        if train_times:
            ms_per_step = 1000.0 * np.mean(train_times) / TOTAL_TIMESTEPS_PER_TASK
            print(f"  training step time:  {ms_per_step:.3f} ms/step  "
                  f"(avg over {len(train_times)} task run(s))")
        else:
            print("  training step time: n/a -- did you apply the run_sac.py timing edits?")

        if distill_times:
            ms_per_distill_step = 1000.0 * np.mean(distill_times) / DISTILL_EXTRA_STEPS
            print(f"  distill-buffer step time: {ms_per_distill_step:.3f} ms/step  "
                  f"(avg over {len(distill_times)} task run(s))")
        else:
            print("  distill-buffer step time: n/a (distillation off for this mode, or no data)")

        if merge_times:
            print(f"  merge time: {np.mean(merge_times):.4f}s avg over {len(merge_times)} merge(s) "
                  f"(min {min(merge_times):.4f}s, max {max(merge_times):.4f}s)")
        else:
            print("  merge time: n/a -- no merges observed (unexpected with pool_size=1; check logs)")

    print("\n*** To estimate a real run's wall-clock: ***")
    print("***   (ms_per_step / 1000) * planned_total_timesteps ***")
    print("***   + (distill ms/step / 1000) * planned_distill_extra_steps  [if distillation]***")
    print("***   + merge_time * expected_number_of_merges_in_that_chain ***")
    print("*** ...summed per task, times number of tasks, times number of conditions. ***")
