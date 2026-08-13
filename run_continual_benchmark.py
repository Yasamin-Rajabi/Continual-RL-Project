"""
Continual RL benchmark: CKA-RL baseline (simple averaging) vs our
distillation-based merging, run side-by-side across a sequential task chain.

WHAT THIS DOES
--------------
1. Trains a SEQUENCE of tasks, one after another, TWICE (once per condition):
     - "cka_baseline"   : distillation=False  -> merges via simple averaging
     - "cka_distill"    : distillation=True   -> merges via your new
                          single-shared-pair + buffer-pool distillation merge
   Each task in the chain uses ALL previously trained tasks as `--prev-units`
   (base = first task in the chain, latest = most recently trained task),
   exactly matching how run_sac.py already interprets `prev_units`.

2. While training, run_sac.py already writes actor_loss, qf_loss (critic
   loss), episodic_return, and success to TensorBoard. This script pulls
   those out AFTER each task finishes and plots BOTH conditions overlaid,
   per task, so you can see the actual training curves (not a bar chart).

3. At the END of the whole chain, it loads the FINAL merged model from both
   conditions and evaluates it on EVERY task in the sequence. This is the
   real test of the merge: does distillation retain earlier tasks' behavior
   better than simple averaging, or does it just look fine on whichever task
   happened to be trained last? (See note below on why your previous 2-task
   run looked so lopsided on Task 3 vs Task 5 -- a longer chain evaluated at
   the end avoids that recency bias.)

BEFORE YOU RUN
---------------
- Runtime: with the defaults below (5 tasks x 150k steps x 2 conditions =
  1.5M env steps total), expect several hours on a Kaggle GPU. Cut
  TOTAL_TIMESTEPS_PER_TASK and/or len(TASK_SEQUENCE) for a faster smoke test
  first (see QUICK_TEST below) before committing to the full run.
- Valid task ids come straight from your tasks.py:
    0 hammer-v2, 1 faucet-close-v2, 2 stick-pull-v2, 3 handle-press-side-v2,
    4 push-v2, 5 window-close-v2, 6 peg-unplug-side-v2
- This script assumes it lives next to run_sac.py, cka_rl.py, tasks.py, etc.
  (same directory you already run run_sac.py from on Kaggle).
- If TASK_SEQUENCE repeats a task_id (as the default below does, to test
  relearning speed), every occurrence gets its OWN save directory and
  TensorBoard tag, keyed by its POSITION in the sequence (`seq{i}`), not
  just by task_id. Without this, the resumability check would find the
  first occurrence's checkpoint already on disk and silently skip training
  every later occurrence of that task_id -- reusing the wrong-generation
  checkpoint with no warning. If you've already run an earlier version of
  this script without the `seq{i}` directories, it won't be recognized as
  "already done" -- it'll retrain from scratch, which is the correct
  behavior given the earlier run's later occurrences may have been silently
  skipped.
"""

import os
import glob
import json
import subprocess
import pathlib

import numpy as np
import torch
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

from cka_rl import CkaRlAgent, FrozenCkaPolicy
from tasks import get_task, get_task_name

# ==========================================================================
# CONFIG
# ==========================================================================
QUICK_TEST = False          

SEED = 42
POOL_SIZE = 5                
LEARNING_STARTS = 5_000
DISTILL_EXTRA_STEPS = 15_000
NUM_EVAL_EPISODES = 10

TASK_SEQUENCE = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5]
TOTAL_TIMESTEPS_PER_TASK = 150_000

SAVE_ROOT = "agents"
RUNS_ROOT = "runs"
PLOTS_ROOT = "plots_continual_benchmark"

CONDITIONS = {
    "Mode1_Baseline":     {"distillation": False, "fusion_mode": "classic_cka",  "tag": "Mode1_Baseline"},
    "Mode2_DistillOnly":  {"distillation": True,  "fusion_mode": "classic_cka",  "tag": "Mode2_DistillOnly"},
    "Mode3_WeightOnly":   {"distillation": False, "fusion_mode": "weight_delta", "tag": "Mode3_WeightOnly"},
    "Mode4_OursCombined": {"distillation": True,  "fusion_mode": "weight_delta", "tag": "Mode4_OursCombined"},
}

METRICS = {
    "losses/actor_loss":       ("Actor Loss", "actor_loss"),
    "losses/qf_loss":          ("Critic Loss", "critic_loss"),
    "charts/episodic_return":  ("Episodic Return", "reward"),
    "charts/success":          ("Success Rate", "success"),
    "charts/test_success":     ("Evaluation Success", "eval_success"), # اضافه شدن متریک ارزیابی دوره‌ای
}
ROLLING_WINDOW = 25
# ==========================================================================
# TRAINING: run the full task chain for one condition
# ==========================================================================
def run_task_chain(condition_name, cfg):
    tag = cfg["tag"]
    save_dir = f"{SAVE_ROOT}/{tag}"
    prev_units = []  # cumulative list of completed run dirs in this chain

    for seq_idx, task_id in enumerate(TASK_SEQUENCE):
        # IMPORTANT: run_name/save_dir/tag must be unique per POSITION in
        # TASK_SEQUENCE, not just per task_id -- the same task_id appears
        # multiple times by design (to test relearning speed). Without the
        # `seq{seq_idx}` segment here, the 2nd/3rd occurrence of a task_id
        # would resolve to the exact same directory as the 1st, the
        # resumability check below would find it "already exists", and that
        # occurrence would be silently SKIPPED -- no training at all, and
        # the wrong-generation checkpoint reused. (This is the exact bug you
        # flagged -- it was present here too, not just in the merged
        # version.)
        position_tag = f"{tag}_seq{seq_idx}"
        position_save_dir = f"{save_dir}/seq{seq_idx}"
        run_name = f"task_{task_id}__cka-rl__run_sac__{SEED}"
        run_dir = pathlib.Path(f"{position_save_dir}/{run_name}")

        if run_dir.exists():
            print(f"[{condition_name}] seq{seq_idx} (task {task_id}) already trained, skipping -> {run_dir}")
        else:
            cmd = [
                "python3", "run_sac.py",
                "--model-type=cka-rl",
                f"--task-id={task_id}",
                f"--seed={SEED}",
                f"--tag={position_tag}",
                f"--total-timesteps={TOTAL_TIMESTEPS_PER_TASK}",
                f"--learning-starts={LEARNING_STARTS}",
                f"--distill-extra-steps={DISTILL_EXTRA_STEPS}",
                f"--eval-every=10000",
                f"--pool-size={POOL_SIZE}",
                f"--save-dir={position_save_dir}",
                f"--fusion-mode={cfg['fusion_mode']}",
                "--distillation" if cfg["distillation"] else "--no-distillation",
            ]
            if prev_units:
                cmd.append("--prev-units")
                cmd.extend(str(p) for p in prev_units)

            print(f"\n>>> [{condition_name}] training seq{seq_idx}: task {task_id} "
                  f"({get_task_name(task_id)}), prev_units={len(prev_units)} <<<")
            subprocess.run(cmd, check=True)

        prev_units.append(run_dir)

    return prev_units  # final cumulative chain, in training order


# ==========================================================================
# TENSORBOARD EXTRACTION + DURING-TRAINING PLOTS
# ==========================================================================
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


def plot_during_training(seq_idx, task_id, task_name):
    os.makedirs(PLOTS_ROOT, exist_ok=True)
    run_name = f"task_{task_id}__cka-rl__run_sac__{SEED}"

    for scalar_tag, (title, fname) in METRICS.items():
        plt.figure(figsize=(10, 5))
        plotted_anything = False
        for cond_name, cfg in CONDITIONS.items():
            position_tag = f"{cfg['tag']}_seq{seq_idx}"
            pattern = f"{RUNS_ROOT}/{position_tag}/{run_name}"
            steps, values = load_scalar(pattern, scalar_tag)
            if steps is None:
                print(f"  (no data for {cond_name} / {scalar_tag} / seq{seq_idx} task {task_id})")
                continue
            plotted_anything = True
            if "reward" in fname or "success" in fname:
                smoothed = rolling_mean(values, ROLLING_WINDOW)
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
        out_path = f"{PLOTS_ROOT}/{fname}_seq{seq_idx}_task_{task_id}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  saved {out_path}")


# ==========================================================================
# END-OF-CHAIN RETENTION EVAL: does the FINAL merged model still work on
# EARLIER tasks, not just the most recent one?
# ==========================================================================
def evaluate_final_model_on_task(prev_units, condition_cfg, eval_task_id, device):
    latest_dir = prev_units[-1]

    env = get_task(eval_task_id)
    obs_dim = np.array(env.observation_space.shape).prod()
    act_dim = np.prod(env.action_space.shape)

    # Evaluate the exact pre-finalize policy snapshot.  The finalized HeadPool
    # checkpoint is the continual-learning state for constructing the NEXT task;
    # its old alpha no longer necessarily represents the just-trained policy.
    agent = FrozenCkaPolicy.load(
        dirname=str(latest_dir),
        map_location=device,
    ).to(device)
    agent.eval()

    episode_rewards, episode_successes = [], []
    for ep in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset(seed=SEED + ep)
        ep_ret, success = 0.0, 0
        while True:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, _ = agent(obs_tensor)
            act_mid = (env.action_space.high + env.action_space.low) / 2.0
            act_rng = (env.action_space.high - env.action_space.low) / 2.0
            action = (torch.tanh(mean)[0].cpu().numpy() * act_rng) + act_mid
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            success = max(success, int(info.get("success", 0)))
            if terminated or truncated:
                episode_rewards.append(ep_ret)
                episode_successes.append(success)
                break
    env.close()
    return float(np.mean(episode_rewards)), float(np.mean(episode_successes))


def plot_retention(all_prev_units, device):
    os.makedirs(PLOTS_ROOT, exist_ok=True)
    results = {cond: {"reward": [], "success": []} for cond in CONDITIONS}

    for cond_name, cfg in CONDITIONS.items():
        prev_units = all_prev_units[cond_name]
        print(f"\n>>> Retention eval for [{cond_name}] on every task in the chain <<<")
        for task_id in TASK_SEQUENCE:
            reward, success = evaluate_final_model_on_task(prev_units, cfg, task_id, device)
            print(f"  task {task_id} ({get_task_name(task_id)}): "
                  f"reward={reward:.1f}, success={success:.2f}")
            results[cond_name]["reward"].append(reward)
            results[cond_name]["success"].append(success)

    with open(f"{PLOTS_ROOT}/retention_results.json", "w") as f:
        json.dump(results, f, indent=2)

    x = np.arange(len(TASK_SEQUENCE))
    labels = [f"T{t}\n{get_task_name(t)}" for t in TASK_SEQUENCE]

    for metric in ["reward", "success"]:
        plt.figure(figsize=(10, 5.5))
        for cond_name in CONDITIONS:
            plt.plot(x, results[cond_name][metric], marker="o", linewidth=2,
                      label=cond_name)
        plt.xticks(x, labels)
        plt.title(f"Retention across the task chain (evaluated with the FINAL merged model) - {metric}")
        plt.xlabel("Task (in training order)")
        plt.ylabel(metric)
        plt.legend(loc="best")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        out_path = f"{PLOTS_ROOT}/retention_{metric}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"saved {out_path}")


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"*** Device: {device} ***")
    print(f"*** Task sequence: {[(t, get_task_name(t)) for t in TASK_SEQUENCE]} ***")

    all_prev_units = {}
    for cond_name, cfg in CONDITIONS.items():
        print(f"\n========== Training chain for condition: {cond_name} "
              f"(distillation={cfg['distillation']}) ==========")
        all_prev_units[cond_name] = run_task_chain(cond_name, cfg)

    print("\n========== Plotting during-training curves (both conditions overlaid) ==========")
    for seq_idx, task_id in enumerate(TASK_SEQUENCE):
        print(f"-- seq{seq_idx}: task {task_id} ({get_task_name(task_id)}) --")
        plot_during_training(seq_idx, task_id, get_task_name(task_id))

    print("\n========== End-of-chain retention evaluation ==========")
    plot_retention(all_prev_units, device)

    print(f"\n*** Done. All plots are in ./{PLOTS_ROOT}/ ***")
