"""
Continual RL benchmark: CKA-RL baseline (simple averaging) vs our
distillation-based merging, run side-by-side across a sequential task chain.

WHAT THIS DOES
--------------
1. Trains a SEQUENCE of tasks, one after another, per condition in
   CONDITIONS. Each task in the chain uses ALL previously trained tasks as
   `--prev-units` (base = first task in the chain, latest = most recently
   trained task), exactly matching how run_sac.py already interprets
   `prev_units`.

2. While training, run_sac.py already writes actor_loss, qf_loss (critic
   loss), episodic_return, and success to TensorBoard. plot_during_training
   (delegating to plots.py) pulls those out after each task finishes and
   plots every condition overlaid, per task.

3. At the END of the whole chain, evaluate_final_model_on_task /
   collect_retention_results load the FINAL pre-finalize policy snapshot
   from each condition and evaluate it on EVERY task in the sequence --
   does distillation retain earlier tasks' behavior better than simple
   averaging? plot_retention (delegating to plots.py) draws the result.

NOTE ON plots.py: all actual plotting (matplotlib/TensorBoard-reading code)
now lives in plots.py, as pure functions with no side effects beyond
writing PNG files. This file keeps everything that trains or evaluates
(subprocess calls, environment rollouts, model loading) and calls into
plots.py for the drawing itself. plot_during_training/plot_retention below
keep the EXACT same call signature as before (thin wrappers), so any script
that already does `benchmark.plot_during_training(...)` or
`benchmark.plot_retention(...)` keeps working unchanged. load_scalar and
rolling_mean are also re-exported here (imported from plots.py) for the
same reason.

BEFORE YOU RUN
---------------
- Runtime: with the defaults below, expect several hours on a Kaggle GPU.
  Cut TOTAL_TIMESTEPS_PER_TASK and/or len(TASK_SEQUENCE) for a faster smoke
  test first before committing to the full run.
- Valid task ids come straight from your tasks.py.
- This script assumes it lives next to run_sac.py, cka_rl.py, tasks.py,
  plots.py, etc. (same directory you already run run_sac.py from on
  Kaggle).
- If TASK_SEQUENCE repeats a task_id (to test relearning speed), every
  occurrence gets its OWN save directory and TensorBoard tag, keyed by its
  POSITION in the sequence (`seq{i}`), not just by task_id. Without this,
  the resumability check would find the first occurrence's checkpoint
  already on disk and silently skip training every later occurrence of
  that task_id -- reusing the wrong-generation checkpoint with no warning.
"""

import pathlib
import subprocess

import numpy as np
import torch

from cka_rl import CkaRlAgent, FrozenCkaPolicy
from tasks import get_task, get_task_name
import plots
from plots import load_scalar, rolling_mean  # re-exported for backward compatibility

# ==========================================================================
# CONFIG
# ==========================================================================
QUICK_TEST = False

SEED = 42
POOL_SIZE = 7
LEARNING_STARTS = 5_000
DISTILL_EXTRA_STEPS = 15_000
NUM_EVAL_EPISODES = 10

TASK_SEQUENCE = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1]
TOTAL_TIMESTEPS_PER_TASK = 130000

SAVE_ROOT = "agents"
RUNS_ROOT = "runs"
PLOTS_ROOT = "plots_continual_benchmark"

CONDITIONS = {
    "Mode1_Baseline":     {"use_alpha_mass": False, "distillation": False,
                           "fusion_mode": "classic_cka",  "tag": "Mode1_Baseline"},

    "Mode2_DistillOnly":  {"use_alpha_mass": False, "distillation": True,
                           "fusion_mode": "classic_cka",  "tag": "Mode2_DistillOnly"},

    "Mode3_WeightOnly":   {"use_alpha_mass": True, "distillation": False,
                           "fusion_mode": "weight_delta", "tag": "Mode3_WeightOnly"},

    "Mode4_OursCombined": {"use_alpha_mass": True, "distillation": True,
                           "fusion_mode": "weight_delta", "tag": "Mode4_OursCombined"},
}

METRICS = {
    "losses/actor_loss":        ("Actor Loss", "actor_loss"),
    "losses/qf_loss":           ("Critic Loss", "critic_loss"),
    "charts/episodic_return":   ("Episodic Return", "reward"),
    "charts/success":           ("Success Rate", "success"),
    "charts/test_success":      ("Evaluation Success", "eval_success"),
    "charts/velocity_error":       ("Velocity Tracking Error", "velocity_error"),
    "charts/test_velocity_error":  ("Evaluation Velocity Error", "eval_velocity_error"),
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
        # TASK_SEQUENCE, not just per task_id -- see module docstring.
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
                "--use-alpha-mass" if cfg["use_alpha_mass"] else "--no-use-alpha-mass",
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
# DURING-TRAINING PLOTS -- thin wrapper, delegates to plots.py
# ==========================================================================
def plot_during_training(seq_idx, task_id, task_name):
    plots.plot_during_training(
        seq_idx, task_id, task_name,
        conditions=CONDITIONS, runs_root=RUNS_ROOT, seed=SEED,
        metrics=METRICS, rolling_window=ROLLING_WINDOW, plots_root=PLOTS_ROOT,
    )


# ==========================================================================
# END-OF-CHAIN RETENTION EVAL: does the FINAL merged model still work on
# EARLIER tasks, not just the most recent one? (evaluation lives here;
# plotting is delegated to plots.py)
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


def collect_retention_results(all_prev_units, device):
    """Runs the actual evaluation (env rollouts) for every condition x every
    task in TASK_SEQUENCE. Returns the raw results dict -- no plotting here,
    see plot_retention below."""
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

    return results


def plot_retention(all_prev_units, device):
    """Thin wrapper: evaluate (this file), then delegate the drawing to
    plots.py. Kept as ONE function with this exact signature for backward
    compatibility with scripts that already call benchmark.plot_retention(...)."""
    results = collect_retention_results(all_prev_units, device)
    plots.plot_retention(
        results,
        conditions=CONDITIONS, task_sequence=TASK_SEQUENCE,
        get_task_name=get_task_name, plots_root=PLOTS_ROOT,
    )


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
