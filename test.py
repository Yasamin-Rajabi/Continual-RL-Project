import os
import subprocess
import torch
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from cka_rl import CkaRlAgent
from tasks import get_task

import glob
from tensorboard.backend.event_processing import event_accumulator

# ==========================================
# CONFIGURATION FOR CLEAN INTEGRATION TEST
# ==========================================
TAG = "SimilarTasksBenchmark"
SAVE_DIR = f"agents/{TAG}"
SEED = 42
TOTAL_TIMESTEPS = 300_000   # افزایش استپ‌ها به ۳۰۰ هزار برای همگرایی واقعی Teacher
LEARNING_STARTS = 5000
DISTILL_STEPS = 15_000      # افزایش استپ‌های بافر جامع برای پوشش کامل فضای حالت
NUM_EVAL_EPISODES = 10 # تعداد اپیزودهای ارزیابی برای رسم نمودار خطی

TASK_A = 3  
TASK_B = 5

def plot_training_metrics(tag, task_id, task_name, save_dir="plots_training"):
    """استخراج و رسم منحنی‌های آموزش (Reward, Actor Loss, Critic Loss) از فایل‌های TensorBoard"""
    os.makedirs(save_dir, exist_ok=True)
    run_dir_pattern = f"runs/{tag}/task_{task_id}__*"
    run_dirs = glob.glob(run_dir_pattern)
    
    if not run_dirs:
        print(f"⚠️ دایرکتوری لاگ TensorBoard برای تسک {task_id} پیدا نشد.")
        return

    event_file = glob.glob(f"{run_dirs[0]}/*events.out*")
    if not event_file:
        return

    ea = event_accumulator.EventAccumulator(event_file[0])
    ea.Reload()

    tags_to_extract = {
        'charts/episodic_return': ('Episodic Return during Training', 'Episodic Return', 'reward_train'),
        'losses/actor_loss': ('Actor Loss during Training', 'Actor Loss', 'actor_loss'),
        'losses/qf_loss': ('Critic (Q-Function) Loss during Training', 'Critic Loss', 'critic_loss')
    }

    available_tags = ea.Tags().get('scalars', [])

    for tag_name, (title, ylabel, filename) in tags_to_extract.items():
        if tag_name in available_tags:
            events = ea.Scalars(tag_name)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            plt.figure(figsize=(9, 4.5))
            plt.plot(steps, values, color='purple' if 'loss' in filename else 'teal', alpha=0.85, linewidth=1.5)
            plt.title(f'{title} - Task {task_id} ({task_name})')
            plt.xlabel('Timesteps')
            plt.ylabel(ylabel)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{filename}_task_{task_id}.png", dpi=300)
            plt.close()
            print(f"📊 نمودار آموزش ذخیره شد: {save_dir}/{filename}_task_{task_id}.png")

def run_sac_for_task(task_id):
    """اجرای استاندارد SAC روی تسک مشخص شده و ذخیره خروجی‌ها"""
    run_name = f"task_{task_id}__cka-rl__run_sac__42"
    cmd = [
        "python3", "run_sac.py",
        f"--model-type=cka-rl",
        f"--task-id={task_id}",
        f"--seed={SEED}",
        f"--tag={TAG}",
        f"--total-timesteps={TOTAL_TIMESTEPS}",
        f"--learning_starts={LEARNING_STARTS}",
        f"--distill_extra_steps={DISTILL_STEPS}",
        f"--save-dir={SAVE_DIR}"
    ]
    
    # برای تسک ۱، آدرس تسک ۰ را به عنوان یونیت قبلی می‌فرستیم
    if task_id == TASK_B:
        prev_run = f"task_{TASK_A}__cka-rl__run_sac__42"
        cmd.extend(["--prev-units", f"{SAVE_DIR}/{prev_run}"])
        
    print(f"\n>>> Running SAC Training for Task {task_id} ({TOTAL_TIMESTEPS} steps) <<<")
    subprocess.run(cmd, check=True)

def evaluate_policy_variant(prev_units, task_id, mode: str):
    """
    ارزیابی کاملاً ایزوله برای واریانت‌های مختلف سیاست
    mode: 'distill' | 'simple' | 'original'
    """
    env = get_task(task_id)
    obs_dim = np.array(env.observation_space.shape).prod()
    act_dim = np.prod(env.action_space.shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == 'original':
        agent = CkaRlAgent.load(
            dirname=str(prev_units[task_id]),
            obs_dim=obs_dim,
            act_dim=act_dim,
            map_location=device,
            reset_heads=False
        ).to(device)
    else:
        base_dir = prev_units[TASK_A]
        latest_dir = prev_units[TASK_B]
        distillation_flag = (mode == 'distill')
        
        agent = CkaRlAgent(
            base_dir=base_dir,
            latest_dir=latest_dir,
            obs_dim=obs_dim,
            act_dim=act_dim,
            fuse_shared=False,
            fuse_heads=True,
            pool_size=2,
            prev_units_paths=prev_units,
            distillation=distillation_flag
        ).to(device)
        
    agent.eval()
    
    episode_rewards = []
    for ep in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset(seed=SEED + ep)
        ep_ret = 0
        while True:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, _ = agent(obs_tensor)
            
            act_mid = (env.action_space.high + env.action_space.low) / 2.0
            act_rng = (env.action_space.high - env.action_space.low) / 2.0
            action = (torch.tanh(mean)[0].cpu().numpy() * act_rng) + act_mid
            
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            if terminated or truncated:
                episode_rewards.append(ep_ret)
                break
    env.close()
    return episode_rewards

if __name__ == "__main__":
    # ۱. تغییر هایپرپارامترهای محلی برای تسک‌های مشابه جدید
    
    # تابع run_sac_for_task را برای تسک‌های ۳ و ۵ بازنویسی موقت می‌کنیم تا با تگ جدید بخواند
    def run_sac_for_similar_tasks(task_id):
        cmd = [
            "python3", "run_sac.py",
            f"--model-type=cka-rl",
            f"--task-id={task_id}",
            f"--seed={SEED}",
            f"--tag={TAG}",
            f"--total-timesteps={TOTAL_TIMESTEPS}",
            f"--learning_starts={LEARNING_STARTS}",
            f"--distill_extra_steps={DISTILL_STEPS}",
            f"--save-dir={SAVE_DIR}"
        ]
        if task_id == 5: # تسک دوم ما ۵ است، پس یونیت قبلی آن تسک ۳ خواهد بود
            prev_run = f"task_3__cka-rl__run_sac__42"
            cmd.extend(["--prev-units", f"{SAVE_DIR}/{prev_run}"])
            
        print(f"\n>>> Running SAC Training for Similar Task {task_id} ({TOTAL_TIMESTEPS} steps) <<<")
        subprocess.run(cmd, check=True)

    # ۲. اجرای فرآیند آموزش از صفر برای دو تسک شبیه به هم
    run_sac_for_similar_tasks(TASK_A) # Handle Press Side
    run_sac_for_similar_tasks(TASK_B) # Window Close

    print("\n--- Plotting Training Curves from TensorBoard Logs ---")
    plot_training_metrics(TAG, TASK_A, "Handle Press Side")
    plot_training_metrics(TAG, TASK_B, "Window Close")


    run_a_name = f"task_{TASK_A}__cka-rl__run_sac__42"
    run_b_name = f"task_{TASK_B}__cka-rl__run_sac__42"
    
    prev_units = {
        TASK_A: pathlib.Path(f"{SAVE_DIR}/{run_a_name}"),
        TASK_B: pathlib.Path(f"{SAVE_DIR}/{run_b_name}")
    }
    
    print("\n--- Starting Comprehensive Similar Tasks Benchmark Evaluation ---")
    
    # ۳. ارزیابی هر ۳ حالت روی تسک اول جدید 
    rewards_ta_distill = evaluate_policy_variant(prev_units, task_id=TASK_A, mode='distill')
    rewards_ta_simple  = evaluate_policy_variant(prev_units, task_id=TASK_A, mode='simple')
    rewards_ta_original = evaluate_policy_variant(prev_units, task_id=TASK_A, mode='original')
    
    # ۳. ارزیابی هر ۳ حالت روی تسک دوم (Window Close)
    rewards_tb_distill = evaluate_policy_variant(prev_units, task_id=TASK_B, mode='distill')
    rewards_tb_simple  = evaluate_policy_variant(prev_units, task_id=TASK_B, mode='simple')
    rewards_tb_original = evaluate_policy_variant(prev_units, task_id=TASK_B, mode='original')
    
    # ۵. رسم پلات‌های جدید
    os.makedirs("plots_similar", exist_ok=True)
    episodes = np.arange(1, NUM_EVAL_EPISODES + 1)
    
    # نمودار تسک Handle Press
    plt.figure(figsize=(10, 5.5))
    plt.plot(episodes, rewards_ta_distill, label='Policy Distillation (Ours)', marker='o', color='blue', linewidth=2)
    plt.plot(episodes, rewards_ta_simple, label='Simple Weight Averaging', marker='s', color='orange', linestyle='--', linewidth=2)
    plt.plot(episodes, rewards_ta_original, label='Original Policy (Upper Bound)', marker='^', color='green', linestyle=':', linewidth=2)
    plt.title(f'Merged Policy Performance: Task {TASK_A} (Handle Press Side)')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Total Episodic Reward')
    plt.xticks(episodes)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f'plots_similar/reward_timeline_task_{TASK_A}.png', dpi=300)
    plt.close()
    
    # نمودار تسک Window Close
    plt.figure(figsize=(10, 5.5))
    plt.plot(episodes, rewards_tb_distill, label='Policy Distillation (Ours)', marker='o', color='blue', linewidth=2)
    plt.plot(episodes, rewards_tb_simple, label='Simple Weight Averaging', marker='s', color='orange', linestyle='--', linewidth=2)
    plt.plot(episodes, rewards_tb_original, label='Original Policy (Upper Bound)', marker='^', color='green', linestyle=':', linewidth=2)
    plt.title(f'Merged Policy Performance: Task {TASK_B} (Window Close)')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Total Episodic Reward')
    plt.xticks(episodes)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(f'plots_similar/reward_timeline_task_{TASK_B}.png', dpi=300)
    plt.close()
    
    print("\n*** Full pipeline execution complete! Beautiful plots generated inside `./plots_similar/` ***")