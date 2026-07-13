import os
import subprocess
import torch
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from cka_rl import CkaRlAgent
from tasks import get_task

# ==========================================
# CONFIGURATION FOR CLEAN INTEGRATION TEST
# ==========================================
TAG = "UltimateBenchmark"
SAVE_DIR = f"agents/{TAG}"
SEED = 42
TOTAL_TIMESTEPS = 300_000   # افزایش استپ‌ها به ۳۰۰ هزار برای همگرایی واقعی Teacher
LEARNING_STARTS = 5000
DISTILL_STEPS = 15_000      # افزایش استپ‌های بافر جامع برای پوشش کامل فضای حالت
NUM_EVAL_EPISODES = 10 # تعداد اپیزودهای ارزیابی برای رسم نمودار خطی

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
    if task_id == 1:
        prev_run = f"task_0__cka-rl__run_sac__42"
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

    # کانفیگ دینامیک ایجنت بر اساس نوع ارزیابی برای جلوگیری از تداخل اندیس‌ها
    if mode == 'original':
        # مدل خالص و دست‌نخورده همان تسک بدون اعمال فرآیند مرج دیسک
        base_dir = None
        latest_dir = prev_units[task_id]
        distillation_flag = False
        pool_size = 99  # مقدار بزرگ برای عدم فعال‌شدن مرج
    elif mode == 'distill':
        base_dir = prev_units[0]
        latest_dir = prev_units[-1]
        distillation_flag = True
        pool_size = 2
    elif mode == 'simple':
        base_dir = prev_units[0]
        latest_dir = prev_units[-1]
        distillation_flag = False
        pool_size = 2

    agent = CkaRlAgent(
        base_dir=base_dir,
        latest_dir=latest_dir,
        obs_dim=obs_dim,
        act_dim=act_dim,
        fuse_shared=False,
        fuse_heads=True,
        pool_size=pool_size,
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
            action = torch.tanh(mean)[0].cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            if terminated or truncated:
                episode_rewards.append(ep_ret)
                break
    env.close()
    return episode_rewards

if __name__ == "__main__":
    # ۱. اجرای فرآیند آموزش از صفر برای هر دو تسک
    run_sac_for_task(0)
    run_sac_for_task(1)
    
    run0_name = "task_0__cka-rl__run_sac__42"
    run1_name = "task_1__cka-rl__run_sac__42"
    prev_units = (
        pathlib.Path(f"{SAVE_DIR}/{run0_name}"),
        pathlib.Path(f"{SAVE_DIR}/{run1_name}")
    )
    
    print("\n--- Starting Comprehensive Benchmark Evaluation ---")
    
    # ۲. ارزیابی هر ۳ حالت روی تسک اول (Task 0)
    rewards_t0_distill = evaluate_policy_variant(prev_units, task_id=0, mode='distill')
    rewards_t0_simple  = evaluate_policy_variant(prev_units, task_id=0, mode='simple')
    rewards_t0_original = evaluate_policy_variant(prev_units, task_id=0, mode='original')
    
    # ۳. ارزیابی هر ۳ حالت روی تسک دوم (Task 1)
    rewards_t1_distill = evaluate_policy_variant(prev_units, task_id=1, mode='distill')
    rewards_t1_simple  = evaluate_policy_variant(prev_units, task_id=1, mode='simple')
    rewards_t1_original = evaluate_policy_variant(prev_units, task_id=1, mode='original')
    
    # ۴. رسم پلات‌ها
    os.makedirs("plots", exist_ok=True)
    episodes = np.arange(1, NUM_EVAL_EPISODES + 1)
    
    # 📊 نمودار تسک ۰
    plt.figure(figsize=(10, 5.5))
    plt.plot(episodes, rewards_t0_distill, label='Policy Distillation (Ours)', marker='o', color='blue', linewidth=2)
    plt.plot(episodes, rewards_t0_simple, label='Simple Weight Averaging', marker='s', color='orange', linestyle='--', linewidth=2)
    plt.plot(episodes, rewards_t0_original, label='Original Task 0 Policy (Upper Bound)', marker='^', color='green', linestyle=':', linewidth=2)
    plt.title('Merged Policy Performance: Task 0 (Hammer)')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Total Episodic Reward')
    plt.xticks(episodes)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('plots/reward_timeline_task_0.png', dpi=300)
    plt.close()
    
    # 📊 نمودار تسک ۱
    plt.figure(figsize=(10, 5.5))
    plt.plot(episodes, rewards_t1_distill, label='Policy Distillation (Ours)', marker='o', color='blue', linewidth=2)
    plt.plot(episodes, rewards_t1_simple, label='Simple Weight Averaging', marker='s', color='orange', linestyle='--', linewidth=2)
    plt.plot(episodes, rewards_t1_original, label='Original Task 1 Policy (Upper Bound)', marker='^', color='green', linestyle=':', linewidth=2)
    plt.title('Merged Policy Performance: Task 1 (Faucet Close)')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Total Episodic Reward')
    plt.xticks(episodes)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('plots/reward_timeline_task_1.png', dpi=300)
    plt.close()
    
    print("\n*** Full pipeline execution complete! Beautiful plots generated inside `./plots/` ***")