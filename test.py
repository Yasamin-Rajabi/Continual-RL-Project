import os
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
from cka_rl import CkaRlAgent
from tasks import get_task

# تنظیمات اصلی برای اجرای تست مینیاتوری سریع
TAG = "MiniTest"
SAVE_DIR = f"agents/{TAG}"
SEED = 42
TOTAL_TIMESTEPS = 50
LEARNING_STARTS = 5
DISTILL_STEPS = 100 # تعداد استپ‌های سریع برای پر شدن بافر تست
NUM_EVAL_EPISODES = 3

def run_sac_for_task(task_id):
    """اجرای خودکار SAC روی تسک مشخص شده و تولید بافر و وزن‌ها"""
    run_name = f"task_{task_id}__cka-rl__run_sac__{SEED}"
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
    
    # اگر تسک ۱ بود، مسیر تسک ۰ را به عنوان یونیت قبلی پاس می‌دهیم
    if task_id == 1:
        prev_run = f"task_0__cka-rl__run_sac__{SEED}"
        cmd.extend(["--prev-units", f"{SAVE_DIR}/{prev_run}"])
        
    print(f"\n--- Running SAC for Task {task_id} ---")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

def evaluate_merged_policy(prev_units, distillation_mode, task_id):
    """ارزیابی ریوارد مدل مرج شده روی یک تسک مشخص"""
    env = get_task(task_id)
    obs_dim = np.array(env.observation_space.shape).prod()
    act_dim = np.prod(env.action_space.shape)
    
    base_dir = prev_units[0]
    latest_dir = prev_units[-1]
    
    # ساخت ایجنت با متد مرج انتخابی
    agent = CkaRlAgent(
        base_dir=base_dir,
        latest_dir=latest_dir,
        obs_dim=obs_dim,
        act_dim=act_dim,
        fuse_shared=False,
        fuse_heads=True,
        pool_size=2,
        prev_units_paths=prev_units,
        distillation=distillation_mode # سوئیچ بین تقطیر و میانگین‌گیری
    )
    agent.eval()
    
    episode_rewards = []
    for _ in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset(seed=SEED)
        ep_ret = 0
        while True:
            obs_tensor = torch.Tensor(obs).unsqueeze(0)
            with torch.no_grad():
                mean, _ = agent(obs_tensor)
            action = torch.tanh(mean)[0].numpy() # اعمال اکشن دترمینستیک زمان تست
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            if terminated or truncated:
                episode_rewards.append(ep_ret)
                break
    env.close()
    return np.mean(episode_rewards)

if __name__ == "__main__":
    # گام ۱: ران کردن SAC برای هر دو تسک و ذخیره داده‌ها روی دیسک
    run_sac_for_task(0)
    run_sac_for_task(1)
    
    # آدرس یونیت‌های تولید شده
    run0_name = f"task_0__cka-rl__run_sac__{SEED}"
    run1_name = f"task_1__cka-rl__run_sac__{SEED}"
    prev_units = (
        pathlib.Path(f"{SAVE_DIR}/{run0_name}"),
        pathlib.Path(f"{SAVE_DIR}/{run1_name}")
    )
    
    print("\n--- Evaluating Merged Policies ---")
    # گام ۲: ارزیابی عملکرد متد تقطیر (Distillation Method)
    distill_reward_task0 = evaluate_merged_policy(prev_units, distillation_mode=True, task_id=0)
    distill_reward_task1 = evaluate_merged_policy(prev_units, distillation_mode=True, task_id=1)
    
    # گام ۳: ارزیابی عملکرد متد میانگین‌گیری ساده (Simple Averaging Method)
    simple_reward_task0 = evaluate_merged_policy(prev_units, distillation_mode=False, task_id=0)
    simple_reward_task1 = evaluate_merged_policy(prev_units, distillation_mode=False, task_id=1)
    
    print(f"\nResults Task 0 -> Distillation: {distill_reward_task0:.2f} | Simple Avg: {simple_reward_task0:.2f}")
    print(f"Results Task 1 -> Distillation: {distill_reward_task1:.2f} | Simple Avg: {simple_reward_task1:.2f}")
    
    # گام ۴: رسم پلات‌ها و ذخیره‌سازی بصری خروجی
    os.makedirs("plots", exist_ok=True)
    methods = ['Policy Distillation', 'Simple Averaging']
    
    # پلات اول: عملکرد روی تسک اول (Task 0)
    plt.figure(figsize=(6, 5))
    rewards_t0 = [distill_reward_task0, simple_reward_task0]
    plt.bar(methods, rewards_t0, color=['blue', 'orange'], width=0.4)
    plt.title('Merged Policy Performance on Task 0')
    plt.ylabel('Mean Episodic Reward')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('plots/performance_task_0.png')
    plt.close()
    
    # پلات دوم: عملکرد روی تسک دوم (Task 1)
    plt.figure(figsize=(6, 5))
    rewards_t1 = [distill_reward_task1, simple_reward_task1]
    plt.bar(methods, rewards_t1, color=['blue', 'orange'], width=0.4)
    plt.title('Merged Policy Performance on Task 1')
    plt.ylabel('Mean Episodic Reward')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('plots/performance_task_1.png')
    plt.close()
    
    print("\n*** Integration testing finished successfully! Plots saved in `./plots/` directory ***")