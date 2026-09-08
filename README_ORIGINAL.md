# 🧠 CKARL — Continual Knowledge-Aware Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2%2B-ee4c2c.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-brightgreen.svg)
![MetaWorld](https://img.shields.io/badge/Env-MetaWorld-orange.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

> Official implementation of the **CKARL** framework for continual reinforcement learning, evaluated on **MetaWorld** robotic manipulation benchmarks.

This repository explores advanced knowledge fusion mechanisms to adapt to new tasks while mitigating catastrophic forgetting. It features the original **CKA (Centered Kernel Alignment)** representation fusion, alongside our novel ✨ **Weight-Space Delta Fusion** ✨ approach, which strictly isolates base weights and optimizes local variations.

---

## 🚀 Key Highlights

- 🔬 **Dual-Mode Fusion:** Seamlessly switch between `classic_cka` and our proposed `weight_delta` method via CLI.
- 🤖 **MetaWorld Benchmarks:** Evaluated on complex, continuous robotic manipulation sequences.
- 📊 **Comprehensive Evaluation:** Built-in tools for tracking *Forward Transfer* and *Forgetting Rate*.
- 🛠️ **Modular Design:** Easily extensible to new continual RL baselines.

---

## 📚 Supported Algorithms

| Algorithm | Description |
| :--- | :--- |
| 👑 `simple` | Single-task SAC (Oracle Upper Bound) |
| 🔧 `finetune` | Standard Fine-tuning Baseline |
| 🧠 `cka-rl` | **CKARL** — Supports `classic_cka` & `weight_delta` fusion modes (**Ours**) |
| 🧩 `componet` | CompoNet |
| 📦 `packnet` | PackNet |
| 📈 `prognet` | Progressive Neural Networks |
| 🎭 `masknet` | MaskNet |
| 🔄 `cbpnet` | CBP (Continual Backpropagation) |
| ⚡ `crelus` | CReLUs |

---

## 📂 Project Structure

```text
Continual-RL-Project/
├── cka_rl.py            # 🧠 Main CKARL agent (Weight decomposition & distillation)
├── fuse_module.py       # 🔗 FuseLinear/FuseShared layers (Dual-mode fusion logic)
├── shared_arch.py       # 🏗️ Shared 2-layer MLP encoder architecture
├── run_sac.py           # 🏃‍♂️ SAC training loop, buffer generation, and evaluation
├── run_experiments.py   # 🔄 Automated pipeline for sequential task execution
├── AdamGnT.py           # ⚙️ Adam optimizer variant with Generate-and-Test
├── tasks.py             # 🎯 MetaWorld task definitions (7 manipulation skills)
├── test.py              # 🧪 Evaluation script for policy variant testing & plotting
└── process_results.py   # 📊 Aggregates results (Forward Transfer, Forgetting Rate)

```

---

## 🤖 Tasks (MetaWorld)

We evaluate our method on 7 continuous robotic manipulation tasks:
`hammer-v2`, `faucet-close-v2`, `stick-pull-v2`, `handle-press-side-v2`, `push-v2`, `window-close-v2`, `peg-unplug-side-v2`.

The full continual sequence consists of **17 modes** (all 7 tasks sequentially, repeated, followed by 3 extended tasks). Custom task sequences can be easily defined via the CLI.

---

## 💻 Usage

Our automated pipeline allows you to seamlessly switch between baseline methods and our proposed architecture.

### 1️⃣ Train using Our Proposed Method (Weight-Space Delta)

```bash
python run_experiments.py \
    --algorithm cka-rl \
    --fusion-mode weight_delta \
    --task-sequence 1 3 5 \
    --seed 42 \
    --tag Ours_WeightDelta \
    --pool_size 2

```

### 2️⃣ Train using the Original Baseline (Classic CKA)

```bash
python run_experiments.py \
    --algorithm cka-rl \
    --fusion-mode classic_cka \
    --task-sequence 1 3 5 \
    --seed 42 \
    --tag Baseline_Run \
    --pool_size 2

```

### 3️⃣ Other Baselines & Oracles

```bash
# Train with a different continual RL baseline (e.g., PackNet)
python run_experiments.py --algorithm packnet --seed 42

# Single-task training (Oracle upper bound)
python run_experiments.py --algorithm simple --seed 42

```

---

## 🛠️ Requirements

See `requirements.txt` for the full list of dependencies. Key requirements include:

* `torch >= 2.1.2`
* `gymnasium >= 0.29.1`
* `metaworld` (Farama-Foundation fork)
* `stable-baselines3 >= 2.2.1`
* `mujoco >= 3.0.0`

---
