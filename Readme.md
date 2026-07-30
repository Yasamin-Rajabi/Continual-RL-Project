# CKARL — Continual Knowledge-Aware Reinforcement Learning

Implementation of the **CKARL** method for continual reinforcement learning, evaluated on **MetaWorld** robotic manipulation benchmarks.

This repository explores knowledge fusion mechanisms for adapting to new tasks while mitigating catastrophic forgetting. It includes the original **CKA (Centered Kernel Alignment)** representation fusion, as well as our novel **Weight-Space Delta Fusion** approach which strictly isolates base weights and optimizes local variations.

## Supported Algorithms

| Algorithm | Description |
|-----------|-------------|
| `simple` | Single-task SAC (upper bound) |
| `finetune` | Fine-tuning baseline |
| `cka-rl` | **CKARL** — Supports both `classic_cka` and our novel `weight_delta` fusion modes |
| `componet` | CompoNet |
| `packnet` | PackNet |
| `prognet` | Progressive Neural Networks |
| `masknet` | MaskNet |
| `cbpnet` | CBP (Continual Backpropagation) |
| `crelus` | CReLUs |

## Project Structure

| File | Description |
|------|-------------|
| `cka_rl.py` | CKARL agent — handles weight decomposition, dynamic alpha fusion, and policy distillation |
| `fuse_module.py` | `FuseLinear` / `FuseShared` layers supporting dual-mode fusion (`classic_cka` and `weight_delta`) |
| `shared_arch.py` | Shared 2-layer MLP encoder (256→256) |
| `run_sac.py` | SAC training loop with comprehensive distillation buffer generation and evaluation |
| `run_experiments.py` | Automated pipeline to run custom task sequences with isolated tagging |
| `AdamGnT.py` | Adam variant with Generate-and-Test support |
| `tasks.py` | MetaWorld task definitions (7 manipulation skills) |
| `test.py` | Evaluation script for isolated policy variant testing and plotting |
| `extract_results.py` | Extract metrics from saved TensorBoard logs |
| `process_results.py` | Aggregate and tabulate results (Forward Transfer, Forgetting Rate) |

## Tasks (MetaWorld)

7 continuous robotic manipulation tasks: `hammer-v2`, `faucet-close-v2`, `stick-pull-v2`, `handle-press-side-v2`, `push-v2`, `window-close-v2`, `peg-unplug-side-v2`.

The full continual sequence consists of 17 modes: all 7 tasks once, then again, followed by 3 extended tasks. Custom sequences can be defined via CLI.

## Usage

You can easily switch between the classic method and our novel weight-delta approach using the `--fusion-mode` flag.

```bash
# 1. Train using the Original Baseline (Classic CKA)
python run_experiments.py --algorithm cka-rl --fusion-mode classic_cka --task-sequence 1 3 5 --seed 42 --tag Baseline_Run --pool_size 2

# 2. Train using Our Proposed Method (Weight-Space Delta)
python run_experiments.py --algorithm cka-rl --fusion-mode weight_delta --task-sequence 1 3 5 --seed 42 --tag Ours_WeightDelta --pool_size 2

# Train with a different continual RL baseline (e.g., PackNet)
python run_experiments.py --algorithm packnet --seed 42

# Single-task training (Oracle upper bound)
python run_experiments.py --algorithm simple --seed 42


## Requirements

See `requirements.txt`. Key dependencies:

* PyTorch >= 2.1.2
* Gymnasium >= 0.29.1
* MetaWorld (Farama-Foundation fork)
* Stable-Baselines3 >= 2.2.1
* MuJoCo >= 3.0.0
س