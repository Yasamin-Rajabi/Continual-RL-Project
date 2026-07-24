# CKARL — Continual Knowledge-Aware Reinforcement Learning

Implementation of the **CKARL** method for continual reinforcement learning, evaluated on **MetaWorld** robotic manipulation benchmarks.

CKARL uses **CKA (Centered Kernel Alignment)** to fuse knowledge from previously learned policies when adapting to new tasks, mitigating catastrophic forgetting.

## Supported Algorithms

| Algorithm | Description |
|-----------|-------------|
| `simple` | Single-task SAC (upper bound) |
| `finetune` | Fine-tuning baseline |
| `cka-rl` | **CKARL** — CKA-based knowledge fusion (ours) |
| `componet` | CompoNet |
| `packnet` | PackNet |
| `prognet` | Progressive Neural Networks |
| `masknet` | MaskNet |
| `cbpnet` | CBP (Continual Backpropagation) |
| `crelus` | CReLUs |

## Project Structure

| File | Description |
|------|-------------|
| `cka_rl.py` | CKARL agent — weight decomposition, alpha fusion, CKA-based reuse |
| `fuse_module.py` | `FuseLinear` / `FuseShared` layers with learnable alpha-weighted fusion |
| `shared_arch.py` | Shared 2-layer MLP encoder (256→256) |
| `run_sac.py` | SAC training loop with replay buffer, evaluation, and logging |
| `run_experiments.py` | Sequentially runs all tasks for a given algorithm |
| `AdamGnT.py` | Adam variant with Generate-and-Test support |
| `tasks.py` | MetaWorld task definitions (7 manipulation skills) |
| `test.py` | Evaluation script |
| `extract_results.py` | Extract metrics from saved logs |
| `process_results.py` | Aggregate and tabulate results |

## Tasks (MetaWorld)

7 continuous robotic manipulation tasks: `hammer-v2`, `faucet-close-v2`, `stick-pull-v2`, `handle-press-side-v2`, `push-v2`, `window-close-v2`, `peg-unplug-side-v2`.

The full continual sequence is 17 modes: all 7 tasks once, then again, then 3 extended tasks.

## Usage

```bash
# Train CKARL on all tasks sequentially
python run_experiments.py --algorithm cka-rl --seed 42

# Train with a different method
python run_experiments.py --algorithm packnet --seed 42

# Single-task training (oracle upper bound)
python run_experiments.py --algorithm simple --seed 42

# Run a single SAC training session manually
python run_sac.py --model-type cka-rl --save-dir ./logs/ckarl_mode0
```

## Requirements

See `requirements.txt`. Key dependencies: PyTorch 2.1.2, Gymnasium 0.29.1, MetaWorld, Stable-Baselines3 2.2.1, MuJoCo 2.3.7.