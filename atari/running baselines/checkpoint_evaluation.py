"""Method-aware checkpoint evaluation for the legacy Atari baselines.

All reported evaluation uses raw full-game Atari score (no reward clipping and
no EpisodicLife termination) and the same deterministic argmax policy by
default.  Optional test-time adaptation consumes exactly ``test_adapt_steps``
environment interactions before evaluation and never writes the adapted model
back to disk.

The loader respects each method's storage semantics:
* FT-N/CReLUs/CbpNet/CKA-RL: evaluate the stage checkpoint directly.
* MaskNet: evaluate the stage checkpoint with the target task mask selected.
* PackNet: use the stage checkpoint's consolidated masked encoder and the
  target task's preserved actor head.
* ProgNet/CompoNet: evaluate the target task's preserved module/column, because
  later modules do not replace that task-specific policy.
"""
from __future__ import annotations

import pathlib
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from benchmark_protocol import (
    TASK_MODULE_METHODS,
    canonical_method,
    env_id as resolve_env_id,
    evaluation_seed,
    task_slot_map,
)

try:
    import gymnasium as gym
    from stable_baselines3.common.atari_wrappers import (
        FireResetEnv,
        MaxAndSkipEnv,
        NoopResetEnv,
    )
except ImportError as exc:  # pragma: no cover - exercised only without Atari deps
    gym = None
    FireResetEnv = MaxAndSkipEnv = NoopResetEnv = None
    _ATARI_IMPORT_ERROR = exc
else:
    _ATARI_IMPORT_ERROR = None

try:  # Gymnasium/ALE registration is version dependent.
    import ale_py
except ImportError:
    ale_py = None
else:
    if gym is not None and hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)

# These names match the user's existing baseline project.  Importing this file
# does not instantiate any model; the imports only need to succeed when metric
# evaluation is actually run in that project.
try:
    from models import (
        CnnSimpleAgent,
        CnnCompoNetAgent,
        ProgressiveNetAgent,
        PackNetAgent,
        CkaRlAgent,
        CnnMaskAgent,
        CnnCbpAgent,
        CReLUsAgent,
    )
except ImportError:  # allow syntax/unit testing of numeric helpers elsewhere
    CnnSimpleAgent = CnnCompoNetAgent = ProgressiveNetAgent = None
    PackNetAgent = CkaRlAgent = CnnMaskAgent = CnnCbpAgent = CReLUsAgent = None



@contextmanager
def _legacy_pickle_loads():
    """Make legacy project torch.load calls explicit under PyTorch >=2.6.

    The uploaded baseline classes serialize whole nn.Module objects.  PyTorch
    2.6 changed torch.load's default to weights_only=True, which rejects those
    trusted local checkpoints unless weights_only=False is explicit.
    """
    original = torch.load

    def compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = compat
    try:
        yield
    finally:
        torch.load = original

def _torch_load(path, map_location=None):
    kwargs = {} if map_location is None else {"map_location": map_location}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _grayscale_wrapper(env):
    cls = getattr(gym.wrappers, "GrayScaleObservation", None)
    if cls is None:
        cls = getattr(gym.wrappers, "GrayscaleObservation")
    return cls(env)


def _frame_stack_wrapper(env, n=4):
    cls = getattr(gym.wrappers, "FrameStack", None)
    if cls is not None:
        return cls(env, n)
    cls = getattr(gym.wrappers, "FrameStackObservation")
    return cls(env, stack_size=n)


def make_eval_env(env, task_id: int):
    """Raw full-game Atari evaluator with the benchmark preprocessing."""
    if _ATARI_IMPORT_ERROR is not None:
        raise ImportError(
            "Atari evaluation requires gymnasium, stable-baselines3 and ALE."
        ) from _ATARI_IMPORT_ERROR

    base = gym.make(
        resolve_env_id(env),
        mode=int(task_id),
        frameskip=1,
        repeat_action_probability=0.25,
    )
    base = NoopResetEnv(base, noop_max=30)
    base = MaxAndSkipEnv(base, skip=4)
    if "FIRE" in base.unwrapped.get_action_meanings():
        base = FireResetEnv(base)
    base = gym.wrappers.ResizeObservation(base, (84, 84))
    base = _grayscale_wrapper(base)
    base = _frame_stack_wrapper(base, 4)
    return base


def _env_spec(env):
    """Minimal object expected by the legacy model constructors/loaders."""
    return SimpleNamespace(single_action_space=env.action_space)


def _require(path: pathlib.Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


def _latest_occurrence(task_sequence: Sequence[int], task_id: int, stage_idx: int):
    matches = [
        i
        for i, value in enumerate(task_sequence[: stage_idx + 1])
        if int(value) == int(task_id)
    ]
    return None if not matches else matches[-1]


def checkpoint_complete(method: str, directory) -> bool:
    method = canonical_method(method)
    directory = pathlib.Path(directory)
    if not directory.exists():
        return False
    if method == "PackNet":
        return (directory / "packnet.pt").is_file()
    if method in {"ProgNet", "CompoNet"}:
        return all((directory / name).is_file() for name in ("actor.pt", "encoder.pt"))
    return all((directory / name).is_file() for name in ("actor.pt", "encoder.pt"))


def _load_prognet(directory, previous_dirs, env_spec, device):
    if ProgressiveNetAgent is None:
        raise ImportError("could not import ProgressiveNetAgent from models")
    with _legacy_pickle_loads():
        model = ProgressiveNetAgent(
            envs=env_spec,
            prevs_paths=[str(p) for p in previous_dirs],
            map_location=device,
        )
    previous_models = model.encoder.previous_models
    model.actor = _torch_load(pathlib.Path(directory) / "actor.pt", device)
    model.encoder = _torch_load(pathlib.Path(directory) / "encoder.pt", device)
    model.encoder.previous_models = previous_models
    return model


def _load_componet(directory, previous_dirs, env_spec, device):
    if CnnCompoNetAgent is None:
        raise ImportError("could not import CnnCompoNetAgent from models")
    with _legacy_pickle_loads():
        return CnnCompoNetAgent.load(
            str(directory),
            env_spec,
            prevs_paths=[str(p) for p in previous_dirs],
            map_location=device,
        )


def _restore_packnet_view(model, target_slot: int):
    # save() applies a task view before serialization.  Restore the unmasked
    # tensor first, then apply the requested semantic task's view.
    if getattr(model.network, "view", None) is not None:
        model.network.set_view(None)
    model.task_id = int(target_slot) + 1
    model.network.task_id = int(target_slot) + 1
    model.network.set_view(int(target_slot) + 1)


def load_policy_for_task(
    method: str,
    checkpoint_dirs: Sequence[pathlib.Path],
    task_sequence: Sequence[int],
    stage_idx: int,
    eval_task_id: int,
    env_spec,
    device,
):
    """Load the policy that method would use for task ``eval_task_id`` at stage."""
    method = canonical_method(method)
    stage_idx = int(stage_idx)
    if not 0 <= stage_idx < len(checkpoint_dirs):
        raise IndexError("stage_idx outside checkpoint sequence")

    target_occurrence = _latest_occurrence(task_sequence, eval_task_id, stage_idx)
    if target_occurrence is None:
        return None  # task was not encountered yet

    stage_dir = pathlib.Path(checkpoint_dirs[stage_idx])
    target_dir = pathlib.Path(checkpoint_dirs[target_occurrence])
    slots = task_slot_map(task_sequence)
    target_slot = slots[int(eval_task_id)]
    num_tasks = len(slots)

    if method in TASK_MODULE_METHODS:
        # Frozen task-specific modules are the method's retained policy for that
        # task; later modules are additional columns/components, not replacements.
        previous_dirs = [pathlib.Path(p) for p in checkpoint_dirs[:target_occurrence]]
        if method == "ProgNet":
            model = _load_prognet(target_dir, previous_dirs, env_spec, device)
        else:
            model = _load_componet(target_dir, previous_dirs, env_spec, device)
        return model.to(device)

    if method == "PackNet":
        _require(stage_dir / "packnet.pt", "PackNet stage checkpoint")
        model = _torch_load(stage_dir / "packnet.pt", device)
        # The shared masked encoder must come from the current stage, while the
        # task-specific actor head comes from that task's latest occurrence.
        target_saved = _torch_load(
            _require(target_dir / "packnet.pt", "PackNet task checkpoint"),
            device,
        )
        model.actor = target_saved.actor
        _restore_packnet_view(model, target_slot)
        return model.to(device)

    if method == "MaskNet":
        if CnnMaskAgent is None:
            raise ImportError("could not import CnnMaskAgent from models")
        model = CnnMaskAgent(env_spec, num_tasks=num_tasks)
        model.network = _torch_load(_require(stage_dir / "encoder.pt", "MaskNet encoder"), device)
        model.actor = _torch_load(_require(stage_dir / "actor.pt", "MaskNet actor"), device)
        model.set_task(target_slot, new_task=False)
        return model.to(device)

    if method in {"FT-N", "Baseline"}:
        if CnnSimpleAgent is None:
            raise ImportError("could not import CnnSimpleAgent from models")
        model = CnnSimpleAgent(env_spec)
        model.network = _torch_load(_require(stage_dir / "encoder.pt", "simple encoder"), device)
        model.actor = _torch_load(_require(stage_dir / "actor.pt", "simple actor"), device)
        return model.to(device)

    if method == "CReLUs":
        model = CReLUsAgent(env_spec)
        model.network = _torch_load(_require(stage_dir / "encoder.pt", "CReLUs encoder"), device)
        model.actor = _torch_load(_require(stage_dir / "actor.pt", "CReLUs actor"), device)
        return model.to(device)

    if method == "CbpNet":
        model = CnnCbpAgent(env_spec)
        model.network = _torch_load(_require(stage_dir / "encoder.pt", "CbpNet encoder"), device)
        model.actor = _torch_load(_require(stage_dir / "actor.pt", "CbpNet actor"), device)
        return model.to(device)

    if method == "CKA-RL":
        # For evaluation we need the serialized encoder+actor behavior, not the
        # constructor's historical-vector reconstruction. This also avoids old
        # torch.load defaults in the legacy static load method.
        model = CkaRlAgent(env_spec, None, None)
        model.network = _torch_load(_require(stage_dir / "encoder.pt", "CKA-RL encoder"), device)
        model.actor = _torch_load(_require(stage_dir / "actor.pt", "CKA-RL actor"), device)
        return model.to(device)

    raise ValueError(method)


def _distribution(agent, method: str, obs: torch.Tensor) -> Categorical:
    """Return the exact categorical action distribution of a legacy agent."""
    method = canonical_method(method)

    if method == "CompoNet":
        values = agent.actor(obs)
        probs = values[0]
        return Categorical(probs=probs)

    if method == "ProgNet":
        hidden = agent.encoder(obs)
        logits = agent.actor(hidden)
        return Categorical(logits=logits)

    # Every other uploaded Atari baseline uses .network followed by .actor.
    hidden = agent.network(obs)
    logits = agent.actor(hidden)
    return Categorical(logits=logits)


def _adaptation_parameters(agent, method: str):
    """Inference-time adaptation parameters.

    To keep test-time interactions comparable without rewriting each continual
    method, the generic baselines adapt only their action-head parameters.  For
    CKA-RL, where the reusable-selection mechanism is explicit, adaptation is
    restricted to alpha/alpha-scale parameters, matching the CKA evaluation
    protocol rather than fine-tuning stored experts.
    """
    method = canonical_method(method)

    if method == "CKA-RL":
        selected = []
        seen = set()
        for name, param in agent.actor.named_parameters():
            if "alpha" in name and param.requires_grad and id(param) not in seen:
                selected.append(param)
                seen.add(id(param))
        return selected

    encoder_ids = set()
    if method == "CompoNet":
        # CompoNet registers the encoder inside actor as well.  The adaptation
        # protocol is head/routing-only, so explicitly exclude encoder tensors.
        actor_encoder = getattr(agent.actor, "encoder", None)
        if isinstance(actor_encoder, torch.nn.Module):
            encoder_ids.update(id(p) for p in actor_encoder.parameters())
        if isinstance(getattr(agent, "encoder", None), torch.nn.Module):
            encoder_ids.update(id(p) for p in agent.encoder.parameters())

    selected = []
    seen = set()
    for param in agent.actor.parameters():
        if param.requires_grad and id(param) not in encoder_ids and id(param) not in seen:
            selected.append(param)
            seen.add(id(param))
    return selected


def _obs_tensor(obs, device):
    return torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device).unsqueeze(0) / 255.0


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def evaluate_loaded_policy(
    agent,
    method: str,
    env,
    task_id: int,
    episodes: int,
    device,
    *,
    seed: int,
    action_mode: str = "deterministic",
    success_threshold=None,
    adapt_steps: int = 0,
    adapt_lr: float = 1e-2,
):
    if action_mode not in {"deterministic", "stochastic"}:
        raise ValueError("action_mode must be deterministic or stochastic")
    if episodes < 1 or adapt_steps < 0 or adapt_lr <= 0:
        raise ValueError("invalid evaluation/adaptation budget")

    method = canonical_method(method)
    agent.eval()

    # Preserve which parameters were eligible before freezing the loaded model.
    selected = _adaptation_parameters(agent, method) if adapt_steps else []
    for param in agent.parameters():
        param.requires_grad_(False)
    for param in selected:
        param.requires_grad_(True)
    optimizer = torch.optim.Adam(selected, lr=float(adapt_lr)) if selected else None

    adaptation_interactions = 0
    evaluation_interactions = 0
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed) + 10_000 * int(task_id))

        if adapt_steps:
            obs, _ = env.reset(seed=int(seed) + 1_000_000 + int(task_id))
            for _ in range(int(adapt_steps)):
                x = _obs_tensor(obs, device)
                dist = _distribution(agent, method, x)
                action = dist.sample()
                next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                adaptation_interactions += 1

                if optimizer is not None:
                    # Same deliberately lightweight one-step score-function
                    # heuristic used by the CKA checkpoint evaluator.
                    loss = -dist.log_prob(action).mean() * float(reward)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                if terminated or truncated:
                    obs, _ = env.reset()
                else:
                    obs = next_obs

        for param in agent.parameters():
            param.requires_grad_(False)
        agent.eval()

        returns = []
        successes = []
        for episode in range(int(episodes)):
            obs, _ = env.reset(seed=evaluation_seed(task_id, episode))
            total = 0.0
            while True:
                x = _obs_tensor(obs, device)
                with torch.no_grad():
                    dist = _distribution(agent, method, x)
                    action = (
                        dist.sample()
                        if action_mode == "stochastic"
                        else torch.argmax(dist.probs, dim=-1)
                    )
                obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                evaluation_interactions += 1
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
            if success_threshold is not None:
                successes.append(float(total >= float(success_threshold)))

    value = _finite_mean(returns)
    success = _finite_mean(successes)
    return {
        "return": value,
        "reward": value,
        "success": success,
        "adaptation_interactions": int(adaptation_interactions),
        "evaluation_interactions": int(evaluation_interactions),
    }


def evaluate_checkpoint(
    method: str,
    checkpoint_dirs: Sequence[pathlib.Path],
    task_sequence: Sequence[int],
    stage_idx: int,
    eval_task_id: int,
    episodes: int,
    seed: int,
    device,
    *,
    test_adapt_steps: int = 0,
    test_adapt_lr: float = 1e-2,
    action_mode: str = "deterministic",
    success_threshold=None,
    env="Freeway",
):
    """Evaluate one method at one continual stage on one semantic task."""
    device = torch.device(device)
    evaluator = make_eval_env(env, eval_task_id)
    try:
        model = load_policy_for_task(
            method,
            checkpoint_dirs,
            task_sequence,
            stage_idx,
            eval_task_id,
            _env_spec(evaluator),
            device,
        )
        if model is None:
            return {
                "return": float("nan"),
                "reward": float("nan"),
                "success": float("nan"),
                "adaptation_interactions": 0,
                "evaluation_interactions": 0,
            }
        return evaluate_loaded_policy(
            model,
            method,
            evaluator,
            eval_task_id,
            episodes,
            device,
            seed=seed,
            action_mode=action_mode,
            success_threshold=success_threshold,
            adapt_steps=test_adapt_steps,
            adapt_lr=test_adapt_lr,
        )
    finally:
        evaluator.close()


@torch.no_grad()
def evaluate_live_agent(
    agent,
    method: str,
    env,
    task_id: int,
    episodes: int,
    device,
    *,
    action_mode: str = "deterministic",
    success_threshold=None,
):
    """Raw full-game evaluation during training for FT learning curves."""
    device = torch.device(device)
    evaluator = make_eval_env(env, task_id)
    was_training = bool(agent.training)
    agent.eval()
    try:
        returns, successes = [], []
        for episode in range(int(episodes)):
            obs, _ = evaluator.reset(seed=evaluation_seed(task_id, episode))
            total = 0.0
            while True:
                x = _obs_tensor(obs, device)
                dist = _distribution(agent, method, x)
                action = (
                    dist.sample()
                    if action_mode == "stochastic"
                    else torch.argmax(dist.probs, dim=-1)
                )
                obs, reward, terminated, truncated, _ = evaluator.step(int(action.item()))
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
            if success_threshold is not None:
                successes.append(float(total >= float(success_threshold)))
    finally:
        evaluator.close()
        agent.train(was_training)

    value = _finite_mean(returns)
    return {
        "return": value,
        "reward": value,
        "success": _finite_mean(successes),
    }
