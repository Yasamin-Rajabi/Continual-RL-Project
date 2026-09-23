"""Shared benchmark naming, task, path, and success-threshold helpers.

This module is intentionally method-agnostic.  The continual baselines and the
post-hoc metric code import the same helpers so repeated task occurrences are
never confused with semantic Atari mode IDs.
"""
from __future__ import annotations

import json
import pathlib
from typing import Iterable, Mapping

ENV_IDS = {
    "Freeway": "ALE/Freeway-v5",
    "SpaceInvaders": "ALE/SpaceInvaders-v5",
}
ENV_NAMES = {value: key for key, value in ENV_IDS.items()}

METHODS = (
    "FT-N",
    "ProgNet",
    "PackNet",
    "MaskNet",
    "CReLUs",
    "CompoNet",
    "CbpNet",
    "CKA-RL",
)

# Training code historically called FT-N "Finetune".  Keep one canonical
# external label while accepting the old spelling where needed.
_METHOD_ALIASES = {
    "Finetune": "FT-N",
    "FTN": "FT-N",
    "FT-N": "FT-N",
    "ProgressiveNet": "ProgNet",
    "ProgNet": "ProgNet",
    "Packnet": "PackNet",
    "PackNet": "PackNet",
    "MaskNet": "MaskNet",
    "CReLUs": "CReLUs",
    "CompoNet": "CompoNet",
    "CbpNet": "CbpNet",
    "CKA": "CKA-RL",
    "CKA-RL": "CKA-RL",
    "Baseline": "Baseline",
}

# Methods whose current task is represented by a dedicated frozen module.
# When evaluating an old task after later training, use that task's latest
# module rather than the final active module.
TASK_MODULE_METHODS = frozenset({"ProgNet", "CompoNet"})

# Methods that continue from one latest checkpoint.
LATEST_ONLY_METHODS = frozenset({"FT-N", "PackNet", "MaskNet", "CReLUs", "CbpNet"})

# Methods that need the complete prior module list during construction.
ALL_PREVIOUS_METHODS = frozenset({"ProgNet", "CompoNet", "CKA-RL"})


def canonical_method(name: str) -> str:
    try:
        return _METHOD_ALIASES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown method {name!r}") from exc


def trainer_method(name: str) -> str:
    """Name understood by the legacy model-selection branch in run_ppo.py."""
    name = canonical_method(name)
    return "Finetune" if name == "FT-N" else name


def canonical_env_name(env: str) -> str:
    env = str(env)
    if env in ENV_IDS:
        return env
    if env in ENV_NAMES:
        return ENV_NAMES[env]
    low = env.lower().replace("_", "")
    if low == "freeway":
        return "Freeway"
    if low in {"spaceinvaders", "spaceinvader"}:
        return "SpaceInvaders"
    raise ValueError(f"unknown Atari environment/suite {env!r}")


def env_id(env: str) -> str:
    return ENV_IDS[canonical_env_name(env)]


def unique_in_order(values: Iterable[int]) -> list[int]:
    seen = set()
    out = []
    for value in values:
        value = int(value)
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def task_slot_map(task_sequence: Iterable[int]) -> dict[int, int]:
    """Map semantic task/mode IDs to stable first-encounter task slots."""
    return {task_id: slot for slot, task_id in enumerate(unique_in_order(task_sequence))}


def run_name(env: str, task_id: int, method: str, seed: int) -> str:
    env = canonical_env_name(env)
    method = canonical_method(method)
    return f"{env}_{int(task_id)}_{method}_{int(seed)}"


def checkpoint_dir(save_root, env: str, tag: str, method: str, seed: int, seq_idx: int, task_id: int) -> pathlib.Path:
    env = canonical_env_name(env)
    method = canonical_method(method)
    return (
        pathlib.Path(save_root)
        / env
        / str(tag)
        / method
        / f"seed_{int(seed)}"
        / f"seq_{int(seq_idx)}"
        / run_name(env, task_id, method, seed)
    )


def event_dir(runs_root, env: str, tag: str, method: str, seed: int, seq_idx: int, task_id: int) -> pathlib.Path:
    env = canonical_env_name(env)
    method = canonical_method(method)
    return (
        pathlib.Path(runs_root)
        / env
        / str(tag)
        / method
        / f"seed_{int(seed)}"
        / f"seq_{int(seq_idx)}"
        / run_name(env, task_id, method, seed)
    )


def scratch_checkpoint_dir(save_root, env: str, method: str, task_id: int, total_timesteps: int, seed: int) -> pathlib.Path:
    env = canonical_env_name(env)
    method = canonical_method(method)
    return (
        pathlib.Path(save_root)
        / env
        / method
        / f"task_{int(task_id)}"
        / f"steps_{int(total_timesteps)}"
        / f"seed_{int(seed)}"
        / run_name(env, task_id, method, seed)
    )


def scratch_event_dir(runs_root, env: str, method: str, task_id: int, total_timesteps: int, seed: int) -> pathlib.Path:
    env_lower = canonical_env_name(env).lower()  
    
    return (
        pathlib.Path(runs_root)
        / "scratch"
        / env_lower
        / f"task_{int(task_id)}"
        / f"steps_{int(total_timesteps)}"
        / f"seed_{int(seed)}"
        / f"{env_lower}__task_{int(task_id)}__cka-rl__run_ppo__{int(seed)}"
    )


def metric_root(plots_root, env: str, method: str, seed: int) -> pathlib.Path:
    return pathlib.Path(plots_root) / canonical_env_name(env) / canonical_method(method) / f"seed_{int(seed)}"


def load_success_thresholds(spec) -> dict:
    """Load fixed task thresholds from a dict, JSON string, or JSON file.

    Expected shape:
        {"Freeway": {"0": 20.0, ...}, "SpaceInvaders": {"0": 300.0, ...}}

    No thresholds are inferred from method results: doing so would make the
    success definition depend on the methods being compared.
    """
    if spec is None or spec == "":
        return {}
    if isinstance(spec, Mapping):
        return dict(spec)
    path = pathlib.Path(str(spec))
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return json.loads(str(spec))


def success_threshold(mapping: Mapping, env: str, task_id: int):
    if not mapping:
        return None
    env = canonical_env_name(env)
    row = mapping.get(env, mapping.get(ENV_IDS[env], {}))
    value = row.get(str(int(task_id)), row.get(int(task_id))) if isinstance(row, Mapping) else None
    return None if value is None else float(value)


def evaluation_seed(task_id: int, episode: int) -> int:
    return 10_000 + 10_000 * int(task_id) + int(episode)
