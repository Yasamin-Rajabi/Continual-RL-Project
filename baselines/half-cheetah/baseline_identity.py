"""Run/checkpoint identity for the continual-RL BASELINES.

This is the baseline-side twin of experiment_identity.py, and it exists as a
separate module for one specific reason.

``experiment_identity.SOURCE_CANDIDATES`` fingerprints the files that define
the CKA-RL method's training semantics. Adding the baseline files to that tuple
would change ``source_fingerprint()`` and mark EVERY CKA-RL checkpoint already
trained as stale, forcing a full retrain of completed work. Conversely,
fingerprinting the baselines with the method's file list would make a harmless
edit to cka_rl.py invalidate finished baseline runs.

So the two sets are tracked independently:

    experiment_identity.SOURCE_CANDIDATES   the method's semantics
    baseline_identity.SOURCE_CANDIDATES     the baselines' semantics
                                            + the shared files both depend on

The shared files (tasks.py, the env module, shared_arch.py, policy_utils.py,
policy_composition.py, training_protocol.py) appear in BOTH lists, which is
correct: a change to the task definitions or the action distribution really
does invalidate both sides.

Reuses experiment_identity's generic helpers (_jsonable, sha256_file,
runtime_versions, MANIFEST_SCHEMA_VERSION) rather than duplicating them, so the
manifest format stays in lockstep with the method's.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Mapping, Sequence

from experiment_identity import (
    MANIFEST_NAME,
    MANIFEST_SCHEMA_VERSION,
    _canonical_json,
    _jsonable,
    runtime_versions,
)

# Knobs that can change a baseline's learned policy. Logging and output-path
# settings are deliberately excluded so moving an experiment directory does not
# invalidate its model identity.
TRAINING_KEYS = (
    "model_type",
    "method",
    "task_aware",
    "task_suite",
    "task_id",
    "seq_idx",
    "seed",
    "torch_deterministic",
    "cuda",
    "total_timesteps",
    "frozen_tail_steps",
    "buffer_size",
    "gamma",
    "tau",
    "batch_size",
    "learning_starts",
    "random_actions_end",
    "policy_lr",
    "q_lr",
    "policy_frequency",
    "target_network_frequency",
    "alpha",
    "autotune",
    "autotune_init_from_alpha",
    "hidden_dim",
    "encoder_linear_out",
    "eval_action_mode",
    # Do not change gradient updates, but they define the periodic learning
    # curves Forward Transfer integrates, so they belong to run identity.
    "eval_every",
    "num_evals",
    # Method-specific capacity knobs.
    "prognet_adapter_dim",
    "packnet_keep_fraction",
    "packnet_retrain_fraction",
    "masknet_gate_init",
    "masknet_sparsity_reg",
)

# Baseline-owned files plus the shared modules the baselines genuinely depend
# on. Plotting and orchestration files are excluded so a cosmetic change does
# not force retraining.
SOURCE_CANDIDATES = (
    "run_baseline.py",
    "baselines/__init__.py",
    "baselines/common/lifecycle.py",
    "baselines/common/sac_core.py",
    "baselines/common/masks.py",
    "baselines/common/snapshot.py",
    "baselines/ft_n.py",
    "baselines/prognet.py",
    "baselines/packnet.py",
    "baselines/masknet.py",
    # Shared with the method. A change here really does invalidate both sides.
    "tasks.py",
    "halfcheetah_envs.py",
    "metaworld_envs.py",
    "shared_arch.py",
    "policy_utils.py",
    "policy_composition.py",
    "training_protocol.py",
)

# baseline_defaults.py is deliberately NOT fingerprinted. It only supplies
# argparse/tyro defaults, and every value it supplies is recorded in
# training_config once resolved, so a default that actually changed a run is
# already caught by the config comparison. Fingerprinting it would mean editing
# DEFAULT_SEEDS -- which cannot change a trained policy -- invalidated finished
# checkpoints.

CHECKPOINT_FILES = ("policy_snapshot.pt", "agent_state.pt")


def source_fingerprint(root=None) -> str:
    """Hash the baseline training semantics.

    Unlike the method's fingerprint this does no AST stripping: the baseline
    runner keeps all its logging in sac_core.py behind a writer object, so a
    logging change does not touch the files listed here in the first place.
    """
    root = pathlib.Path(root or pathlib.Path(__file__).resolve().parent)
    h = hashlib.sha256()
    found = 0
    for name in SOURCE_CANDIDATES:
        path = root / name
        if not path.exists():
            continue
        found += 1
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    if found == 0:
        raise RuntimeError(f"no baseline source files found under {root}")
    return h.hexdigest()


def training_config(mapping: Mapping[str, Any]) -> dict:
    return {key: _jsonable(mapping[key]) for key in TRAINING_KEYS if key in mapping}


def load_manifest(run_dir) -> dict | None:
    path = pathlib.Path(run_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:
        return None
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return None
    # A CKA-RL manifest in a baseline output directory means two different
    # experiments were pointed at the same path. Refuse it rather than resume.
    if data.get("experiment_family") != "baseline":
        return None
    return data


def checkpoint_signature(run_dir) -> str | None:
    manifest = load_manifest(run_dir)
    return None if manifest is None else manifest.get("run_signature")


def parent_signatures(parent_dirs: Sequence[Any]) -> list[str | None]:
    return [checkpoint_signature(path) for path in parent_dirs]


def build_manifest(args_mapping: Mapping[str, Any], *, parent_dirs=(), root=None) -> dict:
    root = pathlib.Path(root or pathlib.Path(__file__).resolve().parent)
    identity = {
        "training_config": training_config(args_mapping),
        "source_fingerprint": source_fingerprint(root),
        "runtime_versions": runtime_versions(),
        "parent_signatures": parent_signatures(parent_dirs),
    }
    run_signature = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_family": "baseline",
        "run_signature": run_signature,
        **identity,
        "args": _jsonable(dict(args_mapping)),
    }


def write_manifest(run_dir, args_mapping: Mapping[str, Any], *, parent_dirs=(),
                   root=None) -> dict:
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args_mapping, parent_dirs=parent_dirs, root=root)
    with (run_dir / MANIFEST_NAME).open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def checkpoint_complete(path) -> bool:
    path = pathlib.Path(path)
    if not path.exists():
        return False
    if not all((path / name).exists() for name in CHECKPOINT_FILES):
        return False
    return load_manifest(path) is not None


def checkpoint_matches(run_dir, expected_mapping: Mapping[str, Any], *,
                       parent_dirs=(), root=None,
                       check_runtime: bool = True) -> tuple[bool, str]:
    """Return ``(matches, reason)`` for resumable orchestration."""
    if not checkpoint_complete(run_dir):
        return False, "checkpoint files or a valid baseline run_manifest.json are missing"
    manifest = load_manifest(run_dir)

    expected = training_config(expected_mapping)
    actual = manifest.get("training_config", {})
    for key, value in expected.items():
        if actual.get(key) != value:
            return False, f"config mismatch on {key}: {actual.get(key)!r} != {value!r}"

    root = pathlib.Path(root or pathlib.Path(__file__).resolve().parent)
    current_fingerprint = source_fingerprint(root)
    if manifest.get("source_fingerprint") != current_fingerprint:
        return False, "baseline training sources changed since this checkpoint was written"

    expected_parents = parent_signatures(parent_dirs)
    if manifest.get("parent_signatures") != expected_parents:
        return False, "parent checkpoint chain differs from the one this run was trained on"

    if check_runtime:
        stored = manifest.get("runtime_versions", {})
        current = runtime_versions()
        # Patch-level differences are tolerated; a major/minor change to torch
        # or numpy can change kernel behaviour enough to matter.
        for package in ("torch", "numpy"):
            old, new = stored.get(package), current.get(package)
            if old is None or new is None:
                continue
            if old.split(".")[:2] != new.split(".")[:2]:
                return False, f"{package} changed from {old} to {new} since this checkpoint"

    return True, "ok"
