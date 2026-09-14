"""Run/checkpoint identity helpers for reproducible continual Atari PPO experiments.

A checkpoint is reusable only when:
1) training/evaluation settings that affect the learned policy or FT curves match,
2) source files implementing Atari training/dynamics match, and
3) parent/pretrained checkpoint identities match.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pathlib
from typing import Any, Mapping, Sequence

MANIFEST_NAME = "run_manifest.json"
MANIFEST_SCHEMA_VERSION = 2

# Settings that can alter optimization, continuation state, merge behavior, or
# the periodic evaluation curves used for FT. Pure output/plot settings are
# intentionally excluded.
TRAINING_KEYS = (
    "model_type",
    "task_suite",
    "task_id",
    "seq_idx",
    "seed",
    "torch_deterministic",
    "cuda",
    # PPO
    "total_timesteps",
    "learning_rate",
    "num_envs",
    "num_steps",
    "anneal_lr",
    "gamma",
    "gae_lambda",
    "num_minibatches",
    "update_epochs",
    "norm_adv",
    "clip_coef",
    "clip_vloss",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "target_kl",
    # periodic evaluation curves used by FT
    "eval_every",
    "num_evals",
    "success_threshold",
    # knowledge-pool method
    "fusion_mode",
    "pool_size",
    "alpha_init",
    "alpha_major",
    "alpha_factor",
    "fix_alpha",
    "alpha_learning_rate",
    "alpha_warmup_steps",
    "alpha_entropy_reg",
    "alpha_mass_reg",
    "use_alpha_scale",
    "fix_alpha_scale",
    "use_alpha_mass",
    "constrain_alpha_mass",
    # shared encoder
    "encoder_from_base",
    "train_shared",
    "freeze_root_encoder",
    "shared_dim",
    "head_hidden_dim",
    "distill_encoder_lr_mult",
    "drift_reg",
    # merge / distillation
    "distillation",
    "collect_cosine_buffers",
    "distill_extra_steps",
    "max_distill_buffer",
    "similarity_samples",
    "distill_max_samples",
    "distill_epochs",
    "distill_lr",
    "distill_batch_size",
    "distill_test_frac",
    "distill_select_best_val",
)

# Files that can change an Atari trajectory or the continuation state. Keep
# HalfCheetah-only files out of this fingerprint.
SOURCE_CANDIDATES = (
    "run_ppo_continual.py",
    "cka_rl.py",
    "knowledge_pools.py",
    "shared_arch.py",
    "atari_envs.py",
    "atari_tasks.py",
    "experiment_identity.py",
)


def _jsonable(value: Any):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(k): _jsonable(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def sha256_file(path) -> str | None:
    path = pathlib.Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fingerprint(root=None) -> str:
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
        raise RuntimeError(f"no Atari training source files found under {root}")
    return h.hexdigest()


def runtime_versions() -> dict:
    packages = (
        "python",
        "torch",
        "numpy",
        "gymnasium",
        "ale-py",
        "stable-baselines3",
        "tyro",
    )
    result = {}
    for package in packages:
        if package == "python":
            import sys
            result[package] = sys.version.split()[0]
            continue
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def training_config(mapping: Mapping[str, Any]) -> dict:
    return {
        key: _jsonable(mapping[key])
        for key in TRAINING_KEYS
        if key in mapping
    }


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
    return data


def checkpoint_signature(run_dir) -> str | None:
    manifest = load_manifest(run_dir)
    return None if manifest is None else manifest.get("run_signature")


def parent_signatures(parent_dirs: Sequence[Any]) -> list[str | None]:
    return [checkpoint_signature(path) for path in parent_dirs]


def build_manifest(
    args_mapping: Mapping[str, Any],
    *,
    parent_dirs=(),
    pretrained_encoder=None,
    root=None,
) -> dict:
    root = pathlib.Path(root or pathlib.Path(__file__).resolve().parent)
    identity = {
        "training_config": training_config(args_mapping),
        "source_fingerprint": source_fingerprint(root),
        "runtime_versions": runtime_versions(),
        "pretrained_encoder_sha256": (
            None if pretrained_encoder is None else sha256_file(pretrained_encoder)
        ),
        "parent_signatures": parent_signatures(parent_dirs),
    }
    run_signature = hashlib.sha256(
        _canonical_json(identity).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_signature": run_signature,
        **identity,
        "args": _jsonable(dict(args_mapping)),
    }


def write_manifest(
    run_dir,
    args_mapping: Mapping[str, Any],
    *,
    parent_dirs=(),
    pretrained_encoder=None,
    root=None,
) -> dict:
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        args_mapping,
        parent_dirs=parent_dirs,
        pretrained_encoder=pretrained_encoder,
        root=root,
    )
    with (run_dir / MANIFEST_NAME).open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def checkpoint_matches(
    run_dir,
    expected_mapping: Mapping[str, Any],
    *,
    parent_dirs=(),
    pretrained_encoder=None,
    root=None,
) -> tuple[bool, str]:
    manifest = load_manifest(run_dir)
    if manifest is None:
        return False, "missing/invalid Atari run_manifest.json"

    expected = training_config(expected_mapping)
    actual = manifest.get("training_config", {})
    for key, value in expected.items():
        if actual.get(key) != value:
            return (
                False,
                f"training config mismatch for {key}: "
                f"saved={actual.get(key)!r}, expected={value!r}",
            )

    current_source = source_fingerprint(root)
    if manifest.get("source_fingerprint") != current_source:
        return False, "Atari training source fingerprint changed"

    current_runtime = runtime_versions()
    if manifest.get("runtime_versions") != current_runtime:
        return False, "Python/package runtime versions changed"

    expected_pretrained = (
        None if pretrained_encoder is None else sha256_file(pretrained_encoder)
    )
    if manifest.get("pretrained_encoder_sha256") != expected_pretrained:
        return False, "pretrained encoder contents changed"

    expected_parents = parent_signatures(parent_dirs)
    if manifest.get("parent_signatures") != expected_parents:
        return False, "parent checkpoint identity changed"

    return True, "match"
