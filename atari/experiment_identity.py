"""Run/checkpoint identity helpers for reproducible continual Atari PPO experiments.

A checkpoint is reusable only when three things still match:
1. the training-relevant CLI configuration,
2. the source files implementing training/environment dynamics, and
3. the parent/pretrained checkpoints the run actually depended on.

Pure logging/output changes are excluded from the training fingerprint so
switching TensorBoard writers or adding scalar logs does not invalidate an
otherwise identical trained checkpoint.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import pathlib
from typing import Any, Mapping, Sequence

MANIFEST_NAME = "run_manifest.json"
MANIFEST_SCHEMA_VERSION = 2

# Only knobs that can change the learned PPO policy/pool, continuation state, or
# the periodic learning curves used for Forward Transfer belong here.
TRAINING_KEYS = (
    "model_type",
    "task_suite",
    "task_id",
    "seq_idx",
    "seed",
    "torch_deterministic",
    "cuda",

    # New shared composition-space semantics.
    "fusion_mode",
    "composition_space",
    "policy_student_replay",
    "projection_epochs",
    "projection_max_samples",
    "eval_action_mode",

    # PPO.
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

    # Periodic evaluation curves used by FT.
    "eval_every",
    "num_evals",
    "success_threshold",

    # Knowledge pool / routing.
    "pool_size",
    "alpha_init",
    "alpha_major",
    "alpha_factor",
    "fix_alpha",
    "alpha_learning_rate",
    "alpha_mass_learning_rate",
    "alpha_warmup_steps",
    "alpha_entropy_reg",
    "alpha_mass_reg",
    "use_alpha_scale",
    "fix_alpha_scale",
    "use_alpha_mass",
    "constrain_alpha_mass",

    # Shared encoder.
    "encoder_from_base",
    "train_shared",
    "freeze_root_encoder",
    "shared_dim",
    "head_hidden_dim",
    "distill_encoder_lr_mult",
    "drift_reg",

    # Merge / distillation / frozen-tail protocol.
    "distillation",
    "collect_cosine_buffers",
    "distill_extra_steps",
    "max_distill_buffer",
    "similarity_samples",
    "balance_source_lineages",
    "distill_max_samples",
    "distill_epochs",
    "distill_lr",
    "distill_batch_size",
    "distill_test_frac",
    "distill_select_best_val",
)

# Files that can affect the Atari training trajectory, bounded-pool state, or
# the stored analysis needed for lineage/KL diagnostics. Plotting-only and
# checkpoint-evaluation-only files are intentionally excluded.
SOURCE_CANDIDATES = (
    "run_ppo_continual.py",
    "cka_rl.py",
    "policy_composition.py",
    "policy_space.py",
    "training_protocol.py",
    "knowledge_pools.py",
    "shared_arch.py",
    "policy_utils.py",
    "atari_tasks.py",
    "atari_envs.py",
    "analysis_logging.py",
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
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
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


class _RunPpoLoggingStripper(ast.NodeTransformer):
    """Strip pure scalar-writer plumbing from run_ppo_continual.py identity.

    Training/evaluation calls remain fingerprinted. Only the writer import,
    writer construction, and direct writer.* output calls are removed.
    """

    _WRITER_METHODS = {"add_scalar", "add_text", "flush", "close"}

    def visit_ImportFrom(self, node):
        if node.module in {"torch.utils.tensorboard", "csv_summary_writer"}:
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(
            isinstance(target, ast.Name) and target.id == "writer"
            for target in node.targets
        ):
            value = node.value
            if isinstance(value, ast.Call):
                func = value.func
                if (
                    isinstance(func, ast.Name)
                    and func.id in {"SummaryWriter", "CsvSummaryWriter"}
                ):
                    return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            owner = value.func.value
            if (
                isinstance(owner, ast.Name)
                and owner.id == "writer"
                and value.func.attr in self._WRITER_METHODS
            ):
                return None
        return self.generic_visit(node)


def _semantic_source_bytes(name: str, path: pathlib.Path) -> bytes:
    """Canonical source bytes used by the training identity fingerprint."""
    if name != "run_ppo_continual.py":
        return path.read_bytes()

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    tree = _RunPpoLoggingStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(
        tree,
        annotate_fields=True,
        include_attributes=False,
    ).encode("utf-8")


def source_fingerprint(root=None) -> str:
    """Fingerprint Atari training semantics while ignoring pure writer changes."""
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
        h.update(_semantic_source_bytes(name, path))
        h.update(b"\0")

    if found == 0:
        raise RuntimeError(f"no Atari training source files found under {root}")
    return h.hexdigest()


def runtime_versions() -> dict:
    result = {}
    for package in (
        "python",
        "torch",
        "numpy",
        "gymnasium",
        "ale-py",
        "stable-baselines3",
        "tyro",
    ):
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
    config = {
        key: _jsonable(mapping[key])
        for key in TRAINING_KEYS
        if key in mapping
    }

    # Match the HalfCheetah convention: a missing/None dedicated mass LR means
    # "reuse the alpha LR", so explicit equality and the default are identical.
    if (
        "alpha_mass_learning_rate" in config
        and config["alpha_mass_learning_rate"] is None
    ):
        config["alpha_mass_learning_rate"] = config.get("alpha_learning_rate")

    return config


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
            None
            if pretrained_encoder is None
            else sha256_file(pretrained_encoder)
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
    check_runtime: bool = True,
) -> tuple[bool, str]:
    """Return (matches, reason) for resumable orchestration."""
    manifest = load_manifest(run_dir)
    if manifest is None:
        return False, "missing/invalid Atari run_manifest.json"

    expected = training_config(expected_mapping)
    actual = manifest.get("training_config", {})

    for key, value in expected.items():
        # Before the dedicated mass-LR knob exists in an Atari manifest, its
        # effective behavior is exactly alpha_learning_rate.
        if (
            key == "alpha_mass_learning_rate"
            and key not in actual
            and value == actual.get("alpha_learning_rate")
        ):
            continue

        if actual.get(key) != value:
            return (
                False,
                f"training config mismatch for {key}: "
                f"saved={actual.get(key)!r}, expected={value!r}",
            )

    if manifest.get("source_fingerprint") != source_fingerprint(root):
        return False, "Atari training source fingerprint changed"

    # Post-hoc metric code can deliberately skip the current-runtime check while
    # still comparing manifests produced by the original training runtimes.
    if check_runtime:
        if manifest.get("runtime_versions") != runtime_versions():
            return False, "Python/package runtime versions changed"

    expected_pretrained = (
        None
        if pretrained_encoder is None
        else sha256_file(pretrained_encoder)
    )
    if manifest.get("pretrained_encoder_sha256") != expected_pretrained:
        return False, "pretrained encoder contents changed"

    expected_parents = parent_signatures(parent_dirs)
    if manifest.get("parent_signatures") != expected_parents:
        return False, "parent checkpoint identity changed"

    return True, "match"
