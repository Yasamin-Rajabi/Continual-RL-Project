"""Run/checkpoint identity helpers for reproducible, resumable experiments.

A checkpoint is only reusable when three things still match:
1. the training-relevant CLI configuration,
2. the source files that implement training/environment dynamics, and
3. the parent/pretrained checkpoints the run actually depended on.

This prevents a stale directory from being silently treated as a completed
experiment after changing e.g. timesteps, encoder settings, fusion logic, or a
pretrained encoder file while keeping the same output path.
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

# Only knobs that can change the learned policy/pool are part of the training
# signature. Logging/output-path settings are intentionally excluded so moving
# an experiment directory does not invalidate its model identity.
TRAINING_KEYS = (
    "model_type",
    "task_suite",
    "task_id",
    "seq_idx",
    "seed",
    "torch_deterministic",
    "cuda",
    "fusion_mode",
    "composition_space",
    "policy_student_replay",
    "projection_epochs",
    "projection_max_samples",
    "eval_action_mode",
    "total_timesteps",
    "buffer_size",
    "gamma",
    "tau",
    "batch_size",
    "learning_starts",
    "random_actions_end",
    "policy_lr",
    "alpha_lr",
    "alpha_warmup_steps",
    "alpha_entropy_reg",
    "distill_encoder_lr_mult",
    "q_lr",
    "policy_frequency",
    "target_network_frequency",
    "alpha",
    "autotune",
    "autotune_init_from_alpha",
    "pool_size",
    # These do not change gradient updates, but they define the periodic learning
    # curves used by Forward Transfer and therefore belong to reusable run identity.
    "eval_every",
    "num_evals",
    "encoder_from_base",
    "freeze_root_encoder",
    "distillation",
    "use_alpha_mass",
    "use_alpha_scale",
    "fix_alpha_scale",
    "alpha_mass_reg",
    "drift_reg",
    "constrain_alpha_mass",
    "train_shared",
    "encoder_linear_out",
    "distill_observation_skip",
    "distill_extra_steps",
    "collect_cosine_buffers",
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

# Files that can affect a training trajectory or the stored analysis needed by
# the upcoming lineage/KL investigations. Plotting-only files are excluded so a
# cosmetic plot edit does not force model retraining.
PRE_LINEAGE_BALANCING_SOURCE_FINGERPRINTS = {'8ab92dde4d65d370311a3dce70871804ad3ae6ef426c9e38ae63ea7348b7485c', '1e598563580b6ffbba6830f48cc866bb0e3cce81c43aea7f7213e4cad2ee2702', '29ed1d675cfe9ce90ad07b288d084a2e7548e833f4d9032960d9fe24c4c520c1'}

SOURCE_CANDIDATES = (
    "run_sac.py",
    "cka_rl.py",
    "policy_composition.py",
    "policy_space.py",
    "training_protocol.py",
    "knowledge_pools.py",
    "shared_arch.py",
    "policy_utils.py",
    "tasks.py",
    'walker2d_envs.py',
    "analysis_logging.py",
)


def _jsonable(value: Any):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path) -> str | None:
    path = pathlib.Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class _RunSacLoggingStripper(ast.NodeTransformer):
    """Remove pure logging plumbing from run_sac.py before identity hashing.

    Training semantics still remain fingerprinted.  This only strips the writer
    import/construction and direct writer.* calls, so switching TensorBoard to a
    CSV-mirroring writer or adding/removing scalar logging does not make trained
    checkpoints look stale.
    """

    _WRITER_METHODS = {"add_scalar", "add_text", "flush", "close"}

    def visit_ImportFrom(self, node):
        if node.module in {"torch.utils.tensorboard", "csv_summary_writer"}:
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        # Strip only the top-level writer construction, not arbitrary assignments.
        if any(isinstance(t, ast.Name) and t.id == "writer" for t in node.targets):
            value = node.value
            if isinstance(value, ast.Call):
                func = value.func
                if isinstance(func, ast.Name) and func.id in {"SummaryWriter", "CsvSummaryWriter"}:
                    return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            owner = value.func.value
            if (isinstance(owner, ast.Name) and owner.id == "writer"
                    and value.func.attr in self._WRITER_METHODS):
                return None
        return self.generic_visit(node)


def _semantic_source_bytes(name: str, path: pathlib.Path) -> bytes:
    """Canonical bytes for training identity.

    run_sac.py is parsed so pure metric-output statements do not affect the
    training fingerprint. Other training-relevant files remain byte-exact.
    """
    if name != "run_sac.py":
        return path.read_bytes()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    tree = _RunSacLoggingStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False).encode("utf-8")


def source_fingerprint(root=None) -> str:
    """Fingerprint training semantics while ignoring pure run_sac logging changes."""
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
        raise RuntimeError(f"no training source files found under {root}")
    return h.hexdigest()


def _raw_source_fingerprint(root: pathlib.Path, run_sac_bytes: bytes | None = None) -> str:
    """Legacy schema-v2 byte fingerprint used by already-trained checkpoints."""
    h = hashlib.sha256()
    found = 0
    for name in SOURCE_CANDIDATES:
        path = root / name
        if not path.exists():
            continue
        found += 1
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        if name == "run_sac.py" and run_sac_bytes is not None:
            h.update(run_sac_bytes)
        else:
            h.update(path.read_bytes())
        h.update(b"\0")
    if found == 0:
        raise RuntimeError(f"no training source files found under {root}")
    return h.hexdigest()


def _compatible_legacy_source_fingerprints(root=None) -> set[str]:
    """Raw hashes accepted only for old manifests.

    The CSV logger changed exactly the SummaryWriter import and constructor.
    Reconstruct both byte-level variants from the current source so checkpoints
    produced immediately before or after that logging-only change remain valid.
    Any other training-source change still fails validation.
    """
    root = pathlib.Path(root or pathlib.Path(__file__).resolve().parent)
    run_sac = root / "run_sac.py"
    variants = {_raw_source_fingerprint(root)}
    if not run_sac.exists():
        return variants
    text = run_sac.read_text(encoding="utf-8")

    tb = text.replace(
        "from csv_summary_writer import CsvSummaryWriter",
        "from torch.utils.tensorboard import SummaryWriter",
    ).replace("writer = CsvSummaryWriter(", "writer = SummaryWriter(")
    variants.add(_raw_source_fingerprint(root, tb.encode("utf-8")))

    csv = text.replace(
        "from torch.utils.tensorboard import SummaryWriter",
        "from csv_summary_writer import CsvSummaryWriter",
    ).replace("writer = SummaryWriter(", "writer = CsvSummaryWriter(")
    variants.add(_raw_source_fingerprint(root, csv.encode("utf-8")))
    return variants


def runtime_versions() -> dict:
    result = {}
    for package in ("python", "torch", "numpy", "gymnasium", "mujoco", "stable-baselines3"):
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
    return data


def checkpoint_signature(run_dir) -> str | None:
    manifest = load_manifest(run_dir)
    return None if manifest is None else manifest.get("run_signature")


def parent_signatures(parent_dirs: Sequence[Any]) -> list[str | None]:
    return [checkpoint_signature(path) for path in parent_dirs]


def build_manifest(args_mapping: Mapping[str, Any], *, parent_dirs=(), pretrained_encoder=None, root=None) -> dict:
    root = pathlib.Path(root or pathlib.Path(__file__).resolve().parent)
    identity = {
        "training_config": training_config(args_mapping),
        "source_fingerprint": source_fingerprint(root),
        "runtime_versions": runtime_versions(),
        "pretrained_encoder_sha256": None if pretrained_encoder is None else sha256_file(pretrained_encoder),
        "parent_signatures": parent_signatures(parent_dirs),
    }
    run_signature = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_signature": run_signature,
        **identity,
        "args": _jsonable(dict(args_mapping)),
    }


def write_manifest(run_dir, args_mapping: Mapping[str, Any], *, parent_dirs=(), pretrained_encoder=None, root=None) -> dict:
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
        return False, "missing/invalid run_manifest.json"

    expected = training_config(expected_mapping)
    actual = manifest.get("training_config", {})
    for key, value in expected.items():
        # Checkpoints created before the lineage-balancing ablation existed are
        # exactly the current default when the new flag is False.
        if key == "balance_source_lineages" and key not in actual and value is False:
            continue
        if actual.get(key) != value:
            return False, f"training config mismatch for {key}: saved={actual.get(key)!r}, expected={value!r}"

    saved_source = manifest.get("source_fingerprint")
    current_source = source_fingerprint(root)
    if saved_source != current_source:
        # Backward compatibility for logging-only changes and for the exact
        # pre-lineage-balancing implementation. The latter is accepted only
        # when the expected config keeps balance_source_lineages disabled.
        allow_pre_lineage = (
            not bool(expected.get("balance_source_lineages", False))
            and saved_source in PRE_LINEAGE_BALANCING_SOURCE_FINGERPRINTS
        )
        if (
            saved_source not in _compatible_legacy_source_fingerprints(root)
            and not allow_pre_lineage
        ):
            return False, "training source fingerprint changed"

    # Runtime equality is required when a checkpoint may be resumed/reused for
    # training or policy execution.  Post-hoc metrics that only read already
    # recorded learning curves can set check_runtime=False and compare the
    # *training* runtimes of the scratch and continual manifests instead.
    if check_runtime:
        current_runtime = runtime_versions()
        if manifest.get("runtime_versions") != current_runtime:
            return False, "Python/package runtime versions changed"

    expected_pretrained = None if pretrained_encoder is None else sha256_file(pretrained_encoder)
    if manifest.get("pretrained_encoder_sha256") != expected_pretrained:
        return False, "pretrained encoder contents changed"

    expected_parents = parent_signatures(parent_dirs)
    if manifest.get("parent_signatures") != expected_parents:
        return False, "parent checkpoint identity changed"

    return True, "match"
