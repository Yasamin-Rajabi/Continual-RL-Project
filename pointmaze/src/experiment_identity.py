"""Reproducible checkpoint manifests.

A process can finish with exit code 0 and still have written a checkpoint for
a different configuration than the one that was requested -- a stale flag, a
resumed session with edited defaults, a half-deleted directory.  Every
checkpoint therefore carries a manifest, and the orchestrator refuses to build
on one whose signature does not match what it asked for.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
from typing import Iterable, Optional

MANIFEST_NAME = "manifest.json"

# Fields that define *what was trained*.  Logging paths and plotting options
# are deliberately excluded: re-running with a different --runs-root must not
# invalidate an otherwise identical checkpoint.
IDENTITY_FIELDS = (
    "method",
    "suite",
    "task_id",
    "seq_idx",
    "seed",
    "total_timesteps",
    "distill_extra_steps",
    "fusion_mode",
    "composition_space",
    "policy_student_replay",
    "pool_size",
    "distillation",
    "use_alpha_mass",
    "use_alpha_scale",
    "fix_alpha_scale",
    "constrain_alpha_mass",
    "alpha_init",
    "alpha_warmup_steps",
    "balance_source_lineages",
    "max_distill_buffer",
    "distill_max_samples",
    "distill_epochs",
    "similarity_samples",
    "projection_epochs",
    "projection_max_samples",
    "shared_dim",
    "head_hidden_dim",
    "train_shared",
    "freeze_root_encoder",
    "encoder_from_base",
    "learning_rate",
    "batch_size",
    "gamma",
    "tau",
    "learning_starts",
    "num_tasks",
    # PackNet-only: a width or capacity-rule change makes an old checkpoint
    # incompatible, so it must invalidate the resume.
    "packnet_width",
    "packnet_capacity_mode",
    "packnet_keep_frac",
    "packnet_retrain_frac",
)


def _identity_payload(config: dict, parent_dirs, pretrained_encoder) -> dict:
    payload = {k: config.get(k) for k in IDENTITY_FIELDS if k in config}
    payload["parents"] = [os.path.basename(str(p)) for p in (parent_dirs or [])]
    payload["pretrained_encoder"] = (
        None if pretrained_encoder is None else os.path.basename(str(pretrained_encoder))
    )
    return payload


def run_signature(config: dict, parent_dirs=None, pretrained_encoder=None) -> str:
    payload = _identity_payload(config, parent_dirs, pretrained_encoder)
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def write_manifest(
    run_dir,
    config: dict,
    parent_dirs: Optional[Iterable] = None,
    pretrained_encoder: Optional[str] = None,
) -> dict:
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_signature": run_signature(config, parent_dirs, pretrained_encoder),
        "identity": _identity_payload(config, parent_dirs, pretrained_encoder),
        "config": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in config.items()},
        "parents": [str(p) for p in (parent_dirs or [])],
        "python": platform.python_version(),
    }
    with (run_dir / MANIFEST_NAME).open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def read_manifest(run_dir) -> Optional[dict]:
    path = pathlib.Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def manifest_matches(run_dir, config: dict, parent_dirs=None, pretrained_encoder=None):
    """Return (ok, reason) for an existing checkpoint against a requested config.

    Comparison is a *subset* check, not a hash equality check: the orchestrator
    names the identity fields it controls, and every one of them must match
    what the checkpoint recorded.  Fields the orchestrator leaves to the
    trainer's own defaults are not compared.

    A hash over "whichever identity fields happen to be present" would be
    wrong here, because the trainer writes more fields than the orchestrator
    specifies.  The two key sets would never coincide, no checkpoint would ever
    match, and every resumed session would silently retrain the whole chain
    from task 0 -- which is exactly the failure a resume mechanism exists to
    prevent.
    """
    manifest = read_manifest(run_dir)
    if manifest is None:
        return False, f"no {MANIFEST_NAME} in {run_dir}"

    have = dict(manifest.get("identity") or {})
    stored_config = manifest.get("config") or {}
    wanted = _identity_payload(config, parent_dirs, pretrained_encoder)

    diffs = []
    for key, value in wanted.items():
        found = have.get(key, stored_config.get(key, _MISSING))
        if found is _MISSING:
            diffs.append(f"{key}: not recorded in the checkpoint (requested {value!r})")
        elif not _equal(found, value):
            diffs.append(f"{key}: checkpoint={found!r} requested={value!r}")
    if diffs:
        return False, "; ".join(diffs)
    return True, "ok"


_MISSING = object()


def _equal(found, wanted) -> bool:
    """Compare manifest values tolerantly across JSON round-tripping."""
    if found == wanted:
        return True
    if isinstance(wanted, bool) or isinstance(found, bool):
        return bool(found) == bool(wanted)
    if isinstance(wanted, (int, float)) and isinstance(found, (int, float)):
        return abs(float(found) - float(wanted)) <= 1e-9 * max(1.0, abs(float(wanted)))
    # Paths survive JSON as strings; compare by basename where that is enough.
    if isinstance(wanted, list) and isinstance(found, list):
        return [str(x) for x in found] == [str(x) for x in wanted]
    return str(found) == str(wanted)
