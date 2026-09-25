"""Disk compaction for completed continual-RL checkpoints.

A completed predecessor checkpoint contains two logically different kinds of
state:

1. evaluation state (encoder + finalized policy-pool weights), which is needed
   for retention matrices, A_N/FG/BWT, and optional test-time routing; and
2. retained rollout buffers, which are needed only while that checkpoint is the
   latest resumable training state.

After the next task checkpoint has been written successfully, those buffers
have already been inherited into the new latest checkpoint.  Removing them from
older checkpoints therefore preserves post-hoc evaluation while avoiding an
O(sequence_length * pool_size * buffer_size) duplication on disk.
"""
from __future__ import annotations

import json
import os
import pathlib
from datetime import datetime, timezone

import torch


MARKER_NAME = "storage_compacted.json"
_POOL_FILES = ("mean_pool.pt", "logstd_pool.pt", "policy_pool.pt")


def is_compacted(run_dir) -> bool:
    return (pathlib.Path(run_dir) / MARKER_NAME).is_file()


def _buffer_nbytes(buffer) -> int:
    if buffer is None:
        return 0
    total = 0
    seen = set()
    if isinstance(buffer, dict):
        values = buffer.values()
    else:
        values = (buffer,)
    for value in values:
        obj_id = id(value)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        nbytes = getattr(value, "nbytes", None)
        if nbytes is not None:
            total += int(nbytes)
    return total


def _strip_pool_buffers(pool):
    """Remove only rollout buffers; leave all policy/evaluation state intact."""
    stripped_entries = 0
    logical_buffer_bytes = 0
    seen_buffers = set()

    for entry in getattr(pool, "pool", ()):
        if not isinstance(entry, dict):
            continue
        buffer = entry.get("buffer")
        if buffer is None:
            continue
        buf_id = id(buffer)
        if buf_id not in seen_buffers:
            logical_buffer_bytes += _buffer_nbytes(buffer)
            seen_buffers.add(buf_id)
        entry["buffer"] = None
        stripped_entries += 1

    own_buffer = getattr(pool, "own_buffer", None)
    if own_buffer is not None:
        buf_id = id(own_buffer)
        if buf_id not in seen_buffers:
            logical_buffer_bytes += _buffer_nbytes(own_buffer)
            seen_buffers.add(buf_id)
        pool.own_buffer = None

    return stripped_entries, logical_buffer_bytes


def compact_checkpoint(run_dir):
    """Strip training-only rollout buffers from one *non-latest* checkpoint.

    The checkpoint remains fully usable by the benchmark's frozen-pool
    evaluation path, including test-time alpha adaptation.  It is intentionally
    marked non-resumable because future distillation/merging needs the buffers.
    """
    run_dir = pathlib.Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    marker = run_dir / MARKER_NAME
    if marker.exists():
        try:
            with marker.open() as f:
                return json.load(f)
        except Exception:
            pass

    files = [run_dir / name for name in _POOL_FILES if (run_dir / name).is_file()]
    if not files:
        raise FileNotFoundError(f"no policy-pool checkpoint found under {run_dir}")

    before_bytes = sum(path.stat().st_size for path in files)
    stripped_entries = 0
    logical_buffer_bytes = 0
    rewritten = []

    for path in files:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        count, logical = _strip_pool_buffers(obj)
        if count == 0 and logical == 0:
            continue
        tmp = path.with_name(path.name + ".compact_tmp")
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
        stripped_entries += count
        logical_buffer_bytes += logical
        rewritten.append(path.name)

    after_bytes = sum(path.stat().st_size for path in files)
    report = {
        "schema_version": 1,
        "compacted_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_state_preserved": True,
        "resumable_training_state_preserved": False,
        "stripped_pool_entries": int(stripped_entries),
        "logical_buffer_bytes_removed": int(logical_buffer_bytes),
        "pool_files_before_bytes": int(before_bytes),
        "pool_files_after_bytes": int(after_bytes),
        "disk_bytes_saved": int(max(before_bytes - after_bytes, 0)),
        "rewritten_files": rewritten,
    }
    with marker.open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report


def require_resumable(run_dir):
    """Fail clearly if a compacted checkpoint is about to be used as latest."""
    if is_compacted(run_dir):
        raise RuntimeError(
            f"{run_dir} is an evaluation-only compacted checkpoint. Its rollout "
            "buffers were intentionally removed after a later task completed, so "
            "it cannot be used as the latest training predecessor. Keep/use the "
            "newest full checkpoint, or rerun the chain with --no-compact-storage."
        )
