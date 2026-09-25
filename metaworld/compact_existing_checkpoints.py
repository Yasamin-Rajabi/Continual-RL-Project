"""Compact already-finished continual-RL result trees without losing metrics.

Run this from an environment directory, for example:

    python compact_existing_checkpoints.py --save-root /path/to/RUN_ROOT/agents --dry-run
    python compact_existing_checkpoints.py --save-root /path/to/RUN_ROOT/agents

By default the newest sequence checkpoint of every suite/condition/seed chain is
left untouched and fully resumable. Older checkpoints lose only retained
rollout buffers; finalized pool weights, policy snapshots, manifests, scalar
logs, and cached metric files are not removed.
"""
from __future__ import annotations

import argparse
import pathlib

import storage_compaction


def _checkpoint_records(save_root: pathlib.Path):
    groups = {}
    for manifest in save_root.rglob("run_manifest.json"):
        run_dir = manifest.parent
        try:
            rel = run_dir.relative_to(save_root)
        except ValueError:
            continue
        parts = rel.parts
        seq_pos = next((i for i, part in enumerate(parts) if part.startswith("seq_")), None)
        if seq_pos is None:
            continue
        try:
            seq_idx = int(parts[seq_pos].split("_", 1)[1])
        except Exception:
            continue
        key = parts[:seq_pos]
        groups.setdefault(key, []).append((seq_idx, run_dir, rel))
    return groups


def _pool_bytes(run_dir: pathlib.Path):
    total = 0
    for name in ("mean_pool.pt", "logstd_pool.pt", "policy_pool.pt"):
        path = run_dir / name
        if path.is_file():
            total += path.stat().st_size
    return total


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--save-root", required=True, type=pathlib.Path)
    p.add_argument("--analysis-root", type=pathlib.Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--include-latest", action="store_true",
        help="Also compact the newest checkpoint in each chain. This saves more disk but makes the chain non-resumable.",
    )
    p.add_argument(
        "--drop-analysis-snapshots", action="store_true",
        help="Also delete start/pre_finalize/post_finalize .pt diagnostics. Scalar metrics and merge metadata in finalized pools remain available.",
    )
    args = p.parse_args()

    root = args.save_root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    groups = _checkpoint_records(root)
    if not groups:
        print(f"No run_manifest.json checkpoints found under {root}")

    total_before = total_after = compacted = skipped_latest = 0
    analysis_deleted = 0
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row[0])
        latest_seq = rows[-1][0]
        for seq_idx, run_dir, rel in rows:
            before = _pool_bytes(run_dir)
            total_before += before
            if seq_idx == latest_seq and not args.include_latest:
                total_after += before
                skipped_latest += 1
                print(f"KEEP latest resumable: {run_dir}")
                continue

            if args.dry_run:
                status = "already compacted" if storage_compaction.is_compacted(run_dir) else "would compact"
                print(f"{status}: {run_dir}  pool_files={before / (1024 ** 2):.1f} MiB")
                total_after += before
            else:
                report = storage_compaction.compact_checkpoint(run_dir)
                after = _pool_bytes(run_dir)
                total_after += after
                compacted += 1
                print(
                    f"COMPACTED: {run_dir}  "
                    f"saved={report.get('disk_bytes_saved', max(before-after, 0)) / (1024 ** 2):.1f} MiB"
                )

    if args.drop_analysis_snapshots and args.analysis_root is not None:
        analysis_root = args.analysis_root.resolve()
        if analysis_root.exists():
            snapshots = []
            for name in ("start.pt", "pre_finalize.pt", "post_finalize.pt"):
                snapshots.extend(analysis_root.rglob(name))
            snapshots = sorted(set(snapshots))
            for path in snapshots:
                if args.dry_run:
                    print(f"would delete analysis snapshot: {path}")
                else:
                    path.unlink()
                    analysis_deleted += 1

    if args.dry_run:
        print(
            f"Dry run complete: {len(groups)} chains, {skipped_latest} latest checkpoints kept fully resumable."
        )
    else:
        print(
            f"Done: compacted {compacted} checkpoints; kept {skipped_latest} latest checkpoints fully resumable; "
            f"pool-file disk saved={(total_before-total_after)/(1024**3):.3f} GiB; "
            f"analysis snapshots deleted={analysis_deleted}."
        )


if __name__ == "__main__":
    main()
