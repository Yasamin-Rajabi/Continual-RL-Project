"""Export selected TensorBoard scalar events from the new benchmark to CSV."""
from __future__ import annotations

import argparse
import csv
import pathlib

from tensorboard.backend.event_processing import event_accumulator

DEFAULT_SCALARS = [
    "charts/episodic_return",
    "charts/test_episodic_return",
    "charts/test_velocity_error",
    "charts/test_success",
    "losses/actor_loss",
    "losses/qf_loss",
    "analysis/merge/symmetric_kl",
    "distillation/policy/distill_test_kl",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--output", default="tensorboard_scalars.csv")
    p.add_argument("--scalars", nargs="+", default=DEFAULT_SCALARS)
    return p.parse_args()


def main():
    args = parse_args()
    root = pathlib.Path(args.runs_root)
    event_files = list(root.rglob("events.out.tfevents.*"))
    rows = []
    seen_dirs = sorted({path.parent for path in event_files})
    for directory in seen_dirs:
        try:
            ea = event_accumulator.EventAccumulator(
                str(directory), size_guidance={event_accumulator.SCALARS: 0}
            )
            ea.Reload()
        except Exception as exc:
            print(f"Skipping {directory}: {exc}")
            continue
        available = set(ea.Tags().get("scalars", []))
        rel = directory.relative_to(root)
        for tag in args.scalars:
            if tag not in available:
                continue
            for event in ea.Scalars(tag):
                rows.append({
                    "run_dir": str(rel),
                    "scalar": tag,
                    "step": event.step,
                    "value": event.value,
                    "wall_time": event.wall_time,
                })
    if not rows:
        raise SystemExit(f"No requested scalar events found under {root}")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
