"""Verdict for the 2-task alpha/alpha_mass smoke test.

Run this after run_alpha_fix_smoke.py finishes. It reports, for each of the
2 trained tasks:

  1. The final mixture_weights vector saved in policy_snapshot.pt (as
     diagnose_alpha_mass.py did before) -- novel_weight = 1 - mass.
  2. The alpha_mass history recorded DURING training (from the CSV scalar
     log), so you can see whether it's actually moving instead of just
     comparing one before/after number.

Before the fix: novel_weight was pinned at ~0.0001 for every task,
regardless of how good the freshly trained expert actually was.
After the fix, a real difference between tasks (or at least visible
movement away from ~0 during training) is the sign it's working.

Usage:
    !python check_alpha_fix_smoke.py
"""
import csv
import pathlib

import torch

SUITE = "pointmaze_goal"
TAG = "alpha_fix_smoke"
METHOD = "Ours"
SEED = 1

AGENTS_CANDIDATES = [
    pathlib.Path("/kaggle/working/agents_pointmaze_smoke"),
    pathlib.Path("agents_pointmaze_smoke"),
]
RUNS_CANDIDATES = [
    pathlib.Path("/kaggle/working/runs_pointmaze_smoke"),
    pathlib.Path("runs_pointmaze_smoke"),
]


def _first_existing(candidates):
    return next((p for p in candidates if p.is_dir()), None)


def report_checkpoints(agents_root: pathlib.Path):
    base = agents_root / SUITE / TAG / METHOD / f"seed_{SEED}"
    if not base.is_dir():
        print(f"[!] no checkpoints found under {base}")
        return
    print(f"\n=== final mixture_weights per task ({base}) ===")
    for seq_dir in sorted(base.glob("seq_*")):
        for task_dir in sorted(seq_dir.glob("task_*")):
            snap_path = task_dir / "policy_snapshot.pt"
            if not snap_path.is_file():
                print(seq_dir.name, task_dir.name, "-- no policy_snapshot.pt")
                continue
            snap = torch.load(snap_path, map_location="cpu", weights_only=False)
            w = snap.get("mixture_weights")
            if w is None:
                print(seq_dir.name, task_dir.name, "-- no mixture_weights in snapshot")
                continue
            w = w.detach().cpu().tolist()
            n = len(w)
            novel_w = w[-1] if n > 1 else w[0]
            mass = 1.0 - novel_w if n > 1 else float("nan")
            print(
                f"{seq_dir.name}/{task_dir.name}: n_components={n} "
                f"weights={['%.4f' % x for x in w]} "
                f"novel(last)_weight={novel_w:.4f} implied_mass={mass:.4f} "
                f"uniform_would_be={1.0 / n:.4f}"
            )


def report_training_curve(runs_root: pathlib.Path):
    base = runs_root / TAG
    if not base.is_dir():
        print(f"[!] no run logs found under {base}")
        return
    print(f"\n=== alpha_mass during training ({base}) ===")
    for run_dir in sorted(base.glob(f"{SUITE}__task_*__{METHOD}__{SEED}")):
        csv_path = run_dir / "scalars.csv"
        if not csv_path.is_file():
            print(run_dir.name, "-- no scalars.csv")
            continue
        rows = []
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if row["tag"] == "analysis/policy/alpha_mass":
                    rows.append((int(row["step"]), float(row["value"])))
        if not rows:
            print(run_dir.name, "-- no analysis/policy/alpha_mass rows logged")
            continue
        rows.sort()
        print(f"{run_dir.name}:")
        for step, value in rows:
            print(f"    step {step:>7}: alpha_mass(effective) = {value:.4f}")


if __name__ == "__main__":
    agents_root = _first_existing(AGENTS_CANDIDATES)
    runs_root = _first_existing(RUNS_CANDIDATES)
    if agents_root is None and runs_root is None:
        raise FileNotFoundError(
            "found neither agents_pointmaze_smoke nor runs_pointmaze_smoke; "
            "did run_alpha_fix_smoke.py finish?"
        )
    if agents_root is not None:
        report_checkpoints(agents_root)
    if runs_root is not None:
        report_training_curve(runs_root)
