"""Diagnostic: inspect the exported mixture_weights of a few Ours checkpoints.

Run this from inside the Kaggle notebook's working directory (where the
pointmaze package and agents_pointmaze/ checkpoints live), e.g. add a cell:

    !python diagnose_alpha_mass.py

or in a code cell:

    import subprocess, sys
    print(subprocess.run([sys.executable, "diagnose_alpha_mass.py"],
                          capture_output=True, text=True).stdout)

It prints, for a few task checkpoints of the "Ours" chain, the exported
mixture_weights vector and which index (if any) is the novel/just-trained
component -- the last entry, when use_alpha_mass is on. If that last weight
is small (e.g. << 1/num_pool_members) right after training that very task,
that confirms the hypothesis: alpha_mass never converged toward trusting the
freshly trained expert, so the zero-adaptation "peak" evaluation is scoring a
policy still dominated by irrelevant, previously-trained experts.
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path("/kaggle/working/agents_pointmaze/pointmaze_goal/main/Ours/seed_1")
if not ROOT.is_dir():
    ROOT = pathlib.Path("agents_pointmaze/pointmaze_goal/main/Ours/seed_1")

for seq_dir in sorted(ROOT.glob("seq_*")):
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
        print(
            f"{seq_dir.name}/{task_dir.name}: n_components={n} "
            f"weights={['%.4f' % x for x in w]} "
            f"novel(last)_weight={novel_w:.4f} "
            f"uniform_would_be={1.0 / n:.4f}"
        )
