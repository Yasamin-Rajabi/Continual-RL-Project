"""Diagnose why specific tasks (e.g. seq 2, 7, 8) score badly on the full
10-task "Ours" chain -- and separate two different possible causes:

  (a) routing/mixture problem: at PEAK-eval time (right after training that
      task, zero test-time adaptation), the mixture is still leaning on a
      bad historical component instead of the freshly trained novel expert.
  (b) training/difficulty problem: the freshly trained novel expert itself
      just didn't solve that particular goal well within the fixed
      260k-step budget, regardless of any routing.

It cross-checks metrics.py's peak_per_task (computed by evaluate_chain,
using the FINAL merged/projected policy_snapshot for that task) against
run_sac_continual.py's own internal end-of-training eval
(charts/final_return in that task's scalars.csv, computed by the training
script itself right when that task's training loop ends, before any later
merge/distillation touches it). If the two agree, the peak is genuinely a
training/difficulty ceiling (b). If metrics.py's peak is much worse than the
internal final_return, something in projection/evaluation is losing
performance that training itself actually reached (a).

Also reports the exported mixture_weights for that task's own checkpoint
(novel component's weight) and, for context, charts/zero_shot_return (the
mixture's performance on the task BEFORE any training on it -- pure
transfer from the historical pool).

Usage (from inside the pointmaze code directory, after a full or partial
Ours run under tag "main"):
    !python diagnose_hard_tasks.py
"""
import csv
import json
import pathlib

import torch

SUITE = "pointmaze_goal"
TAG = "main"
METHOD = "Ours"
SEED = 1

SAVE_CANDIDATES = [pathlib.Path("/kaggle/working/agents_pointmaze"), pathlib.Path("agents_pointmaze")]
RUNS_CANDIDATES = [pathlib.Path("/kaggle/working/runs_pointmaze"), pathlib.Path("runs_pointmaze")]
RESULTS_CANDIDATES = [pathlib.Path("/kaggle/working/results_pointmaze"), pathlib.Path("results_pointmaze")]


def _first_existing(candidates):
    return next((p for p in candidates if p.is_dir()), None)


def _last_scalar(csv_path, tag):
    if not csv_path.is_file():
        return None
    best_step, best_val = None, None
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["tag"] == tag:
                step = int(row["step"])
                if best_step is None or step >= best_step:
                    best_step, best_val = step, float(row["value"])
    return best_val


def _first_scalar(csv_path, tag):
    if not csv_path.is_file():
        return None
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["tag"] == tag:
                return float(row["value"])
    return None


def main():
    save_root = _first_existing(SAVE_CANDIDATES)
    runs_root = _first_existing(RUNS_CANDIDATES)
    results_root = _first_existing(RESULTS_CANDIDATES)
    if save_root is None or runs_root is None:
        raise FileNotFoundError("could not find agents_pointmaze / runs_pointmaze")

    metrics_path = None
    if results_root is not None:
        for p in sorted(results_root.glob(f"{SUITE}__{METHOD}__seed{SEED}__metrics.json")):
            metrics_path = p
    peak_per_task = final_per_task = None
    if metrics_path is not None:
        with metrics_path.open() as f:
            m = json.load(f)
        peak_per_task = m.get("peak_per_task")
        final_per_task = m.get("final_per_task")

    chain_base = save_root / SUITE / TAG / METHOD / f"seed_{SEED}"
    print(f"{'task':<6}{'metrics.peak':>14}{'internal.final':>16}{'zero_shot':>12}"
          f"{'novel_weight':>14}{'n_components':>14}")
    print("-" * 76)
    for seq_dir in sorted(chain_base.glob("seq_*"), key=lambda p: int(p.name.split("_")[1])):
        for task_dir in sorted(seq_dir.glob("task_*")):
            task_id = int(task_dir.name.split("_")[1])
            seq_idx = int(seq_dir.name.split("_")[1])

            snap_path = task_dir / "policy_snapshot.pt"
            novel_w = n_comp = None
            if snap_path.is_file():
                snap = torch.load(snap_path, map_location="cpu", weights_only=False)
                w = snap.get("mixture_weights")
                if w is not None:
                    w = w.detach().cpu().tolist()
                    n_comp = len(w)
                    novel_w = w[-1] if n_comp > 1 else w[0]

            run_name = f"{SUITE}__task_{task_id}__{METHOD}__{SEED}"
            csv_path = runs_root / TAG / run_name / "scalars.csv"
            internal_final = _last_scalar(csv_path, "charts/final_return")
            zero_shot = _first_scalar(csv_path, "charts/zero_shot_return")

            m_peak = peak_per_task[seq_idx] if peak_per_task and seq_idx < len(peak_per_task) else None

            def fmt(v):
                return "n/a" if v is None else f"{v:.2f}"

            print(f"{seq_idx:<6}{fmt(m_peak):>14}{fmt(internal_final):>16}{fmt(zero_shot):>12}"
                  f"{fmt(novel_w):>14}{('n/a' if n_comp is None else n_comp):>14}")

    print(
        "\nRead: if 'metrics.peak' ~= 'internal.final', the low score is a "
        "training/difficulty ceiling for that task (more steps or reward "
        "shaping would help, not routing). If 'metrics.peak' is much worse "
        "than 'internal.final', something in the merge/projection or the "
        "peak-eval mixture is losing performance training itself reached -- "
        "check 'novel_weight' for that task (should be close to 1 for a "
        "healthy peak eval, as seen in the smoke test)."
    )


if __name__ == "__main__":
    main()
