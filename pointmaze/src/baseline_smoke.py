"""Tiny end-to-end run of every method.  Needs torch; no GPU required.

This is the check to run on Kaggle *before* spending a session: it exercises
construction, one task boundary, the merge/projection path, saving, reloading
and scoring, for every method, at a budget small enough to finish in minutes.

    python baseline_smoke.py --method all
    python baseline_smoke.py --method Ours --stages 3
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile
import traceback

ROOT = pathlib.Path(__file__).resolve().parent

ALL = ["Ours", "Ours-parameter", "CKA-RL", "FT-N", "ProgNet", "PackNet",
       "MaskNet", "CReLUs", "CompoNet", "CbpNet"]

# Budgets small enough to be fast, large enough that every code path runs:
# learning starts, at least one policy update, the frozen tail, the merge.
TINY = dict(
    total_timesteps=1_400,
    distill_extra_steps=200,
    learning_starts=200,
    batch_size=32,
    buffer_size=2_000,
    eval_every=0,
    num_evals=1,
    shared_dim=32,
    head_hidden_dim=32,
    pool_size=2,
    alpha_warmup_steps=100,
    projection_epochs=2,
    projection_max_samples=100,
    max_distill_buffer=400,
    distill_max_samples=200,
    similarity_samples=64,
    distill_epochs=2,
)


def run(cmd, cwd=ROOT):
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
    return proc.returncode


def smoke_method(method: str, stages: int, workdir: pathlib.Path) -> bool:
    print(f"\n=== {method} ===")
    cmd = [
        sys.executable, "run_continual_benchmark.py",
        "--methods", method,
        "--seeds", "1",
        f"--total-timesteps={TINY['total_timesteps']}",
        f"--distill-extra-steps={TINY['distill_extra_steps']}",
        f"--learning-starts={TINY['learning_starts']}",
        f"--batch-size={TINY['batch_size']}",
        f"--buffer-size={TINY['buffer_size']}",
        f"--eval-every={TINY['eval_every']}",
        f"--num-evals={TINY['num_evals']}",
        f"--shared-dim={TINY['shared_dim']}",
        f"--head-hidden-dim={TINY['head_hidden_dim']}",
        f"--pool-size={TINY['pool_size']}",
        f"--alpha-warmup-steps={TINY['alpha_warmup_steps']}",
        f"--projection-epochs={TINY['projection_epochs']}",
        f"--projection-max-samples={TINY['projection_max_samples']}",
        f"--max-distill-buffer={TINY['max_distill_buffer']}",
        f"--distill-max-samples={TINY['distill_max_samples']}",
        f"--similarity-samples={TINY['similarity_samples']}",
        f"--distill-epochs={TINY['distill_epochs']}",
        "--test-adapt-steps=100",
        f"--save-root={workdir / 'agents'}",
        f"--runs-root={workdir / 'runs'}",
        f"--results-root={workdir / 'results'}",
        f"--analysis-root={workdir / 'analysis'}",
        "--tag=smoke",
        "--no-cuda",
    ]
    code = run(cmd)
    if code != 0:
        print(f"[FAIL] {method}")
        return False
    print(f"[ok] {method}")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="all")
    p.add_argument("--stages", type=int, default=None, help="unused; kept for symmetry")
    p.add_argument("--keep", action="store_true", help="keep the temporary work dir")
    args = p.parse_args()

    methods = ALL if args.method == "all" else [args.method]
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="pointmaze_smoke_"))
    print(f"work dir: {workdir}")

    passed, failed = [], []
    try:
        for method in methods:
            try:
                (passed if smoke_method(method, args.stages, workdir) else failed).append(method)
            except Exception:
                traceback.print_exc()
                failed.append(method)
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)

    print(f"\npassed: {passed}")
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print("all methods smoke-tested OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
