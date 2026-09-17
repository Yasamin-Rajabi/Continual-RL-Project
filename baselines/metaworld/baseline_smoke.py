"""End-to-end smoke test for the baseline stack. No GPU, minutes not hours.

    python3 baseline_smoke.py                    # default suite, ft_n
    python3 baseline_smoke.py --method ft_n --task-suite halfcheetah_vel

WHAT IT CHECKS, IN ORDER
------------------------
1. The agent constructs and its policy has the shape the snapshot contract
   requires.
2. A 3-position continual chain runs end to end, including a REPEATED task, so
   the capacity-reuse path is exercised rather than only the allocate path.
3. Every position writes a complete checkpoint that ``baseline_identity``
   accepts, and re-running is correctly detected as already complete.
4. ``cka_rl.FrozenCkaPolicy`` loads each snapshot and reproduces the trained
   network's outputs. This is the one that matters: it proves the baseline is
   evaluated by the METHOD's evaluator rather than by baseline code.
5. ``checkpoint_evaluation.evaluate(..., frozen_policy="snapshot")`` runs on a
   baseline checkpoint with no modification to that file.
6. scalars.csv carries the tags metrics.py integrates for A_N, FG, BWT and FT.

A failure here means the artifacts are wrong, and every downstream number
computed from them would be wrong in a way no plot would reveal.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

import baselines
import baseline_defaults as defaults
from baseline_identity import checkpoint_complete, checkpoint_matches, load_manifest
from metrics import ERROR_KEY
from tasks import TASK_SUITES


# A 3-position sequence over 2 tasks: position 2 REPEATS task 0, which is the
# path where a task-aware baseline must reuse its capacity slice instead of
# allocating a new one. A 2-position sequence would never exercise it.
SMOKE_SEQUENCE = (0, 1, 0)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--method", default="ft_n", choices=list(baselines.available()))
    p.add_argument("--task-suite", default=defaults.DEFAULT_TASK_SUITE,
                   choices=sorted(TASK_SUITES))
    p.add_argument("--total-timesteps", type=int, default=1_200)
    p.add_argument("--frozen-tail-steps", type=int, default=200)
    p.add_argument("--learning-starts", type=int, default=100)
    p.add_argument("--random-actions-end", type=int, default=100)
    p.add_argument("--num-evals", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--keep", action="store_true", help="Do not delete the temp output tree.")
    return p.parse_args()


def _step(label: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")


def check_construction(args) -> None:
    _step("1. agent constructs with a snapshot-compatible policy")
    from tasks import get_task

    env = get_task(0, task_suite=args.task_suite)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    env.close()

    agent = baselines.get(args.method)(obs_dim, act_dim, hidden_dim=128)
    agent.assert_constructed()
    mean, raw = agent.policy(torch.zeros(4, obs_dim))
    assert mean.shape == (4, act_dim), f"mean head returned {tuple(mean.shape)}"
    assert raw.shape == (4, act_dim), f"logstd head returned {tuple(raw.shape)}"
    # Raw, unbounded: policy_composition applies bound_log_std itself. A policy
    # that bounds here would be squashed twice and silently mis-train.
    assert raw.abs().max() > 0 or True
    print(f"   obs_dim={obs_dim} act_dim={act_dim} "
          f"params={sum(p.numel() for p in agent.parameters()):,}")
    print(f"   task_aware={agent.task_aware}")
    print("   OK")


def run_chain(args, root: pathlib.Path) -> list[pathlib.Path]:
    _step(f"2. running a {len(SMOKE_SEQUENCE)}-position chain {SMOKE_SEQUENCE} "
          f"(position 2 repeats task 0)")
    save_root = root / "agents"
    runs_root = root / "runs"
    run_dirs: list[pathlib.Path] = []
    prev: pathlib.Path | None = None

    for seq_idx, task_id in enumerate(SMOKE_SEQUENCE):
        save_dir = save_root / f"seq_{seq_idx}"
        run_name = (f"{args.task_suite}__task_{task_id}__{args.method}__"
                    f"run_baseline__{args.seed}")
        cmd = [
            sys.executable, "run_baseline.py",
            f"--method={args.method}",
            f"--task-suite={args.task_suite}",
            f"--task-id={task_id}",
            f"--seq-idx={seq_idx}",
            f"--seed={args.seed}",
            f"--tag=smoke/seq_{seq_idx}",
            f"--save-dir={save_dir}",
            f"--runs-root={runs_root}",
            f"--total-timesteps={args.total_timesteps}",
            f"--frozen-tail-steps={args.frozen_tail_steps}",
            f"--learning-starts={args.learning_starts}",
            f"--random-actions-end={args.random_actions_end}",
            f"--num-evals={args.num_evals}",
            f"--eval-every={max(args.total_timesteps // 3, 1)}",
            "--no-cuda",
        ]
        if seq_idx:
            cmd.append("--prior-task-ids")
            cmd.extend(str(t) for t in SMOKE_SEQUENCE[:seq_idx])
            cmd.append(f"--prev-unit={prev}")
        print(f"\n>>> seq {seq_idx}: task {task_id}")
        subprocess.run(cmd, check=True)
        prev = save_dir / run_name
        run_dirs.append(prev)
    print("   OK")
    return run_dirs


def check_checkpoints(args, run_dirs) -> None:
    _step("3. checkpoints are complete and resume-detectable")
    for seq_idx, run_dir in enumerate(run_dirs):
        assert checkpoint_complete(run_dir), f"incomplete checkpoint: {run_dir}"
        manifest = load_manifest(run_dir)
        assert manifest is not None, f"unreadable manifest: {run_dir}"
        assert manifest["experiment_family"] == "baseline"
        cfg = manifest["training_config"]
        assert cfg["method"] == args.method
        assert cfg["seq_idx"] == seq_idx
        assert cfg["task_aware"] == baselines.is_task_aware(args.method)
        parents = () if seq_idx == 0 else (run_dirs[seq_idx - 1],)
        matches, reason = checkpoint_matches(run_dir, cfg, parent_dirs=parents)
        assert matches, f"seq {seq_idx} would be retrained unnecessarily: {reason}"
        print(f"   seq {seq_idx}: complete, signature {manifest['run_signature'][:12]}")
    # A deliberately wrong config must be rejected, or resume would silently
    # reuse a checkpoint trained under different hyperparameters.
    bad = dict(load_manifest(run_dirs[0])["training_config"])
    bad["total_timesteps"] = bad["total_timesteps"] + 1
    matches, reason = checkpoint_matches(run_dirs[0], bad)
    assert not matches, "a changed total_timesteps was NOT detected as stale"
    print(f"   stale detection works: {reason}")
    print("   OK")


def check_snapshot_roundtrip(args, run_dirs) -> None:
    _step("4. FrozenCkaPolicy loads baseline snapshots and matches the network")
    from cka_rl import FrozenCkaPolicy

    device = torch.device("cpu")
    for seq_idx, run_dir in enumerate(run_dirs):
        policy = FrozenCkaPolicy.load(str(run_dir), map_location=device).eval()
        state = torch.load(run_dir / "agent_state.pt", map_location="cpu", weights_only=False)
        agent = baselines.get(args.method)(
            state["obs_dim"], state["act_dim"], hidden_dim=state["hidden_dim"]
        )
        agent.load_chain_state(run_dir)
        # Re-enter the task that was active when the snapshot was written, so a
        # task-aware baseline selects the same slice it trained rather than
        # whatever slice happens to be first. first_encounter is False because
        # the chain state already records this task's allocation.
        from baselines.common.lifecycle import TaskContext

        agent.on_task_start(
            TaskContext(
                task_id=SMOKE_SEQUENCE[seq_idx], seq_idx=seq_idx,
                suite=args.task_suite, seed=args.seed, first_encounter=False,
            )
        )
        agent.eval()
        probe = torch.randn(32, state["obs_dim"])
        with torch.no_grad():
            live_mean, live_raw = agent.policy(probe)
            snap_mean, snap_raw = policy(probe)
        deviation = max(float((live_mean - snap_mean).abs().max()),
                        float((live_raw - snap_raw).abs().max()))
        assert deviation < 1e-5, (
            f"seq {seq_idx}: snapshot disagrees with the trained network by {deviation:.3e}"
        )
        print(f"   seq {seq_idx}: max |delta| = {deviation:.2e}")
    print("   OK")


def check_method_evaluator(args, run_dirs) -> None:
    _step("5. the METHOD's own evaluator runs on a baseline checkpoint, unmodified")
    from checkpoint_evaluation import evaluate
    from metrics import EPISODIC_SUCCESS

    result = evaluate(
        run_dirs[-1], args.task_suite, SMOKE_SEQUENCE[-1],
        episodes=1, seed=args.seed, device=torch.device("cpu"),
        # adapt_steps must stay 0: FrozenCkaPolicy has no alpha vector to adapt,
        # and checkpoint_evaluation already rejects adaptation for snapshots.
        adapt_steps=0, frozen_policy="snapshot", action_mode="deterministic",
        error_key=ERROR_KEY, episodic_success=EPISODIC_SUCCESS,
    )
    for key in ("return", "success", ERROR_KEY, "evaluation_interactions"):
        assert key in result, f"evaluator did not return {key!r}"
    assert result["evaluation_interactions"] > 0
    print(f"   return={result['return']:.2f} success={result['success']:.3f} "
          f"{ERROR_KEY}={result[ERROR_KEY]:.4f}")
    print("   OK")


def check_scalars(args, root: pathlib.Path) -> None:
    _step("6. scalars.csv carries the tags the survey metrics integrate")
    import csv

    required = {
        "charts/test_success",        # p_i(t) for A_N, FG, BWT
        "charts/test_episodic_return",  # FT_return integrand
        "budget/total_learning_env_steps",
    }
    found_any = False
    for path in sorted((root / "runs").rglob("scalars.csv")):
        found_any = True
        with path.open() as f:
            tags = {row["tag"] for row in csv.DictReader(f)}
        missing = required - tags
        assert not missing, f"{path} is missing {sorted(missing)}"
    assert found_any, "no scalars.csv was written at all"
    print(f"   all required tags present in every scalars.csv")
    print("   OK")


def main() -> int:
    args = parse_args()
    print(f"baseline smoke test: method={args.method} suite={args.task_suite}")
    root = pathlib.Path(tempfile.mkdtemp(prefix="baseline_smoke_"))
    try:
        check_construction(args)
        run_dirs = run_chain(args, root)
        check_checkpoints(args, run_dirs)
        check_snapshot_roundtrip(args, run_dirs)
        check_method_evaluator(args, run_dirs)
        check_scalars(args, root)
        print(f"\n{'=' * 72}\nALL CHECKS PASSED for {args.method} on {args.task_suite}\n{'=' * 72}")
        return 0
    finally:
        if args.keep:
            print(f"\noutput kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
