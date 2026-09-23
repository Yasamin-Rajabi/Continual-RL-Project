"""End-to-end smoke test for the baseline stack. No GPU, minutes not hours.

    python3 baseline_smoke.py                      # default suite, ft_n
    python3 baseline_smoke.py --method all         # every baseline in turn
    python3 baseline_smoke.py --method packnet --task-suite mw_easy4

WHAT IT CHECKS, IN ORDER
------------------------
1. The agent constructs and its policy has the shape the snapshot contract
   requires.
2. A 3-position continual chain runs end to end, including a REPEATED task, so
   the capacity-reuse path is exercised rather than only the allocate path.
3. Every position writes a complete checkpoint that ``baseline_identity``
   accepts, and re-running is correctly detected as already complete.
4. ``cka_rl.FrozenCkaPolicy`` loads each snapshot and reproduces the trained
   network's outputs. This is the one that matters most: it proves the baseline
   is evaluated by the METHOD's evaluator rather than by baseline code.
5. PARAMETER ISOLATION. After task 1 has trained, task 0's policy is
   re-exported from the newer chain state and compared against what task 0
   actually saved. See check_isolation for why this is the test that earns its
   runtime.
6. ``checkpoint_evaluation.evaluate(..., frozen_policy="snapshot")`` runs on a
   baseline checkpoint with no modification to that file.
7. scalars.csv carries the tags metrics.py integrates for A_N, FG, BWT and FT.

A failure here means the artifacts are wrong, and every downstream number
computed from them would be wrong in a way no plot would reveal.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

import baseline_defaults as defaults
import baselines
from baseline_identity import checkpoint_complete, checkpoint_matches, load_manifest
from baselines.common.lifecycle import TaskContext
from baselines.common.snapshot import HEAD_KEYS
from metrics import ERROR_KEY
from tasks import TASK_SUITES

# A 3-position sequence over 2 tasks. Position 2 REPEATS task 0, which is the
# path where a task-aware baseline must reuse its capacity slice instead of
# allocating a new one; a 2-position sequence would never exercise it.
SMOKE_SEQUENCE = (0, 1, 0)

# Methods whose architecture promises that training a later task cannot change
# an earlier one's policy at all. MaskNet is deliberately not here: its
# protection is proportional to how strongly a unit was claimed, so partial
# drift is the designed behaviour rather than a defect. FT-N is not here either,
# and is checked for the opposite.
EXACT_ISOLATION_METHODS = ("prognet", "packnet")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--method", default="ft_n",
                   choices=list(baselines.available()) + ["all"])
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


def _build_agent(method: str, run_dir: pathlib.Path):
    """Reconstruct an agent from a saved chain state."""
    state = torch.load(run_dir / "agent_state.pt", map_location="cpu", weights_only=False)
    agent = baselines.get(method)(
        state["obs_dim"], state["act_dim"], hidden_dim=state["hidden_dim"],
        encoder_linear_out=state["encoder_linear_out"],
    )
    agent.load_chain_state(run_dir)
    return agent, state


def check_construction(args, method: str) -> None:
    _step(f"1. [{method}] agent constructs with a snapshot-compatible policy")
    from tasks import get_task

    env = get_task(0, task_suite=args.task_suite)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    env.close()

    agent = baselines.get(method)(obs_dim, act_dim, hidden_dim=128)
    agent.assert_constructed()
    # Task-aware policies refuse to run before a task is selected, which is
    # itself worth confirming: a silent default would mean the oracle was not
    # actually reaching the forward pass.
    agent.on_task_start(
        TaskContext(task_id=0, seq_idx=0, suite=args.task_suite,
                    seed=args.seed, first_encounter=True)
    )
    mean, raw = agent.policy(torch.zeros(4, obs_dim))
    assert mean.shape == (4, act_dim), f"mean head returned {tuple(mean.shape)}"
    assert raw.shape == (4, act_dim), f"logstd head returned {tuple(raw.shape)}"
    print(f"   obs_dim={obs_dim} act_dim={act_dim} "
          f"params={sum(p.numel() for p in agent.parameters()):,}")
    print(f"   task_aware={agent.task_aware}  "
          f"effective_hidden_dim={agent.effective_hidden_dim()}")
    print("   OK")


def run_chain(args, method: str, root: pathlib.Path) -> list[pathlib.Path]:
    _step(f"2. [{method}] running a {len(SMOKE_SEQUENCE)}-position chain "
          f"{SMOKE_SEQUENCE} (position 2 repeats task 0)")
    save_root = root / "agents"
    runs_root = root / "runs"
    run_dirs: list[pathlib.Path] = []
    prev: pathlib.Path | None = None

    for seq_idx, task_id in enumerate(SMOKE_SEQUENCE):
        save_dir = save_root / f"seq_{seq_idx}"
        run_name = (f"{args.task_suite}__task_{task_id}__{method}__"
                    f"run_baseline__{args.seed}")
        cmd = [
            sys.executable, "run_baseline.py",
            f"--method={method}",
            f"--task-suite={args.task_suite}",
            f"--task-id={task_id}",
            f"--seq-idx={seq_idx}",
            f"--seed={args.seed}",
            f"--tag=smoke/{method}/seq_{seq_idx}",
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


def check_checkpoints(args, method: str, run_dirs) -> None:
    _step(f"3. [{method}] checkpoints are complete and resume-detectable")
    for seq_idx, run_dir in enumerate(run_dirs):
        assert checkpoint_complete(run_dir), f"incomplete checkpoint: {run_dir}"
        manifest = load_manifest(run_dir)
        assert manifest is not None, f"unreadable manifest: {run_dir}"
        assert manifest["experiment_family"] == "baseline"
        cfg = manifest["training_config"]
        assert cfg["method"] == method
        assert cfg["seq_idx"] == seq_idx
        assert cfg["task_aware"] == baselines.is_task_aware(method), (
            "the manifest disagrees with the registry about whether this run "
            "had the task oracle"
        )
        parents = () if seq_idx == 0 else (run_dirs[seq_idx - 1],)
        matches, reason = checkpoint_matches(run_dir, cfg, parent_dirs=parents)
        assert matches, f"seq {seq_idx} would be retrained unnecessarily: {reason}"
        print(f"   seq {seq_idx}: complete, signature {manifest['run_signature'][:12]}")

    bad = dict(load_manifest(run_dirs[0])["training_config"])
    bad["total_timesteps"] = bad["total_timesteps"] + 1
    matches, reason = checkpoint_matches(run_dirs[0], bad)
    assert not matches, "a changed total_timesteps was NOT detected as stale"
    print(f"   stale detection works: {reason}")
    print("   OK")


def check_snapshot_roundtrip(args, method: str, run_dirs) -> None:
    _step(f"4. [{method}] FrozenCkaPolicy loads baseline snapshots and matches "
          f"the network")
    from cka_rl import FrozenCkaPolicy

    device = torch.device("cpu")
    for seq_idx, run_dir in enumerate(run_dirs):
        policy = FrozenCkaPolicy.load(str(run_dir), map_location=device).eval()
        agent, state = _build_agent(method, run_dir)
        # Re-enter the task that was active when the snapshot was written, so a
        # task-aware baseline selects the slice it trained rather than whichever
        # happens to be first. first_encounter is False because the chain state
        # already records this task's allocation.
        agent.on_task_start(
            TaskContext(task_id=SMOKE_SEQUENCE[seq_idx], seq_idx=seq_idx,
                        suite=args.task_suite, seed=args.seed,
                        first_encounter=False)
        )
        agent.prepare_for_evaluation()
        agent.eval()
        probe = torch.randn(32, state["obs_dim"])
        with torch.no_grad():
            live_mean, live_raw = agent.policy(probe)
            snap_mean, snap_raw = policy(probe)
        deviation = max(float((live_mean - snap_mean).abs().max()),
                        float((live_raw - snap_raw).abs().max()))
        assert deviation < 1e-5, (
            f"seq {seq_idx}: snapshot disagrees with the trained network by "
            f"{deviation:.3e}"
        )
        print(f"   seq {seq_idx}: max |delta| = {deviation:.2e}")
    print("   OK")


def check_isolation(args, method: str, run_dirs) -> None:
    """Does training task 1 leave task 0's policy alone?

    This is the check worth its runtime. Every failure mode these three
    architectures have is a silent one: gradient masking that Adam's momentum
    walks straight through, a gate tensor whose other rows were never protected,
    a lateral that reaches back into a frozen column. None of them raise, none
    of them dent the training curves, and all of them show up only as a
    retention matrix that looks like a property of the method rather than a bug.

    So instead of trusting the mechanism, measure the invariant directly.
    Position 0 trained task 0 and saved its policy. Position 1 then trained task
    1. Reload the chain state as it stood AFTER position 1, select task 0 again,
    export its head, and compare against what task 0 actually saved.

    ProgNet and PackNet promise this is exactly zero. MaskNet does not: its
    protection scales with how strongly each unit was claimed, so partial drift
    is designed behaviour and gets reported rather than asserted. FT-N promises
    the opposite, and is checked for it -- if FT-N came back unchanged, the test
    itself would be measuring nothing.
    """
    _step(f"5. [{method}] parameter isolation: task 1 trains, does task 0 move?")

    saved = torch.load(run_dirs[0] / "policy_snapshot.pt",
                       map_location="cpu", weights_only=False)
    agent, _ = _build_agent(method, run_dirs[1])
    agent.on_task_start(
        TaskContext(task_id=SMOKE_SEQUENCE[0], seq_idx=1, suite=args.task_suite,
                    seed=args.seed, first_encounter=False)
    )
    agent.prepare_for_evaluation()
    agent.eval()
    with torch.no_grad():
        after = agent.export_policy_snapshot()

    deltas = {}
    for head in ("mean", "logstd"):
        for key in HEAD_KEYS:
            before_t = saved[head][key]
            after_t = after[head][key]
            assert before_t.shape == after_t.shape, (
                f"{head}.{key} changed shape from {tuple(before_t.shape)} to "
                f"{tuple(after_t.shape)}; task 0's head was rebuilt, not reused"
            )
            deltas[f"{head}.{key}"] = float((before_t - after_t).abs().max())
    for key, tensor in saved["fc_state_dict"].items():
        deltas[f"encoder.{key}"] = float(
            (tensor - after["fc_state_dict"][key]).abs().max()
        )

    worst_key = max(deltas, key=deltas.get)
    worst = deltas[worst_key]
    print(f"   largest drift in task 0's policy: {worst:.3e}  ({worst_key})")

    if method in EXACT_ISOLATION_METHODS:
        assert worst == 0.0, (
            f"{method} promises exact parameter isolation but task 0's policy "
            f"moved by {worst:.3e} at {worst_key} while task 1 was training. "
            "The usual cause is Adam momentum carrying protected parameters "
            "past a masked gradient; masking alone is not enough, the "
            "after_optimizer_step restore is what enforces this."
        )
        print("   OK  exactly zero, as this architecture guarantees")
    elif method == "ft_n":
        assert worst > 0.0, (
            "FT-N left task 0's policy untouched, which it should not be able "
            "to do. Either nothing trained at position 1, or this check is "
            "comparing something that cannot move and is therefore vacuous "
            "for the other baselines too."
        )
        print("   OK  nonzero, as sequential fine-tuning implies "
              "(this is the forgetting FT-N exists to measure)")
    else:
        print(f"   reported, not asserted: {method} protects the backbone in "
              f"proportion to how strongly each unit was claimed, so some "
              f"drift is by design")


def check_method_evaluator(args, method: str, run_dirs) -> None:
    _step(f"6. [{method}] the METHOD's own evaluator runs on a baseline "
          f"checkpoint, unmodified")
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


def check_scalars(args, method: str, root: pathlib.Path) -> None:
    _step(f"7. [{method}] scalars.csv carries the tags the survey metrics "
          f"integrate")
    required = {
        "charts/test_success",           # p_i(t) for A_N, FG, BWT
        "charts/test_episodic_return",   # FT_return integrand
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
    print("   all required tags present in every scalars.csv")
    print("   OK")


def run_one(args, method: str) -> None:
    print(f"\n\n{'#' * 72}\n# baseline smoke test: method={method} "
          f"suite={args.task_suite}\n{'#' * 72}")
    root = pathlib.Path(tempfile.mkdtemp(prefix=f"baseline_smoke_{method}_"))
    try:
        check_construction(args, method)
        run_dirs = run_chain(args, method, root)
        check_checkpoints(args, method, run_dirs)
        check_snapshot_roundtrip(args, method, run_dirs)
        check_isolation(args, method, run_dirs)
        check_method_evaluator(args, method, run_dirs)
        check_scalars(args, method, root)
        print(f"\n{'=' * 72}\nALL CHECKS PASSED for {method} on "
              f"{args.task_suite}\n{'=' * 72}")
    finally:
        if args.keep:
            print(f"\noutput kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    args = parse_args()
    methods = list(baselines.available()) if args.method == "all" else [args.method]
    for method in methods:
        run_one(args, method)
    if len(methods) > 1:
        print(f"\n{'=' * 72}\nALL {len(methods)} BASELINES PASSED on "
              f"{args.task_suite}: {', '.join(methods)}\n{'=' * 72}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
