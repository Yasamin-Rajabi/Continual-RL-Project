"""Train ONE baseline on ONE task of the continual sequence.

The baseline twin of run_sac.py, and deliberately parallel to it: one
subprocess per position in the sequence, continuity carried only through
checkpoint directories passed as ``--prev-unit``. Task boundaries are process
boundaries here for the same reason they are in run_sac.py, so the two sides of
the comparison have identical process structure and identical replay-buffer
lifetimes.

    python3 run_baseline.py --method=ft_n --task-suite=halfcheetah_vel \
        --task-id=0 --seq-idx=0 --seed=1 --save-dir=agents_baselines/... \
        --total-timesteps=300000

Normally driven by baseline_benchmark.py rather than invoked by hand.

WHAT THIS FILE GUARANTEES
-------------------------
1. Budget parity. ``TaskBudget(total_timesteps, frozen_tail_steps)`` is the
   same object run_sac.py uses, so every condition gets Delta - B optimization
   steps and B frozen-tail interactions.
2. Environment parity. Environments are built by sac_core from
   ``tasks.get_task``; no environment name appears anywhere in the baselines.
3. Evaluation parity. The saved ``policy_snapshot.pt`` is in the format
   ``cka_rl.FrozenCkaPolicy`` reads, so the retention matrix is computed by the
   method's own evaluator, not by baseline code.
4. Oracle honesty. A task-blind method that is handed a task id, or a
   task-aware one that is not, is rejected before training starts, and the
   information set is recorded in the manifest.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import tyro

import baselines
import baseline_defaults as defaults
from baseline_identity import write_manifest
from baselines.common.lifecycle import TaskContext
from baselines.common.sac_core import train_task
from baselines.common.snapshot import save_snapshot, verify_snapshot
from csv_summary_writer import CsvSummaryWriter
from tasks import get_task, get_task_name
from training_protocol import TaskBudget


@dataclass
class Args:
    method: str = "ft_n"
    """Which baseline: ft_n | prognet | packnet | masknet."""
    model_type: str = "baseline"
    task_suite: str = defaults.DEFAULT_TASK_SUITE
    """Defaults come from baseline_defaults.py, the one suite-specific file in
    the baseline stack. They mirror run_sac.py's defaults for this folder, so a
    baseline run and a method run share a budget unless told otherwise."""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    task_id: int = 0
    seq_idx: int = 0
    """Unique occurrence index in the continual sequence. task_id repeats;
    seq_idx does not. Capacity is keyed by task_id, paths by seq_idx."""
    prior_task_ids: Tuple[int, ...] = ()
    """Task ids at sequence positions BEFORE this one, in order. Determines
    TaskContext.first_encounter, which is what makes a recurring task reuse its
    column/mask/gate instead of allocating a new one."""
    prev_unit: Optional[pathlib.Path] = None
    """Checkpoint directory of the immediately preceding position. Unlike
    run_sac.py the baselines need only the latest state, never the root: none
    of them keeps an immutable base the way the knowledge pool does."""

    save_dir: Optional[str] = None
    runs_root: str = "runs_baselines"
    tag: str = "Debug"

    # --- budget -------------------------------------------------------
    total_timesteps: int = defaults.DEFAULT_TOTAL_TIMESTEPS
    frozen_tail_steps: int = defaults.DEFAULT_FROZEN_TAIL_STEPS
    """B: the final frozen-policy interactions INSIDE total_timesteps, never
    added on top. Named to match run_sac.py's --distill-extra-steps, which is
    the same quantity under a name that only makes sense for the method."""

    # --- SAC ----------------------------------------------------------
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = defaults.DEFAULT_LEARNING_STARTS
    random_actions_end: int = defaults.DEFAULT_RANDOM_ACTIONS_END
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    autotune_init_from_alpha: bool = False

    # --- architecture -------------------------------------------------
    hidden_dim: int = 128
    """Head width. 128 matches CkaRlAgent's default, so a result difference is
    never a capacity difference."""
    encoder_linear_out: bool = False

    # --- evaluation ---------------------------------------------------
    eval_action_mode: Literal["deterministic", "stochastic"] = "deterministic"
    eval_every: int = defaults.DEFAULT_EVAL_EVERY
    num_evals: int = defaults.DEFAULT_NUM_EVALS

    # --- method-specific ----------------------------------------------
    prognet_adapter_dim: int = 64
    packnet_keep_fraction: float = 0.5
    packnet_retrain_fraction: float = 0.3
    masknet_gate_init: float = 2.0
    masknet_sparsity_reg: float = 0.0

    verify_snapshot_tolerance: float = 1e-5
    """Max allowed deviation between the saved snapshot and the live network.
    Set to 0 to skip the check; leaving it on is strongly recommended."""


def _validate_args(args: Args) -> None:
    if args.method not in baselines.available():
        raise ValueError(
            f"unknown --method {args.method!r}; available: {list(baselines.available())}"
        )
    # Raises on B >= Delta or B < 0.
    budget = TaskBudget(args.total_timesteps, args.frozen_tail_steps)
    if args.learning_starts < 0 or args.random_actions_end < 0:
        raise ValueError("learning_starts and random_actions_end must be nonnegative")
    if budget.training <= args.learning_starts + 1:
        raise ValueError(
            "Delta - B must exceed learning_starts + 1 so SAC can perform updates"
        )
    if args.policy_lr <= 0 or args.q_lr <= 0:
        raise ValueError("policy and q learning rates must be > 0")
    if args.total_timesteps < 1 or args.num_evals < 1:
        raise ValueError("total_timesteps and num_evals must be >= 1")
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if args.hidden_dim < 1:
        raise ValueError("hidden_dim must be >= 1")
    if args.seq_idx < 0 or args.task_id < 0:
        raise ValueError("task_id and seq_idx must be >= 0")
    if len(args.prior_task_ids) != args.seq_idx:
        raise ValueError(
            f"--prior-task-ids has {len(args.prior_task_ids)} entries but "
            f"--seq-idx={args.seq_idx}; it must list every earlier position so "
            "first_encounter can be determined."
        )
    if args.seq_idx > 0 and args.prev_unit is None:
        raise ValueError(
            "--prev-unit is required for seq_idx > 0; a continual chain cannot "
            "skip a position."
        )
    if args.seq_idx == 0 and args.prev_unit is not None:
        raise ValueError("--prev-unit must not be set for the root task")
    if not 0.0 < args.packnet_keep_fraction <= 1.0:
        raise ValueError("packnet_keep_fraction must be in (0, 1]")
    if not 0.0 <= args.packnet_retrain_fraction < 1.0:
        raise ValueError("packnet_retrain_fraction must be in [0, 1)")
    if args.prognet_adapter_dim < 1:
        raise ValueError("prognet_adapter_dim must be >= 1")
    if args.masknet_sparsity_reg < 0:
        raise ValueError("masknet_sparsity_reg must be >= 0")
    if args.verify_snapshot_tolerance < 0:
        raise ValueError("verify_snapshot_tolerance must be >= 0")


def main() -> None:
    args = tyro.cli(Args)
    _validate_args(args)

    agent_cls = baselines.get(args.method)
    task_aware = baselines.is_task_aware(args.method)
    if bool(agent_cls.task_aware) != task_aware:
        raise RuntimeError(
            f"{args.method} declares task_aware={agent_cls.task_aware} on the class "
            f"but the registry says {task_aware}. These must agree: the manifest "
            "records which information set the run actually had."
        )

    run_name = (
        f"{args.task_suite}__task_{args.task_id}__{args.method}__"
        f"{args.exp_name}__{args.seed}"
    )
    task_name = get_task_name(args.task_id, args.task_suite)
    print(f"\n*** Run name: {run_name} | {task_name} ***")
    print(
        f"*** Method: {args.method} "
        f"({'task-aware upper bound' if task_aware else 'task-blind'}) ***\n"
    )

    writer = CsvSummaryWriter(str(pathlib.Path(args.runs_root) / args.tag / run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"*** Device: {device}")

    # Read the observation and action dimensions off the real env rather than
    # hardcoding them, so this file never needs to know which suite it is in.
    probe_env = get_task(args.task_id, task_suite=args.task_suite)
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    act_dim = int(np.prod(probe_env.action_space.shape))
    probe_env.close()

    agent = agent_cls(
        obs_dim,
        act_dim,
        hidden_dim=args.hidden_dim,
        encoder_linear_out=args.encoder_linear_out,
        prognet_adapter_dim=args.prognet_adapter_dim,
        packnet_keep_fraction=args.packnet_keep_fraction,
        packnet_retrain_fraction=args.packnet_retrain_fraction,
        masknet_gate_init=args.masknet_gate_init,
        masknet_sparsity_reg=args.masknet_sparsity_reg,
    )

    if args.prev_unit is not None:
        print(f"*** Continuing the chain from {args.prev_unit}")
        agent.load_chain_state(args.prev_unit, map_location="cpu")

    ctx = TaskContext(
        task_id=args.task_id,
        seq_idx=args.seq_idx,
        suite=args.task_suite,
        seed=args.seed,
        first_encounter=args.task_id not in set(args.prior_task_ids),
    )
    print(
        f"*** Position {ctx.seq_idx}: task {ctx.task_id} "
        f"({'first encounter' if ctx.first_encounter else 'REPEAT, reusing capacity'}) ***"
    )

    final_eval = train_task(agent, ctx, args, writer, device)

    if args.save_dir is not None:
        run_dir = pathlib.Path(args.save_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving trained agent in `{args.save_dir}` with name `{run_name}`")

        save_snapshot(agent, run_dir)
        if args.verify_snapshot_tolerance > 0:
            deviation = verify_snapshot(
                agent, run_dir, device, atol=args.verify_snapshot_tolerance
            )
            writer.add_scalar("analysis/snapshot_max_deviation", deviation, args.total_timesteps)
            print(f"*** Snapshot verified against the live network: max |delta| = {deviation:.3e}")
        agent.save_chain_state(run_dir)

        budget = TaskBudget(args.total_timesteps, args.frozen_tail_steps)
        with (run_dir / "interaction_budget.json").open("w") as f:
            json.dump(
                {
                    "Delta": budget.total,
                    "optimization_phase_steps": budget.training,
                    "frozen_tail_steps": budget.frozen_tail,
                    "monitor_evaluation_steps": int(final_eval.get("monitor_evaluation_steps", 0)),
                    "evaluation_updates_policy": False,
                },
                f,
                indent=2,
            )

        config = dict(vars(args))
        config["task_aware"] = task_aware
        parents = () if args.prev_unit is None else (args.prev_unit,)
        manifest = write_manifest(run_dir, config, parent_dirs=parents)
        print(f"*** RUN_SIGNATURE: {manifest['run_signature']} ***")

    writer.close()


if __name__ == "__main__":
    main()
