"""The protocol every continual-RL baseline implements.

DESIGN RULE
-----------
The SAC training loop in sac_core.py is IDENTICAL for all four baselines. A
baseline customizes behaviour only by overriding the hooks below. Nothing in
the environment construction, the replay buffer, the budget accounting or the
evaluation protocol is reachable from here, which is what makes environmental
parity with the CKA-RL method structural instead of a promise.

CALL ORDER FOR ONE TASK
-----------------------
    agent = Method(obs_dim, act_dim, cfg)          # fresh process per task
    agent.load_chain_state(prev_dir)               # if seq_idx > 0
    agent.on_task_start(ctx)                       # allocate/select capacity
    for step in range(budget.training):            # Delta - B optimization steps
        ...
        agent.on_phase_boundary(step, budget)      # every step; PackNet prunes here
        ...                                        # actor/critic losses
        agent.before_optimizer_step()              # mask gradients
        optimizer.step()
        agent.after_optimizer_step()               # re-impose frozen weights
    #  frozen tail: B steps, no optimizer call at all
    agent.on_task_end(ctx)                         # commit masks/columns
    agent.export_policy_snapshot()                 # FrozenCkaPolicy-compatible
    agent.save_chain_state(run_dir)

TASK BOUNDARIES ARE PROCESS BOUNDARIES
--------------------------------------
Exactly as in run_sac.py, each position in the continual sequence is a separate
subprocess. Continuity flows only through ``save_chain_state`` /
``load_chain_state``. There is no in-process continual loop, so an agent must
serialize everything it needs: capacity allocations, per-task masks, the
task_id -> slice mapping, and the optimizer-independent parts of its own state.

THE TASK ORACLE
---------------
``TaskContext.task_id`` is the oracle. Per the approved design, ProgNet,
PackNet and MaskNet consume it at both train and eval time and are reported as
Task-Aware Upper Bounds; FT-N ignores it entirely. A task-blind agent must
declare ``task_aware = False`` so run_baseline.py can assert it never reads the
field, and so the manifest records which information set the run actually had.

REPEATED TASKS
--------------
Both continual sequences revisit every task (HalfCheetah: 12 positions over 6
tasks; Meta-World mw_easy4: 8 positions over 4 tasks). Capacity is keyed by
``task_id``, never by ``seq_idx``: the second encounter of task t REUSES the
column, mask or gate allocated on the first encounter. That is what makes the
second pass a measurement of retention and backward transfer rather than a
measurement of how much fresh capacity the method was handed.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TaskContext:
    """One position in the continual sequence.

    task_id   Index into the suite's task list. REPEATS across the sequence,
              and is the key under which capacity is allocated and reused.
    seq_idx   Unique position index. Never repeats. Used for logging lineage
              and output paths, never for capacity allocation.
    suite     Task-suite name, e.g. "halfcheetah_vel" or "mw_easy4".
    seed      Run seed.
    first_encounter
              True when this task_id has not appeared earlier in the sequence.
              Task-aware baselines allocate on True and reuse on False.
    """

    task_id: int
    seq_idx: int
    suite: str
    seed: int
    first_encounter: bool

    @property
    def is_root(self) -> bool:
        return self.seq_idx == 0


class ContinualAgent(nn.Module):
    """Base class for the four baselines.

    Subclasses MUST override:
        build_policy()              -> nn.Module returning (mean, raw_log_std)
        on_task_start(ctx)
        on_task_end(ctx)

    Subclasses MAY override:
        before_optimizer_step()
        after_optimizer_step()
        on_phase_boundary(step, budget)
        trainable_actor_parameters()
        extra_state() / load_extra_state()
        scalars()

    Subclasses MUST NOT override:
        export_policy_snapshot()    the on-disk evaluation contract
        save_chain_state() / load_chain_state()
    """

    #: Whether this baseline reads TaskContext.task_id. See module docstring.
    task_aware: bool = False

    #: Filename for the full continual state carried between subprocesses.
    CHAIN_STATE_NAME = "agent_state.pt"

    def __init__(self, obs_dim: int, act_dim: int, *, hidden_dim: int = 128,
                 encoder_linear_out: bool = False):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder_linear_out = bool(encoder_linear_out)
        # Set by on_task_start; read by task-aware forward passes so the policy
        # module keeps the single-argument forward(obs) signature that
        # policy_composition.components() requires.
        self.active_task: Optional[int] = None
        self._seen_tasks: Dict[int, int] = {}
        # Shared critic state carried between subprocesses. ProgNet keeps one
        # critic across the whole sequence by approved design (grow the actor
        # only); FT-N carries it because fine-tuning the critic is part of what
        # naive fine-tuning IS. PackNet and MaskNet also carry it so the only
        # difference between the four baselines is actor-side capacity.
        self._critic_state: Optional[Dict[str, Any]] = None
        # Subclasses MUST assign this at the end of their __init__:
        #     self.policy = self.build_policy()
        # It is what policy_composition sees, so its forward(obs) must return
        # (mean, raw_log_std). Declared here so the attribute always exists.
        self.policy: Optional[nn.Module] = None

    def assert_constructed(self) -> None:
        """Fail loudly if a subclass forgot to assign ``self.policy``."""
        if self.policy is None:
            raise RuntimeError(
                f"{type(self).__name__}.__init__ did not assign self.policy. "
                "Every baseline must end its __init__ with "
                "`self.policy = self.build_policy()`."
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def build_policy(self) -> nn.Module:
        """Return the policy module.

        Its ``forward(obs)`` must return ``(mean, raw_log_std)`` with RAW,
        unbounded log-std: ``policy_composition.components()`` applies
        ``bound_log_std`` itself, exactly as it does for ``CkaRlAgent``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_task_start(self, ctx: TaskContext) -> None:
        """Allocate or select capacity for this task.

        Called once, before the first environment step. Task-aware baselines
        allocate a new slice when ``ctx.first_encounter`` is True and select
        the existing one otherwise.
        """
        raise NotImplementedError

    def on_task_end(self, ctx: TaskContext) -> None:
        """Commit whatever has to survive into the next subprocess.

        Called after the frozen tail, before the snapshot is exported. No
        optimizer runs after this point.
        """
        raise NotImplementedError

    def before_optimizer_step(self) -> None:
        """Zero the gradients of parameters this task must not move.

        Default is a no-op. PackNet and ProgNet use it to protect weights
        belonging to earlier tasks. Called after ``backward()`` and before
        ``optimizer.step()``, so masking a gradient here is sufficient for SGD
        but NOT for Adam, whose momentum can still move a zero-gradient
        parameter. ``after_optimizer_step`` is what actually guarantees
        invariance; this hook is the cheap first line of defence.
        """
        return None

    def after_optimizer_step(self) -> None:
        """Restore any parameter that must be bit-identical to its frozen value.

        Default is a no-op. This is the hook that genuinely guarantees
        parameter isolation under Adam.
        """
        return None

    def on_phase_boundary(self, step: int, budget) -> None:
        """Called every optimization step with the step index and TaskBudget.

        Default is a no-op. PackNet uses it to trigger pruning at its
        train/retrain split, which is carved strictly out of ``budget.training``
        so compute parity with every other condition is preserved.
        """
        return None

    # ------------------------------------------------------------------
    # Optimizer wiring
    # ------------------------------------------------------------------
    def trainable_actor_parameters(self):
        """Parameters the actor optimizer should receive.

        Default is every parameter with ``requires_grad``. ProgNet narrows this
        to the active column plus its lateral adapters.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def scalars(self) -> Dict[str, float]:
        """Extra scalars to log under ``baseline/``. Default: none."""
        return {}

    # ------------------------------------------------------------------
    # The evaluation contract. Do not override.
    # ------------------------------------------------------------------
    def export_policy_snapshot(self) -> Dict[str, Any]:
        """Build the dict saved as ``policy_snapshot.pt``.

        The format is exactly what ``cka_rl.FrozenCkaPolicy.__init__`` reads in
        the ``composition_space == "parameter"`` branch, so a baseline
        checkpoint loads through the UNMODIFIED
        ``checkpoint_evaluation.evaluate(..., frozen_policy="snapshot")``.

        ``distillation`` is False so ``FrozenCkaPolicy.forward`` feeds the
        encoder output straight to the head with no observation skip
        connection, which is what every baseline here does.
        """
        from baselines.common.snapshot import export_snapshot

        return export_snapshot(self)

    def effective_head_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return ``{"mean": {...}, "logstd": {...}}`` for the ACTIVE task.

        Keys within each head are ``l0_weight``, ``l0_bias``, ``l2_weight``,
        ``l2_bias``, matching ``cka_rl._HEAD_KEYS``. Because the snapshot is a
        flat two-layer head, a baseline whose forward pass is not literally
        two linear layers (ProgNet, with its lateral adapters) must return the
        parameters of an EQUIVALENT two-layer head; see prognet.py.
        """
        raise NotImplementedError

    def shared_encoder(self) -> nn.Module:
        """Return the ``shared_arch.shared`` encoder for the snapshot."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Chain state. Do not override; use extra_state / load_extra_state.
    # ------------------------------------------------------------------
    def extra_state(self) -> Dict[str, Any]:
        """Baseline-specific state to carry into the next subprocess.

        Subclasses that override this MUST merge in ``super().extra_state()``
        so the shared critic survives the task boundary.
        """
        return {"critic_state": self._critic_state}

    def load_extra_state(self, state: Dict[str, Any]) -> None:
        """Restore whatever ``extra_state`` produced.

        Subclasses that override this MUST call ``super().load_extra_state``.
        """
        self._critic_state = state.get("critic_state")

    def stash_critic_state(self, critic_state: Dict[str, Any]) -> None:
        """Hand the trained critic forward to the next task in the chain.

        Called by sac_core.train_task after the final evaluation. The critic is
        shared across the whole sequence for every baseline, so the only thing
        that distinguishes the four methods is actor-side capacity handling.
        """
        self._critic_state = critic_state

    def save_chain_state(self, run_dir) -> pathlib.Path:
        run_dir = pathlib.Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / self.CHAIN_STATE_NAME
        payload = {
            "format_version": 1,
            "method": type(self).__name__,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "hidden_dim": self.hidden_dim,
            "encoder_linear_out": self.encoder_linear_out,
            "task_aware": bool(self.task_aware),
            "seen_tasks": dict(self._seen_tasks),
            "state_dict": {k: v.detach().cpu().clone()
                           for k, v in self.state_dict().items()},
            "extra": self.extra_state(),
        }
        torch.save(payload, path)
        return path

    def load_chain_state(self, run_dir, *, map_location="cpu") -> None:
        """Restore the previous task's state into this fresh process.

        ``strict=False`` on purpose: ProgNet grows its state_dict by one column
        per unseen task, so the incoming checkpoint is legitimately a subset of
        the current module. Every other baseline has a fixed parameter set and
        is checked below.
        """
        path = pathlib.Path(run_dir) / self.CHAIN_STATE_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; the continual chain cannot be continued. "
                "Re-run the sequence from seq_idx 0."
            )
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("format_version") != 1:
            raise ValueError(f"unsupported chain-state format in {path}")
        for field in ("obs_dim", "act_dim", "hidden_dim"):
            if int(payload[field]) != int(getattr(self, field)):
                raise ValueError(
                    f"{path} has {field}={payload[field]} but this run has "
                    f"{getattr(self, field)}; start a fresh chain."
                )
        if bool(payload["encoder_linear_out"]) != bool(self.encoder_linear_out):
            raise ValueError(
                f"{path} has encoder_linear_out={payload['encoder_linear_out']} "
                f"but this run was configured with {self.encoder_linear_out}. "
                "The two variants have identical parameter shapes and different "
                "forward functions, so this must match."
            )
        self._seen_tasks = {int(k): int(v) for k, v in payload["seen_tasks"].items()}
        self.load_extra_state(payload.get("extra", {}))
        missing, unexpected = self.load_state_dict(payload["state_dict"], strict=False)
        if unexpected:
            raise ValueError(
                f"{path} carries parameters this agent does not define: "
                f"{sorted(unexpected)[:8]}"
            )
        if missing and not self._allows_grown_parameters():
            raise ValueError(
                f"{path} is missing parameters this agent requires: "
                f"{sorted(missing)[:8]}"
            )

    def _allows_grown_parameters(self) -> bool:
        """True for architectures that legitimately add parameters per task."""
        return False

    # ------------------------------------------------------------------
    # Capacity bookkeeping shared by the task-aware baselines
    # ------------------------------------------------------------------
    def slice_for_task(self, task_id: int) -> Optional[int]:
        """Return the capacity slice already allocated to ``task_id``."""
        return self._seen_tasks.get(int(task_id))

    def allocate_slice(self, task_id: int, slice_index: int) -> int:
        """Record that ``task_id`` owns ``slice_index``.

        Re-allocating a task to a different slice is a bug: it would silently
        discard the capacity holding that task's learned behaviour and make the
        second pass through the sequence measure fresh capacity rather than
        retention.
        """
        task_id, slice_index = int(task_id), int(slice_index)
        existing = self._seen_tasks.get(task_id)
        if existing is not None and existing != slice_index:
            raise RuntimeError(
                f"task {task_id} already owns slice {existing}; refusing to "
                f"reallocate it to {slice_index}. Recurring tasks must reuse "
                "their capacity."
            )
        self._seen_tasks[task_id] = slice_index
        return slice_index

    def num_allocated_slices(self) -> int:
        return len(self._seen_tasks)
