"""MaskNet: task-conditioned gating over one shared backbone.

WHAT THIS IS
------------
A single backbone, plus one learned gate vector per unique task over the hidden
units of each head. The gate selects which units that task uses, so different
tasks occupy overlapping-but-distinguishable subnetworks of the same weights.

    h = relu(W1 z + b1)
    out = W2 (h * g_k) + b2,     g_k = sigmoid(s * logits_k)

Unlike PackNet the subnetworks are soft and may overlap. Unlike ProgNet nothing
is added to the parameter count except one gate vector per task.

Per the approved design this baseline receives the task oracle at train and
eval time and is reported as a Task-Aware Upper Bound. The critic is shared
across the sequence.

GATES ALONE DO NOT PREVENT FORGETTING
-------------------------------------
Worth being blunt about, because it is what makes a naive version of this
baseline meaningless. A gate chooses which units a task READS. It does nothing
to stop a later task from overwriting the weights underneath those units.
Implemented as gates and nothing else, this method forgets about as badly as
FT-N and would be a second lower bound wearing the label of an upper bound.

So the backbone is protected the way HAT protects it: gradients are attenuated
by how strongly earlier tasks claimed each unit. For unit u let

    m_u = max over previously seen tasks j of g_j[u]

and scale the gradient of every weight attached to unit u by (1 - m_u). A unit
an earlier task relies on completely is frozen; one no earlier task uses is
free; in between it moves proportionally. Rows of W1 and b1 produce unit u and
column u of W2 reads it, so those are what get scaled.

Note what this does NOT block. A later task may still open its own gate on a
claimed unit and read it. Attenuation restricts writing, never reading, and
that asymmetry is the whole transfer mechanism: shared features stay available
to everyone while staying stable for whoever depends on them.

THREE THINGS THAT WOULD FAIL SILENTLY IF DONE THE OBVIOUS WAY
--------------------------------------------------------------
1. Every task's gate logits live in ONE parameter tensor, one row per task, so
   requires_grad cannot protect earlier rows: it is all-or-nothing for the
   whole tensor. Left alone, training task k rewrites every earlier task's gate
   and each of them silently starts reading a different subnetwork than the one
   it was trained on. The rows are therefore protected the same way PackNet
   protects committed weights, by masking the gradient AND copying the rows
   back after the optimiser step. The second half is the one that matters:
   Adam moves a parameter with zero gradient whenever it still carries
   momentum, so masking alone leaves earlier gates drifting for many steps.

2. The attenuation mask for a finished task has to be read at the slope that
   task was hardened at, not at whatever the current task's ramp has reached.
   Read at the current soft slope early in a task, a hardened gate of zero
   comes back as one half and the mask under-protects exactly when the
   backbone is moving fastest.

3. Gate initialisation decides whether this method works at all. See
   TaskGates in baselines/common/masks.py: a constant positive init saturates
   every gate open, task 0 claims the entire backbone, and every later task is
   frozen solid with no error and no obviously broken curve.

THE SPARSITY REGULARISER
------------------------
Attenuation only helps later tasks if earlier ones leave something unclaimed.
--masknet-sparsity-reg adds a penalty on the mean open gate, weighted by how
much room is left:

    L_sparsity = lambda * sum_u g_k[u] (1 - m_u) / sum_u (1 - m_u)

The (1 - m_u) weighting is the detail that matters. Penalising the raw gate
would punish a task for reusing a unit an earlier task already owns, which is
exactly the transfer this architecture exists to permit. Weighted this way the
penalty pushes back only on claiming FRESH capacity.

It defaults to 0, which makes this a pure gating baseline with attenuation and
no capacity pressure. With logits starting near zero a task claims roughly half
the units on its own, so the free pool halves per task: comfortable for the
four unique Meta-World tasks, tight by the sixth HalfCheetah task. Turn the
regulariser on if the later tasks in a longer sequence come out starved.

THE TEMPERATURE RAMP
--------------------
Gates have to end close to binary or m_u is a smear of half-open units and the
attenuation protects nothing in particular. Following HAT, the sigmoid slope is
annealed from SLOPE_MIN to SLOPE_MAX across the task's optimisation phase, so
gradients flow freely early and the gate hardens as the task settles.

The ramp reaches SLOPE_MAX exactly at the last optimisation step, on purpose.
If it hardened afterwards, the final evaluation and the exported snapshot would
describe a different policy than the one just trained, and the logged
final_success would not match the checkpoint sitting beside it.

WHY THE FORWARD PASS FOLDS THE GATE INTO W2
--------------------------------------------
Scaling the hidden vector and scaling the columns of W2 are the same function:

    W2 (h * g) == (W2 * g) h

The second form is what the snapshot must store, since policy_snapshot.pt has
to be a plain two-layer head. Writing the LIVE forward pass the same way makes
the trained network and the exported checkpoint perform the identical
floating-point operation rather than merely equivalent ones, so verify_snapshot
reports exact zero instead of a residue that would make a real mismatch
indistinguishable from rounding.

WHAT HAPPENS WHEN A TASK RECURS
-------------------------------
It reuses its gate, per the approved design, and the attenuation mask excludes
its own earlier claim: m_u is a max over OTHER seen tasks. Otherwise a task
would be frozen out of the very units it chose for itself and a revisit could
not improve at all.

THE ENCODER
-----------
Frozen after the root task, as in ProgNet. It sits upstream of every gate, so
letting it drift would change every task's input at once and no gate could
protect against that.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.lifecycle import ContinualAgent, TaskContext
from baselines.common.masks import TaskGates
from shared_arch import shared


class GatedHead(nn.Module):
    """Two-layer head whose hidden units are gated per task."""

    def __init__(self, in_dim: int, hidden_dim: int, act_dim: int):
        super().__init__()
        self.l0 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, act_dim)

    def gated_parameters(self, gate: torch.Tensor) -> Dict[str, torch.Tensor]:
        """The effective two-layer head for this gate.

        The gate scales the COLUMNS of l2_weight, one per hidden unit.
        """
        return {
            "l0_weight": self.l0.weight,
            "l0_bias": self.l0.bias,
            "l2_weight": self.l2.weight * gate.unsqueeze(0),
            "l2_bias": self.l2.bias,
        }

    def forward(self, z: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        p = self.gated_parameters(gate)
        h = F.relu(F.linear(z, p["l0_weight"], p["l0_bias"]))
        return F.linear(h, p["l2_weight"], p["l2_bias"])


class MaskNetPolicy(nn.Module):
    """Shared backbone plus one gate bank per head."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int,
                 encoder_linear_out: bool, gate_init: float):
        super().__init__()
        self.fc = shared(obs_dim, linear_out=encoder_linear_out)
        self.mean_head = GatedHead(256, hidden_dim, act_dim)
        self.logstd_head = GatedHead(256, hidden_dim, act_dim)
        # One row per unique task, grown on first encounter.
        self.mean_gates = TaskGates(1, hidden_dim, init_bias=gate_init)
        self.logstd_gates = TaskGates(1, hidden_dim, init_bias=gate_init)
        self.active_slice: Optional[int] = None
        # Sigmoid slope, annealed during training. A buffer so it travels with
        # the module into the chain state.
        self.register_buffer("slope", torch.tensor(1.0))

    def gate_banks(self) -> Dict[str, TaskGates]:
        return {"mean": self.mean_gates, "logstd": self.logstd_gates}

    def gates_for_active(self) -> Dict[str, torch.Tensor]:
        if self.active_slice is None:
            raise RuntimeError(
                "MaskNetPolicy used before a task was selected; "
                "on_task_start must run first."
            )
        slope = float(self.slope)
        return {name: bank.gate_at(self.active_slice, slope)
                for name, bank in self.gate_banks().items()}

    def forward(self, obs: torch.Tensor):
        z = self.fc(obs)
        gates = self.gates_for_active()
        return (self.mean_head(z, gates["mean"]),
                self.logstd_head(z, gates["logstd"]))


class MaskNetAgent(ContinualAgent):
    """Task-conditioned gating with HAT-style backbone protection."""

    task_aware = True

    #: Sigmoid slope at the start and end of a task's optimisation phase.
    #: HAT uses a much larger maximum together with a gradient-compensation
    #: term; without that compensation a very large slope kills gate learning
    #: outright, so this stops somewhere the gate still hardens but keeps a
    #: usable gradient through most of the ramp.
    SLOPE_MIN = 1.0
    SLOPE_MAX = 50.0
    #: A gate above this counts as claimed, for reporting only. The attenuation
    #: itself is continuous.
    CLAIM_THRESHOLD = 0.5

    def __init__(self, obs_dim: int, act_dim: int, *, hidden_dim: int = 128,
                 encoder_linear_out: bool = False,
                 masknet_gate_init: float = 0.0,
                 masknet_sparsity_reg: float = 0.0,
                 **unused):
        super().__init__(
            obs_dim, act_dim,
            hidden_dim=hidden_dim,
            encoder_linear_out=encoder_linear_out,
        )
        if masknet_sparsity_reg < 0:
            raise ValueError("masknet_sparsity_reg must be >= 0")
        self.gate_init = float(masknet_gate_init)
        self.sparsity_reg = float(masknet_sparsity_reg)
        self.policy = self.build_policy()
        self.assert_constructed()
        # Per head: (1 - strongest claim any OTHER seen task makes on each unit).
        self._attenuation: Dict[str, torch.Tensor] = {}
        # Per head: gate logits as they stood at task start, used to hold every
        # inactive row exactly still.
        self._gate_reference: Dict[str, torch.Tensor] = {}

    def build_policy(self) -> nn.Module:
        return MaskNetPolicy(
            self.obs_dim, self.act_dim, self.hidden_dim,
            self.encoder_linear_out, self.gate_init,
        )

    def _allows_grown_parameters(self) -> bool:
        # The gate banks gain a row per unique task, so a checkpoint from
        # earlier in the sequence legitimately carries narrower gate tensors.
        return True

    # ------------------------------------------------------------------
    # Attenuation
    # ------------------------------------------------------------------
    def _rebuild_attenuation(self) -> None:
        """Compute (1 - m_u) per head over every task except the active one.

        Other tasks are read at SLOPE_MAX because that is the slope they were
        hardened at and the slope they will be evaluated at. Reading them at
        the current ramp position would report a hardened-closed gate as half
        open early in training, under-protecting the backbone at exactly the
        point it moves fastest.
        """
        active = int(self.policy.active_slice)
        self._attenuation = {}
        for head_name, bank in self.policy.gate_banks().items():
            with torch.no_grad():
                others = [
                    bank.gate_at(slot, self.SLOPE_MAX)
                    for slot in self._seen_tasks.values()
                    if int(slot) != active and int(slot) < bank.num_slices
                ]
                if others:
                    claimed = torch.stack(others, dim=0).max(dim=0).values
                else:
                    claimed = torch.zeros(
                        bank.width, device=bank.logits.device,
                        dtype=bank.logits.dtype,
                    )
                self._attenuation[head_name] = (1.0 - claimed).clamp_(0.0, 1.0)

    def before_optimizer_step(self) -> None:
        """Attenuate backbone gradients, and freeze every inactive gate row."""
        if self.policy.active_slice is None:
            return
        active = int(self.policy.active_slice)

        for head_name, head in (("mean", self.policy.mean_head),
                                ("logstd", self.policy.logstd_head)):
            free = self._attenuation.get(head_name)
            if free is None:
                continue
            free = free.to(head.l0.weight.device)
            # Row u of l0 produces unit u; column u of l2 reads it.
            if head.l0.weight.grad is not None:
                head.l0.weight.grad.mul_(free.unsqueeze(1))
            if head.l0.bias is not None and head.l0.bias.grad is not None:
                head.l0.bias.grad.mul_(free)
            if head.l2.weight.grad is not None:
                head.l2.weight.grad.mul_(free.unsqueeze(0))

        # Every task's gate lives in one tensor, so protect the other rows by
        # hand. requires_grad cannot express "this row only".
        for bank in self.policy.gate_banks().values():
            if bank.logits.grad is None:
                continue
            keep = torch.zeros_like(bank.logits, dtype=torch.bool)
            keep[active] = True
            bank.logits.grad.masked_fill_(~keep, 0.0)

    def after_optimizer_step(self) -> None:
        """Copy every inactive gate row back, bit for bit.

        Masking the gradient above is not sufficient under Adam: momentum
        accumulated while a row WAS active keeps moving it for many steps after
        its gradient goes to zero. This is what actually holds earlier tasks'
        gates still.
        """
        if self.policy.active_slice is None or not self._gate_reference:
            return
        active = int(self.policy.active_slice)
        with torch.no_grad():
            for head_name, bank in self.policy.gate_banks().items():
                reference = self._gate_reference.get(head_name)
                if reference is None:
                    continue
                reference = reference.to(bank.logits.device)
                rows = min(reference.shape[0], bank.logits.shape[0])
                for row in range(rows):
                    if row != active:
                        bank.logits.data[row].copy_(reference[row])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_task_start(self, ctx: TaskContext) -> None:
        device = next(self.parameters()).device
        existing = self.slice_for_task(ctx.task_id)

        if existing is None:
            if not ctx.first_encounter:
                raise RuntimeError(
                    f"task {ctx.task_id} is marked as a repeat but owns no gate. "
                    "The chain was probably resumed from the wrong checkpoint."
                )
            index = self.num_allocated_slices()
            # Slot 0 exists from construction; every later task adds one.
            while self.policy.mean_gates.num_slices <= index:
                self.policy.mean_gates.grow(init_bias=self.gate_init)
                self.policy.logstd_gates.grow(init_bias=self.gate_init)
            self.to(device)
            self.allocate_slice(ctx.task_id, index)
            print(f"*** MaskNet: new gate {index} for task {ctx.task_id} ***")
        else:
            index = existing
            print(f"*** MaskNet: reusing gate {index} for task {ctx.task_id} ***")

        self.active_task = int(ctx.task_id)
        self.policy.active_slice = index
        # Start each task at the soft end of the ramp so gradients flow.
        self.policy.slope.fill_(self.SLOPE_MIN)

        # Encoder trains on the root task only; it is upstream of every gate.
        for param in self.parameters():
            param.requires_grad_(True)
        for param in self.policy.fc.parameters():
            param.requires_grad_(ctx.is_root)

        self._rebuild_attenuation()
        self._gate_reference = {
            name: bank.logits.detach().clone()
            for name, bank in self.policy.gate_banks().items()
        }

        free_fraction = float(
            torch.stack([v.mean() for v in self._attenuation.values()]).mean()
        )
        print(f"*** MaskNet: {free_fraction:.1%} of backbone capacity unclaimed by "
              f"earlier tasks | encoder "
              f"{'trainable (root task)' if ctx.is_root else 'frozen'} ***")

    def on_phase_boundary(self, step: int, budget) -> None:
        """Anneal the sigmoid slope, reaching SLOPE_MAX at the final step.

        Ending the ramp exactly at the last optimisation step keeps the final
        evaluation, the frozen tail and the exported snapshot all describing one
        and the same policy.
        """
        total = max(int(budget.training) - 1, 1)
        progress = min(max(step / total, 0.0), 1.0)
        self.policy.slope.fill_(
            self.SLOPE_MIN + (self.SLOPE_MAX - self.SLOPE_MIN) * progress
        )

    def auxiliary_loss(self) -> Optional[torch.Tensor]:
        """Pressure to leave capacity for later tasks. Zero-weighted by default.

        Weighted by (1 - m_u) so reusing a unit an earlier task already claimed
        is free and only claiming fresh capacity is penalised.
        """
        if self.sparsity_reg <= 0 or self.policy.active_slice is None:
            return None
        gates = self.policy.gates_for_active()
        terms = []
        for head_name, gate in gates.items():
            free = self._attenuation.get(head_name)
            if free is None:
                continue
            free = free.to(gate.device)
            denominator = free.sum().clamp_min(1e-6)
            terms.append((gate * free).sum() / denominator)
        if not terms:
            return None
        return self.sparsity_reg * torch.stack(terms).mean()

    def prepare_for_evaluation(self) -> None:
        """Harden the gates, as they were when the snapshot was written."""
        self.policy.slope.fill_(self.SLOPE_MAX)
        if self.policy.active_slice is not None:
            self._rebuild_attenuation()

    def on_task_end(self, ctx: TaskContext) -> None:
        self.policy.slope.fill_(self.SLOPE_MAX)
        self._rebuild_attenuation()
        occupancy = {
            head_name: float((gate > self.CLAIM_THRESHOLD).float().mean())
            for head_name, gate in self.policy.gates_for_active().items()
        }
        print(f"*** MaskNet: task {ctx.task_id} closed with gate occupancy "
              f"mean={occupancy['mean']:.1%} logstd={occupancy['logstd']:.1%} ***")

    # ------------------------------------------------------------------
    # Snapshot contract
    # ------------------------------------------------------------------
    def shared_encoder(self) -> nn.Module:
        return self.policy.fc

    def effective_head_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self.policy.active_slice is None:
            raise RuntimeError("no active gate; on_task_start must run first")
        gates = self.policy.gates_for_active()
        return {
            "mean": self.policy.mean_head.gated_parameters(gates["mean"]),
            "logstd": self.policy.logstd_head.gated_parameters(gates["logstd"]),
        }

    # ------------------------------------------------------------------
    # Chain state
    # ------------------------------------------------------------------
    def extra_state(self) -> Dict[str, Any]:
        state = super().extra_state()
        state["num_gates"] = self.policy.mean_gates.num_slices
        state["gate_init"] = self.gate_init
        return state

    def load_extra_state(self, state: Dict[str, Any]) -> None:
        super().load_extra_state(state)
        # Grow the banks before load_state_dict so the stored rows have
        # somewhere to land. The base class calls this first for that reason.
        target = int(state.get("num_gates", 1))
        while self.policy.mean_gates.num_slices < target:
            self.policy.mean_gates.grow(init_bias=self.gate_init)
            self.policy.logstd_gates.grow(init_bias=self.gate_init)
        if self.policy.mean_gates.num_slices != target:
            raise ValueError(
                f"cannot shrink from {self.policy.mean_gates.num_slices} gates "
                f"to {target}; start a fresh chain."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def scalars(self) -> Dict[str, float]:
        out = {
            "gates": float(self.policy.mean_gates.num_slices),
            "active_gate": float(-1 if self.policy.active_slice is None
                                 else self.policy.active_slice),
            "slope": float(self.policy.slope),
        }
        if self._attenuation:
            out["free_capacity"] = float(
                torch.stack([v.mean() for v in self._attenuation.values()]).mean()
            )
        if self.policy.active_slice is not None:
            with torch.no_grad():
                gates = self.policy.gates_for_active()
                out["gate_occupancy"] = float(
                    torch.stack([
                        (g > self.CLAIM_THRESHOLD).float().mean()
                        for g in gates.values()
                    ]).mean()
                )
        return out
