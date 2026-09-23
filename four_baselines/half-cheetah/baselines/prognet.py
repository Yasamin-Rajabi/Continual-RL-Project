"""ProgNet: Progressive Neural Networks, adapted to a continuous-control actor.

WHAT THIS IS
------------
One actor column per UNIQUE task. When a new task arrives, a fresh column is
allocated and every earlier column is frozen; the new column reads the earlier
columns through lateral connections, so it can reuse what they learned without
being able to damage it. Forgetting is structurally zero, and the price is that
the parameter count grows with the number of tasks.

Per the approved design this baseline receives the task oracle at train and
eval time and is reported as a Task-Aware Upper Bound. The critic is shared
across the whole sequence: only the actor grows.

WHERE THE LATERALS GO, AND WHY IT MATTERS HERE
----------------------------------------------
The original formulation puts laterals into every layer beyond the first:

    h_k^(i) = f( W_k^(i) h_k^(i-1) + sum_{j<k} U_k^(i:j) h_j^(i-1) )

Layer 1 has none, because every column reads the SAME input. That detail is not
cosmetic in this codebase. All columns share the encoder output z, so a lateral
into layer 1 would compute U z, which just adds to W_k^(1) z; it is a
reparameterisation of the same function and buys nothing. So for the two-layer
head used here there is exactly one place laterals belong: the output layer,
reading the previous columns' hidden activations.

    h_k = relu(W1_k z + b1_k)                       (per column, no laterals)
    out_k = W2_k h_k + sum_{j<k} L_kj h_j + b2_k    (laterals into the output)

THE FLATTENING, WHICH IS WHAT MAKES THE EVALUATION CONTRACT WORK
-----------------------------------------------------------------
policy_snapshot.pt has to be a plain two-layer head, because that is what
cka_rl.FrozenCkaPolicy evaluates and the whole point is that baselines go
through the method's own evaluator. A network with lateral connections looks
like it cannot be written that way. It can, exactly, and here is why.

Every column's hidden layer is relu of a linear map of the same z. ReLU is
elementwise, so stacking the columns' hidden vectors is the same as applying
one wider layer:

    H = [h_0 ; h_1 ; ... ; h_k] = relu( [W1_0 ; ... ; W1_k] z + [b1_0 ; ... ; b1_k] )

and the output is a single linear map of that stack:

    out_k = [L_k0 | L_k1 | ... | L_k(k-1) | W2_k] H + b2_k

So the exported head is a two-layer head of width (k+1) * hidden_dim. Not an
approximation, not a distillation: the same function, rewritten.

Two consequences worth stating plainly. ``effective_hidden_dim`` returns the
stacked width, which is what the snapshot validator checks against. And the
live forward pass below BUILDS those stacked matrices and issues one F.linear,
rather than looping over columns and summing. Written that way the training
network and the exported snapshot perform the identical floating-point
operation, so verify_snapshot reports exactly zero rather than a small
accumulation-order residue that would leave a real mismatch indistinguishable
from rounding.

LATERAL ADAPTERS ARE LINEAR, DELIBERATELY
-----------------------------------------
The paper puts a nonlinear bottleneck on each lateral to keep parameter growth
manageable. A nonlinearity there would break the flattening above: relu(V h_j)
composed with the column's own relu is three layers deep, not two, and no
two-layer head reproduces it. So the adapter here is a LINEAR low-rank
bottleneck, L_kj = U_kj V_kj with inner dimension --prognet-adapter-dim. That
keeps growth sub-quadratic for the same reason the paper's adapter does, while
staying exactly expressible in the evaluation format. It is a real deviation
from the published architecture and belongs in the paper's baseline description.

WHAT HAPPENS WHEN A TASK RECURS
-------------------------------
Both sequences revisit every task, and per the approved design a recurring task
reuses its existing column rather than allocating a new one. That raises a
question the first pass never asks: may the revisit retrain that column?

Not freely. Column m > k reads h_k through its lateral L_mk. Retraining column
k's FIRST layer would move h_k and silently change column m's output, which is
precisely the forgetting ProgNet exists to rule out. So on a revisit the
column's first layer is frozen whenever any later column exists, and only the
output layer and the column's own incoming laterals are trained. Those affect
nothing except this task's own output, because laterals read hidden
activations, never outputs.

The rule collapses to one line: the first layer is trainable exactly when this
column is the most recently allocated one. On a first encounter that is true by
construction, since the column was just appended.

A revisit of an early column therefore trains only a few thousand parameters
and will barely move. That is not a bug to be worked around; it is what
ProgNet's zero-forgetting guarantee costs on a sequence with repeats, and the
retention matrix should show it.

THE ENCODER
-----------
Frozen after the root task. It sits upstream of every column, so letting it
keep training would change all of them at once and defeat the whole
construction. This matches the method's own default (train_shared=False:
learn the root encoder on task 0, freeze it thereafter), so the two sides stay
comparable.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.lifecycle import ContinualAgent, TaskContext
from shared_arch import shared


class LateralAdapter(nn.Module):
    """Low-rank LINEAR lateral from one earlier column's hidden layer.

    ``effective_weight()`` collapses the bottleneck into the single
    ``[act_dim, hidden_dim]`` matrix the flattening needs. There is no
    nonlinearity and no bias here, both on purpose: see the module docstring.
    """

    def __init__(self, hidden_dim: int, act_dim: int, adapter_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, adapter_dim, bias=False)
        self.up = nn.Linear(adapter_dim, act_dim, bias=False)
        # Start at zero so a freshly allocated column begins as an exact copy of
        # a plain independent column. If laterals started random, the new task's
        # first gradient steps would be fighting noise injected from frozen
        # columns, which reads as negative transfer that the architecture did
        # not actually imply.
        nn.init.zeros_(self.up.weight)

    def effective_weight(self) -> torch.Tensor:
        return self.up.weight @ self.down.weight


class ProgressiveColumn(nn.Module):
    """One task's column: a two-layer head plus its incoming laterals."""

    def __init__(self, in_dim: int, hidden_dim: int, act_dim: int,
                 adapter_dim: int, num_previous: int):
        super().__init__()
        self.l0 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, act_dim)
        self.laterals = nn.ModuleList(
            LateralAdapter(hidden_dim, act_dim, adapter_dim)
            for _ in range(num_previous)
        )


class ProgressiveHead(nn.Module):
    """A growing stack of columns for one head (mean or log-std)."""

    def __init__(self, in_dim: int, hidden_dim: int, act_dim: int, adapter_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.act_dim = int(act_dim)
        self.adapter_dim = int(adapter_dim)
        self.columns = nn.ModuleList()

    @property
    def num_columns(self) -> int:
        return len(self.columns)

    def grow(self) -> int:
        """Append a column that reads every existing one. Returns its index."""
        index = len(self.columns)
        self.columns.append(
            ProgressiveColumn(
                self.in_dim, self.hidden_dim, self.act_dim,
                self.adapter_dim, num_previous=index,
            )
        )
        return index

    # ------------------------------------------------------------------
    # The stacked form. Used by BOTH the forward pass and the snapshot, so
    # the two are the same arithmetic rather than merely the same function.
    # ------------------------------------------------------------------
    def stacked_parameters(self, slice_index: int) -> Dict[str, torch.Tensor]:
        k = int(slice_index)
        if not 0 <= k < len(self.columns):
            raise IndexError(f"column {k} out of range for {len(self.columns)} columns")
        active = self.columns[k]
        if len(active.laterals) != k:
            raise RuntimeError(
                f"column {k} carries {len(active.laterals)} laterals, expected {k}. "
                "The chain state is inconsistent; restart the sequence from seq 0."
            )

        l0_weight = torch.cat([self.columns[j].l0.weight for j in range(k + 1)], dim=0)
        l0_bias = torch.cat([self.columns[j].l0.bias for j in range(k + 1)], dim=0)
        # Column order in the output block must match the row order above.
        blocks = [active.laterals[j].effective_weight() for j in range(k)]
        blocks.append(active.l2.weight)
        l2_weight = torch.cat(blocks, dim=1)
        return {
            "l0_weight": l0_weight,
            "l0_bias": l0_bias,
            "l2_weight": l2_weight,
            "l2_bias": active.l2.bias,
        }

    def forward(self, z: torch.Tensor, slice_index: int) -> torch.Tensor:
        p = self.stacked_parameters(slice_index)
        h = F.relu(F.linear(z, p["l0_weight"], p["l0_bias"]))
        return F.linear(h, p["l2_weight"], p["l2_bias"])


class ProgNetPolicy(nn.Module):
    """Shared encoder plus one progressive stack per head.

    ``active_slice`` is set by the agent's ``on_task_start``. Keeping it on the
    module lets ``forward(obs)`` stay single-argument, which is what
    ``policy_composition.components`` requires.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int,
                 adapter_dim: int, encoder_linear_out: bool):
        super().__init__()
        self.fc = shared(obs_dim, linear_out=encoder_linear_out)
        self.mean_head = ProgressiveHead(256, hidden_dim, act_dim, adapter_dim)
        self.logstd_head = ProgressiveHead(256, hidden_dim, act_dim, adapter_dim)
        self.active_slice: int | None = None

    def forward(self, obs: torch.Tensor):
        if self.active_slice is None:
            raise RuntimeError(
                "ProgNetPolicy.forward called before a task was selected; "
                "on_task_start must run first."
            )
        z = self.fc(obs)
        return (self.mean_head(z, self.active_slice),
                self.logstd_head(z, self.active_slice))


class ProgNetAgent(ContinualAgent):
    """Progressive Neural Networks. Task-aware upper bound."""

    task_aware = True

    def __init__(self, obs_dim: int, act_dim: int, *, hidden_dim: int = 128,
                 encoder_linear_out: bool = False, prognet_adapter_dim: int = 64,
                 **unused):
        super().__init__(
            obs_dim, act_dim,
            hidden_dim=hidden_dim,
            encoder_linear_out=encoder_linear_out,
        )
        self.adapter_dim = int(prognet_adapter_dim)
        if self.adapter_dim < 1:
            raise ValueError("prognet_adapter_dim must be >= 1")
        self.policy = self.build_policy()
        self.assert_constructed()

    def build_policy(self) -> nn.Module:
        return ProgNetPolicy(
            self.obs_dim, self.act_dim, self.hidden_dim,
            self.adapter_dim, self.encoder_linear_out,
        )

    def _allows_grown_parameters(self) -> bool:
        # Columns are rebuilt in load_extra_state before the state dict is
        # applied, so nothing should actually be missing. Kept permissive
        # because this architecture is the one where a shape surprise is
        # legitimate rather than a corrupted checkpoint.
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_task_start(self, ctx: TaskContext) -> None:
        existing = self.slice_for_task(ctx.task_id)

        if existing is None:
            if not ctx.first_encounter:
                raise RuntimeError(
                    f"task {ctx.task_id} is marked as a repeat but owns no column. "
                    "The chain was probably resumed from the wrong checkpoint."
                )
            device = next(self.parameters()).device
            index = self.policy.mean_head.grow()
            logstd_index = self.policy.logstd_head.grow()
            if index != logstd_index:
                raise RuntimeError(
                    f"head stacks fell out of step: mean grew to {index}, "
                    f"logstd to {logstd_index}"
                )
            # grow() builds on CPU; the rest of the agent may already be on GPU.
            self.to(device)
            self.allocate_slice(ctx.task_id, index)
            print(f"*** ProgNet: allocated column {index} for task {ctx.task_id} "
                  f"({index} lateral(s) from earlier columns) ***")
        else:
            index = existing
            print(f"*** ProgNet: reusing column {index} for task {ctx.task_id} ***")

        self.active_task = int(ctx.task_id)
        self.policy.active_slice = index
        self._apply_freezing(ctx, index)

    def _apply_freezing(self, ctx: TaskContext, index: int) -> None:
        """Set requires_grad so this task can only touch what it is allowed to.

        Trainable:
          * the encoder, ONLY on the root task;
          * the active column's output layer and its incoming laterals, always;
          * the active column's first layer, only when no later column exists.

        Everything else is frozen. The first-layer rule is the one that matters:
        a later column reads this column's hidden activations through a lateral,
        so moving them would change that later column's output.
        """
        for param in self.parameters():
            param.requires_grad_(False)

        encoder_trainable = ctx.is_root
        for param in self.policy.fc.parameters():
            param.requires_grad_(encoder_trainable)

        num_columns = self.policy.mean_head.num_columns
        first_layer_trainable = index == num_columns - 1

        for head in (self.policy.mean_head, self.policy.logstd_head):
            column = head.columns[index]
            for param in column.l2.parameters():
                param.requires_grad_(True)
            for lateral in column.laterals:
                for param in lateral.parameters():
                    param.requires_grad_(True)
            for param in column.l0.parameters():
                param.requires_grad_(first_layer_trainable)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        note = "" if first_layer_trainable else (
            " | first layer FROZEN: a later column reads this column's hidden units"
        )
        print(
            f"*** ProgNet: {trainable:,} / {total:,} actor parameters trainable"
            f"{note} | encoder {'trainable (root task)' if encoder_trainable else 'frozen'} ***"
        )

    def on_task_end(self, ctx: TaskContext) -> None:
        """Freeze everything. The next subprocess re-opens only what it may use."""
        for param in self.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Snapshot contract
    # ------------------------------------------------------------------
    def effective_hidden_dim(self) -> int:
        if self.policy.active_slice is None:
            return self.hidden_dim
        return (int(self.policy.active_slice) + 1) * self.hidden_dim

    def shared_encoder(self) -> nn.Module:
        return self.policy.fc

    def effective_head_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self.policy.active_slice is None:
            raise RuntimeError("no active column; on_task_start must run first")
        k = self.policy.active_slice
        return {
            "mean": self.policy.mean_head.stacked_parameters(k),
            "logstd": self.policy.logstd_head.stacked_parameters(k),
        }

    # ------------------------------------------------------------------
    # Chain state
    # ------------------------------------------------------------------
    def extra_state(self) -> Dict[str, Any]:
        state = super().extra_state()
        state["num_columns"] = self.policy.mean_head.num_columns
        state["adapter_dim"] = self.adapter_dim
        return state

    def load_extra_state(self, state: Dict[str, Any]) -> None:
        super().load_extra_state(state)
        stored_adapter = int(state.get("adapter_dim", self.adapter_dim))
        if stored_adapter != self.adapter_dim:
            raise ValueError(
                f"checkpoint was trained with prognet_adapter_dim={stored_adapter} "
                f"but this run uses {self.adapter_dim}; start a fresh chain."
            )
        # Rebuild the columns BEFORE load_state_dict runs, so every stored
        # tensor has somewhere to land. The base class calls this first for
        # exactly this reason.
        target = int(state.get("num_columns", 0))
        while self.policy.mean_head.num_columns < target:
            self.policy.mean_head.grow()
            self.policy.logstd_head.grow()
        if self.policy.mean_head.num_columns != target:
            raise ValueError(
                f"cannot shrink from {self.policy.mean_head.num_columns} columns "
                f"to {target}; start a fresh chain."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def scalars(self) -> Dict[str, float]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "columns": float(self.policy.mean_head.num_columns),
            "active_column": float(-1 if self.policy.active_slice is None
                                   else self.policy.active_slice),
            "trainable_parameters": float(trainable),
            "total_parameters": float(total),
        }
