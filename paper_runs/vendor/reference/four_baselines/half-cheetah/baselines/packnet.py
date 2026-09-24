"""PackNet: iterative pruning, parameter isolation, and binary masks.

WHAT THIS IS
------------
One fixed-size network, carved up between tasks. Task k trains in whatever
weights are still unclaimed, then keeps the largest fraction of them by
magnitude and commits those to itself permanently; the rest are released back
to the free pool for later tasks. At evaluation time task k runs with the
weights owned by tasks 0..k and nothing else.

Because a committed weight is never written again, forgetting is exactly zero.
The cost is capacity: the free pool shrinks geometrically, and a sequence long
enough will run out.

Per the approved design this baseline receives the task oracle at train and
eval time and is reported as a Task-Aware Upper Bound. The critic is shared
across the sequence; only the actor is masked.

THE THREE PHASES OF A TASK, ALL INSIDE THE SAME BUDGET
------------------------------------------------------
Per the approved design the prune-and-retrain cycle is carved out of the
(Delta - B) optimisation budget rather than added on top, so the comparison
with every other condition stays step-for-step fair.

    step 0                          ... training on the free pool
    step floor((Delta-B)*(1-r))     PRUNE: keep the top fraction by magnitude,
                                    commit them, zero the rest
    ... to step (Delta-B)           retrain, now confined to the committed mask
    then B frozen-tail steps, no optimiser at all

with r = --packnet-retrain-fraction. The pruning happens in on_phase_boundary,
which sac_core calls every step with the TaskBudget, so the split is derived
from the same budget object every other baseline uses.

WHY GRADIENT MASKING ALONE IS NOT ISOLATION
--------------------------------------------
This is the part that would fail silently if it were done the obvious way.

Zeroing a frozen weight's gradient before optimizer.step() is not enough under
Adam. Adam's update is m_hat / (sqrt(v_hat) + eps), and m and v are running
averages. A weight that was trained during an earlier task carries non-zero
momentum into this one. Feeding it a zero gradient does not produce a zero
update; it produces a decaying non-zero update that keeps moving the weight for
many steps afterwards. The network would still train, the loss would still go
down, and an earlier task's committed weights would be quietly drifting the
whole time. Nothing in the training curves would show it. The forgetting would
surface only in the retention matrix, looking like a property of the method.

So isolation is enforced twice, in two different hooks:

    before_optimizer_step   zero the gradients of protected weights
    after_optimizer_step    copy protected weights back, bit for bit

The second one is what actually guarantees the invariant. The first is cheap
and keeps Adam's second-moment estimates from being polluted by gradients that
are going to be discarded anyway.

The frozen reference is re-snapshotted at the start of each task, so it always
holds the committed values rather than whatever the weights happened to be.

BIASES
------
Biases are not pruned: masking them buys almost no capacity and destabilises
the head far more than it saves. But a shared trainable bias would break
isolation just as surely as a shared weight, since task k's output depends on
it. So biases train on the root task and are frozen for the rest of the
sequence. Every task then sees the same biases, which is consistent, and the
isolation guarantee holds. The encoder is frozen after the root task for the
same reason it is in ProgNet: it sits upstream of everything.

WHAT THE MASK DOES IN THE FORWARD PASS
--------------------------------------
Masks are applied functionally, as F.linear(x, w * mask, b), rather than by
zeroing the stored weights. Two reasons. Weights belonging to LATER tasks have
to be excluded from task k's forward pass without being destroyed, which
zeroing would do. And multiplying in the forward pass means the gradient
reaching w is already masked, so the masking is consistent between the forward
and backward pass by construction rather than by a second mechanism that could
drift out of sync with the first.

The snapshot then stores w * mask directly, which is the identical product, so
the exported checkpoint and the trained network perform the same arithmetic and
verify_snapshot reports exact zero.

WHAT HAPPENS WHEN A TASK RECURS
-------------------------------
A recurring task reuses the mask it already owns, per the approved design. It
does not prune again and it does not take new weights from the free pool: it
retrains inside its existing subnetwork for the full (Delta - B) steps. That
keeps the capacity accounting honest, since otherwise a task appearing twice
would claim twice the share, and it makes the second pass a measurement of
what the isolated subnetwork can recover rather than of extra capacity.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common.lifecycle import ContinualAgent, TaskContext
from baselines.common.masks import (
    FREE,
    OwnershipBook,
    magnitude_keep_mask,
    prunable_parameters,
    snapshot_weights,
)
from shared_arch import shared


class MaskedHead(nn.Module):
    """Two-layer head whose weights are masked functionally in the forward pass."""

    def __init__(self, in_dim: int, hidden_dim: int, act_dim: int):
        super().__init__()
        self.l0 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, act_dim)

    def forward(self, z: torch.Tensor, masks: Dict[str, torch.Tensor],
                prefix: str) -> torch.Tensor:
        w0 = self.l0.weight
        m0 = masks.get(f"{prefix}.l0.weight")
        if m0 is not None:
            w0 = w0 * m0
        h = F.relu(F.linear(z, w0, self.l0.bias))

        w2 = self.l2.weight
        m2 = masks.get(f"{prefix}.l2.weight")
        if m2 is not None:
            w2 = w2 * m2
        return F.linear(h, w2, self.l2.bias)

    def masked_parameters(self, masks: Dict[str, torch.Tensor],
                          prefix: str) -> Dict[str, torch.Tensor]:
        w0 = self.l0.weight
        m0 = masks.get(f"{prefix}.l0.weight")
        if m0 is not None:
            w0 = w0 * m0
        w2 = self.l2.weight
        m2 = masks.get(f"{prefix}.l2.weight")
        if m2 is not None:
            w2 = w2 * m2
        return {
            "l0_weight": w0,
            "l0_bias": self.l0.bias,
            "l2_weight": w2,
            "l2_bias": self.l2.bias,
        }


class PackNetPolicy(nn.Module):
    """Encoder plus two masked heads. One fixed-size network for every task."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int,
                 encoder_linear_out: bool):
        super().__init__()
        self.fc = shared(obs_dim, linear_out=encoder_linear_out)
        self.mean_head = MaskedHead(256, hidden_dim, act_dim)
        self.logstd_head = MaskedHead(256, hidden_dim, act_dim)
        # name -> float mask, set by the agent. Empty means "use everything",
        # which is only ever true before the first task has been set up.
        self.forward_masks: Dict[str, torch.Tensor] = {}

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs
        for index, layer in enumerate(self.fc):
            if isinstance(layer, nn.Linear):
                w = layer.weight
                mask = self.forward_masks.get(f"fc.{index}.weight")
                if mask is not None:
                    w = w * mask
                x = F.linear(x, w, layer.bias)
            else:
                x = layer(x)
        return x

    def forward(self, obs: torch.Tensor):
        z = self.encode(obs)
        return (self.mean_head(z, self.forward_masks, "mean_head"),
                self.logstd_head(z, self.forward_masks, "logstd_head"))


class PackNetAgent(ContinualAgent):
    """Iterative pruning with parameter isolation. Task-aware upper bound."""

    task_aware = True

    #: Submodules the ownership machinery does not touch. The shared
    #: encoder trains on the root task and is frozen afterwards, exactly as
    #: in ProgNet and MaskNet, so every baseline shares the same feature
    #: extractor policy and only the head treatment differs.
    EXCLUDE_FROM_PRUNING = ("fc.",)

    def __init__(self, obs_dim: int, act_dim: int, *, hidden_dim: int = 128,
                 encoder_linear_out: bool = False,
                 packnet_keep_fraction: float = 0.5,
                 packnet_retrain_fraction: float = 0.3,
                 **unused):
        super().__init__(
            obs_dim, act_dim,
            hidden_dim=hidden_dim,
            encoder_linear_out=encoder_linear_out,
        )
        if not 0.0 < packnet_keep_fraction <= 1.0:
            raise ValueError("packnet_keep_fraction must be in (0, 1]")
        if not 0.0 <= packnet_retrain_fraction < 1.0:
            raise ValueError("packnet_retrain_fraction must be in [0, 1)")
        self.keep_fraction = float(packnet_keep_fraction)
        self.retrain_fraction = float(packnet_retrain_fraction)

        self.policy = self.build_policy()
        self.assert_constructed()

        # The shared encoder is frozen after the root task, so pruning it
        # could only destroy capacity nothing can ever reclaim. Heads only.
        self.book = OwnershipBook(self.policy, exclude_prefixes=self.EXCLUDE_FROM_PRUNING)
        self.active_slice: Optional[int] = None
        self._frozen_reference: Dict[str, torch.Tensor] = {}
        self._pruned_this_task = False
        self._prune_step: Optional[int] = None
        self._is_repeat = False

    def build_policy(self) -> nn.Module:
        return PackNetPolicy(
            self.obs_dim, self.act_dim, self.hidden_dim, self.encoder_linear_out
        )

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    def _rebuild_forward_masks(self, *, include_free: bool) -> None:
        """Recompute which weights this task's forward pass may use.

        Weights owned by slices 0..active are always in. Free weights are in
        only while the task is still training in the free pool, i.e. before it
        prunes. Weights owned by any LATER slice are always out, which is what
        makes evaluating an early task after the sequence has moved on give
        that task's own subnetwork rather than a mixture.
        """
        if self.active_slice is None:
            self.policy.forward_masks = {}
            return
        active = int(self.active_slice)
        masks: Dict[str, torch.Tensor] = {}
        for name, param in prunable_parameters(self.policy, self.EXCLUDE_FROM_PRUNING):
            owner = self.book.owner_tensor(name)
            usable = (owner != FREE) & (owner <= active + 1)
            if include_free:
                usable = usable | (owner == FREE)
            masks[name] = usable.to(dtype=param.dtype, device=param.device)
        self.policy.forward_masks = masks

    def _trainable_mask(self, name: str) -> torch.Tensor:
        """Weights this task's optimiser may move.

        Before pruning: the free pool plus anything this slice already owns.
        After pruning, and on a repeat: only what this slice owns.
        """
        owner = self.book.owner_tensor(name)
        mine = owner == (int(self.active_slice) + 1)
        if self._pruned_this_task or self._is_repeat:
            return mine
        return mine | (owner == FREE)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_task_start(self, ctx: TaskContext) -> None:
        device = next(self.parameters()).device
        self.book.to(device)

        existing = self.slice_for_task(ctx.task_id)
        if existing is None:
            if not ctx.first_encounter:
                raise RuntimeError(
                    f"task {ctx.task_id} is marked as a repeat but owns no mask. "
                    "The chain was probably resumed from the wrong checkpoint."
                )
            index = self.allocate_slice(ctx.task_id, self.num_allocated_slices())
            self._is_repeat = False
            self._pruned_this_task = False
            print(f"*** PackNet: new slice {index} for task {ctx.task_id}; "
                  f"will prune to the top {self.keep_fraction:.0%} of the free pool ***")
        else:
            index = existing
            self._is_repeat = True
            # A repeat has already pruned, in an earlier subprocess. Treat it as
            # done so the trainable mask stays confined to this slice for the
            # whole task and no second prune fires.
            self._pruned_this_task = True
            print(f"*** PackNet: reusing slice {index} for task {ctx.task_id}; "
                  f"retraining inside its existing mask, no new capacity ***")

        self.active_task = int(ctx.task_id)
        self.active_slice = index

        # Biases and the encoder train on the root task only. After that a
        # shared trainable parameter would break isolation exactly as a shared
        # weight would.
        root = ctx.is_root
        prunable_names = {name for name, _ in prunable_parameters(self.policy, self.EXCLUDE_FROM_PRUNING)}
        for name, param in self.policy.named_parameters():
            param.requires_grad_(True if name in prunable_names else root)
        for param in self.policy.fc.parameters():
            param.requires_grad_(root)

        self._rebuild_forward_masks(include_free=not self._pruned_this_task)
        self._frozen_reference = snapshot_weights(self.policy)

        report = self.book.capacity_report()
        print(f"*** PackNet: {report['owned_fraction']:.1%} of prunable weights "
              f"already owned, {int(report['free_weights']):,} free ***")

    def on_phase_boundary(self, step: int, budget) -> None:
        """Prune once, at the point that leaves the configured retrain tail.

        Carved out of budget.training, never added to it.
        """
        if self._pruned_this_task or self.active_slice is None:
            return
        if self._prune_step is None:
            self._prune_step = int(budget.training * (1.0 - self.retrain_fraction))
            # A degenerate split would either skip pruning or leave no retrain
            # steps; clamp it into the interior so both phases exist.
            self._prune_step = max(1, min(self._prune_step, budget.training - 1))
        if step < self._prune_step:
            return
        self._prune(step, budget)

    def _prune(self, step: int, budget) -> None:
        committed_total = 0
        released_total = 0
        with torch.no_grad():
            for name, param in prunable_parameters(self.policy, self.EXCLUDE_FROM_PRUNING):
                free = self.book.free_mask(name)
                keep = magnitude_keep_mask(param, free, self.keep_fraction)
                committed_total += self.book.commit(name, keep, self.active_slice)
                # Whatever was in the free pool and not kept goes back to zero
                # and back to the pool. Zeroing matters: those weights were
                # trained for this task, and leaving them at their trained
                # values would hand the next task a warm start it did not earn
                # while also making this task's subnetwork ill-defined.
                released = free & ~keep
                released_total += int(released.sum())
                param.data.masked_fill_(released, 0.0)

        self._pruned_this_task = True
        # From here on: no free weights in the forward pass, and only this
        # slice's own weights are trainable.
        self._rebuild_forward_masks(include_free=False)
        self._frozen_reference = snapshot_weights(self.policy)

        retrain_steps = budget.training - step
        report = self.book.capacity_report()
        print(
            f"\n*** PackNet PRUNE at step {step}/{budget.training}: "
            f"committed {committed_total:,} weights to slice {self.active_slice}, "
            f"released {released_total:,} back to the pool. "
            f"{report['owned_fraction']:.1%} of the network now owned. "
            f"{retrain_steps:,} retrain steps remain INSIDE this task's budget ***\n"
        )

    def before_optimizer_step(self) -> None:
        """Zero the gradients of every weight this task must not move."""
        if self.active_slice is None:
            return
        for name, param in prunable_parameters(self.policy, self.EXCLUDE_FROM_PRUNING):
            if param.grad is None:
                continue
            param.grad.masked_fill_(~self._trainable_mask(name), 0.0)

    def after_optimizer_step(self) -> None:
        """Restore protected weights bit for bit.

        This, not the gradient masking above, is what makes the isolation real:
        Adam moves parameters whose current gradient is zero whenever it still
        carries momentum for them.
        """
        if self.active_slice is None:
            return
        with torch.no_grad():
            for name, param in prunable_parameters(self.policy, self.EXCLUDE_FROM_PRUNING):
                protected = ~self._trainable_mask(name)
                if not bool(protected.any()):
                    continue
                param.data.copy_(
                    torch.where(protected,
                                self._frozen_reference[name].to(param.device),
                                param.data)
                )

    def on_task_end(self, ctx: TaskContext) -> None:
        """A first encounter that never reached its prune step still has to commit.

        That happens when the budget is tiny, as in the smoke test. Without this
        the task would own nothing and evaluating it later would run an empty
        subnetwork.
        """
        if not self._pruned_this_task and self.active_slice is not None:
            print("*** PackNet: task ended before the prune step; committing now ***")

            class _Budget:
                training = 1

            self._prune(0, _Budget())
        self._rebuild_forward_masks(include_free=False)

    # ------------------------------------------------------------------
    # Snapshot contract
    # ------------------------------------------------------------------
    def shared_encoder(self) -> nn.Module:
        """A copy of the encoder with the active task's mask folded in.

        export_snapshot reads a state_dict, so the mask has to be baked into
        the tensors rather than applied at call time.
        """
        encoder = copy.deepcopy(self.policy.fc)
        with torch.no_grad():
            for index, layer in enumerate(encoder):
                if not isinstance(layer, nn.Linear):
                    continue
                mask = self.policy.forward_masks.get(f"fc.{index}.weight")
                if mask is not None:
                    layer.weight.mul_(mask.to(layer.weight.device))
        return encoder

    def effective_head_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self.active_slice is None:
            raise RuntimeError("no active slice; on_task_start must run first")
        return {
            "mean": self.policy.mean_head.masked_parameters(
                self.policy.forward_masks, "mean_head"),
            "logstd": self.policy.logstd_head.masked_parameters(
                self.policy.forward_masks, "logstd_head"),
        }

    # ------------------------------------------------------------------
    # Chain state
    # ------------------------------------------------------------------
    def extra_state(self) -> Dict[str, Any]:
        state = super().extra_state()
        state["ownership"] = self.book.state_dict()
        state["keep_fraction"] = self.keep_fraction
        return state

    def load_extra_state(self, state: Dict[str, Any]) -> None:
        super().load_extra_state(state)
        stored_keep = float(state.get("keep_fraction", self.keep_fraction))
        if abs(stored_keep - self.keep_fraction) > 1e-9:
            raise ValueError(
                f"checkpoint was trained with packnet_keep_fraction={stored_keep} "
                f"but this run uses {self.keep_fraction}; the capacity split "
                "would not line up. Start a fresh chain."
            )
        ownership = state.get("ownership")
        if ownership is None:
            raise ValueError("PackNet chain state carries no ownership map")
        self.book.load_state_dict(ownership)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def scalars(self) -> Dict[str, float]:
        report = self.book.capacity_report()
        return {
            "owned_fraction": report["owned_fraction"],
            "free_weights": report["free_weights"],
            "active_slice": float(-1 if self.active_slice is None else self.active_slice),
            "pruned": float(self._pruned_this_task),
        }
