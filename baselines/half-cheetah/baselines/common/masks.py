"""Pruning, parameter-isolation bookkeeping and task-conditioned gating.

Shared by PackNet (binary masks over weights) and MaskNet (gates over units).

THE OWNERSHIP TENSOR
--------------------
PackNet's bookkeeping is one ``int16`` tensor per prunable parameter, the same
shape as the parameter:

    owner[i] == FREE (0)   weight i belongs to no task yet and is trainable
    owner[i] == k   (>0)   weight i was committed to capacity slice k and is
                           frozen forever

A slice index is ``task_slice + 1`` so that slice 0 is distinguishable from
FREE. ``num_owned``/``num_free`` report capacity pressure, which matters here:
the head is 256 -> hidden_dim -> act_dim, and with 6 unique HalfCheetah tasks
or 4 Meta-World tasks a per-task prune ratio that is too aggressive exhausts
the free pool before the sequence ends. ``capacity_report`` surfaces that as a
logged scalar rather than as a silent failure to learn.

RECURRING TASKS
---------------
Both sequences revisit every task. A second encounter of task t must REUSE
slice t's existing mask and retrain only within it, never allocate fresh free
weights. ``select_existing`` is what enforces that; see packnet.py.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

FREE = 0


def prunable_parameters(module: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """Return the weight tensors PackNet/MaskNet operate on.

    Biases and 1-D parameters are excluded: pruning them buys almost no
    capacity and destabilizes the head far more than it saves.
    """
    result = []
    for name, param in module.named_parameters():
        if param.dim() >= 2:
            result.append((name, param))
    return result


class OwnershipBook:
    """Tracks which capacity slice owns each weight of each parameter."""

    def __init__(self, module: nn.Module):
        self._owner: Dict[str, torch.Tensor] = {
            name: torch.full_like(param, FREE, dtype=torch.int16)
            for name, param in prunable_parameters(module)
        }

    # -- state -------------------------------------------------------
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu().clone()
                for name, tensor in self._owner.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor], device=None) -> None:
        missing = set(self._owner) - set(state)
        if missing:
            raise ValueError(f"ownership state is missing parameters: {sorted(missing)[:8]}")
        for name in self._owner:
            tensor = state[name].to(dtype=torch.int16)
            if tensor.shape != self._owner[name].shape:
                raise ValueError(
                    f"ownership tensor for {name} has shape {tuple(tensor.shape)}, "
                    f"expected {tuple(self._owner[name].shape)}"
                )
            self._owner[name] = tensor if device is None else tensor.to(device)

    def to(self, device):
        self._owner = {name: tensor.to(device) for name, tensor in self._owner.items()}
        return self

    def names(self) -> Iterable[str]:
        return self._owner.keys()

    # -- queries -----------------------------------------------------
    def free_mask(self, name: str) -> torch.Tensor:
        """Boolean mask of weights owned by nobody."""
        return self._owner[name] == FREE

    def owned_by(self, name: str, slice_index: int) -> torch.Tensor:
        """Boolean mask of weights committed to ``slice_index``."""
        return self._owner[name] == (int(slice_index) + 1)

    def frozen_mask(self, name: str, *, active_slice: int | None) -> torch.Tensor:
        """Weights this task must NOT move: owned by any OTHER slice.

        When ``active_slice`` is None (first pass over a task, still training in
        the free pool), every owned weight is frozen. When retraining a task
        that already owns a slice, that slice's own weights stay trainable.
        """
        owner = self._owner[name]
        frozen = owner != FREE
        if active_slice is not None:
            frozen = frozen & (owner != (int(active_slice) + 1))
        return frozen

    def trainable_mask(self, name: str, *, active_slice: int | None) -> torch.Tensor:
        return ~self.frozen_mask(name, active_slice=active_slice)

    # -- mutation ----------------------------------------------------
    def commit(self, name: str, keep: torch.Tensor, slice_index: int) -> int:
        """Assign the currently-free weights selected by ``keep`` to a slice.

        Only free weights can be committed; a weight already owned by another
        task is never reassigned, which is what makes the isolation permanent.
        Returns how many weights were committed.
        """
        free = self.free_mask(name)
        newly = keep & free
        self._owner[name] = torch.where(
            newly,
            torch.full_like(self._owner[name], int(slice_index) + 1),
            self._owner[name],
        )
        return int(newly.sum())

    def release_free(self, name: str) -> torch.Tensor:
        """Boolean mask of weights that are still free after a commit."""
        return self.free_mask(name)

    # -- reporting ---------------------------------------------------
    def capacity_report(self) -> Dict[str, float]:
        owned = sum(int((tensor != FREE).sum()) for tensor in self._owner.values())
        total = sum(int(tensor.numel()) for tensor in self._owner.values())
        return {
            "owned_weights": float(owned),
            "free_weights": float(total - owned),
            "owned_fraction": float(owned) / float(max(total, 1)),
        }


def magnitude_keep_mask(param: torch.Tensor, candidate: torch.Tensor,
                        keep_fraction: float) -> torch.Tensor:
    """Select the top ``keep_fraction`` of ``candidate`` weights by |value|.

    ``candidate`` is the boolean mask of weights eligible for selection (the
    free pool). Returns a boolean mask, a subset of ``candidate``.

    Ties are broken by ``torch.topk``'s ordering, which is deterministic for a
    fixed input, so a rerun with the same seed produces the same mask.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
    eligible = int(candidate.sum())
    if eligible == 0:
        return torch.zeros_like(candidate)
    keep_count = max(1, int(round(eligible * float(keep_fraction))))
    keep_count = min(keep_count, eligible)

    scores = param.detach().abs().masked_fill(~candidate, float("-inf"))
    flat_indices = torch.topk(scores.reshape(-1), keep_count).indices
    keep = torch.zeros_like(candidate.reshape(-1))
    keep[flat_indices] = True
    return keep.reshape(candidate.shape)


def apply_masked_gradients(module: nn.Module, book: OwnershipBook,
                           *, active_slice: int | None) -> None:
    """Zero the gradient of every weight frozen for the current task."""
    for name, param in prunable_parameters(module):
        if param.grad is None:
            continue
        frozen = book.frozen_mask(name, active_slice=active_slice)
        param.grad.masked_fill_(frozen, 0.0)


def restore_frozen_weights(module: nn.Module, book: OwnershipBook,
                           reference: Dict[str, torch.Tensor],
                           *, active_slice: int | None) -> None:
    """Rewrite every frozen weight back to its committed value.

    Masking gradients is not sufficient under Adam: a parameter whose current
    gradient is zero still moves if its momentum buffers are non-zero. Copying
    the reference value back after ``optimizer.step()`` is what actually makes
    parameter isolation exact, and it is cheap relative to the SAC update.
    """
    with torch.no_grad():
        for name, param in prunable_parameters(module):
            frozen = book.frozen_mask(name, active_slice=active_slice)
            if not bool(frozen.any()):
                continue
            param.data = torch.where(frozen, reference[name].to(param.device), param.data)


def snapshot_weights(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone the prunable weights, for use as the frozen reference."""
    return {name: param.detach().clone()
            for name, param in prunable_parameters(module)}


# ======================================================================
# MaskNet gating
# ======================================================================
class TaskGates(nn.Module):
    """One learned gate vector per capacity slice, over a hidden layer.

    MaskNet's subnetwork selection: the gate multiplies the hidden activations,
    so each task learns which units of a single shared backbone it uses. Gates
    are real-valued during training and passed through a sigmoid, which keeps
    the whole thing differentiable; ``binarize`` thresholds them for reporting
    and for the exported snapshot.

    ``num_slices`` grows as unseen tasks arrive. A recurring task selects its
    existing gate rather than adding one, so gate count equals the number of
    UNIQUE tasks, not the sequence length.
    """

    def __init__(self, num_slices: int, width: int, *, init_bias: float = 2.0):
        super().__init__()
        if num_slices < 1 or width < 1:
            raise ValueError("num_slices and width must be >= 1")
        self.width = int(width)
        # Positive init so every unit starts roughly open and the task has to
        # learn what to switch off. Starting near zero silences the backbone
        # and the task never gets a learning signal.
        self.logits = nn.Parameter(torch.full((int(num_slices), int(width)), float(init_bias)))

    @property
    def num_slices(self) -> int:
        return int(self.logits.shape[0])

    def grow(self, additional: int = 1, *, init_bias: float = 2.0) -> None:
        """Append ``additional`` new gate rows, preserving existing ones."""
        if additional < 1:
            return
        with torch.no_grad():
            extra = torch.full(
                (int(additional), self.width), float(init_bias),
                device=self.logits.device, dtype=self.logits.dtype,
            )
            grown = torch.cat([self.logits.data, extra], dim=0)
        self.logits = nn.Parameter(grown)

    def gate(self, slice_index: int) -> torch.Tensor:
        if not 0 <= int(slice_index) < self.num_slices:
            raise IndexError(
                f"gate slice {slice_index} out of range for {self.num_slices} slices"
            )
        return torch.sigmoid(self.logits[int(slice_index)])

    def binarize(self, slice_index: int, threshold: float = 0.5) -> torch.Tensor:
        return (self.gate(slice_index) > float(threshold)).to(self.logits.dtype)

    def occupancy(self, slice_index: int, threshold: float = 0.5) -> float:
        """Fraction of units this task keeps open. Logged as a diagnostic."""
        return float(self.binarize(slice_index, threshold).mean())
