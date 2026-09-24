"""Continual-RL baseline agents for PointMaze (SAC, squashed-Gaussian actors).

Each class is the continuous-control counterpart of the project's Atari
baseline of the same name.  Where the Atari version used a CNN encoder and a
categorical head, these use the shared MLP encoder and a Gaussian head; the
continual mechanism in each case is unchanged.

    FT-N      fine-tune everything, one output head per task
    ProgNet   one frozen column per task plus lateral adapters
    PackNet   iterative magnitude pruning into per-task subnetworks
    MaskNet   fixed random weights, learned per-task supermasks
    CReLUs    CReLU activations to preserve plasticity
    CompoNet  attention-based composition of frozen previous policies
    CbpNet    continual backprop (generate-and-test on hidden units)
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared_arch import layer_init

from .common import BaselineAgent, GaussianHead
from .cbp_modules import CbpGaussianActor, GnT
from .componet_modules import CompoNet, FirstModuleWrapper, Identity
from .mask_modules import (
    NEW_MASK_LINEAR_COMB,
    MultitaskMaskLinear,
    consolidate_mask,
    set_model_task,
    set_num_tasks_learned,
)


# ---------------------------------------------------------------------------
# FT-N
# ---------------------------------------------------------------------------
class FtNAgent(BaselineAgent):
    """Fine-tune the whole network, with one output head per task.

    This is the strongest naive baseline and the one the task ordering is
    hardest on: it carries forward exactly one solution, and in this suite the
    most recent task is never the closest one to the task starting.
    """

    def __init__(self, obs_dim, act_dim, num_tasks, shared_dim=256, hidden_dim=256):
        super().__init__(obs_dim, act_dim, shared_dim, hidden_dim)
        self.num_tasks = int(num_tasks)
        self.heads = nn.ModuleList(
            [GaussianHead(shared_dim, hidden_dim, act_dim) for _ in range(self.num_tasks)]
        )
        self.task_slot = 0

    def set_task(self, task_slot: int, new_task: bool = False):
        if not 0 <= int(task_slot) < self.num_tasks:
            raise ValueError(f"task_slot {task_slot} outside 0..{self.num_tasks - 1}")
        self.task_slot = int(task_slot)

    def actor_raw(self, features):
        return self.heads[self.task_slot](features)


# ---------------------------------------------------------------------------
# CReLUs
# ---------------------------------------------------------------------------
class CReLU(nn.Module):
    """CReLU(x) = [ReLU(x), ReLU(-x)]; doubles the feature width."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((F.relu(x), F.relu(-x)), dim=-1)


class CReLUsAgent(BaselineAgent):
    """Fine-tuning with CReLU activations, which slow plasticity loss."""

    def __init__(self, obs_dim, act_dim, shared_dim=256, hidden_dim=256):
        super().__init__(obs_dim, act_dim, shared_dim, hidden_dim)
        half = int(hidden_dim) // 2
        self.actor = nn.Sequential(
            layer_init(nn.Linear(shared_dim, half)),
            CReLU(),
            layer_init(nn.Linear(2 * half, 2 * int(act_dim)), std=0.01),
        )

    def actor_raw(self, features):
        return self.actor(features)


# ---------------------------------------------------------------------------
# CbpNet
# ---------------------------------------------------------------------------
class CbpNetAgent(BaselineAgent):
    """Continual backprop: low-utility hidden units are regenerated."""

    def __init__(
        self,
        obs_dim,
        act_dim,
        shared_dim=256,
        hidden_dim=256,
        replacement_rate=1e-4,
        maturity_threshold=100,
        decay_rate=0.99,
    ):
        super().__init__(obs_dim, act_dim, shared_dim, hidden_dim)
        self.actor = CbpGaussianActor(shared_dim, hidden_dim, act_dim)
        self.replacement_rate = float(replacement_rate)
        self.maturity_threshold = int(maturity_threshold)
        self.decay_rate = float(decay_rate)
        self._gnt = None

    def attach_gnt(self, optimizer, device):
        self._gnt = GnT(
            self.actor,
            optimizer,
            decay_rate=self.decay_rate,
            replacement_rate=self.replacement_rate,
            maturity_threshold=self.maturity_threshold,
            device=device,
        )

    def actor_raw(self, features):
        return self.actor(features)

    def after_update(self):
        if self._gnt is not None:
            self._gnt.step()

    def save(self, dirname):
        # The GnT holds a reference to the live optimizer, which must not be
        # pickled into the checkpoint.
        gnt, self._gnt = self._gnt, None
        try:
            super().save(dirname)
        finally:
            self._gnt = gnt


# ---------------------------------------------------------------------------
# PackNet
# ---------------------------------------------------------------------------
class PackNetAgent(BaselineAgent):
    """Iterative magnitude pruning into disjoint per-task subnetworks.

    Capacity rule (``capacity_mode``)
    ---------------------------------
    ``equal_share`` (default)
        Every task reserves the same slice of the whole network:
        ``keep_frac / total_tasks``.  With ``keep_frac=1.0`` and ten tasks that
        is 10% each and the network ends fully used.  This is the most generous
        allocation for the *late* tasks, which is where a ten-task chain hurts
        a fixed-capacity method.

    ``geometric``
        Textbook PackNet: keep ``keep_frac`` of whatever is still free.  The
        free pool decays as ``(1-keep_frac)^k``, so a large ``keep_frac`` over
        ten tasks starves the tail badly -- at 0.75 the tenth task receives
        zero weights.  Provided for faithfulness, not recommended at N=10.

    Use ``capacity_report()`` to see the exact per-task weight counts before
    committing to a run.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        task_slot,
        total_tasks,
        shared_dim=256,
        hidden_dim=256,
        capacity_mode: str = "equal_share",
        keep_frac: float = 1.0,
    ):
        super().__init__(obs_dim, act_dim, shared_dim, hidden_dim)
        self.task_slot = int(task_slot)
        self.total_tasks = int(total_tasks)
        if capacity_mode not in ("equal_share", "geometric"):
            raise ValueError(f"unknown capacity_mode={capacity_mode!r}")
        if not 0.0 < float(keep_frac) <= 1.0:
            raise ValueError("keep_frac must be in (0, 1]")
        self.capacity_mode = str(capacity_mode)
        self.keep_frac = float(keep_frac)
        self.retrain_mode = False
        self.view = None
        self._saved_params = []

        self.actor = GaussianHead(shared_dim, hidden_dim, act_dim)
        # Masks cover the encoder and the actor: the weights this method packs.
        self.masks = [
            torch.zeros_like(p, dtype=torch.long) for p in self._maskable_parameters()
        ]
        if self.task_slot > 0:
            for name, param in self._named_maskable():
                if name.endswith("bias"):
                    param.requires_grad = False

    def _named_maskable(self):
        out = []
        for prefix, module in (("encoder", self.encoder), ("actor", self.actor)):
            for name, param in module.named_parameters():
                out.append((f"{prefix}.{name}", param))
        return out

    def _maskable_parameters(self):
        return [p for _, p in self._named_maskable()]

    def actor_raw(self, features):
        return self.actor(features)

    def _weight_mask_pairs(self):
        for (name, param), mask in zip(self._named_maskable(), self.masks):
            if name.endswith("weight"):
                yield param, mask

    @torch.no_grad()
    def before_update(self):
        """Zero the gradient of every weight this task is not allowed to touch."""
        mask_id = self.task_slot + 1 if self.retrain_mode else 0
        for param, mask in self._weight_mask_pairs():
            if param.grad is not None:
                param.grad = param.grad * (mask == mask_id)

    @torch.no_grad()
    def prune(self):
        """Reserve this task's allocation from the weights that are still free.

        Selection is restricted to genuinely free positions.  Ranking a tensor
        that has had reserved weights zeroed out would, once the free pool is
        smaller than the allocation, start handing out positions that earlier
        tasks already own -- silently destroying those tasks instead of just
        running out of room.
        """
        for param, mask in self._weight_mask_pairs():
            flat_mask = mask.flatten()
            free_idx = torch.nonzero(flat_mask == 0, as_tuple=False).flatten()
            if free_idx.numel() == 0:
                continue
            n_keep = min(self._allocation(param.numel(), free_idx.numel()), free_idx.numel())
            if n_keep <= 0:
                continue
            magnitudes = param.flatten()[free_idx].abs()
            order = torch.argsort(magnitudes, descending=True)
            flat_mask[free_idx[order[:n_keep]]] = self.task_slot + 1

    def _allocation(self, total_weights: int, free_weights: int) -> int:
        """How many weights this task reserves, under the configured rule."""
        if self.capacity_mode == "equal_share":
            # Every task reserves the same slice of the whole network.  With N
            # tasks and keep_frac=1.0 that is exactly 1/N each and the network
            # is fully used; keep_frac<1 leaves the rest permanently free.
            return int(self.keep_frac * total_weights / float(self.total_tasks))
        if self.capacity_mode == "geometric":
            # Classic PackNet: keep a fraction of whatever is still free.
            return int(self.keep_frac * free_weights)
        raise ValueError(f"unknown capacity_mode={self.capacity_mode!r}")

    def capacity_report(self) -> dict:
        """What each task would receive under the current rule; for logging."""
        total = sum(p.numel() for p, _ in self._weight_mask_pairs())
        free = total
        per_task = []
        for _ in range(self.total_tasks):
            take = min(self._allocation(total, free), free)
            per_task.append(int(take))
            free -= take
        return {
            "capacity_mode": self.capacity_mode,
            "keep_frac": float(self.keep_frac),
            "total_maskable_weights": int(total),
            "per_task_weights": per_task,
            "last_task_weights": per_task[-1] if per_task else 0,
            "last_task_fraction": (per_task[-1] / total) if per_task and total else 0.0,
            "fraction_of_network_used": (total - free) / total if total else 0.0,
        }

    def start_retraining(self):
        if self.retrain_mode:
            return
        self.retrain_mode = True
        self.prune()
        self.set_view(self.task_slot + 1)

    @torch.no_grad()
    def set_view(self, task_view):
        """Zero every weight outside the allocation of tasks <= ``task_view``.

        ``task_view=None`` means "no restriction, restore the full network".
        It must be idempotent: on_task_end() and save() both call
        set_view(None) back to back, and the second call must be a no-op
        rather than falling into the "set a view" branch with a None target
        (which previously crashed on ``mask <= None``).
        """
        if task_view is None:
            if self.view is not None:
                for saved, (param, mask) in zip(self._saved_params, self._weight_mask_pairs()):
                    keep = torch.logical_and(mask <= self.view, mask > 0)
                    param.data += saved.data * torch.logical_not(keep)
                self._saved_params = []
                self.view = None
            return

        if not self._saved_params:
            self._saved_params = [
                copy.deepcopy(param) for param, _ in self._weight_mask_pairs()
            ]
        for param, mask in self._weight_mask_pairs():
            param.data *= torch.logical_and(mask <= task_view, mask > 0)
        self.view = task_view

    def on_task_end(self):
        # Restore the free weights so the next task starts from a full network
        # with only the reserved allocations locked.
        self.set_view(None)

    def save(self, dirname):
        self.set_view(None)
        super().save(dirname)


# ---------------------------------------------------------------------------
# MaskNet
# ---------------------------------------------------------------------------
class MaskNetAgent(BaselineAgent):
    """Fixed random weights; a learned binary supermask per task."""

    def __init__(self, obs_dim, act_dim, num_tasks, shared_dim=256, hidden_dim=256):
        super().__init__(obs_dim, act_dim, shared_dim, hidden_dim)
        self.num_tasks = int(num_tasks)
        self.actor = nn.Sequential(
            MultitaskMaskLinear(
                shared_dim, hidden_dim, discrete=True,
                num_tasks=self.num_tasks, new_mask_type=NEW_MASK_LINEAR_COMB,
            ),
            nn.ReLU(),
            MultitaskMaskLinear(
                hidden_dim, 2 * int(act_dim), discrete=True,
                num_tasks=self.num_tasks, new_mask_type=NEW_MASK_LINEAR_COMB,
            ),
        )
        self.task_slot = 0
        set_model_task(self, 0, new_task=True)

    def set_task(self, task_slot: int, new_task: bool = False):
        self.task_slot = int(task_slot)
        set_model_task(self, self.task_slot, new_task=new_task)

    def set_num_tasks_learned(self, n: int):
        set_num_tasks_learned(self, int(n))

    def actor_raw(self, features):
        return self.actor(features)

    def on_task_end(self):
        consolidate_mask(self)
        set_num_tasks_learned(self, self.task_slot + 1)


# ---------------------------------------------------------------------------
# ProgNet
# ---------------------------------------------------------------------------
class ProgressiveColumn(nn.Module):
    """One column: two hidden layers plus lateral adapters from earlier columns."""

    def __init__(self, obs_dim, hidden_dim, out_dim, n_prev: int):
        super().__init__()
        self.l1 = layer_init(nn.Linear(int(obs_dim), int(hidden_dim)))
        self.l2 = layer_init(nn.Linear(int(hidden_dim), int(hidden_dim)))
        self.u2 = nn.ModuleList(
            [layer_init(nn.Linear(hidden_dim, hidden_dim, bias=False)) for _ in range(n_prev)]
        )
        self.out = layer_init(nn.Linear(int(hidden_dim), int(out_dim)), std=0.01)
        self.uo = nn.ModuleList(
            [layer_init(nn.Linear(hidden_dim, out_dim, bias=False)) for _ in range(n_prev)]
        )
        self.act = nn.ReLU()

    def forward_first(self, x):
        h1 = self.act(self.l1(x))
        h2 = self.act(self.l2(h1))
        return self.out(h2), h1, h2

    def forward_other(self, x, h1s, h2s):
        if len(h1s) != len(self.u2):
            raise AssertionError(
                "number of previous activations does not match the adapter count"
            )
        h1 = self.act(self.l1(x))
        h2 = self.act(self.l2(h1) + sum(u(prev) for u, prev in zip(self.u2, h1s)))
        out = self.out(h2) + sum(u(prev) for u, prev in zip(self.uo, h2s))
        return out, h1, h2


class ProgNetAgent(BaselineAgent):
    """A frozen column per previous task, plus a trainable current column."""

    def __init__(self, obs_dim, act_dim, previous_columns=None, shared_dim=256, hidden_dim=256):
        super().__init__(obs_dim, act_dim, shared_dim, hidden_dim)
        # The generic encoder is unused: ProgNet's representation lives in the
        # columns.  It is removed so its parameters never reach an optimizer.
        self.encoder = nn.Identity()

        previous_columns = list(previous_columns or [])
        for column in previous_columns:
            column.eval()
            for p in column.parameters():
                p.requires_grad = False
        self.previous_columns = nn.ModuleList(previous_columns)
        self.column = ProgressiveColumn(
            obs_dim, hidden_dim, 2 * int(act_dim), n_prev=len(previous_columns)
        )
        # The critic consumes the current column's second hidden layer.
        from shared_arch import TwinCritic

        self.critic = TwinCritic(hidden_dim, act_dim, hidden_dim)
        self.critic_target = TwinCritic(hidden_dim, act_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

    def _run_columns(self, obs):
        if not self.previous_columns:
            return self.column.forward_first(obs)
        with torch.no_grad():
            out, h1, h2 = self.previous_columns[0].forward_first(obs)
            h1s, h2s = [h1], [h2]
            for column in self.previous_columns[1:]:
                out, h1, h2 = column.forward_other(obs, h1s, h2s)
                h1s.append(h1)
                h2s.append(h2)
        return self.column.forward_other(obs, h1s, h2s)

    def encode(self, obs):
        # Features for the critic; the actor output is produced alongside.
        return self._run_columns(obs)[2]

    def actor_raw(self, features):  # pragma: no cover - not the ProgNet path
        raise RuntimeError("ProgNet produces its action output together with features")

    def action_distribution(self, obs):
        from .common import raw_to_distribution

        raw, _, _ = self._run_columns(obs)
        return raw_to_distribution(raw, self.act_dim)

    def forward_actor_and_features(self, obs):
        raw, _, h2 = self._run_columns(obs)
        return raw, h2

    def current_column(self):
        return self.column


# ---------------------------------------------------------------------------
# CompoNet
# ---------------------------------------------------------------------------
class CompoNetAgent(BaselineAgent):
    """Attention-based composition over frozen previous policy modules."""

    def __init__(self, obs_dim, act_dim, previous_units=None, shared_dim=256, hidden_dim=256):
        super().__init__(obs_dim, act_dim, shared_dim, hidden_dim)
        out_dim = 2 * int(act_dim)
        previous_units = list(previous_units or [])

        if previous_units:
            internal = nn.Sequential(
                layer_init(nn.Linear(shared_dim + hidden_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, out_dim), std=0.01),
            )
            self.unit = CompoNet(
                previous_units=previous_units,
                input_dim=shared_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                internal_policy=internal,
                encoder=None,
            )
            self.first_head = None
        else:
            self.unit = None
            self.first_head = GaussianHead(shared_dim, hidden_dim, act_dim)

    def actor_raw(self, features):
        if self.unit is None:
            return self.first_head(features)
        return self.unit(features)[0]

    def as_previous_unit(self):
        """Return this task's module, frozen, for the next task to compose."""
        if self.unit is None:
            wrapper = FirstModuleWrapper(copy.deepcopy(self.first_head), encoder=None)
            wrapper.is_prev = True
            for p in wrapper.parameters():
                p.requires_grad = False
            return wrapper
        unit = copy.deepcopy(self.unit)
        unit.is_prev = True
        for p in unit.parameters():
            p.requires_grad = False
        return unit
