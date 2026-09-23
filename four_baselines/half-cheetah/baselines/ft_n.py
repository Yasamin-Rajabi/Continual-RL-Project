"""FT-N: sequential fine-tuning across N tasks.

WHAT THIS IS
------------
One network. Train it on task 0, carry every weight forward, train it on task
1, and so on to the end of the sequence. No capacity isolation, no
regularization, no replay, no task input. Whatever the last task needed is
what the weights encode by the end.

This is the naive lower bound the continual-RL literature calls Finetuning,
and its job in the paper is to establish what catastrophic forgetting costs on
these two sequences when nothing at all is done about it. Expect A_N near the
final task's solo performance, strongly negative BWT, and FG close to the full
per-task score for everything except the last few positions.

TASK-BLIND BY DESIGN
--------------------
Per the approved experiment design, FT-N does NOT receive the task oracle. It
is the one baseline that faces exactly the information set the CKA-RL method
faces: a 17-D (HalfCheetah) or 39-D (Meta-World) observation with no task
identifier anywhere in it, and a task boundary it is never told about.

That makes FT-N the only strictly like-for-like comparison in the baseline set.
ProgNet, PackNet and MaskNet all get the oracle and are reported as Task-Aware
Upper Bounds. When writing the results section, FT-N is the lower bracket and
those three are the upper bracket; the method sits between them WITHOUT the
oracle, which is the claim.

ARCHITECTURE PARITY
-------------------
Identical shape to the method's actor so a difference in results is never a
difference in capacity:

    encoder  shared_arch.shared(obs_dim)      Linear(obs,256) ReLU Linear(256,256) [ReLU]
    mean     Linear(256,hidden) ReLU Linear(hidden,act)
    logstd   Linear(256,hidden) ReLU Linear(hidden,act)

with hidden_dim=128, matching CkaRlAgent's default. The encoder is trainable
throughout: freezing it would make this a different, gentler baseline than
naive fine-tuning, and the point of FT-N is that nothing protects anything.

WHAT CARRIES ACROSS A TASK BOUNDARY
-----------------------------------
Everything. Encoder, both heads and the shared critic are all restored from
the previous task's ``agent_state.pt``. The replay buffer does not carry over,
which matches run_sac.py exactly: SB3's ReplayBuffer is constructed fresh in
each subprocess and dies with it.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from baselines.common.lifecycle import ContinualAgent, TaskContext
from shared_arch import shared


class GaussianHead(nn.Module):
    """Two-layer head, shape-identical to one ``knowledge_pools.HeadPool`` slot.

    Parameter names match ``cka_rl._HEAD_KEYS`` so ``effective_head_parameters``
    is a direct read with no renaming, and the exported snapshot is exactly what
    ``FrozenCkaPolicy._head`` evaluates:

        h = relu(linear(z, l0_weight, l0_bias))
        out = linear(h, l2_weight, l2_bias)
    """

    def __init__(self, in_dim: int, hidden_dim: int, act_dim: int):
        super().__init__()
        self.l0 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, act_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.relu(self.l0(z)))

    def head_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            "l0_weight": self.l0.weight,
            "l0_bias": self.l0.bias,
            "l2_weight": self.l2.weight,
            "l2_bias": self.l2.bias,
        }


class FineTunePolicy(nn.Module):
    """Encoder plus mean and log-std heads.

    ``forward`` returns RAW, unbounded log-std. ``policy_composition.components``
    applies ``bound_log_std`` itself, exactly as it does for ``CkaRlAgent``;
    bounding here would apply the tanh squash twice.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int,
                 encoder_linear_out: bool):
        super().__init__()
        self.fc = shared(obs_dim, linear_out=encoder_linear_out)
        self.mean_head = GaussianHead(256, hidden_dim, act_dim)
        self.logstd_head = GaussianHead(256, hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor):
        z = self.fc(obs)
        return self.mean_head(z), self.logstd_head(z)


class FineTuneAgent(ContinualAgent):
    """Sequential fine-tuning. The naive lower bound."""

    task_aware = False

    def __init__(self, obs_dim: int, act_dim: int, *, hidden_dim: int = 128,
                 encoder_linear_out: bool = False, **unused):
        # **unused absorbs the method-specific knobs run_baseline.py passes to
        # every constructor (prune fractions, gate init, ...). FT-N has none by
        # definition: the whole point is that it does nothing special.
        super().__init__(
            obs_dim, act_dim,
            hidden_dim=hidden_dim,
            encoder_linear_out=encoder_linear_out,
        )
        self.policy = self.build_policy()
        self.assert_constructed()

    def build_policy(self) -> nn.Module:
        return FineTunePolicy(
            self.obs_dim, self.act_dim, self.hidden_dim, self.encoder_linear_out
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_task_start(self, ctx: TaskContext) -> None:
        """Nothing to allocate: one network serves every task.

        ``active_task`` is recorded for logging only. The forward pass never
        reads it, which is what makes this baseline task-blind, and the
        assertion below is what keeps that honest if the class is ever edited.
        """
        assert not self.task_aware, "FT-N must stay task-blind"
        self.active_task = int(ctx.task_id)
        # Bookkeeping so num_allocated_slices reports unique tasks seen. This
        # is never consumed by the forward pass; it only makes the FT-N run
        # manifest comparable with the task-aware baselines.
        if self.slice_for_task(ctx.task_id) is None:
            self.allocate_slice(ctx.task_id, self.num_allocated_slices())

    def on_task_end(self, ctx: TaskContext) -> None:
        """Nothing to commit: the weights are already what they are."""
        return None

    # ------------------------------------------------------------------
    # Snapshot contract
    # ------------------------------------------------------------------
    def shared_encoder(self) -> nn.Module:
        return self.policy.fc

    def effective_head_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "mean": self.policy.mean_head.head_parameters(),
            "logstd": self.policy.logstd_head.head_parameters(),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def scalars(self) -> Dict[str, float]:
        return {"unique_tasks_seen": float(self.num_allocated_slices())}
