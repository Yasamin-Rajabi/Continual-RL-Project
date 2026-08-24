"""Aligned knowledge-pool storage for one policy output head.

A CKA-RL policy has two output heads (mean and log-std), but a pool slot is a
*whole policy knowledge item*. Pair selection therefore happens once at the
agent level using both heads jointly. This module only owns the tensors for one
head and applies the pair chosen by :class:`CkaRlAgent`.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init

BASE_FUSION_MODE = "classic_cka"
_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


class HeadPool(nn.Module):
    def __init__(
        self,
        head_type: str,
        shared_dim: int,
        hidden_dim: int,
        act_dim: int,
        fusion_mode: str = BASE_FUSION_MODE,
        pool_size: int = 5,
        distillation: bool = True,
        max_distill_buffer: int = 50_000,
        use_alpha_mass: bool = False,
        distill_test_frac: float = 0.2,
    ):
        super().__init__()
        assert head_type in ("mean", "logstd"), head_type
        assert fusion_mode in (BASE_FUSION_MODE, "weight_delta"), fusion_mode
        if fusion_mode == BASE_FUSION_MODE and use_alpha_mass:
            raise ValueError(
                "use_alpha_mass changes the original CKA weighting rule and is only "
                "available with fusion_mode='weight_delta'."
            )

        self.format_version = 2
        self.head_type = head_type
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.fusion_mode = fusion_mode
        self.pool_size = int(pool_size)
        self.distillation = bool(distillation)
        self.max_distill_buffer = int(max_distill_buffer)
        self.use_alpha_mass = bool(use_alpha_mass)
        self.distill_test_frac = float(distill_test_frac)

        # Frozen task-1 base head.
        self.register_buffer("base_l0_weight", torch.zeros(hidden_dim, shared_dim))
        self.register_buffer("base_l0_bias", torch.zeros(hidden_dim))
        self.register_buffer("base_l2_weight", torch.zeros(act_dim, hidden_dim))
        self.register_buffer("base_l2_bias", torch.zeros(act_dim))

        # Current task's trainable contribution.
        self.own_l0_weight = Parameter(torch.empty(hidden_dim, shared_dim))
        self.own_l0_bias = Parameter(torch.empty(hidden_dim))
        self.own_l2_weight = Parameter(torch.empty(act_dim, hidden_dim))
        self.own_l2_bias = Parameter(torch.empty(act_dim))
        init.kaiming_uniform_(self.own_l0_weight, a=math.sqrt(5))
        fan_in0, _ = init._calculate_fan_in_and_fan_out(self.own_l0_weight)
        bound0 = 1 / math.sqrt(fan_in0) if fan_in0 > 0 else 0
        init.uniform_(self.own_l0_bias, -bound0, bound0)
        init.kaiming_uniform_(self.own_l2_weight, a=math.sqrt(5))
        fan_in2, _ = init._calculate_fan_in_and_fan_out(self.own_l2_weight)
        bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
        init.uniform_(self.own_l2_bias, -bound2, bound2)

        # Entries are aligned across mean/logstd pools by CkaRlAgent.
        self.pool = []
        self.own_buffer = None

        # The SAME alpha objects are attached to both heads by CkaRlAgent.
        self.alpha = None
        self.alpha_scale = None
        self.alpha_mass = None

        self.last_merge_info = None
        self.last_distill_train_kl = None
        self.last_distill_test_kl = None
        # Kept for compatibility with old logging code; the new distillation
        # objective is KL, not independent head MSE.
        self.last_distill_train_mse = None
        self.last_distill_test_mse = None

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        for entry in self.pool:
            for key in _HEAD_KEYS:
                entry[key] = fn(entry[key])
        return self

    # ------------------------------------------------------------------
    # Current policy construction
    # ------------------------------------------------------------------
    def set_alpha(self, alpha, alpha_scale, alpha_mass=None):
        self.alpha = alpha
        self.alpha_scale = alpha_scale
        self.alpha_mass = alpha_mass

    def _historical(self):
        if not self.pool:
            return {key: 0.0 for key in _HEAD_KEYS}
        if self.alpha is None or self.alpha.numel() != len(self.pool):
            raise RuntimeError(
                f"{self.head_type} alpha length does not match pool length: "
                f"alpha={None if self.alpha is None else self.alpha.numel()}, pool={len(self.pool)}"
            )
        scale = 1.0 if self.alpha_scale is None else self.alpha_scale
        weights = F.softmax(self.alpha * scale, dim=0)
        if self.use_alpha_mass and self.alpha_mass is not None:
            weights = self.alpha_mass * weights

        out = {}
        for name, ndim in (("l0_weight", 2), ("l0_bias", 1), ("l2_weight", 2), ("l2_bias", 1)):
            stacked = torch.stack([entry[name] for entry in self.pool], dim=0)
            out[name] = (weights.view((-1,) + (1,) * ndim) * stacked).sum(dim=0)
        return out

    def _effective(self):
        hist = self._historical()
        w0 = self.own_l0_weight + hist["l0_weight"]
        b0 = self.own_l0_bias + hist["l0_bias"]
        w2 = self.own_l2_weight + hist["l2_weight"]
        b2 = self.own_l2_bias + hist["l2_bias"]
        if self.fusion_mode == BASE_FUSION_MODE:
            w0 = self.base_l0_weight + w0
            b0 = self.base_l0_bias + b0
            w2 = self.base_l2_weight + w2
            b2 = self.base_l2_bias + b2
        return w0, b0, w2, b2

    @staticmethod
    def _forward_with_weights(shared_features, weights):
        w0, b0, w2, b2 = weights
        h = F.relu(F.linear(shared_features, w0, b0))
        return F.linear(h, w2, b2)

    def forward(self, shared_features):
        return self._forward_with_weights(shared_features, self._effective())

    def entry_effective_weights(self, entry) -> Tuple[torch.Tensor, ...]:
        """Weights represented by one pool slot when used as a standalone item."""
        if self.fusion_mode == BASE_FUSION_MODE:
            return (
                self.base_l0_weight + entry["l0_weight"],
                self.base_l0_bias + entry["l0_bias"],
                self.base_l2_weight + entry["l2_weight"],
                self.base_l2_bias + entry["l2_bias"],
            )
        return tuple(entry[key] for key in _HEAD_KEYS)

    def forward_entry(self, shared_features, entry_index: int):
        return self._forward_with_weights(
            shared_features, self.entry_effective_weights(self.pool[entry_index])
        )

    # ------------------------------------------------------------------
    # Pool lifecycle
    # ------------------------------------------------------------------
    def inherit_pool_from(self, latest_pool: "HeadPool"):
        if getattr(latest_pool, "format_version", 1) != self.format_version:
            raise RuntimeError(
                "Checkpoint uses the legacy independent-head pool format. "
                "Start a fresh continual chain with this version so behavioral-KL "
                "pair alignment is guaranteed."
            )
        self.pool = []
        for entry in latest_pool.pool:
            copied = {key: entry[key].detach().clone() for key in _HEAD_KEYS}
            # Buffers are NumPy arrays and immutable during optimization; a
            # shallow dict copy avoids duplicating tens of thousands of rows in RAM.
            copied["buffer"] = entry.get("buffer")
            self.pool.append(copied)
        self.base_l0_weight.copy_(latest_pool.base_l0_weight)
        self.base_l0_bias.copy_(latest_pool.base_l0_bias)
        self.base_l2_weight.copy_(latest_pool.base_l2_weight)
        self.base_l2_bias.copy_(latest_pool.base_l2_bias)

    def reset_own_to_zero(self):
        with torch.no_grad():
            self.own_l0_weight.zero_()
            self.own_l0_bias.zero_()
            self.own_l2_weight.zero_()
            self.own_l2_bias.zero_()

    def set_own_buffer(self, buffer):
        self.own_buffer = buffer

    def pool_length(self):
        return len(self.pool)

    def needs_merge(self):
        return len(self.pool) > self.pool_size

    def set_base(self):
        """Turn the trained root policy into theta_base and seed pool slot v1."""
        self.base_l0_weight.copy_(self.own_l0_weight.data)
        self.base_l0_bias.copy_(self.own_l0_bias.data)
        self.base_l2_weight.copy_(self.own_l2_weight.data)
        self.base_l2_bias.copy_(self.own_l2_bias.data)

        if self.fusion_mode == BASE_FUSION_MODE:
            entry = {
                "l0_weight": torch.zeros_like(self.own_l0_weight.data),
                "l0_bias": torch.zeros_like(self.own_l0_bias.data),
                "l2_weight": torch.zeros_like(self.own_l2_weight.data),
                "l2_bias": torch.zeros_like(self.own_l2_bias.data),
            }
        else:
            entry = {
                "l0_weight": self.own_l0_weight.data.clone(),
                "l0_bias": self.own_l0_bias.data.clone(),
                "l2_weight": self.own_l2_weight.data.clone(),
                "l2_bias": self.own_l2_bias.data.clone(),
            }
        entry["buffer"] = self.own_buffer
        self.pool = [entry]
        self.reset_own_to_zero()

    def finalize_own_contribution(self):
        """Insert the current task entry WITHOUT selecting/merging a pair.

        Pair selection is intentionally agent-level so mean and log-std cannot
        accidentally merge different task slots.
        """
        if self.fusion_mode == "weight_delta":
            hist = self._historical()
            entry = {
                "l0_weight": (self.own_l0_weight.data + hist["l0_weight"]).clone(),
                "l0_bias": (self.own_l0_bias.data + hist["l0_bias"]).clone(),
                "l2_weight": (self.own_l2_weight.data + hist["l2_weight"]).clone(),
                "l2_bias": (self.own_l2_bias.data + hist["l2_bias"]).clone(),
            }
        else:
            entry = {
                "l0_weight": self.own_l0_weight.data.clone(),
                "l0_bias": self.own_l0_bias.data.clone(),
                "l2_weight": self.own_l2_weight.data.clone(),
                "l2_bias": self.own_l2_bias.data.clone(),
            }
        entry["buffer"] = self.own_buffer
        self.pool = [entry] + self.pool
        self.reset_own_to_zero()

    def average_pair_params(self, idx1: int, idx2: int) -> Dict[str, torch.Tensor]:
        return {
            key: 0.5 * (self.pool[idx1][key] + self.pool[idx2][key])
            for key in _HEAD_KEYS
        }

    def replace_pair(
        self,
        idx1: int,
        idx2: int,
        params: Dict[str, torch.Tensor],
        merged_buffer,
        merge_info: dict,
    ):
        n = len(self.pool)
        entry = {key: params[key].detach().clone() for key in _HEAD_KEYS}
        entry["buffer"] = merged_buffer
        self.pool = [self.pool[i] for i in range(n) if i not in (idx1, idx2)] + [entry]
        self.last_merge_info = dict(merge_info)

    @staticmethod
    def merge_buffers(buf1, buf2, max_rows: int):
        """Merge two lineage buffers without letting the larger parent dominate.

        If truncation is required, reserve roughly half the budget for each
        parent and only use spare capacity when one parent is too small. The
        same sampled row indices are applied to every stored array.
        """
        if buf1 is None and buf2 is None:
            return None
        if buf1 is None:
            return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in buf2.items()}
        if buf2 is None:
            return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in buf1.items()}

        keys = [
            key for key in buf1.keys()
            if key in buf2 and isinstance(buf1[key], np.ndarray) and isinstance(buf2[key], np.ndarray)
        ]
        if "obs" not in keys:
            raise ValueError("pool buffers must contain a NumPy 'obs' array")
        n1, n2 = len(buf1["obs"]), len(buf2["obs"])
        if n1 + n2 <= max_rows:
            return {key: np.concatenate([buf1[key], buf2[key]], axis=0) for key in keys}

        half = max_rows // 2
        take1 = min(n1, half)
        take2 = min(n2, max_rows - take1)
        # If parent 2 was too small, give its unused budget back to parent 1.
        if take1 + take2 < max_rows:
            take1 = min(n1, take1 + (max_rows - take1 - take2))
        # If parent 1 was too small, give its unused budget to parent 2.
        if take1 + take2 < max_rows:
            take2 = min(n2, take2 + (max_rows - take1 - take2))

        idx1 = np.random.choice(n1, size=take1, replace=False)
        idx2 = np.random.choice(n2, size=take2, replace=False)
        return {
            key: np.concatenate([buf1[key][idx1], buf2[key][idx2]], axis=0)
            for key in keys
        }
