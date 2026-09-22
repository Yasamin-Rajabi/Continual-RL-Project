"""Bounded knowledge-pool storage for one categorical Atari policy head."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init

BASE_FUSION_MODE = "classic_cka"
_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


def balanced_lineage_indices(source_ids, max_rows: int):
    """Sample up to max_rows as evenly as possible across source lineages."""
    source_ids = np.asarray(source_ids).reshape(-1)
    n_rows = int(source_ids.shape[0])
    target = min(max(int(max_rows), 0), n_rows)
    if target == 0:
        return np.empty((0,), dtype=np.int64)

    unique_ids = np.unique(source_ids)
    groups = {sid: np.flatnonzero(source_ids == sid) for sid in unique_ids}
    allocations = {sid: 0 for sid in unique_ids}
    remaining = target
    active = list(unique_ids)

    while remaining > 0 and active:
        active = list(np.random.permutation(active))
        share = remaining // len(active)
        if share == 0:
            for sid in active[:remaining]:
                allocations[sid] += 1
            remaining = 0
            break
        next_active = []
        used = 0
        for sid in active:
            capacity = len(groups[sid]) - allocations[sid]
            take = min(capacity, share)
            allocations[sid] += take
            used += take
            if allocations[sid] < len(groups[sid]):
                next_active.append(sid)
        remaining -= used
        active = next_active
        if used == 0:
            break

    chosen = []
    for sid in unique_ids:
        take = allocations[sid]
        if take > 0:
            chosen.append(np.random.choice(groups[sid], size=take, replace=False))
    if not chosen:
        return np.empty((0,), dtype=np.int64)
    idx = np.concatenate(chosen).astype(np.int64, copy=False)
    np.random.shuffle(idx)
    return idx


class HeadPool(nn.Module):
    """Discrete counterpart of the HalfCheetah HeadPool: one pool = one policy."""

    FORMAT_VERSION = 4

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
        constrain_alpha_mass: bool = True,
        distill_test_frac: float = 0.2,
    ):
        super().__init__()
        if head_type != "logits":
            raise ValueError("Atari HeadPool requires head_type='logits'")
        if fusion_mode not in (BASE_FUSION_MODE, "weight_delta"):
            raise ValueError(f"unknown fusion_mode={fusion_mode!r}")
        if fusion_mode == BASE_FUSION_MODE and use_alpha_mass:
            raise ValueError(
                "use_alpha_mass is only available with fusion_mode='weight_delta'"
            )

        self.format_version = self.FORMAT_VERSION
        self.force_unit_mass = False
        self.composition_space = "parameter"
        self.head_type = "logits"
        self.shared_dim = int(shared_dim)
        self.hidden_dim = int(hidden_dim)
        self.act_dim = int(act_dim)
        self.fusion_mode = fusion_mode
        self.pool_size = int(pool_size)
        self.distillation = bool(distillation)
        self.max_distill_buffer = int(max_distill_buffer)
        self.use_alpha_mass = bool(use_alpha_mass)
        self.constrain_alpha_mass = bool(constrain_alpha_mass)
        self.distill_test_frac = float(distill_test_frac)

        self.register_buffer("base_l0_weight", torch.zeros(hidden_dim, shared_dim))
        self.register_buffer("base_l0_bias", torch.zeros(hidden_dim))
        self.register_buffer("base_l2_weight", torch.zeros(act_dim, hidden_dim))
        self.register_buffer("base_l2_bias", torch.zeros(act_dim))

        self.own_l0_weight = Parameter(torch.empty(hidden_dim, shared_dim))
        self.own_l0_bias = Parameter(torch.empty(hidden_dim))
        self.own_l2_weight = Parameter(torch.empty(act_dim, hidden_dim))
        self.own_l2_bias = Parameter(torch.empty(act_dim))
        self._reset_own_parameters()

        self.pool = []
        self.own_buffer = None
        self.alpha = None
        self.alpha_scale = None
        self.alpha_mass = None
        self.last_merge_info = None
        self.last_distill_train_kl = None
        self.last_distill_test_kl = None
        self.last_distill_train_mse = None
        self.last_distill_test_mse = None

    def _reset_own_parameters(self):
        init.kaiming_uniform_(self.own_l0_weight, a=math.sqrt(5))
        fan_in0, _ = init._calculate_fan_in_and_fan_out(self.own_l0_weight)
        bound0 = 1 / math.sqrt(fan_in0) if fan_in0 > 0 else 0
        init.uniform_(self.own_l0_bias, -bound0, bound0)
        init.kaiming_uniform_(self.own_l2_weight, a=math.sqrt(5))
        fan_in2, _ = init._calculate_fan_in_and_fan_out(self.own_l2_weight)
        bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
        init.uniform_(self.own_l2_bias, -bound2, bound2)

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        for entry in self.pool:
            for key in _HEAD_KEYS:
                entry[key] = fn(entry[key])
        return self

    def set_alpha(self, alpha, alpha_scale, alpha_mass=None):
        self.alpha = alpha
        self.alpha_scale = alpha_scale
        self.alpha_mass = alpha_mass

    def effective_alpha_mass(self):
        if self.alpha_mass is None:
            return None
        if self.force_unit_mass:
            return torch.ones_like(self.alpha_mass)
        if not self.constrain_alpha_mass:
            return self.alpha_mass
        return torch.sigmoid(self.alpha_mass)

    def _historical(self):
        if not self.pool:
            return {key: 0.0 for key in _HEAD_KEYS}
        if self.alpha is None or self.alpha.numel() != len(self.pool):
            raise RuntimeError(
                f"alpha length does not match pool length: "
                f"alpha={None if self.alpha is None else self.alpha.numel()}, pool={len(self.pool)}"
            )
        scale = 1.0 if self.alpha_scale is None else self.alpha_scale
        weights = F.softmax(self.alpha * scale, dim=0)
        if self.use_alpha_mass and self.alpha_mass is not None:
            weights = self.effective_alpha_mass() * weights

        out = {}
        for name, ndim in (
            ("l0_weight", 2), ("l0_bias", 1), ("l2_weight", 2), ("l2_bias", 1)
        ):
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
    def _forward_with_weights(features, weights):
        w0, b0, w2, b2 = weights
        return F.linear(F.relu(F.linear(features, w0, b0)), w2, b2)

    def forward(self, features):
        return self._forward_with_weights(features, self._effective())

    def entry_effective_weights(self, entry) -> Tuple[torch.Tensor, ...]:
        if self.fusion_mode == BASE_FUSION_MODE:
            return (
                self.base_l0_weight + entry["l0_weight"],
                self.base_l0_bias + entry["l0_bias"],
                self.base_l2_weight + entry["l2_weight"],
                self.base_l2_bias + entry["l2_bias"],
            )
        return tuple(entry[key] for key in _HEAD_KEYS)

    def forward_entry(self, features, entry_index: int):
        return self._forward_with_weights(
            features, self.entry_effective_weights(self.pool[int(entry_index)])
        )

    def inherit_pool_from(self, latest_pool: "HeadPool"):
        if getattr(latest_pool, "format_version", 1) != self.format_version:
            raise RuntimeError(
                "Checkpoint predates the current Atari sigmoid/policy-space pool format. "
                "Start a fresh continual chain."
            )
        if getattr(latest_pool, "head_type", None) != "logits":
            raise RuntimeError("checkpoint is not an Atari categorical logits pool")
        self.pool = []
        for entry in latest_pool.pool:
            copied = {key: entry[key].detach().clone() for key in _HEAD_KEYS}
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

    def replace_pair(self, idx1, idx2, params, merged_buffer, merge_info):
        n = len(self.pool)
        entry = {key: params[key].detach().clone() for key in _HEAD_KEYS}
        entry["buffer"] = merged_buffer
        self.pool = [self.pool[i] for i in range(n) if i not in (idx1, idx2)] + [entry]
        self.last_merge_info = dict(merge_info)

    @staticmethod
    def _trim_buffer(buffer, max_rows):
        if buffer is None:
            return None
        n = len(buffer["obs"])
        if n <= max_rows:
            return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in buffer.items()}
        idx = np.random.choice(n, size=max_rows, replace=False)
        return {
            k: (v[idx] if isinstance(v, np.ndarray) and len(v) == n else v)
            for k, v in buffer.items()
        }

    @staticmethod
    def merge_buffers(buf1, buf2, max_rows: int, balance_source_lineages: bool = False):
        if buf1 is None and buf2 is None:
            return None
        if buf1 is None:
            return HeadPool._trim_buffer(buf2, max_rows)
        if buf2 is None:
            return HeadPool._trim_buffer(buf1, max_rows)

        keys = [
            key for key in buf1
            if key in buf2 and isinstance(buf1[key], np.ndarray)
            and isinstance(buf2[key], np.ndarray)
        ]
        if "obs" not in keys:
            raise ValueError("pool buffers must contain NumPy 'obs'")
        n1, n2 = len(buf1["obs"]), len(buf2["obs"])
        if n1 + n2 <= max_rows:
            return {key: np.concatenate([buf1[key], buf2[key]], axis=0) for key in keys}

        if balance_source_lineages:
            if "source_ids" not in keys:
                raise RuntimeError(
                    "--balance-source-lineages requires source_ids in every retained buffer"
                )
            combined = {key: np.concatenate([buf1[key], buf2[key]], axis=0) for key in keys}
            idx = balanced_lineage_indices(combined["source_ids"], max_rows)
            return {key: combined[key][idx] for key in keys}

        half = max_rows // 2
        take1 = min(n1, half)
        take2 = min(n2, max_rows - take1)
        if take1 + take2 < max_rows:
            take1 = min(n1, take1 + (max_rows - take1 - take2))
        if take1 + take2 < max_rows:
            take2 = min(n2, take2 + (max_rows - take1 - take2))
        idx1 = np.random.choice(n1, size=take1, replace=False)
        idx2 = np.random.choice(n2, size=take2, replace=False)
        return {
            key: np.concatenate([buf1[key][idx1], buf2[key][idx2]], axis=0)
            for key in keys
        }
