"""Bounded knowledge-pool storage for one categorical policy-logit head.

This is the discrete-action counterpart of the HalfCheetah HeadPool.  The
continual-learning mechanism is intentionally unchanged:

    theta_current = theta_own + sum_k alpha_k v_k

with the classic-CKA root/base added in ``classic_cka`` mode, or reconstructed
stored weights used directly in ``weight_delta`` mode.  Pool capacity,
alpha-weighted reuse, arithmetic merging, distillation replacement, and replay
buffer lineage are all preserved.

The only policy-output change relative to the continuous-action implementation
is that one pool represents the entire categorical policy-logit head rather
than one member of an aligned Gaussian mean/log-std pair.
"""
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


class HeadPool(nn.Module):
    """Bounded storage and reuse of categorical policy-head parameters.

    Parameters
    ----------
    head_type:
        Must be ``"logits"`` for the Atari/discrete-action implementation.
    shared_dim:
        Dimensionality of the CNN encoder feature vector.
    hidden_dim:
        Hidden width of the two-layer policy head.
    act_dim:
        Number of discrete actions / output logits.
    """

    FORMAT_VERSION = 3

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
            raise ValueError(
                f"Atari HeadPool stores one categorical logits head; got head_type={head_type!r}"
            )
        if fusion_mode not in (BASE_FUSION_MODE, "weight_delta"):
            raise ValueError(f"unknown fusion_mode={fusion_mode!r}")
        if fusion_mode == BASE_FUSION_MODE and use_alpha_mass:
            raise ValueError(
                "use_alpha_mass changes the original CKA weighting rule and is only "
                "available with fusion_mode='weight_delta'."
            )
        if int(shared_dim) < 1 or int(hidden_dim) < 1 or int(act_dim) < 1:
            raise ValueError("shared_dim, hidden_dim, and act_dim must all be >= 1")
        if int(pool_size) < 1:
            raise ValueError("pool_size must be >= 1")
        if int(max_distill_buffer) < 1:
            raise ValueError("max_distill_buffer must be >= 1")
        if not 0.0 <= float(distill_test_frac) < 1.0:
            raise ValueError("distill_test_frac must be in [0, 1)")

        self.format_version = self.FORMAT_VERSION
        self.head_type = "logits"
        self.shared_dim = int(shared_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.fusion_mode = fusion_mode
        self.pool_size = int(pool_size)
        self.distillation = bool(distillation)
        self.max_distill_buffer = int(max_distill_buffer)
        self.use_alpha_mass = bool(use_alpha_mass)
        self.constrain_alpha_mass = bool(constrain_alpha_mass)
        self.distill_test_frac = float(distill_test_frac)

        # Frozen task-1 base policy-logit head.  In classic_cka this is added to
        # the current residual/history.  weight_delta keeps the fields for
        # checkpoint/layout compatibility but its _effective() does not add them.
        self.register_buffer(
            "base_l0_weight", torch.zeros(self.hidden_dim, self.shared_dim)
        )
        self.register_buffer("base_l0_bias", torch.zeros(self.hidden_dim))
        self.register_buffer(
            "base_l2_weight", torch.zeros(self.act_dim, self.hidden_dim)
        )
        self.register_buffer("base_l2_bias", torch.zeros(self.act_dim))

        # Current task's trainable contribution.
        self.own_l0_weight = Parameter(
            torch.empty(self.hidden_dim, self.shared_dim)
        )
        self.own_l0_bias = Parameter(torch.empty(self.hidden_dim))
        self.own_l2_weight = Parameter(torch.empty(self.act_dim, self.hidden_dim))
        self.own_l2_bias = Parameter(torch.empty(self.act_dim))
        self._reset_own_parameters()

        # Historical entries are plain dictionaries because the project saves
        # the full HeadPool module with torch.save().  _apply() below explicitly
        # moves their tensors when .to(device) is called.
        self.pool = []
        self.own_buffer = None

        # CkaRlAgent attaches these objects after constructing/loading the pool.
        self.alpha = None
        self.alpha_scale = None
        self.alpha_mass = None

        self.last_merge_info = None
        self.last_distill_train_kl = None
        self.last_distill_test_kl = None

    # ------------------------------------------------------------------
    # Initialization / device handling
    # ------------------------------------------------------------------
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
        # nn.Module._apply moves Parameters and registered buffers. Historical
        # pool entries are ordinary tensors, so move them explicitly as well.
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

    def effective_alpha_mass(self):
        if self.alpha_mass is None:
            return None
        if not self.constrain_alpha_mass:
            # Legacy/ablation behavior: the learned scalar is unconstrained.
            return self.alpha_mass

        # Strictly positive semantic mass.  Dividing by softplus(1) preserves
        # the historical initialization raw_mass=1 -> effective_mass=1.
        normalizer = F.softplus(torch.ones_like(self.alpha_mass))
        return F.softplus(self.alpha_mass) / normalizer

    def _historical(self):
        if not self.pool:
            # Returning correctly shaped tensors instead of Python 0.0 keeps
            # dtype/device behavior explicit without changing the mathematics.
            return {
                "l0_weight": torch.zeros_like(self.own_l0_weight),
                "l0_bias": torch.zeros_like(self.own_l0_bias),
                "l2_weight": torch.zeros_like(self.own_l2_weight),
                "l2_bias": torch.zeros_like(self.own_l2_bias),
            }

        if self.alpha is None or self.alpha.numel() != len(self.pool):
            raise RuntimeError(
                f"{self.head_type} alpha length does not match pool length: "
                f"alpha={None if self.alpha is None else self.alpha.numel()}, "
                f"pool={len(self.pool)}"
            )

        scale = 1.0 if self.alpha_scale is None else self.alpha_scale
        weights = F.softmax(self.alpha * scale, dim=0)
        if self.use_alpha_mass and self.alpha_mass is not None:
            weights = self.effective_alpha_mass() * weights

        out = {}
        for name, ndim in (
            ("l0_weight", 2),
            ("l0_bias", 1),
            ("l2_weight", 2),
            ("l2_bias", 1),
        ):
            stacked = torch.stack([entry[name] for entry in self.pool], dim=0)
            out[name] = (
                weights.view((-1,) + (1,) * ndim) * stacked
            ).sum(dim=0)
        return out

    def _effective(self):
        """Return the four parameter tensors of the current effective head."""
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
        # Return RAW logits.  Softmax belongs to Categorical/log-softmax KL,
        # not to the stored head itself.
        return F.linear(h, w2, b2)

    def forward(self, shared_features):
        return self._forward_with_weights(shared_features, self._effective())

    def entry_effective_weights(self, entry) -> Tuple[torch.Tensor, ...]:
        """Weights represented by one historical slot as a standalone policy."""
        if self.fusion_mode == BASE_FUSION_MODE:
            return (
                self.base_l0_weight + entry["l0_weight"],
                self.base_l0_bias + entry["l0_bias"],
                self.base_l2_weight + entry["l2_weight"],
                self.base_l2_bias + entry["l2_bias"],
            )
        return tuple(entry[key] for key in _HEAD_KEYS)

    def forward_entry(self, shared_features, entry_index: int):
        if not 0 <= int(entry_index) < len(self.pool):
            raise IndexError(
                f"entry_index={entry_index} outside pool of length {len(self.pool)}"
            )
        return self._forward_with_weights(
            shared_features,
            self.entry_effective_weights(self.pool[int(entry_index)]),
        )

    # ------------------------------------------------------------------
    # Pool lifecycle
    # ------------------------------------------------------------------
    def _validate_loaded_pool(self, latest_pool: "HeadPool"):
        if getattr(latest_pool, "format_version", 1) != self.format_version:
            raise RuntimeError(
                "Checkpoint uses an incompatible knowledge-pool format. "
                "Start a fresh Atari continual chain with the categorical-logits pool."
            )
        if getattr(latest_pool, "head_type", None) != "logits":
            raise RuntimeError("checkpoint is not a categorical logits pool")
        if getattr(latest_pool, "fusion_mode", None) != self.fusion_mode:
            raise RuntimeError(
                "fusion_mode changed inside one continual chain: "
                f"checkpoint={getattr(latest_pool, 'fusion_mode', None)!r}, "
                f"current={self.fusion_mode!r}"
            )

        expected_shapes = {
            "base_l0_weight": tuple(self.base_l0_weight.shape),
            "base_l0_bias": tuple(self.base_l0_bias.shape),
            "base_l2_weight": tuple(self.base_l2_weight.shape),
            "base_l2_bias": tuple(self.base_l2_bias.shape),
        }
        for name, expected in expected_shapes.items():
            loaded = getattr(latest_pool, name, None)
            if loaded is None or tuple(loaded.shape) != expected:
                got = None if loaded is None else tuple(loaded.shape)
                raise RuntimeError(
                    f"checkpoint {name} shape {got} does not match current {expected}; "
                    "the CNN feature dimension, head hidden size, or action count changed"
                )

        for idx, entry in enumerate(getattr(latest_pool, "pool", [])):
            for key in _HEAD_KEYS:
                if key not in entry:
                    raise RuntimeError(
                        f"checkpoint pool entry {idx} is missing parameter {key!r}"
                    )

    def inherit_pool_from(self, latest_pool: "HeadPool"):
        """Copy historical policy slots and the root/base from the last task.

        Alpha objects are intentionally *not* inherited. CkaRlAgent constructs a
        fresh alpha vector after this call so its length matches the new pool.
        """
        self._validate_loaded_pool(latest_pool)

        self.pool = []
        for entry in latest_pool.pool:
            copied = {key: entry[key].detach().clone() for key in _HEAD_KEYS}
            # Replay buffers are NumPy arrays and are never mutated in-place by
            # HeadPool; sharing the object avoids duplicating large Atari frames.
            copied["buffer"] = entry.get("buffer")
            self.pool.append(copied)

        with torch.no_grad():
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

    @staticmethod
    def _validate_buffer(buffer):
        if buffer is None:
            return
        if not isinstance(buffer, dict):
            raise TypeError("knowledge-pool buffer must be a dict or None")
        if "obs" not in buffer or not isinstance(buffer["obs"], np.ndarray):
            raise ValueError("knowledge-pool buffer must contain NumPy array 'obs'")

        rows = len(buffer["obs"])
        if rows < 1:
            raise ValueError("knowledge-pool buffer 'obs' must contain at least one row")

        # Every ndarray stored as per-transition metadata must remain row-aligned
        # because merge_buffers applies one sampled row index to all of them.
        for key, value in buffer.items():
            if isinstance(value, np.ndarray) and len(value) != rows:
                raise ValueError(
                    f"buffer array {key!r} has {len(value)} rows but obs has {rows}"
                )

    def set_own_buffer(self, buffer):
        self._validate_buffer(buffer)
        self.own_buffer = buffer

    def pool_length(self):
        return len(self.pool)

    def needs_merge(self):
        return len(self.pool) > self.pool_size

    def set_base(self):
        """Turn the trained root policy into theta_base and seed pool slot v1."""
        with torch.no_grad():
            self.base_l0_weight.copy_(self.own_l0_weight)
            self.base_l0_bias.copy_(self.own_l0_bias)
            self.base_l2_weight.copy_(self.own_l2_weight)
            self.base_l2_bias.copy_(self.own_l2_bias)

            if self.fusion_mode == BASE_FUSION_MODE:
                # In classic CKA the root lives in theta_base; the first stored
                # residual is therefore zero.
                entry = {
                    "l0_weight": torch.zeros_like(self.own_l0_weight),
                    "l0_bias": torch.zeros_like(self.own_l0_bias),
                    "l2_weight": torch.zeros_like(self.own_l2_weight),
                    "l2_bias": torch.zeros_like(self.own_l2_bias),
                }
            else:
                # weight_delta slots store reconstructed/effective weights.
                entry = {
                    "l0_weight": self.own_l0_weight.detach().clone(),
                    "l0_bias": self.own_l0_bias.detach().clone(),
                    "l2_weight": self.own_l2_weight.detach().clone(),
                    "l2_bias": self.own_l2_bias.detach().clone(),
                }

        entry["buffer"] = self.own_buffer
        self.pool = [entry]
        self.reset_own_to_zero()

    def finalize_own_contribution(self):
        """Insert the just-trained task entry without choosing a merge pair.

        Pair selection remains agent-level: CkaRlAgent decides whether to use
        parameter cosine similarity or behavioral categorical KL, and then asks
        this pool to replace the selected pair.
        """
        if self.fusion_mode == "weight_delta":
            hist = self._historical()
            entry = {
                "l0_weight": (
                    self.own_l0_weight.detach() + hist["l0_weight"].detach()
                ).clone(),
                "l0_bias": (
                    self.own_l0_bias.detach() + hist["l0_bias"].detach()
                ).clone(),
                "l2_weight": (
                    self.own_l2_weight.detach() + hist["l2_weight"].detach()
                ).clone(),
                "l2_bias": (
                    self.own_l2_bias.detach() + hist["l2_bias"].detach()
                ).clone(),
            }
        else:
            entry = {
                "l0_weight": self.own_l0_weight.detach().clone(),
                "l0_bias": self.own_l0_bias.detach().clone(),
                "l2_weight": self.own_l2_weight.detach().clone(),
                "l2_bias": self.own_l2_bias.detach().clone(),
            }

        entry["buffer"] = self.own_buffer
        self.pool = [entry] + self.pool
        self.reset_own_to_zero()

    def _validate_pair_indices(self, idx1: int, idx2: int):
        idx1, idx2 = int(idx1), int(idx2)
        n = len(self.pool)
        if idx1 == idx2:
            raise ValueError("merge pair must contain two distinct pool entries")
        if not (0 <= idx1 < n and 0 <= idx2 < n):
            raise IndexError(
                f"merge indices ({idx1}, {idx2}) outside pool of length {n}"
            )
        return idx1, idx2

    def average_pair_params(
        self, idx1: int, idx2: int
    ) -> Dict[str, torch.Tensor]:
        idx1, idx2 = self._validate_pair_indices(idx1, idx2)
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
        idx1, idx2 = self._validate_pair_indices(idx1, idx2)
        missing = [key for key in _HEAD_KEYS if key not in params]
        if missing:
            raise ValueError(f"merged parameter dictionary is missing {missing}")
        self._validate_buffer(merged_buffer)

        n = len(self.pool)
        entry = {
            key: params[key].detach().clone()
            for key in _HEAD_KEYS
        }
        entry["buffer"] = merged_buffer

        # Preserve the historical behavior: surviving entries retain order and
        # the newly merged knowledge item is appended at the end.
        self.pool = [
            self.pool[i]
            for i in range(n)
            if i not in (idx1, idx2)
        ] + [entry]
        self.last_merge_info = dict(merge_info)

    # ------------------------------------------------------------------
    # Replay-buffer merging
    # ------------------------------------------------------------------
    @staticmethod
    def _copy_or_trim_buffer(buffer, max_rows: int):
        """Copy one parent buffer while respecting the same hard row budget."""
        if buffer is None:
            return None
        HeadPool._validate_buffer(buffer)
        n = len(buffer["obs"])
        take = min(n, max_rows)
        if take == n:
            return {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in buffer.items()
            }
        idx = np.random.choice(n, size=take, replace=False)
        return {
            key: value[idx].copy() if isinstance(value, np.ndarray) else value
            for key, value in buffer.items()
        }

    @staticmethod
    def merge_buffers(buf1, buf2, max_rows: int):
        """Merge two lineage buffers with approximately equal parent mass.

        If truncation is required, roughly half the row budget is reserved for
        each immediate parent and spare capacity is given back when one parent is
        too small.  The same sampled indices are applied to every row-aligned
        NumPy array, preserving observation/action/task/source correspondence.
        """
        max_rows = int(max_rows)
        if max_rows < 1:
            raise ValueError("max_rows must be >= 1")

        if buf1 is None and buf2 is None:
            return None
        if buf1 is None:
            return HeadPool._copy_or_trim_buffer(buf2, max_rows)
        if buf2 is None:
            return HeadPool._copy_or_trim_buffer(buf1, max_rows)

        HeadPool._validate_buffer(buf1)
        HeadPool._validate_buffer(buf2)

        # Only arrays existing in both parents can remain row-aligned after the
        # merge.  In normal Atari runs these are obs/actions/task_ids/source_ids.
        keys = [
            key
            for key in buf1.keys()
            if key in buf2
            and isinstance(buf1[key], np.ndarray)
            and isinstance(buf2[key], np.ndarray)
        ]
        if "obs" not in keys:
            raise ValueError("both pool buffers must contain a NumPy 'obs' array")

        n1, n2 = len(buf1["obs"]), len(buf2["obs"])
        if n1 + n2 <= max_rows:
            return {
                key: np.concatenate([buf1[key], buf2[key]], axis=0)
                for key in keys
            }

        half = max_rows // 2
        take1 = min(n1, half)
        take2 = min(n2, max_rows - take1)

        # Give unused quota from a short parent to the other parent.
        if take1 + take2 < max_rows:
            take1 = min(n1, take1 + (max_rows - take1 - take2))
        if take1 + take2 < max_rows:
            take2 = min(n2, take2 + (max_rows - take1 - take2))

        idx1 = np.random.choice(n1, size=take1, replace=False)
        idx2 = np.random.choice(n2, size=take2, replace=False)
        return {
            key: np.concatenate(
                [buf1[key][idx1], buf2[key][idx2]], axis=0
            )
            for key in keys
        }
