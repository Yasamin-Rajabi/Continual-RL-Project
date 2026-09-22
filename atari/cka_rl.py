"""Categorical CKA-RL agent for continual Atari PPO.

This is the discrete-action counterpart of the HalfCheetah CKA-RL agent.
Shared continual-learning structure is kept aligned with the HalfCheetah files;
only the policy distribution (categorical), encoder (CNN), PPO value head, and
uint8 Atari replay-state handling are domain-specific.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.distributions import Categorical

from knowledge_pools import (
    BASE_FUSION_MODE,
    HeadPool,
    balanced_lineage_indices,
)
from policy_composition import (
    categorical_mixture_distribution,
    categorical_mixture_logits,
    categorical_mixture_probs,
    stacked_head_forward,
)
from policy_space import PolicySpaceMixin
from policy_utils import categorical_kl, symmetric_categorical_kl
from shared_arch import (
    ENCODER_FORMAT_VERSION,
    layer_init,
    shared,
    validate_shared_encoder,
)

_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")
_POLICY_SNAPSHOT_FORMAT_VERSION = 3


def _torch_load(path, map_location=None):
    kwargs = {} if map_location is None else {"map_location": map_location}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _require_file(path, description: str):
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


class CkaRlAgent(PolicySpaceMixin, nn.Module):
    """Shared Atari CNN + bounded categorical policy pool + task-local critic."""

    def __init__(
        self,
        obs_shape,
        act_dim,
        base_dir=None,
        latest_dir=None,
        pool_size=5,
        alpha_init="Randn",
        alpha_major=0.6,
        alpha_factor=1e-3,
        fix_alpha=False,
        use_alpha_scale=False,
        fix_alpha_scale=False,
        use_alpha_mass=False,
        constrain_alpha_mass=True,
        encoder_from_base=True,
        distillation=True,
        fusion_mode=BASE_FUSION_MODE,
        max_distill_buffer=5_000,
        distill_test_frac=0.2,
        distill_select_best_val=True,
        distill_epochs=8,
        distill_lr=3e-4,
        distill_batch_size=256,
        distill_max_samples=2_000,
        similarity_samples=512,
        balance_source_lineages=False,
        hidden_dim=128,
        shared_dim=512,
        train_shared=False,
        freeze_root_encoder=False,
        pretrained_encoder=None,
        composition_space="parameter",
        projection_epochs=16,
        projection_max_samples=20_000,
        policy_student_replay=False,
        # Explicit compatibility knobs; these are not Atari architecture options.
        distill_observation_skip=False,
        encoder_linear_out=False,
    ):
        super().__init__()

        if composition_space not in ("parameter", "policy"):
            raise ValueError("composition_space must be 'parameter' or 'policy'")
        if composition_space == "policy" and use_alpha_mass and not constrain_alpha_mass:
            raise ValueError("A probability mixture requires bounded sigmoid alpha-mass")

        self.composition_space = composition_space
        self.policy_student_replay = bool(policy_student_replay)
        if self.policy_student_replay and (
            composition_space != "policy"
            or not use_alpha_mass
            or fusion_mode != "weight_delta"
        ):
            raise ValueError(
                "policy_student_replay requires policy composition, "
                "weight_delta, and alpha-mass"
            )

        self.projection_epochs = int(projection_epochs)
        self.projection_max_samples = int(projection_max_samples)
        if self.projection_epochs < 1 or self.projection_max_samples < 2:
            raise ValueError(
                "projection_epochs >= 1 and projection_max_samples >= 2 are required"
            )

        self.mixture_warmup = False
        self.pool_only = False
        self.last_projection_metrics = {}

        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.act_dim = int(act_dim)
        self.shared_dim = int(shared_dim)
        self.hidden_dim = int(hidden_dim)
        self.pool_size = int(pool_size)
        self.distillation = bool(distillation)
        self.fusion_mode = str(fusion_mode)
        self.max_distill_buffer = int(max_distill_buffer)
        self.use_alpha_scale = bool(use_alpha_scale)
        self.fix_alpha_scale = bool(fix_alpha_scale)
        self.use_alpha_mass = bool(use_alpha_mass)
        self.constrain_alpha_mass = bool(constrain_alpha_mass)
        self.distill_test_frac = float(distill_test_frac)
        self.distill_select_best_val = bool(distill_select_best_val)
        self.distill_epochs = int(distill_epochs)
        self.distill_lr = float(distill_lr)
        self.distill_batch_size = int(distill_batch_size)
        self.distill_max_samples = int(distill_max_samples)
        self.similarity_samples = int(similarity_samples)
        self.balance_source_lineages = bool(balance_source_lineages)
        self.train_shared = bool(train_shared)
        self.freeze_root_encoder = bool(freeze_root_encoder)
        self.last_merge_info = None
        self.last_distill_metrics = {}

        self._validate_init_args(
            alpha_init=alpha_init,
            alpha_major=alpha_major,
            distill_observation_skip=distill_observation_skip,
            encoder_linear_out=encoder_linear_out,
        )

        self.policy_pool = HeadPool(
            "logits",
            self.shared_dim,
            self.hidden_dim,
            self.act_dim,
            fusion_mode=self.fusion_mode,
            pool_size=self.pool_size,
            distillation=self.distillation,
            max_distill_buffer=self.max_distill_buffer,
            use_alpha_mass=self.use_alpha_mass,
            constrain_alpha_mass=self.constrain_alpha_mass,
            distill_test_frac=self.distill_test_frac,
        )
        self.policy_pool.composition_space = composition_space

        if latest_dir is not None:
            pool_path = _require_file(
                os.path.join(os.fspath(latest_dir), "policy_pool.pt"),
                "latest categorical policy pool",
            )
            latest_pool = _torch_load(pool_path, map_location="cpu")
            if getattr(latest_pool, "composition_space", "parameter") != composition_space:
                raise ValueError(
                    "Cannot continue a chain with a different composition space; "
                    "start a fresh run"
                )
            self.policy_pool.inherit_pool_from(latest_pool)
            self.policy_pool.reset_own_to_zero()

        self.alpha, self.alpha_scale, self.alpha_mass = self._make_alpha(
            self.policy_pool.pool_length(),
            fix_alpha,
            alpha_init,
            alpha_major,
            alpha_factor,
            use_alpha_scale,
            fix_alpha_scale,
            use_alpha_mass,
        )
        self.policy_pool.set_alpha(
            self.alpha, self.alpha_scale, self.alpha_mass
        )
        self.initialize_policy_space_own()

        logger.info(f"shared alpha: {self.alpha}")
        if use_alpha_mass:
            logger.info(f"shared alpha_mass: {self.alpha_mass}")

        # Match HalfCheetah encoder lifecycle exactly; only shared(...) differs.
        if latest_dir is not None and self.train_shared:
            encoder_path = _require_file(
                os.path.join(os.fspath(latest_dir), "fc.pt"),
                "latest trainable Atari encoder",
            )
            self.fc = _torch_load(encoder_path, map_location="cpu")
            source = f"latest encoder {encoder_path}"
        elif pretrained_encoder is not None:
            encoder_path = _require_file(
                pretrained_encoder, "pretrained Atari encoder"
            )
            self.fc = _torch_load(encoder_path, map_location="cpu")
            source = f"pretrained encoder {encoder_path}"
        elif encoder_from_base and base_dir is not None:
            encoder_path = _require_file(
                os.path.join(os.fspath(base_dir), "fc.pt"),
                "root/base Atari encoder",
            )
            self.fc = _torch_load(encoder_path, map_location="cpu")
            source = f"base encoder {encoder_path}"
        elif latest_dir is not None:
            encoder_path = _require_file(
                os.path.join(os.fspath(latest_dir), "fc.pt"),
                "latest Atari encoder",
            )
            self.fc = _torch_load(encoder_path, map_location="cpu")
            source = f"latest encoder {encoder_path}"
        else:
            self.fc = shared(
                input_shape=self.obs_shape,
                output_dim=self.shared_dim,
            )
            source = "new root Atari CNN encoder"

        validate_shared_encoder(
            self.fc,
            input_shape=self.obs_shape,
            output_dim=self.shared_dim,
            source=source,
        )

        should_freeze = (
            not self.train_shared
            and (
                pretrained_encoder is not None
                or latest_dir is not None
                or self.freeze_root_encoder
            )
        )
        if should_freeze:
            self.fc.requires_grad_(False)

        # Critic is PPO task-local state and never enters the policy pool.
        self.critic = layer_init(nn.Linear(self.shared_dim, 1), std=1.0)

    def _validate_init_args(
        self,
        *,
        alpha_init,
        alpha_major,
        distill_observation_skip,
        encoder_linear_out,
    ):
        if len(self.obs_shape) != 3:
            raise ValueError(
                f"Atari expects CHW observations, got {self.obs_shape}"
            )
        if self.act_dim < 2:
            raise ValueError("act_dim must be >= 2")
        if self.pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        if self.fusion_mode not in (BASE_FUSION_MODE, "weight_delta"):
            raise ValueError(f"unknown fusion_mode={self.fusion_mode!r}")
        if self.fusion_mode == BASE_FUSION_MODE and self.use_alpha_mass:
            raise ValueError(
                "use_alpha_mass is only valid with weight_delta"
            )
        if self.fix_alpha_scale and self.use_alpha_scale:
            raise ValueError(
                "use_alpha_scale=True and fix_alpha_scale=True are mutually exclusive"
            )
        if self.train_shared and self.freeze_root_encoder:
            raise ValueError(
                "train_shared=True and freeze_root_encoder=True are contradictory"
            )
        if alpha_init not in ("Randn", "Major", "Uniform"):
            raise ValueError(f"unknown alpha_init={alpha_init!r}")
        if alpha_init == "Major" and not 0.0 < float(alpha_major) < 1.0:
            raise ValueError("alpha_major must be in (0,1)")
        if self.similarity_samples < 2 or self.distill_max_samples < 2:
            raise ValueError(
                "similarity_samples and distill_max_samples must be >= 2"
            )
        if self.distill_batch_size < 1 or self.distill_lr <= 0:
            raise ValueError("invalid distillation batch size or learning rate")
        if not 0.0 <= self.distill_test_frac < 1.0:
            raise ValueError("distill_test_frac must be in [0,1)")
        if self.distillation and self.distill_epochs < 1:
            raise ValueError(
                "distill_epochs must be >= 1 when distillation is enabled"
            )
        if distill_observation_skip:
            raise ValueError(
                "Atari policy heads consume CNN features only; "
                "distill_observation_skip is unsupported"
            )
        if encoder_linear_out:
            raise ValueError(
                "encoder_linear_out is a HalfCheetah MLP option, not Atari"
            )

    def _make_alpha(
        self,
        num_vectors,
        fix_alpha,
        alpha_init,
        alpha_major,
        alpha_factor,
        use_alpha_scale,
        fix_alpha_scale,
        use_alpha_mass,
    ):
        if num_vectors <= 0:
            return None, None, None
        if fix_alpha:
            alpha = nn.Parameter(
                torch.zeros(num_vectors), requires_grad=False
            )
        elif alpha_init == "Uniform" or num_vectors == 1:
            alpha = nn.Parameter(
                torch.ones(num_vectors) * alpha_factor,
                requires_grad=True,
            )
        elif alpha_init == "Randn":
            alpha = nn.Parameter(
                torch.randn(num_vectors) / max(num_vectors, 1),
                requires_grad=True,
            )
        elif alpha_init == "Major" and num_vectors > 1:
            vals = [
                np.log((1 - alpha_major) / (num_vectors - 1))
                for _ in range(num_vectors - 1)
            ]
            vals.append(np.log(alpha_major))
            alpha = nn.Parameter(
                torch.tensor(vals, dtype=torch.float32),
                requires_grad=True,
            )
        else:
            raise NotImplementedError(alpha_init)

        scale_init = 5.0 if fix_alpha_scale else 1.0
        alpha_scale = nn.Parameter(
            torch.tensor([scale_init], dtype=torch.float32),
            requires_grad=(
                use_alpha_scale
                and not fix_alpha_scale
                and not fix_alpha
            ),
        )
        alpha_mass = (
            nn.Parameter(
                torch.full(
                    (1,),
                    float(np.log(0.95 / 0.05))
                    if self.constrain_alpha_mass
                    else 1.0,
                ),
                requires_grad=not fix_alpha,
            )
            if use_alpha_mass
            else None
        )
        return alpha, alpha_scale, alpha_mass

    # ------------------------------------------------------------------
    # PPO interface
    # ------------------------------------------------------------------
    def encode(self, obs):
        return self.fc(obs)

    def _distribution_at_features(self, features):
        logits, weights = self._components_at_features(features)
        return categorical_mixture_distribution(logits, weights)

    def forward(self, obs):
        features = self.encode(obs)
        if self.composition_space == "policy":
            logits, weights = self._components_at_features(features)
            return categorical_mixture_logits(logits, weights)
        return self.policy_pool(features)

    def action_distribution(self, obs):
        return self._distribution_at_features(self.encode(obs))

    def get_value(self, obs):
        return self.critic(self.encode(obs))

    def get_action_and_value(
        self,
        obs,
        action=None,
        log_writter=None,
        global_step=None,
        **_ignored,
    ):
        del log_writter, global_step, _ignored
        features = self.encode(obs)
        dist = self._distribution_at_features(features)
        if action is None:
            action = dist.sample()
        action = action.long().reshape(-1)
        value = self.critic(features)
        return action, dist.log_prob(action), dist.entropy(), value

    # ------------------------------------------------------------------
    # Pool lifecycle / Atari buffers
    # ------------------------------------------------------------------
    def set_own_buffer(self, buffer):
        if buffer is not None:
            obs = buffer.get("obs") if isinstance(buffer, dict) else None
            if not isinstance(obs, np.ndarray):
                raise ValueError(
                    "Atari merge buffer must contain NumPy array 'obs'"
                )
            if obs.dtype != np.uint8:
                raise ValueError(
                    f"Atari merge observations must be uint8, got {obs.dtype}"
                )
            if tuple(obs.shape[1:]) != self.obs_shape:
                raise ValueError(
                    f"buffer shape {tuple(obs.shape[1:])} != {self.obs_shape}"
                )
        self.policy_pool.set_own_buffer(buffer)

    def set_base(self):
        self.policy_pool.set_base()

    def _sample_reference_observations(self):
        buffers = [entry.get("buffer") for entry in self.policy_pool.pool]
        if any(
            buf is None
            or "obs" not in buf
            or len(buf["obs"]) == 0
            for buf in buffers
        ):
            raise RuntimeError(
                "Behavioral KL merging requires a non-empty buffer for every pool slot"
            )

        per_slot = max(1, self.similarity_samples // 2)
        samples = []
        for buf in buffers:
            obs = buf["obs"]
            if obs.dtype != np.uint8:
                raise RuntimeError("Atari pool observations must be uint8")
            take = min(per_slot, len(obs))
            if self.balance_source_lineages:
                if "source_ids" not in buf:
                    raise RuntimeError(
                        "--balance-source-lineages requires source_ids"
                    )
                idx = balanced_lineage_indices(
                    buf["source_ids"], take
                )
            else:
                idx = np.random.choice(
                    len(obs), size=take, replace=False
                )
            samples.append(obs[idx])
        return samples

    def _encode_obs(self, obs: np.ndarray, batch_size: int = 512):
        if obs.dtype != np.uint8:
            raise ValueError("Atari replay states must be raw uint8")
        device = self.policy_pool.base_l0_weight.device
        chunks = []
        with torch.no_grad():
            for start in range(0, len(obs), batch_size):
                x = torch.as_tensor(
                    obs[start:start + batch_size],
                    dtype=torch.float32,
                    device=device,
                ) / 255.0
                chunks.append(self.fc(x))
        return torch.cat(chunks, dim=0)

    def _entry_logits(self, features, index):
        return self.policy_pool.forward_entry(features, index)

    # ------------------------------------------------------------------
    # Merge-pair selection
    # ------------------------------------------------------------------
    def _select_cosine_pair(self):
        n = self.policy_pool.pool_length()
        if n < 2:
            raise RuntimeError("cannot select pair from fewer than two entries")

        vectors = [
            torch.cat(
                [
                    self.policy_pool.pool[i][key].reshape(-1)
                    for key in _HEAD_KEYS
                ]
            )
            for i in range(n)
        ]
        device = vectors[0].device
        matrix = torch.full(
            (n, n), -float("inf"), device=device
        )
        finite_values = []
        with torch.no_grad():
            for i in range(n):
                for j in range(i + 1, n):
                    score = F.cosine_similarity(
                        vectors[i], vectors[j], dim=0
                    )
                    score = torch.nan_to_num(
                        score, nan=-float("inf")
                    )
                    matrix[i, j] = matrix[j, i] = score
                    if torch.isfinite(score):
                        finite_values.append(float(score))
            flat = int(torch.argmax(matrix))
            idx1, idx2 = divmod(flat, n)
            if idx1 == idx2 or not torch.isfinite(matrix[idx1, idx2]):
                raise RuntimeError("no finite cosine pair")
            selected = float(matrix[idx1, idx2])

        stats = {
            "idx1": idx1,
            "idx2": idx2,
            "similarity_metric": "cosine",
            "cosine_similarity": selected,
            "pairwise_cosine_min": float(np.min(finite_values)),
            "pairwise_cosine_mean": float(np.mean(finite_values)),
            "pairwise_cosine_max": float(np.max(finite_values)),
            "pairwise_cosine_similarity": matrix.detach().cpu().numpy().tolist(),
        }
        return idx1, idx2, stats

    def _select_behavioral_pair(self):
        """HalfCheetah-matched pair-local symmetric KL on both parents' states."""
        n = self.policy_pool.pool_length()
        if n < 2:
            raise RuntimeError("cannot select pair from fewer than two entries")

        obs_by_slot = self._sample_reference_observations()
        features_by_slot = [
            self._encode_obs(obs) for obs in obs_by_slot
        ]

        with torch.no_grad():
            outputs = [
                [
                    self._entry_logits(
                        features_by_slot[source], policy
                    )
                    for source in range(n)
                ]
                for policy in range(n)
            ]
            device = features_by_slot[0].device
            matrix = torch.full(
                (n, n), float("inf"), device=device
            )
            pair_rows = torch.zeros(
                (n, n), dtype=torch.int64, device=device
            )
            finite_values = []

            for i in range(n):
                for j in range(i + 1, n):
                    logits_i = torch.cat(
                        (outputs[i][i], outputs[i][j]), dim=0
                    )
                    logits_j = torch.cat(
                        (outputs[j][i], outputs[j][j]), dim=0
                    )
                    skl = symmetric_categorical_kl(
                        logits_i, logits_j
                    )
                    score = torch.nan_to_num(
                        skl,
                        nan=float("inf"),
                        posinf=float("inf"),
                    ).mean()
                    matrix[i, j] = matrix[j, i] = score
                    pair_rows[i, j] = pair_rows[j, i] = len(logits_i)
                    if torch.isfinite(score):
                        finite_values.append(float(score))

            flat = int(torch.argmin(matrix))
            idx1, idx2 = divmod(flat, n)
            if idx1 == idx2 or not torch.isfinite(matrix[idx1, idx2]):
                raise RuntimeError("no finite behavioral-KL pair")

            selected = float(matrix[idx1, idx2])
            selected_rows = int(pair_rows[idx1, idx2])
            logits_i = torch.cat(
                (outputs[idx1][idx1], outputs[idx1][idx2]), dim=0
            )
            logits_j = torch.cat(
                (outputs[idx2][idx1], outputs[idx2][idx2]), dim=0
            )
            state_kl = symmetric_categorical_kl(
                logits_i, logits_j
            )
            p95 = float(torch.quantile(state_kl, 0.95))
            max_kl = float(state_kl.max())

        stats = {
            "idx1": idx1,
            "idx2": idx2,
            "similarity_metric": "symmetric_kl",
            "symmetric_kl": selected,
            "pairwise_kl_min": float(np.min(finite_values)),
            "pairwise_kl_mean": float(np.mean(finite_values)),
            "pairwise_kl_max": float(np.max(finite_values)),
            "selected_state_kl_p95": p95,
            "selected_state_kl_max": max_kl,
            "similarity_states": selected_rows,
            "reference_rows_per_slot": [
                int(len(x)) for x in obs_by_slot
            ],
            "pairwise_symmetric_kl": matrix.detach().cpu().numpy().tolist(),
        }
        return idx1, idx2, stats

    # ------------------------------------------------------------------
    # Categorical distillation
    # ------------------------------------------------------------------
    @staticmethod
    def _params_to_effective(pool, params):
        if pool.fusion_mode == BASE_FUSION_MODE:
            return (
                pool.base_l0_weight + params["l0_weight"],
                pool.base_l0_bias + params["l0_bias"],
                pool.base_l2_weight + params["l2_weight"],
                pool.base_l2_bias + params["l2_bias"],
            )
        return tuple(params[key] for key in _HEAD_KEYS)

    @staticmethod
    def _head_forward_from_params(pool, features, params):
        return pool._forward_with_weights(
            features,
            CkaRlAgent._params_to_effective(pool, params),
        )

    @staticmethod
    def _buffer_lineage(buffer, key="task_ids"):
        if buffer is None or key not in buffer:
            return {}
        ids, counts = np.unique(
            np.asarray(buffer[key]).reshape(-1),
            return_counts=True,
        )
        return {
            str(int(source_id)): int(count)
            for source_id, count in zip(ids, counts)
        }

    def _balanced_parent_data(self, idx1, idx2):
        buf1 = self.policy_pool.pool[idx1].get("buffer")
        buf2 = self.policy_pool.pool[idx2].get("buffer")
        if buf1 is None or buf2 is None:
            raise RuntimeError(
                "distillation selected an entry without a buffer"
            )

        if self.balance_source_lineages:
            for buf in (buf1, buf2):
                if "source_ids" not in buf:
                    raise RuntimeError(
                        "--balance-source-lineages requires source_ids"
                    )
            n1, n2 = len(buf1["obs"]), len(buf2["obs"])
            # Only the (tiny) integer lineage ids need concatenating to pick a
            # balanced selection. The previous version also concatenated the
            # full 'obs' arrays of both parents just to index a handful of
            # rows back out of them -- a full extra copy of up to both
            # parent buffers (many GB of Atari frames) for a result that
            # keeps at most distill_max_samples rows.
            source_all = np.concatenate(
                [
                    np.asarray(buf1["source_ids"]).reshape(-1),
                    np.asarray(buf2["source_ids"]).reshape(-1),
                ]
            )
            take = min(n1 + n2, self.distill_max_samples)
            idx = balanced_lineage_indices(source_all, take)
            idx1_local = idx[idx < n1]
            idx2_local = idx[idx >= n1] - n1
            obs = HeadPool._gather_two(buf1, buf2, "obs", idx1_local, idx2_local)
            teacher_ids = np.concatenate(
                [
                    np.zeros(len(idx1_local), dtype=np.int64),
                    np.ones(len(idx2_local), dtype=np.int64),
                ]
            )
            source_ids = np.concatenate(
                [
                    np.asarray(buf1["source_ids"]).reshape(-1)[idx1_local],
                    np.asarray(buf2["source_ids"]).reshape(-1)[idx2_local],
                ]
            ).astype(np.int64, copy=False)
            return obs, teacher_ids, source_ids

        max_each = max(1, self.distill_max_samples // 2)
        obs_parts, teacher_ids, source_ids = [], [], []
        for teacher_id, buf in enumerate((buf1, buf2)):
            obs = buf["obs"]
            take = min(len(obs), max_each)
            idx = np.random.choice(
                len(obs), size=take, replace=False
            )
            obs_parts.append(obs[idx])
            teacher_ids.append(
                np.full(take, teacher_id, dtype=np.int64)
            )
            source_ids.append(
                np.asarray(
                    buf.get(
                        "source_ids",
                        np.full(len(obs), teacher_id),
                    )
                )[idx].reshape(-1)
            )
        return (
            np.concatenate(obs_parts, axis=0),
            np.concatenate(teacher_ids),
            np.concatenate(source_ids).astype(
                np.int64, copy=False
            ),
        )

    def _distill_pair(self, idx1, idx2):
        obs, teacher_ids_np, source_ids_np = (
            self._balanced_parent_data(idx1, idx2)
        )
        features = self._encode_obs(obs)
        device = features.device
        teacher_ids = torch.as_tensor(
            teacher_ids_np,
            dtype=torch.long,
            device=device,
        )
        source_ids = torch.as_tensor(
            source_ids_np,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            logits1 = self._entry_logits(features, idx1)
            logits2 = self._entry_logits(features, idx2)
            mask = teacher_ids.unsqueeze(-1).bool()
            teacher_logits = torch.where(
                mask, logits2, logits1
            )

        init = self.policy_pool.average_pair_params(
            idx1, idx2
        )
        params = {
            key: value.detach().clone().requires_grad_(True)
            for key, value in init.items()
        }
        trainables = list(params.values())
        optimizer = torch.optim.Adam(
            trainables, lr=self.distill_lr
        )

        split_groups = (
            torch.unique(source_ids).tolist()
            if self.balance_source_lineages
            else [0, 1]
        )
        split_labels = (
            source_ids
            if self.balance_source_lineages
            else teacher_ids
        )
        train_parts, test_parts = [], []
        for group_id in split_groups:
            idx = torch.nonzero(
                split_labels == int(group_id),
                as_tuple=False,
            ).flatten()
            idx = idx[
                torch.randperm(idx.numel(), device=device)
            ]
            n_test = (
                int(idx.numel() * self.distill_test_frac)
                if self.distill_test_frac > 0
                else 0
            )
            n_test = min(
                n_test, max(idx.numel() - 1, 0)
            )
            test_parts.append(idx[:n_test])
            train_parts.append(idx[n_test:])

        train_idx = torch.cat(train_parts)
        test_idx = torch.cat(test_parts)
        train_idx = train_idx[
            torch.randperm(train_idx.numel(), device=device)
        ]
        if test_idx.numel():
            test_idx = test_idx[
                torch.randperm(test_idx.numel(), device=device)
            ]

        def student_logits(batch_features):
            return self._head_forward_from_params(
                self.policy_pool,
                batch_features,
                params,
            )

        @torch.no_grad()
        def kl_summary(indices):
            if indices.numel() == 0:
                return None
            values = categorical_kl(
                teacher_logits[indices],
                student_logits(features[indices]),
            )
            return {
                "mean": float(values.mean()),
                "p95": float(torch.quantile(values, 0.95)),
                "max": float(values.max()),
            }

        def clone_params():
            return {
                key: value.detach().clone()
                for key, value in params.items()
            }

        validation_idx = (
            test_idx if test_idx.numel() else train_idx
        )
        initial_val = kl_summary(validation_idx)
        best_val = (
            float("inf")
            if initial_val is None
            else initial_val["mean"]
        )
        best_epoch = 0
        best = clone_params()

        for epoch in range(1, self.distill_epochs + 1):
            shuffled = train_idx[
                torch.randperm(
                    train_idx.numel(), device=device
                )
            ]
            for start in range(
                0, shuffled.numel(), self.distill_batch_size
            ):
                idx = shuffled[
                    start:start + self.distill_batch_size
                ]
                loss = categorical_kl(
                    teacher_logits[idx],
                    student_logits(features[idx]),
                ).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainables, 10.0
                )
                optimizer.step()

            val = kl_summary(validation_idx)
            if val is not None and val["mean"] < best_val:
                best_val = val["mean"]
                best_epoch = epoch
                best = clone_params()

        if self.distill_select_best_val:
            with torch.no_grad():
                for key in _HEAD_KEYS:
                    params[key].copy_(best[key])
            selected_epoch = best_epoch
            selected_val = best_val
        else:
            selected_epoch = self.distill_epochs
            val = kl_summary(validation_idx)
            selected_val = (
                float("nan")
                if val is None
                else val["mean"]
            )

        @torch.no_grad()
        def metrics(indices):
            if indices.numel() == 0:
                return None, None, None, None
            student = student_logits(features[indices])
            values = categorical_kl(
                teacher_logits[indices], student
            )
            prob_mse = F.mse_loss(
                F.softmax(student, dim=-1),
                F.softmax(
                    teacher_logits[indices], dim=-1
                ),
            ).item()
            return (
                float(values.mean()),
                float(torch.quantile(values, 0.95)),
                float(values.max()),
                float(prob_mse),
            )

        train = metrics(train_idx)
        test = metrics(test_idx)
        metrics_out = {
            "policy/distill_train_kl": train[0],
            "policy/distill_test_kl": test[0],
            "policy/distill_train_kl_p95": train[1],
            "policy/distill_test_kl_p95": test[1],
            "policy/distill_train_kl_max": train[2],
            "policy/distill_test_kl_max": test[2],
            "policy/distill_train_prob_mse": train[3],
            "policy/distill_test_prob_mse": test[3],
            "policy/distill_best_epoch": int(best_epoch),
            "policy/distill_best_val_kl": float(best_val),
            "policy/distill_selected_epoch": int(selected_epoch),
            "policy/distill_selected_val_kl": float(selected_val),
            "policy/distill_select_best_val": float(
                self.distill_select_best_val
            ),
            "policy/distill_initial_val_kl": (
                None
                if initial_val is None
                else float(initial_val["mean"])
            ),
            "policy/distill_rows": int(len(obs)),
            "policy/distill_source_lineages": int(
                torch.unique(source_ids).numel()
            ),
            "policy/distill_balance_source_lineages": float(
                self.balance_source_lineages
            ),
        }
        out = {
            key: value.detach().clone()
            for key, value in params.items()
        }
        return out, metrics_out

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
    def finalize(self):
        self.last_merge_info = None
        self.last_distill_metrics = {}

        if self.composition_space == "policy":
            if self.policy_student_replay:
                self.store_novel_policy_for_storage()
            else:
                self.project_policy_for_storage()
        else:
            self.policy_pool.finalize_own_contribution()

        if not self.policy_pool.needs_merge():
            return

        if self.distillation:
            idx1, idx2, merge_info = (
                self._select_behavioral_pair()
            )
            merged_params, distill_metrics = (
                self._distill_pair(idx1, idx2)
            )
            self.last_distill_metrics = (
                distill_metrics
            )
            used_distillation = True
        else:
            idx1, idx2, merge_info = (
                self._select_cosine_pair()
            )
            merged_params = (
                self.policy_pool.average_pair_params(
                    idx1, idx2
                )
            )
            used_distillation = False

        buf1 = self.policy_pool.pool[idx1].get("buffer")
        buf2 = self.policy_pool.pool[idx2].get("buffer")
        merged_buffer = HeadPool.merge_buffers(
            buf1,
            buf2,
            self.max_distill_buffer,
            balance_source_lineages=self.balance_source_lineages,
        )
        merge_info.update(
            {
                "used_distillation": used_distillation,
                "balance_source_lineages": bool(
                    self.balance_source_lineages
                ),
                "pool_size_before": int(
                    self.policy_pool.pool_length()
                ),
                "pool_size_after": int(
                    self.policy_pool.pool_length() - 1
                ),
                "parent_1_lineage": self._buffer_lineage(
                    buf1, "task_ids"
                ),
                "parent_2_lineage": self._buffer_lineage(
                    buf2, "task_ids"
                ),
                "merged_lineage": self._buffer_lineage(
                    merged_buffer, "task_ids"
                ),
                "parent_1_source_lineage": self._buffer_lineage(
                    buf1, "source_ids"
                ),
                "parent_2_source_lineage": self._buffer_lineage(
                    buf2, "source_ids"
                ),
                "merged_source_lineage": self._buffer_lineage(
                    merged_buffer, "source_ids"
                ),
            }
        )
        self.policy_pool.replace_pair(
            idx1,
            idx2,
            merged_params,
            merged_buffer,
            merge_info,
        )
        self.last_merge_info = dict(merge_info)

        if "policy/distill_train_kl" in self.last_distill_metrics:
            self.policy_pool.last_distill_train_kl = (
                self.last_distill_metrics[
                    "policy/distill_train_kl"
                ]
            )
            self.policy_pool.last_distill_test_kl = (
                self.last_distill_metrics[
                    "policy/distill_test_kl"
                ]
            )

    def get_distill_metrics(self):
        return dict(self.last_distill_metrics)

    def get_merge_info(self):
        return self.last_merge_info

    # ------------------------------------------------------------------
    # Saving / exact snapshots
    # ------------------------------------------------------------------
    @staticmethod
    def _cpu_clone_dict(d):
        return {
            key: value.detach().cpu().clone()
            for key, value in d.items()
        }

    def export_effective_policy(self):
        if self.composition_space == "policy":
            with torch.no_grad():
                snapshot = self.export_policy_ensemble()
                snapshot["encoder_format_version"] = int(
                    getattr(
                        self.fc,
                        "format_version",
                        ENCODER_FORMAT_VERSION,
                    )
                )
                return snapshot

        with torch.no_grad():
            w0, b0, w2, b2 = (
                self.policy_pool._effective()
            )
            return {
                "format_version": _POLICY_SNAPSHOT_FORMAT_VERSION,
                "encoder_format_version": int(
                    getattr(
                        self.fc,
                        "format_version",
                        ENCODER_FORMAT_VERSION,
                    )
                ),
                "composition_space": "parameter",
                "policy_type": "categorical",
                "obs_shape": tuple(self.obs_shape),
                "act_dim": self.act_dim,
                "shared_dim": self.shared_dim,
                "hidden_dim": self.hidden_dim,
                "distillation": self.distillation,
                "fc_state_dict": self._cpu_clone_dict(
                    self.fc.state_dict()
                ),
                "policy": {
                    "l0_weight": w0.detach().cpu().clone(),
                    "l0_bias": b0.detach().cpu().clone(),
                    "l2_weight": w2.detach().cpu().clone(),
                    "l2_bias": b2.detach().cpu().clone(),
                },
            }

    def save_policy_snapshot(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(
            self.export_effective_policy(),
            os.path.join(dirname, "policy_snapshot.pt"),
        )

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(
            self.fc,
            os.path.join(dirname, "fc.pt"),
        )
        torch.save(
            self.policy_pool,
            os.path.join(dirname, "policy_pool.pt"),
        )

    @staticmethod
    def load(dirname, map_location=None, **_ignored):
        del _ignored
        return FrozenCkaPolicy.load(
            dirname, map_location=map_location
        )


class FrozenCkaPolicy(nn.Module):
    """Inference-only exact categorical policy snapshot."""

    def __init__(self, snapshot):
        super().__init__()
        if int(snapshot.get("format_version", -1)) != (
            _POLICY_SNAPSHOT_FORMAT_VERSION
        ):
            raise ValueError(
                "unsupported Atari policy snapshot format"
            )
        if snapshot.get("policy_type") != "categorical":
            raise ValueError(
                "snapshot is not an Atari categorical policy"
            )

        self.composition_space = snapshot.get(
            "composition_space", "parameter"
        )
        self.obs_shape = tuple(
            int(x) for x in snapshot["obs_shape"]
        )
        self.act_dim = int(snapshot["act_dim"])
        self.shared_dim = int(
            snapshot.get("shared_dim", 512)
        )
        self.hidden_dim = int(
            snapshot.get("hidden_dim", 128)
        )
        self.distillation = bool(
            snapshot.get("distillation", False)
        )

        self.fc = shared(
            input_shape=self.obs_shape,
            output_dim=self.shared_dim,
        )
        self.fc.load_state_dict(
            snapshot["fc_state_dict"]
        )
        validate_shared_encoder(
            self.fc,
            input_shape=self.obs_shape,
            output_dim=self.shared_dim,
            source="policy_snapshot.pt encoder",
        )

        if self.composition_space == "policy":
            self.register_buffer(
                "mixture_weights",
                snapshot["mixture_weights"].clone(),
            )
            for key, value in snapshot[
                "policy_components"
            ].items():
                self.register_buffer(
                    "policy_" + key, value.clone()
                )
        else:
            for key, value in snapshot["policy"].items():
                self.register_buffer(
                    "policy_" + key, value.clone()
                )

    def _parameter_logits(self, features):
        h = F.relu(
            F.linear(
                features,
                self.policy_l0_weight,
                self.policy_l0_bias,
            )
        )
        return F.linear(
            h,
            self.policy_l2_weight,
            self.policy_l2_bias,
        )

    def policy_components(self, obs):
        features = self.fc(obs)
        if self.composition_space == "policy":
            head = {
                key: getattr(self, "policy_" + key)
                for key in _HEAD_KEYS
            }
            return (
                stacked_head_forward(features, head),
                self.mixture_weights,
            )
        logits = self._parameter_logits(features)
        return logits[:, None, :], logits.new_ones(1)

    def forward(self, obs):
        logits, weights = self.policy_components(obs)
        return categorical_mixture_logits(
            logits, weights
        )

    def action_distribution(self, obs):
        logits, weights = self.policy_components(obs)
        return categorical_mixture_distribution(
            logits, weights
        )

    @staticmethod
    def load(dirname, map_location=None):
        path = _require_file(
            os.path.join(
                os.fspath(dirname),
                "policy_snapshot.pt",
            ),
            "categorical policy snapshot",
        )
        snapshot = _torch_load(
            path, map_location=map_location
        )
        return FrozenCkaPolicy(snapshot)
