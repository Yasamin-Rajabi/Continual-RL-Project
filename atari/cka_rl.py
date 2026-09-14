"""Categorical CKA-RL agent for continual Atari PPO.

This module is the discrete-action counterpart of the HalfCheetah CKA-RL
agent.  The continual-learning mechanism is intentionally unchanged:

* one frozen/root base (for classic CKA),
* one trainable current-task contribution,
* a bounded pool of historical policy components,
* one shared alpha mixture over whole policy knowledge items,
* cosine pair selection for non-distillation conditions,
* replay-weighted symmetric policy KL for behavioral pair selection, and
* KL policy distillation when a bounded pool must merge two entries.

The distributional change is only

    diagonal Gaussian policy  ->  categorical policy over Atari actions.

The policy head consumes only the CNN feature vector.  Raw Atari pixels are
kept solely as representative replay states and are re-encoded when behavioral
KL or distillation is computed.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.distributions.categorical import Categorical

from knowledge_pools import BASE_FUSION_MODE, HeadPool
from shared_arch import ENCODER_FORMAT_VERSION, layer_init, shared, validate_shared_encoder

_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")
_POLICY_SNAPSHOT_FORMAT_VERSION = 2


def _torch_load(path, map_location=None):
    """Load project-owned checkpoints across recent PyTorch versions."""
    kwargs = {} if map_location is None else {"map_location": map_location}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:  # PyTorch versions predating weights_only=...
        return torch.load(path, **kwargs)


def _require_file(path, description: str):
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


def categorical_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Return KL(Categorical(p) || Categorical(q)), one value per state.

    Raw logits are used directly.  log_softmax makes the calculation invariant
    to adding a constant to every action logit of one policy at a given state.
    """
    if logits_p.shape != logits_q.shape:
        raise ValueError(
            f"categorical KL requires matching logits shapes; "
            f"got {tuple(logits_p.shape)} and {tuple(logits_q.shape)}"
        )
    if logits_p.ndim < 2:
        raise ValueError(
            f"categorical KL expects [..., num_actions] logits, got {tuple(logits_p.shape)}"
        )

    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)

    # Exact KL is non-negative.  Tiny negative values can occur numerically;
    # NaN/Inf should make a pair maximally unattractive rather than silently win.
    return torch.nan_to_num(
        kl, nan=1e12, posinf=1e12, neginf=0.0
    ).clamp_min(0.0)


def symmetric_categorical_kl(
    logits_a: torch.Tensor, logits_b: torch.Tensor
) -> torch.Tensor:
    """Statewise 0.5 * [KL(a||b) + KL(b||a)], for diagnostics/tests.

    The merge selector below intentionally follows the project's replay-weighted
    equation instead: KL(i||j) is averaged on B_i and KL(j||i) on B_j.
    """
    return 0.5 * (
        categorical_kl(logits_a, logits_b)
        + categorical_kl(logits_b, logits_a)
    )


class CkaRlAgent(nn.Module):
    """Shared Atari CNN + bounded categorical policy pool + PPO value head."""

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
        hidden_dim=128,
        shared_dim=512,
        train_shared=False,
        freeze_root_encoder=False,
        pretrained_encoder=None,
        # Explicit legacy compatibility knobs.  They are not Atari features.
        distill_observation_skip=False,
        encoder_linear_out=False,
        **unexpected_kwargs,
    ):
        super().__init__()

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
        self.train_shared = bool(train_shared)
        self.freeze_root_encoder = bool(freeze_root_encoder)
        self.last_merge_info = None
        self.last_distill_metrics = {}

        self._validate_init_args(
            alpha_init=alpha_init,
            alpha_major=alpha_major,
            distill_observation_skip=distill_observation_skip,
            encoder_linear_out=encoder_linear_out,
            unexpected_kwargs=unexpected_kwargs,
        )

        # One pool slot is now one complete categorical policy head.  There is
        # no separate mean/log-std pool for a discrete policy.
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

        if latest_dir is not None:
            pool_path = _require_file(
                os.path.join(os.fspath(latest_dir), "policy_pool.pt"),
                "latest categorical policy pool",
            )
            latest_policy_pool = _torch_load(pool_path, map_location="cpu")
            self.policy_pool.inherit_pool_from(latest_policy_pool)
            # A new task learns a fresh residual on top of historical knowledge.
            self.policy_pool.reset_own_to_zero()

        # One alpha vector controls whole categorical policy entries, exactly as
        # one alpha controlled an aligned mean/log-std pair in HalfCheetah.
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
        self.policy_pool.set_alpha(self.alpha, self.alpha_scale, self.alpha_mass)

        # Preserve the encoder lifecycle from the continuous implementation.
        # Only the implementation changes from vector MLP to Atari CNN.
        if latest_dir is not None and self.train_shared:
            encoder_path = _require_file(
                os.path.join(os.fspath(latest_dir), "fc.pt"),
                "latest trainable encoder",
            )
            logger.info(f"Loading latest trainable Atari encoder from {encoder_path}")
            self.fc = _torch_load(encoder_path, map_location="cpu")
            source = f"latest encoder {encoder_path}"
        elif pretrained_encoder is not None:
            encoder_path = _require_file(pretrained_encoder, "pretrained Atari encoder")
            logger.info(f"Loading pretrained Atari encoder from {encoder_path}")
            self.fc = _torch_load(encoder_path, map_location="cpu")
            source = f"pretrained encoder {encoder_path}"
        elif encoder_from_base and base_dir is not None:
            encoder_path = _require_file(
                os.path.join(os.fspath(base_dir), "fc.pt"),
                "root/base Atari encoder",
            )
            logger.info(f"Loading frozen root Atari encoder from {encoder_path}")
            self.fc = _torch_load(encoder_path, map_location="cpu")
            source = f"base encoder {encoder_path}"
        elif latest_dir is not None:
            encoder_path = _require_file(
                os.path.join(os.fspath(latest_dir), "fc.pt"),
                "latest Atari encoder",
            )
            logger.info(f"Loading shared Atari encoder from {encoder_path}")
            self.fc = _torch_load(encoder_path, map_location="cpu")
            source = f"latest encoder {encoder_path}"
        else:
            logger.info("Initializing root Atari CNN encoder from scratch")
            self.fc = shared(input_shape=self.obs_shape, output_dim=self.shared_dim)
            source = "new root Atari CNN encoder"

        validate_shared_encoder(
            self.fc,
            input_shape=self.obs_shape,
            output_dim=self.shared_dim,
            source=source,
        )

        # Default policy: task 0 learns the root representation; all later tasks
        # reuse/freeze that basis.  train_shared=True is the explicit continual-
        # finetuning ablation; freeze_root_encoder=True is random-frozen root.
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
            logger.info("Shared Atari CNN encoder frozen")

        # PPO value function is task-local and deliberately not part of policy
        # knowledge storage/behavioral merging, matching the policy-only method.
        self.critic = layer_init(nn.Linear(self.shared_dim, 1), std=1.0)

    # ------------------------------------------------------------------
    # Validation / alpha construction
    # ------------------------------------------------------------------
    def _validate_init_args(
        self,
        *,
        alpha_init,
        alpha_major,
        distill_observation_skip,
        encoder_linear_out,
        unexpected_kwargs,
    ):
        if len(self.obs_shape) != 3:
            raise ValueError(
                f"Atari CKA-RL expects CHW observations, got obs_shape={self.obs_shape}"
            )
        if self.act_dim < 2:
            raise ValueError("act_dim must be >= 2 for a categorical Atari policy")
        if self.shared_dim < 1 or self.hidden_dim < 1:
            raise ValueError("shared_dim and hidden_dim must be >= 1")
        if self.pool_size < 2:
            raise ValueError("pool_size must be >= 2")
        if self.fusion_mode not in (BASE_FUSION_MODE, "weight_delta"):
            raise ValueError(f"unknown fusion_mode={self.fusion_mode!r}")
        if self.fusion_mode == BASE_FUSION_MODE and self.use_alpha_mass:
            raise ValueError(
                "use_alpha_mass is only valid with fusion_mode='weight_delta'"
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
            raise ValueError("alpha_major must be in (0, 1) for alpha_init='Major'")
        if self.max_distill_buffer < 2:
            raise ValueError("max_distill_buffer must be >= 2")
        if self.similarity_samples < 2:
            raise ValueError("similarity_samples must be >= 2")
        if self.distill_max_samples < 2:
            raise ValueError("distill_max_samples must be >= 2")
        if self.distill_batch_size < 1:
            raise ValueError("distill_batch_size must be >= 1")
        if self.distill_lr <= 0:
            raise ValueError("distill_lr must be > 0")
        if not 0.0 <= self.distill_test_frac < 1.0:
            raise ValueError("distill_test_frac must be in [0, 1)")
        if self.distillation and self.distill_epochs < 1:
            raise ValueError("distill_epochs must be >= 1 when distillation is enabled")
        if self.distill_epochs < 0:
            raise ValueError("distill_epochs must be >= 0")

        # Atari deliberately does NOT concatenate 28k raw pixels to CNN output.
        if bool(distill_observation_skip):
            raise ValueError(
                "distill_observation_skip is not supported for Atari. "
                "The policy head must consume CNN features only."
            )
        if bool(encoder_linear_out):
            raise ValueError(
                "encoder_linear_out is a legacy HalfCheetah MLP option and is "
                "not supported by the fixed Atari CNN architecture."
            )
        if unexpected_kwargs:
            names = ", ".join(sorted(str(k) for k in unexpected_kwargs))
            raise TypeError(f"unexpected CkaRlAgent argument(s): {names}")

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
            alpha = nn.Parameter(torch.zeros(num_vectors), requires_grad=False)
        elif alpha_init == "Uniform" or num_vectors == 1:
            alpha = nn.Parameter(
                torch.ones(num_vectors, dtype=torch.float32) * float(alpha_factor),
                requires_grad=True,
            )
        elif alpha_init == "Randn":
            alpha = nn.Parameter(
                torch.randn(num_vectors, dtype=torch.float32) / max(num_vectors, 1),
                requires_grad=True,
            )
        elif alpha_init == "Major" and num_vectors > 1:
            vals = [
                np.log((1.0 - float(alpha_major)) / (num_vectors - 1))
                for _ in range(num_vectors - 1)
            ]
            vals.append(np.log(float(alpha_major)))
            alpha = nn.Parameter(
                torch.tensor(vals, dtype=torch.float32), requires_grad=True
            )
        else:  # guarded by constructor validation
            raise RuntimeError(f"unsupported alpha_init={alpha_init!r}")

        # Explicit regimes preserved from the continuous implementation:
        #   off     -> fixed scale 1
        #   learned -> trainable scale initialized at 1
        #   fixed   -> non-trainable scale 5
        scale_init = 5.0 if fix_alpha_scale else 1.0
        alpha_scale = nn.Parameter(
            torch.tensor([scale_init], dtype=torch.float32),
            requires_grad=(
                bool(use_alpha_scale)
                and not bool(fix_alpha_scale)
                and not bool(fix_alpha)
            ),
        )
        alpha_mass = (
            nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=not fix_alpha)
            if use_alpha_mass
            else None
        )
        return alpha, alpha_scale, alpha_mass

    def log_alphas(self):
        logger.info(f"alpha={self.alpha}")
        logger.info(f"alpha_scale={self.alpha_scale}")
        if self.alpha_mass is not None:
            logger.info(f"alpha_mass(raw)={self.alpha_mass}")
            logger.info(
                f"alpha_mass(effective)={self.policy_pool.effective_alpha_mass()}"
            )

    # ------------------------------------------------------------------
    # PPO policy/value interface
    # ------------------------------------------------------------------
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode already-normalized [0,1] Atari observations."""
        return self.fc(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return raw categorical action logits for normalized observations."""
        return self.policy_pool(self.encode(obs))

    def action_distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encode(obs))

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        log_writter=None,  # retained for compatibility with the old PPO caller
        global_step=None,
        **_ignored,
    ):
        del log_writter, global_step, _ignored
        features = self.encode(obs)
        logits = self.policy_pool(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        action = action.long().reshape(-1)
        value = self.critic(features)
        return action, dist.log_prob(action), dist.entropy(), value

    # ------------------------------------------------------------------
    # Pool lifecycle
    # ------------------------------------------------------------------
    def set_own_buffer(self, buffer):
        """Attach raw uint8 Atari states to the current policy contribution."""
        if buffer is not None:
            obs = buffer.get("obs") if isinstance(buffer, dict) else None
            if not isinstance(obs, np.ndarray):
                raise ValueError("Atari merge buffer must contain NumPy array 'obs'")
            if obs.dtype != np.uint8:
                raise ValueError(
                    f"Atari merge observations must be stored as uint8, got {obs.dtype}. "
                    "Store raw frames and normalize only when they are re-encoded."
                )
            if tuple(obs.shape[1:]) != self.obs_shape:
                raise ValueError(
                    f"buffer observation shape {tuple(obs.shape[1:])} does not match "
                    f"agent obs_shape={self.obs_shape}"
                )
        self.policy_pool.set_own_buffer(buffer)

    def set_base(self):
        self.policy_pool.set_base()

    # ------------------------------------------------------------------
    # Behavioral similarity
    # ------------------------------------------------------------------
    def _sample_reference_observations(self):
        buffers = [entry.get("buffer") for entry in self.policy_pool.pool]
        if any(
            buf is None
            or "obs" not in buf
            or not isinstance(buf["obs"], np.ndarray)
            or len(buf["obs"]) == 0
            for buf in buffers
        ):
            raise RuntimeError(
                "Behavioral KL merging needs a non-empty observation buffer for "
                "every pool slot.  Rerun the continual chain if an old checkpoint "
                "predates Atari merge buffers."
            )

        # similarity_samples is the approximate budget for ONE candidate pair:
        # about half from each immediate parent lineage.
        per_slot = max(1, self.similarity_samples // 2)
        samples = []
        for slot, buf in enumerate(buffers):
            obs = buf["obs"]
            if obs.dtype != np.uint8:
                raise RuntimeError(
                    f"pool slot {slot} stores {obs.dtype} observations; Atari "
                    "behavioral buffers must contain raw uint8 frames"
                )
            if tuple(obs.shape[1:]) != self.obs_shape:
                raise RuntimeError(
                    f"pool slot {slot} observation shape {tuple(obs.shape[1:])} "
                    f"does not match {self.obs_shape}"
                )
            take = min(per_slot, len(obs))
            idx = np.random.choice(len(obs), size=take, replace=False)
            samples.append(obs[idx])
        return samples

    def _encode_obs(self, obs: np.ndarray, batch_size: int = 512) -> torch.Tensor:
        """Encode raw uint8 replay frames exactly as PPO does: float / 255."""
        if not isinstance(obs, np.ndarray) or len(obs) == 0:
            raise ValueError("_encode_obs requires a non-empty NumPy observation array")
        if obs.dtype != np.uint8:
            raise ValueError(f"expected raw uint8 Atari observations, got {obs.dtype}")
        if tuple(obs.shape[1:]) != self.obs_shape:
            raise ValueError(
                f"observation batch shape {tuple(obs.shape[1:])} != {self.obs_shape}"
            )
        if int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1")

        device = self.policy_pool.base_l0_weight.device
        chunks = []
        with torch.no_grad():
            for start in range(0, len(obs), int(batch_size)):
                x = torch.as_tensor(
                    obs[start : start + int(batch_size)],
                    dtype=torch.float32,
                    device=device,
                ) / 255.0
                chunks.append(self.fc(x))
        return torch.cat(chunks, dim=0)

    def _entry_logits(self, features: torch.Tensor, index: int):
        return self.policy_pool.forward_entry(features, index)

    def _select_cosine_pair(self):
        """Select the highest-cosine pair in the stored policy representation."""
        n = self.policy_pool.pool_length()
        if n < 2:
            raise RuntimeError("cannot select a merge pair from fewer than two pool entries")

        vectors = []
        for index in range(n):
            entry = self.policy_pool.pool[index]
            vectors.append(
                torch.cat([entry[key].reshape(-1) for key in _HEAD_KEYS], dim=0)
            )

        device = vectors[0].device
        matrix = torch.full((n, n), -float("inf"), device=device)
        finite_values = []
        with torch.no_grad():
            for i in range(n):
                for j in range(i + 1, n):
                    score = F.cosine_similarity(vectors[i], vectors[j], dim=0)
                    score = torch.nan_to_num(score, nan=-float("inf"))
                    matrix[i, j] = score
                    matrix[j, i] = score
                    if torch.isfinite(score):
                        finite_values.append(float(score.item()))

            flat_idx = int(torch.argmax(matrix).item())
            idx1, idx2 = divmod(flat_idx, n)
            if idx1 == idx2 or not torch.isfinite(matrix[idx1, idx2]):
                raise RuntimeError(
                    "cosine pair selection failed: no finite pairwise similarity"
                )
            selected = float(matrix[idx1, idx2].item())

        report_matrix = matrix.detach().cpu().clone()
        report_matrix.fill_diagonal_(1.0)
        stats = {
            "idx1": int(idx1),
            "idx2": int(idx2),
            "similarity_metric": "cosine",
            "cosine_similarity": selected,
            "pairwise_cosine_min": float(np.min(finite_values)),
            "pairwise_cosine_mean": float(np.mean(finite_values)),
            "pairwise_cosine_max": float(np.max(finite_values)),
            "pairwise_cosine_similarity": report_matrix.numpy().tolist(),
        }
        logger.info(
            f"[cosine merge] pair=({idx1},{idx2}) cosine={selected:.6f}"
        )
        return idx1, idx2, stats

    def _select_behavioral_pair(self):
        """Select minimum replay-weighted symmetric categorical policy KL.

        This follows the requested equation exactly:

            delta(i,j) = 1/2 [
                E_{s~B_i} KL(pi_i(.|s) || pi_j(.|s))
              + E_{s~B_j} KL(pi_j(.|s) || pi_i(.|s))
            ].

        The first direction is therefore evaluated only on B_i and the reverse
        direction only on B_j.  This is intentionally different from merely
        pooling all states and taking a statewise symmetric KL.
        """
        n = self.policy_pool.pool_length()
        if n < 2:
            raise RuntimeError("cannot select a merge pair from fewer than two pool entries")

        obs_by_slot = self._sample_reference_observations()
        features_by_slot = [self._encode_obs(obs) for obs in obs_by_slot]

        with torch.no_grad():
            # outputs[policy_index][state_source_index] = categorical logits.
            outputs = [
                [
                    self._entry_logits(features_by_slot[source], policy)
                    for source in range(n)
                ]
                for policy in range(n)
            ]

            device = features_by_slot[0].device
            matrix = torch.full((n, n), float("inf"), device=device)
            pair_rows = torch.zeros((n, n), dtype=torch.int64, device=device)
            finite_values = []

            for i in range(n):
                for j in range(i + 1, n):
                    kl_i_to_j_on_i = categorical_kl(
                        outputs[i][i], outputs[j][i]
                    )
                    kl_j_to_i_on_j = categorical_kl(
                        outputs[j][j], outputs[i][j]
                    )
                    score = 0.5 * (
                        kl_i_to_j_on_i.mean() + kl_j_to_i_on_j.mean()
                    )
                    score = torch.nan_to_num(
                        score, nan=float("inf"), posinf=float("inf")
                    )
                    matrix[i, j] = score
                    matrix[j, i] = score
                    pair_rows[i, j] = pair_rows[j, i] = int(
                        kl_i_to_j_on_i.numel() + kl_j_to_i_on_j.numel()
                    )
                    if torch.isfinite(score):
                        finite_values.append(float(score.item()))

            flat_idx = int(torch.argmin(matrix).item())
            idx1, idx2 = divmod(flat_idx, n)
            if idx1 == idx2 or not torch.isfinite(matrix[idx1, idx2]):
                raise RuntimeError(
                    "behavioral KL pair selection failed: no finite pairwise KL"
                )

            selected = float(matrix[idx1, idx2].item())
            selected_rows = int(pair_rows[idx1, idx2].item())
            selected_directional_kl = torch.cat(
                [
                    categorical_kl(
                        outputs[idx1][idx1], outputs[idx2][idx1]
                    ),
                    categorical_kl(
                        outputs[idx2][idx2], outputs[idx1][idx2]
                    ),
                ],
                dim=0,
            )
            selected_state_kl_p95 = float(
                torch.quantile(selected_directional_kl, 0.95).item()
            )
            selected_state_kl_max = float(selected_directional_kl.max().item())

        report_matrix = matrix.detach().cpu().clone()
        report_matrix.fill_diagonal_(0.0)
        stats = {
            "idx1": int(idx1),
            "idx2": int(idx2),
            "similarity_metric": "symmetric_kl",
            "symmetric_kl": selected,
            "pairwise_kl_min": float(np.min(finite_values)),
            "pairwise_kl_mean": float(np.mean(finite_values)),
            "pairwise_kl_max": float(np.max(finite_values)),
            "selected_state_kl_p95": selected_state_kl_p95,
            "selected_state_kl_max": selected_state_kl_max,
            "similarity_states": selected_rows,
            "reference_rows_per_slot": [int(len(x)) for x in obs_by_slot],
            "pairwise_symmetric_kl": report_matrix.numpy().tolist(),
        }
        logger.info(
            f"[behavioral merge] pair=({idx1},{idx2}) "
            f"replay-weighted symmetric_KL={selected:.6f} "
            f"p95={selected_state_kl_p95:.6f} "
            f"max={selected_state_kl_max:.6f} over {selected_rows} directional states"
        )
        return idx1, idx2, stats

    # ------------------------------------------------------------------
    # Categorical policy distillation
    # ------------------------------------------------------------------
    @staticmethod
    def _params_to_effective(pool: HeadPool, params: Dict[str, torch.Tensor]):
        if pool.fusion_mode == BASE_FUSION_MODE:
            return (
                pool.base_l0_weight + params["l0_weight"],
                pool.base_l0_bias + params["l0_bias"],
                pool.base_l2_weight + params["l2_weight"],
                pool.base_l2_bias + params["l2_bias"],
            )
        return tuple(params[key] for key in _HEAD_KEYS)

    @staticmethod
    def _head_forward_from_params(
        pool: HeadPool,
        features: torch.Tensor,
        params: Dict[str, torch.Tensor],
    ):
        return pool._forward_with_weights(
            features, CkaRlAgent._params_to_effective(pool, params)
        )

    @staticmethod
    def _buffer_lineage(buffer, key="task_ids"):
        if buffer is None or key not in buffer:
            return {}
        ids, counts = np.unique(
            np.asarray(buffer[key]).reshape(-1), return_counts=True
        )
        return {
            str(int(source_id)): int(count)
            for source_id, count in zip(ids, counts)
        }

    def _balanced_parent_data(self, idx1: int, idx2: int):
        buf1 = self.policy_pool.pool[idx1].get("buffer")
        buf2 = self.policy_pool.pool[idx2].get("buffer")
        if buf1 is None or buf2 is None:
            raise RuntimeError(
                "distillation requested but a selected pool entry has no observation buffer"
            )

        max_each = max(1, self.distill_max_samples // 2)
        obs_parts, teacher_ids = [], []
        for teacher_id, buf in enumerate((buf1, buf2)):
            if "obs" not in buf or not isinstance(buf["obs"], np.ndarray):
                raise RuntimeError("distillation parent buffer is missing NumPy 'obs'")
            obs = buf["obs"]
            if len(obs) == 0:
                raise RuntimeError("distillation parent buffer is empty")
            if obs.dtype != np.uint8:
                raise RuntimeError(
                    f"distillation parent stores {obs.dtype}; expected raw uint8 Atari frames"
                )
            if tuple(obs.shape[1:]) != self.obs_shape:
                raise RuntimeError(
                    f"distillation parent observation shape {tuple(obs.shape[1:])} "
                    f"does not match {self.obs_shape}"
                )
            take = min(len(obs), max_each)
            sample_idx = np.random.choice(len(obs), size=take, replace=False)
            obs_parts.append(obs[sample_idx])
            teacher_ids.append(np.full(take, teacher_id, dtype=np.int64))

        return (
            np.concatenate(obs_parts, axis=0),
            np.concatenate(teacher_ids, axis=0),
        )

    def _distill_pair(self, idx1: int, idx2: int):
        """KL-distill two categorical parent policies into one student head.

        The discrete procedure is the exact analogue of the Gaussian version:
        balanced immediate-parent replay, immediate-parent teacher assignment,
        arithmetic parameter-average initialization, teacher->student KL,
        stratified held-out validation, and optional validation-best selection.
        """
        obs, teacher_ids_np = self._balanced_parent_data(idx1, idx2)
        features = self._encode_obs(obs)
        device = features.device
        teacher_ids = torch.as_tensor(
            teacher_ids_np, dtype=torch.long, device=device
        )

        # Teacher distributions are generated on demand from the two retained
        # parent heads; they are not redundantly stored in replay buffers.
        with torch.no_grad():
            logits1 = self._entry_logits(features, idx1)
            logits2 = self._entry_logits(features, idx2)
            parent2_mask = teacher_ids.unsqueeze(-1).bool()
            teacher_logits = torch.where(parent2_mask, logits2, logits1)

        # Epoch 0 is the arithmetic parent parameter average, exactly as in the
        # continuous implementation.
        initial_params = self.policy_pool.average_pair_params(idx1, idx2)
        student_params = {
            key: value.detach().clone().requires_grad_(True)
            for key, value in initial_params.items()
        }
        trainables = list(student_params.values())
        optimizer = torch.optim.Adam(trainables, lr=self.distill_lr)

        # Stratify by immediate parent so both train and held-out diagnostics
        # contain both teachers whenever the sample count permits it.
        train_parts, test_parts = [], []
        for teacher_id in (0, 1):
            parent_idx = torch.nonzero(
                teacher_ids == teacher_id, as_tuple=False
            ).flatten()
            if parent_idx.numel() == 0:
                raise RuntimeError(
                    f"distillation split unexpectedly has no rows for parent {teacher_id}"
                )
            parent_idx = parent_idx[
                torch.randperm(parent_idx.numel(), device=device)
            ]
            n_parent_test = (
                int(parent_idx.numel() * self.distill_test_frac)
                if self.distill_test_frac > 0
                else 0
            )
            n_parent_test = min(
                n_parent_test, max(parent_idx.numel() - 1, 0)
            )
            test_parts.append(parent_idx[:n_parent_test])
            train_parts.append(parent_idx[n_parent_test:])

        train_idx = torch.cat(train_parts)
        test_idx = torch.cat(test_parts)
        if train_idx.numel() == 0:
            raise RuntimeError("distillation produced an empty training split")
        train_idx = train_idx[
            torch.randperm(train_idx.numel(), device=device)
        ]
        if test_idx.numel() > 0:
            test_idx = test_idx[
                torch.randperm(test_idx.numel(), device=device)
            ]

        def student_logits(batch_features):
            return self._head_forward_from_params(
                self.policy_pool, batch_features, student_params
            )

        @torch.no_grad()
        def kl_summary(indices):
            if indices.numel() == 0:
                return None
            values = categorical_kl(
                teacher_logits[indices], student_logits(features[indices])
            )
            return {
                "mean": float(values.mean().item()),
                "p95": float(torch.quantile(values, 0.95).item()),
                "max": float(values.max().item()),
            }

        def clone_student_params():
            return {
                key: value.detach().clone()
                for key, value in student_params.items()
            }

        validation_idx = test_idx if test_idx.numel() > 0 else train_idx
        initial_val = kl_summary(validation_idx)
        best_val_kl = (
            float("inf") if initial_val is None else initial_val["mean"]
        )
        best_epoch = 0
        best_params = clone_student_params()

        for epoch in range(1, self.distill_epochs + 1):
            shuffled = train_idx[
                torch.randperm(train_idx.numel(), device=device)
            ]
            for start in range(0, shuffled.numel(), self.distill_batch_size):
                idx = shuffled[start : start + self.distill_batch_size]
                loss = categorical_kl(
                    teacher_logits[idx], student_logits(features[idx])
                ).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainables, max_norm=10.0)
                optimizer.step()

            val = kl_summary(validation_idx)
            if val is not None and val["mean"] < best_val_kl:
                best_val_kl = val["mean"]
                best_epoch = epoch
                best_params = clone_student_params()

        if self.distill_select_best_val:
            with torch.no_grad():
                for key in _HEAD_KEYS:
                    student_params[key].copy_(best_params[key])
            selected_epoch = int(best_epoch)
            selected_val_kl = float(best_val_kl)
        else:
            selected_epoch = int(self.distill_epochs)
            final_val = kl_summary(validation_idx)
            selected_val_kl = (
                float("nan") if final_val is None else float(final_val["mean"])
            )

        @torch.no_grad()
        def detailed_metrics(indices):
            if indices.numel() == 0:
                return None, None, None, None
            s_logits = student_logits(features[indices])
            kl_values = categorical_kl(teacher_logits[indices], s_logits)
            teacher_probs = F.softmax(teacher_logits[indices], dim=-1)
            student_probs = F.softmax(s_logits, dim=-1)
            prob_mse = F.mse_loss(student_probs, teacher_probs).item()
            return (
                float(kl_values.mean().item()),
                float(torch.quantile(kl_values, 0.95).item()),
                float(kl_values.max().item()),
                float(prob_mse),
            )

        train_metrics = detailed_metrics(train_idx)
        test_metrics = detailed_metrics(test_idx)
        out_params = {
            key: value.detach().clone()
            for key, value in student_params.items()
        }
        metrics_out = {
            "policy/distill_train_kl": train_metrics[0],
            "policy/distill_test_kl": test_metrics[0],
            "policy/distill_train_kl_p95": train_metrics[1],
            "policy/distill_test_kl_p95": test_metrics[1],
            "policy/distill_train_kl_max": train_metrics[2],
            "policy/distill_test_kl_max": test_metrics[2],
            "policy/distill_train_prob_mse": train_metrics[3],
            "policy/distill_test_prob_mse": test_metrics[3],
            "policy/distill_best_epoch": int(best_epoch),
            "policy/distill_best_val_kl": float(best_val_kl),
            "policy/distill_selected_epoch": int(selected_epoch),
            "policy/distill_selected_val_kl": float(selected_val_kl),
            "policy/distill_select_best_val": float(
                self.distill_select_best_val
            ),
            "policy/distill_initial_val_kl": (
                None if initial_val is None else float(initial_val["mean"])
            ),
            "policy/distill_rows": int(len(obs)),
        }

        logger.info(
            f"[categorical policy distill] rows={len(obs)} "
            f"best_epoch={best_epoch} selected_epoch={selected_epoch} "
            f"train_KL={train_metrics[0]:.6f} "
            f"test_KL={test_metrics[0] if test_metrics[0] is not None else 'n/a'} "
            f"train_p95={train_metrics[1]:.6f} "
            f"test_p95={test_metrics[1] if test_metrics[1] is not None else 'n/a'}"
        )
        return out_params, metrics_out

    # ------------------------------------------------------------------
    # Finalization / bounded merge
    # ------------------------------------------------------------------
    def finalize(self):
        """Insert the new slot and merge exactly one pair if capacity is exceeded."""
        self.last_merge_info = None
        self.last_distill_metrics = {}

        # Under the normal one-task-at-a-time lifecycle, the inherited pool is
        # already bounded.  Failing here catches corrupted/stale checkpoints
        # instead of silently requiring multiple merges in one task finalization.
        if self.policy_pool.pool_length() > self.pool_size:
            raise RuntimeError(
                f"inherited pool length {self.policy_pool.pool_length()} exceeds "
                f"configured capacity {self.pool_size}"
            )

        self.policy_pool.finalize_own_contribution()
        if not self.policy_pool.needs_merge():
            return

        if self.distillation:
            idx1, idx2, merge_info = self._select_behavioral_pair()
            merged_params, distill_metrics = self._distill_pair(idx1, idx2)
            self.last_distill_metrics = distill_metrics
            used_distillation = True
        else:
            idx1, idx2, merge_info = self._select_cosine_pair()
            merged_params = self.policy_pool.average_pair_params(idx1, idx2)
            used_distillation = False

        buf1 = self.policy_pool.pool[idx1].get("buffer")
        buf2 = self.policy_pool.pool[idx2].get("buffer")
        merged_buffer = HeadPool.merge_buffers(
            buf1, buf2, self.max_distill_buffer
        )

        merge_info.update(
            {
                "used_distillation": bool(used_distillation),
                "pool_size_before": int(self.policy_pool.pool_length()),
                "pool_size_after": int(self.policy_pool.pool_length() - 1),
                "parent_1_lineage": self._buffer_lineage(buf1, "task_ids"),
                "parent_2_lineage": self._buffer_lineage(buf2, "task_ids"),
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
            idx1, idx2, merged_params, merged_buffer, merge_info
        )
        self.last_merge_info = dict(merge_info)

        if self.policy_pool.pool_length() > self.pool_size:
            raise RuntimeError("pool remained over capacity after one merge")

        if "policy/distill_train_kl" in self.last_distill_metrics:
            self.policy_pool.last_distill_train_kl = self.last_distill_metrics[
                "policy/distill_train_kl"
            ]
            self.policy_pool.last_distill_test_kl = self.last_distill_metrics[
                "policy/distill_test_kl"
            ]

    def get_distill_metrics(self):
        return dict(self.last_distill_metrics)

    def get_merge_info(self):
        return None if self.last_merge_info is None else dict(self.last_merge_info)

    # ------------------------------------------------------------------
    # Saving / exact inference snapshots
    # ------------------------------------------------------------------
    @staticmethod
    def _cpu_clone_dict(d):
        return {
            key: value.detach().cpu().clone()
            for key, value in d.items()
        }

    def export_effective_policy(self):
        """Export the exact current pre-finalize categorical policy.

        This snapshot is authoritative for evaluation because finalization can
        change pool topology, making the pre-finalize alpha vector semantically
        stale.  Continuation instead loads fc.pt + policy_pool.pt and constructs a
        fresh alpha vector for the inherited pool length.
        """
        with torch.no_grad():
            w0, b0, w2, b2 = self.policy_pool._effective()
            return {
                "format_version": _POLICY_SNAPSHOT_FORMAT_VERSION,
                "encoder_format_version": int(getattr(self.fc, "format_version", ENCODER_FORMAT_VERSION)),
                "policy_type": "categorical",
                "obs_shape": tuple(self.obs_shape),
                "act_dim": int(self.act_dim),
                "shared_dim": int(self.shared_dim),
                "hidden_dim": int(self.hidden_dim),
                "fusion_mode": self.fusion_mode,
                "distillation": bool(self.distillation),
                "fc_state_dict": self._cpu_clone_dict(self.fc.state_dict()),
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
        """Save continuation state; PPO critic is intentionally task-local."""
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.fc, os.path.join(dirname, "fc.pt"))
        torch.save(self.policy_pool, os.path.join(dirname, "policy_pool.pt"))

    @staticmethod
    def load(dirname, map_location=None, **_ignored):
        snapshot_path = _require_file(
            os.path.join(os.fspath(dirname), "policy_snapshot.pt"),
            "categorical policy snapshot",
        )
        del snapshot_path, _ignored
        return FrozenCkaPolicy.load(dirname, map_location=map_location)


class FrozenCkaPolicy(nn.Module):
    """Compact inference-only categorical policy from policy_snapshot.pt."""

    def __init__(self, snapshot):
        super().__init__()
        if not isinstance(snapshot, dict):
            raise TypeError("policy snapshot must be a dictionary")
        version = int(snapshot.get("format_version", -1))
        if version != _POLICY_SNAPSHOT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported policy snapshot format_version={version}; "
                f"expected {_POLICY_SNAPSHOT_FORMAT_VERSION}"
            )
        encoder_version = int(snapshot.get("encoder_format_version", -1))
        if encoder_version != ENCODER_FORMAT_VERSION:
            raise ValueError(
                f"snapshot encoder_format_version={encoder_version}; "
                f"expected {ENCODER_FORMAT_VERSION}"
            )
        if snapshot.get("policy_type", "categorical") != "categorical":
            raise ValueError("snapshot is not a categorical Atari policy")

        self.obs_shape = tuple(int(x) for x in snapshot["obs_shape"])
        self.act_dim = int(snapshot["act_dim"])
        self.shared_dim = int(snapshot.get("shared_dim", 512))
        self.hidden_dim = int(
            snapshot.get(
                "hidden_dim",
                snapshot["policy"]["l0_weight"].shape[0],
            )
        )

        self.fc = shared(
            input_shape=self.obs_shape,
            output_dim=self.shared_dim,
        )
        self.fc.load_state_dict(snapshot["fc_state_dict"])
        validate_shared_encoder(
            self.fc,
            input_shape=self.obs_shape,
            output_dim=self.shared_dim,
            source="policy_snapshot.pt encoder",
        )

        policy = snapshot.get("policy")
        if not isinstance(policy, dict):
            raise ValueError("snapshot is missing the categorical policy head")
        for key in _HEAD_KEYS:
            if key not in policy or not torch.is_tensor(policy[key]):
                raise ValueError(f"snapshot policy is missing tensor {key!r}")

        expected_shapes = {
            "l0_weight": (self.hidden_dim, self.shared_dim),
            "l0_bias": (self.hidden_dim,),
            "l2_weight": (self.act_dim, self.hidden_dim),
            "l2_bias": (self.act_dim,),
        }
        for key, expected in expected_shapes.items():
            got = tuple(policy[key].shape)
            if got != expected:
                raise ValueError(
                    f"snapshot {key} shape {got} does not match expected {expected}"
                )
            self.register_buffer(
                f"policy_{key}", policy[key].detach().clone()
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return logits for already-normalized [0,1] Atari observations."""
        features = self.fc(obs)
        h = F.relu(
            F.linear(features, self.policy_l0_weight, self.policy_l0_bias)
        )
        return F.linear(h, self.policy_l2_weight, self.policy_l2_bias)

    def action_distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(obs))

    @staticmethod
    def load(dirname, map_location=None):
        path = _require_file(
            os.path.join(os.fspath(dirname), "policy_snapshot.pt"),
            "categorical policy snapshot",
        )
        snapshot = _torch_load(path, map_location=map_location)
        return FrozenCkaPolicy(snapshot)
