"""Squashed-Gaussian CKA-RL agent for continual PointMaze SAC.

Continuous-control counterpart of the Atari categorical agent.  The continual
mechanism is deliberately identical:

* one frozen root/base plus one trainable current-task contribution;
* a bounded pool of historical policy heads;
* one shared alpha mixture over whole policy knowledge items;
* cosine pair selection without distillation, replay-weighted symmetric policy
  KL with it;
* KL distillation when a bounded pool must merge two entries;
* parameter-space or exact policy-space composition.

Only the distribution changes: diagonal Gaussian over squashed actions instead
of a categorical over discrete actions.  Because ``tanh`` is an invertible map
and KL is invariant under invertible maps, every divergence below is computed
in closed form on the pre-squash Gaussians and is still exactly the divergence
between the executed policies.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from knowledge_pools import BASE_FUSION_MODE, HeadPool, balanced_lineage_indices
from policy_composition import gaussian_mixture_distribution, stacked_head_forward
from policy_space import PolicySpaceMixin
from policy_utils import clamp_log_std, gaussian_kl, symmetric_gaussian_kl
from shared_arch import (
    ENCODER_FORMAT_VERSION,
    TwinCritic,
    shared,
    validate_shared_encoder,
)

_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")
_POLICY_SNAPSHOT_FORMAT_VERSION = 1


def _torch_load(path, map_location=None):
    kwargs = {} if map_location is None else {"map_location": map_location}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:  # older PyTorch without weights_only
        return torch.load(path, **kwargs)


def _require_file(path, description: str) -> str:
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


class CkaRlAgent(PolicySpaceMixin, nn.Module):
    """Shared MLP encoder + bounded Gaussian policy pool + task-local twin critic."""

    SNAPSHOT_FORMAT_VERSION = _POLICY_SNAPSHOT_FORMAT_VERSION

    def __init__(
        self,
        obs_dim,
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
        max_distill_buffer=20_000,
        distill_test_frac=0.2,
        distill_select_best_val=True,
        distill_epochs=8,
        distill_lr=3e-4,
        distill_batch_size=256,
        distill_max_samples=4_000,
        similarity_samples=1_024,
        balance_source_lineages=True,
        hidden_dim=256,
        shared_dim=256,
        train_shared=False,
        freeze_root_encoder=False,
        pretrained_encoder=None,
        composition_space="parameter",
        projection_epochs=16,
        projection_max_samples=4_000,
        projection_samples=8,
        policy_student_replay=False,
    ):
        super().__init__()

        if composition_space not in ("parameter", "policy"):
            raise ValueError("composition_space must be 'parameter' or 'policy'")
        if composition_space == "policy" and use_alpha_mass and not constrain_alpha_mass:
            raise ValueError("a probability mixture requires bounded sigmoid alpha-mass")

        self.composition_space = composition_space
        self.policy_student_replay = bool(policy_student_replay)
        if self.policy_student_replay and (
            composition_space != "policy"
            or not use_alpha_mass
            or fusion_mode != "weight_delta"
        ):
            raise ValueError(
                "policy_student_replay requires policy composition, weight_delta "
                "and alpha-mass"
            )

        self.obs_dim = int(obs_dim)
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
        self.projection_epochs = int(projection_epochs)
        self.projection_max_samples = int(projection_max_samples)
        self.projection_samples = int(projection_samples)

        self.mixture_warmup = False
        self.pool_only = False
        self.last_projection_metrics = {}
        self.last_merge_info = None
        self.last_distill_metrics = {}

        self._validate_init_args(alpha_init=alpha_init, alpha_major=alpha_major)

        self.policy_pool = HeadPool(
            "gaussian",
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
                "latest Gaussian policy pool",
            )
            latest_pool = _torch_load(pool_path, map_location="cpu")
            if getattr(latest_pool, "composition_space", "parameter") != composition_space:
                raise ValueError(
                    "cannot continue a chain with a different composition space; "
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
        self.policy_pool.set_alpha(self.alpha, self.alpha_scale, self.alpha_mass)

        # ---- encoder lifecycle (identical to the Atari implementation) ----
        if latest_dir is not None and self.train_shared:
            path = _require_file(
                os.path.join(os.fspath(latest_dir), "fc.pt"), "latest trainable encoder"
            )
            self.fc = _torch_load(path, map_location="cpu")
            source = f"latest encoder {path}"
        elif pretrained_encoder is not None:
            path = _require_file(pretrained_encoder, "pretrained encoder")
            self.fc = _torch_load(path, map_location="cpu")
            source = f"pretrained encoder {path}"
        elif encoder_from_base and base_dir is not None:
            path = _require_file(
                os.path.join(os.fspath(base_dir), "fc.pt"), "root/base encoder"
            )
            self.fc = _torch_load(path, map_location="cpu")
            source = f"base encoder {path}"
        elif latest_dir is not None:
            path = _require_file(
                os.path.join(os.fspath(latest_dir), "fc.pt"), "latest encoder"
            )
            self.fc = _torch_load(path, map_location="cpu")
            source = f"latest encoder {path}"
        else:
            self.fc = shared(self.obs_dim, output_dim=self.shared_dim)
            source = "new root encoder"

        validate_shared_encoder(
            self.fc, self.obs_dim, self.shared_dim, source=source
        )

        should_freeze = not self.train_shared and (
            pretrained_encoder is not None
            or latest_dir is not None
            or self.freeze_root_encoder
        )
        if should_freeze:
            self.fc.requires_grad_(False)

        # The SAC critic is task-local and never enters the policy pool.
        self.critic = TwinCritic(self.shared_dim, self.act_dim)
        self.critic_target = TwinCritic(self.shared_dim, self.act_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.initialize_policy_space_own()

    # ------------------------------------------------------------------
    def _validate_init_args(self, *, alpha_init, alpha_major):
        if self.obs_dim < 1 or self.act_dim < 1:
            raise ValueError("obs_dim and act_dim must be >= 1")
        if self.pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        if self.fusion_mode not in (BASE_FUSION_MODE, "weight_delta"):
            raise ValueError(f"unknown fusion_mode={self.fusion_mode!r}")
        if self.fusion_mode == BASE_FUSION_MODE and self.use_alpha_mass:
            raise ValueError("use_alpha_mass is only valid with weight_delta")
        if self.fix_alpha_scale and self.use_alpha_scale:
            raise ValueError("use_alpha_scale and fix_alpha_scale are exclusive")
        if self.train_shared and self.freeze_root_encoder:
            raise ValueError("train_shared and freeze_root_encoder are contradictory")
        if alpha_init not in ("Randn", "Major", "Uniform"):
            raise ValueError(f"unknown alpha_init={alpha_init!r}")
        if alpha_init == "Major" and not 0.0 < float(alpha_major) < 1.0:
            raise ValueError("alpha_major must be in (0, 1)")
        if self.similarity_samples < 2 or self.distill_max_samples < 2:
            raise ValueError("similarity/distillation sample budgets must be >= 2")
        if self.distill_batch_size < 1 or self.distill_lr <= 0:
            raise ValueError("invalid distillation batch size or learning rate")
        if not 0.0 <= self.distill_test_frac < 1.0:
            raise ValueError("distill_test_frac must be in [0, 1)")
        if self.distillation and self.distill_epochs < 1:
            raise ValueError("distill_epochs must be >= 1 when distillation is on")
        if self.projection_epochs < 1 or self.projection_max_samples < 2:
            raise ValueError("projection budgets must be positive")
        if self.projection_samples < 1:
            raise ValueError("projection_samples must be >= 1")

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
            alpha = nn.Parameter(torch.ones(num_vectors) * float(alpha_factor))
        elif alpha_init == "Randn":
            alpha = nn.Parameter(torch.randn(num_vectors) / max(num_vectors, 1))
        elif alpha_init == "Major":
            vals = [np.log((1 - alpha_major) / (num_vectors - 1))] * (num_vectors - 1)
            vals.append(np.log(alpha_major))
            alpha = nn.Parameter(torch.tensor(vals, dtype=torch.float32))
        else:
            raise NotImplementedError(alpha_init)

        alpha_scale = nn.Parameter(
            torch.tensor([5.0 if fix_alpha_scale else 1.0], dtype=torch.float32),
            requires_grad=bool(use_alpha_scale) and not fix_alpha_scale and not fix_alpha,
        )
        alpha_mass = (
            nn.Parameter(
                torch.full(
                    (1,),
                    float(np.log(0.95 / 0.05)) if self.constrain_alpha_mass else 1.0,
                ),
                requires_grad=not fix_alpha,
            )
            if use_alpha_mass
            else None
        )
        return alpha, alpha_scale, alpha_mass

    # ------------------------------------------------------------------
    # SAC interface
    # ------------------------------------------------------------------
    def encode(self, obs):
        return self.fc(obs)

    def _distribution_at_features(self, features):
        raw, weights = self._components_at_features(features)
        return gaussian_mixture_distribution(raw, weights, self.act_dim)

    def forward(self, obs):
        """Raw stacked head output; use ``action_distribution`` for behavior."""
        return self._components_at_features(self.encode(obs))[0]

    def action_distribution(self, obs):
        return self._distribution_at_features(self.encode(obs))

    def act(self, obs, deterministic: bool = False):
        dist = self.action_distribution(obs)
        return dist.deterministic_action() if deterministic else dist.sample()

    def actor_forward(self, obs):
        """Reparameterized action plus log-density, as the SAC actor loss needs."""
        return self.action_distribution(obs).rsample_with_log_prob()

    # ------------------------------------------------------------------
    # Pool lifecycle / replay buffers
    # ------------------------------------------------------------------
    def set_own_buffer(self, buffer):
        if buffer is not None:
            obs = buffer.get("obs") if isinstance(buffer, dict) else None
            if not isinstance(obs, np.ndarray):
                raise ValueError("merge buffer must contain a NumPy array 'obs'")
            if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
                raise ValueError(
                    f"buffer obs shape {obs.shape} does not match obs_dim={self.obs_dim}"
                )
        self.policy_pool.set_own_buffer(buffer)

    def set_base(self):
        self.policy_pool.set_base()

    def _sample_reference_observations(self):
        buffers = [entry.get("buffer") for entry in self.policy_pool.pool]
        if any(b is None or "obs" not in b or len(b["obs"]) == 0 for b in buffers):
            raise RuntimeError(
                "behavioral KL merging needs a non-empty buffer for every pool slot"
            )
        per_slot = max(1, self.similarity_samples // 2)
        samples = []
        for buf in buffers:
            obs = buf["obs"]
            take = min(per_slot, len(obs))
            if self.balance_source_lineages and "source_ids" in buf:
                idx = balanced_lineage_indices(buf["source_ids"], take)
            else:
                idx = np.random.choice(len(obs), size=take, replace=False)
            samples.append(obs[idx])
        return samples

    def _encode_obs(self, obs: np.ndarray, batch_size: int = 4096):
        device = self.policy_pool.base_l0_weight.device
        chunks = []
        with torch.no_grad():
            for start in range(0, len(obs), int(batch_size)):
                x = torch.as_tensor(
                    obs[start:start + int(batch_size)], dtype=torch.float32, device=device
                )
                chunks.append(self.fc(x))
        return torch.cat(chunks, dim=0)

    def _entry_gaussian(self, features, index: int):
        raw = self.policy_pool.forward_entry(features, index)
        mean, log_std = torch.split(raw, self.act_dim, dim=-1)
        return mean, clamp_log_std(log_std)

    # ------------------------------------------------------------------
    # Merge-pair selection
    # ------------------------------------------------------------------
    def _select_cosine_pair(self):
        n = self.policy_pool.pool_length()
        if n < 2:
            raise RuntimeError("cannot select a pair from fewer than two entries")
        vectors = [
            torch.cat([self.policy_pool.pool[i][k].reshape(-1) for k in _HEAD_KEYS])
            for i in range(n)
        ]
        device = vectors[0].device
        matrix = torch.full((n, n), -float("inf"), device=device)
        finite = []
        with torch.no_grad():
            for i in range(n):
                for j in range(i + 1, n):
                    score = torch.nan_to_num(
                        torch.nn.functional.cosine_similarity(vectors[i], vectors[j], dim=0),
                        nan=-float("inf"),
                    )
                    matrix[i, j] = matrix[j, i] = score
                    if torch.isfinite(score):
                        finite.append(float(score))
            flat = int(torch.argmax(matrix))
            idx1, idx2 = divmod(flat, n)
            if idx1 == idx2 or not torch.isfinite(matrix[idx1, idx2]):
                raise RuntimeError("cosine pair selection found no finite pair")
            selected = float(matrix[idx1, idx2])

        report = matrix.detach().cpu().clone()
        report.fill_diagonal_(1.0)
        return idx1, idx2, {
            "idx1": int(idx1),
            "idx2": int(idx2),
            "similarity_metric": "cosine",
            "cosine_similarity": selected,
            "pairwise_cosine_min": float(np.min(finite)),
            "pairwise_cosine_mean": float(np.mean(finite)),
            "pairwise_cosine_max": float(np.max(finite)),
            "pairwise_cosine_similarity": report.numpy().tolist(),
        }

    def _select_behavioral_pair(self):
        """Pick the minimum replay-weighted symmetric policy KL pair.

        For each candidate pair the KL is evaluated on the union of the two
        parents' own replay states, so a pair is judged on the states where
        either of them actually operates rather than on a global sample.
        """
        n = self.policy_pool.pool_length()
        if n < 2:
            raise RuntimeError("cannot select a pair from fewer than two entries")

        obs_by_slot = self._sample_reference_observations()
        features_by_slot = [self._encode_obs(o) for o in obs_by_slot]

        with torch.no_grad():
            # outputs[policy][state source] = (mean, log_std)
            outputs = [
                [self._entry_gaussian(features_by_slot[s], p) for s in range(n)]
                for p in range(n)
            ]
            device = features_by_slot[0].device
            matrix = torch.full((n, n), float("inf"), device=device)
            pair_rows = torch.zeros((n, n), dtype=torch.int64, device=device)
            finite = []

            for i in range(n):
                for j in range(i + 1, n):
                    mi = torch.cat((outputs[i][i][0], outputs[i][j][0]), dim=0)
                    si = torch.cat((outputs[i][i][1], outputs[i][j][1]), dim=0)
                    mj = torch.cat((outputs[j][i][0], outputs[j][j][0]), dim=0)
                    sj = torch.cat((outputs[j][i][1], outputs[j][j][1]), dim=0)
                    skl = symmetric_gaussian_kl(mi, si, mj, sj)
                    score = torch.nan_to_num(
                        skl, nan=float("inf"), posinf=float("inf")
                    ).mean()
                    matrix[i, j] = matrix[j, i] = score
                    pair_rows[i, j] = pair_rows[j, i] = int(mi.shape[0])
                    if torch.isfinite(score):
                        finite.append(float(score))

            flat = int(torch.argmin(matrix))
            idx1, idx2 = divmod(flat, n)
            if idx1 == idx2 or not torch.isfinite(matrix[idx1, idx2]):
                raise RuntimeError("behavioral KL selection found no finite pair")
            selected = float(matrix[idx1, idx2])
            rows = int(pair_rows[idx1, idx2])

            mi = torch.cat((outputs[idx1][idx1][0], outputs[idx1][idx2][0]), dim=0)
            si = torch.cat((outputs[idx1][idx1][1], outputs[idx1][idx2][1]), dim=0)
            mj = torch.cat((outputs[idx2][idx1][0], outputs[idx2][idx2][0]), dim=0)
            sj = torch.cat((outputs[idx2][idx1][1], outputs[idx2][idx2][1]), dim=0)
            state_kl = symmetric_gaussian_kl(mi, si, mj, sj)
            p95 = float(torch.quantile(state_kl, 0.95))
            max_kl = float(state_kl.max())

        report = matrix.detach().cpu().clone()
        report.fill_diagonal_(0.0)
        return idx1, idx2, {
            "idx1": int(idx1),
            "idx2": int(idx2),
            "similarity_metric": "symmetric_kl",
            "symmetric_kl": selected,
            "pairwise_kl_min": float(np.min(finite)),
            "pairwise_kl_mean": float(np.mean(finite)),
            "pairwise_kl_max": float(np.max(finite)),
            "selected_state_kl_p95": p95,
            "selected_state_kl_max": max_kl,
            "similarity_states": rows,
            "reference_rows_per_slot": [int(len(x)) for x in obs_by_slot],
            "pairwise_symmetric_kl": report.numpy().tolist(),
        }

    # ------------------------------------------------------------------
    # Distillation
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
    def _buffer_lineage(buffer, key="task_ids"):
        if buffer is None or key not in buffer:
            return {}
        ids, counts = np.unique(np.asarray(buffer[key]).reshape(-1), return_counts=True)
        return {str(int(i)): int(c) for i, c in zip(ids, counts)}

    def _balanced_parent_data(self, idx1, idx2):
        buf1 = self.policy_pool.pool[idx1].get("buffer")
        buf2 = self.policy_pool.pool[idx2].get("buffer")
        if buf1 is None or buf2 is None:
            raise RuntimeError("distillation selected an entry without a buffer")

        if self.balance_source_lineages and "source_ids" in buf1 and "source_ids" in buf2:
            n1, n2 = len(buf1["obs"]), len(buf2["obs"])
            sources = np.concatenate(
                [
                    np.asarray(buf1["source_ids"]).reshape(-1),
                    np.asarray(buf2["source_ids"]).reshape(-1),
                ]
            )
            take = min(n1 + n2, self.distill_max_samples)
            idx = balanced_lineage_indices(sources, take)
            i1 = idx[idx < n1]
            i2 = idx[idx >= n1] - n1
            obs = HeadPool._gather_two(buf1, buf2, "obs", i1, i2)
            teacher_ids = np.concatenate(
                [np.zeros(len(i1), dtype=np.int64), np.ones(len(i2), dtype=np.int64)]
            )
            source_ids = np.concatenate(
                [
                    np.asarray(buf1["source_ids"]).reshape(-1)[i1],
                    np.asarray(buf2["source_ids"]).reshape(-1)[i2],
                ]
            ).astype(np.int64, copy=False)
            return obs, teacher_ids, source_ids

        max_each = max(1, self.distill_max_samples // 2)
        obs_parts, teacher_ids, source_ids = [], [], []
        for teacher_id, buf in enumerate((buf1, buf2)):
            obs = buf["obs"]
            take = min(len(obs), max_each)
            idx = np.random.choice(len(obs), size=take, replace=False)
            obs_parts.append(obs[idx])
            teacher_ids.append(np.full(take, teacher_id, dtype=np.int64))
            src = np.asarray(
                buf.get("source_ids", np.full(len(obs), teacher_id))
            ).reshape(-1)[idx]
            source_ids.append(src)
        return (
            np.concatenate(obs_parts, axis=0),
            np.concatenate(teacher_ids),
            np.concatenate(source_ids).astype(np.int64, copy=False),
        )

    def _distill_pair(self, idx1, idx2):
        """KL-distill two Gaussian parents into one student head.

        Both teachers and the student are single Gaussians, so the objective is
        the exact closed-form KL -- no sampling noise enters the merge.
        """
        obs, teacher_ids_np, source_ids_np = self._balanced_parent_data(idx1, idx2)
        features = self._encode_obs(obs)
        device = features.device
        teacher_ids = torch.as_tensor(teacher_ids_np, dtype=torch.long, device=device)
        source_ids = torch.as_tensor(source_ids_np, dtype=torch.long, device=device)

        with torch.no_grad():
            m1, s1 = self._entry_gaussian(features, idx1)
            m2, s2 = self._entry_gaussian(features, idx2)
            pick = teacher_ids.unsqueeze(-1).bool()
            teacher_mean = torch.where(pick, m2, m1)
            teacher_log_std = torch.where(pick, s2, s1)

        params = {
            k: v.detach().clone().requires_grad_(True)
            for k, v in self.policy_pool.average_pair_params(idx1, idx2).items()
        }
        trainables = list(params.values())
        optimizer = torch.optim.Adam(trainables, lr=self.distill_lr)

        # Stratify the held-out split by lineage so both train and validation
        # contain every source that the merged entry will have to represent.
        groups = (
            torch.unique(source_ids).tolist()
            if self.balance_source_lineages
            else [0, 1]
        )
        labels = source_ids if self.balance_source_lineages else teacher_ids
        train_parts, test_parts = [], []
        for gid in groups:
            idx = torch.nonzero(labels == int(gid), as_tuple=False).flatten()
            idx = idx[torch.randperm(idx.numel(), device=device)]
            n_test = int(idx.numel() * self.distill_test_frac) if self.distill_test_frac > 0 else 0
            n_test = min(n_test, max(idx.numel() - 1, 0))
            test_parts.append(idx[:n_test])
            train_parts.append(idx[n_test:])
        train_idx = torch.cat(train_parts)
        test_idx = torch.cat(test_parts)
        train_idx = train_idx[torch.randperm(train_idx.numel(), device=device)]

        def student(indices):
            raw = self.policy_pool._forward_with_weights(
                features[indices], self._params_to_effective(self.policy_pool, params)
            )
            mean, log_std = torch.split(raw, self.act_dim, dim=-1)
            return mean, clamp_log_std(log_std)

        @torch.no_grad()
        def kl_summary(indices):
            if indices.numel() == 0:
                return None
            mean, log_std = student(indices)
            values = gaussian_kl(
                teacher_mean[indices], teacher_log_std[indices], mean, log_std
            )
            return {
                "mean": float(values.mean()),
                "p95": float(torch.quantile(values, 0.95)),
                "max": float(values.max()),
            }

        validation_idx = test_idx if test_idx.numel() else train_idx
        initial = kl_summary(validation_idx)
        best_val = float("inf") if initial is None else initial["mean"]
        best_epoch = 0
        best = {k: v.detach().clone() for k, v in params.items()}

        for epoch in range(1, self.distill_epochs + 1):
            shuffled = train_idx[torch.randperm(train_idx.numel(), device=device)]
            for start in range(0, shuffled.numel(), self.distill_batch_size):
                idx = shuffled[start:start + self.distill_batch_size]
                mean, log_std = student(idx)
                loss = gaussian_kl(
                    teacher_mean[idx], teacher_log_std[idx], mean, log_std
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainables, 10.0)
                optimizer.step()

            val = kl_summary(validation_idx)
            if val is not None and val["mean"] < best_val:
                best_val = val["mean"]
                best_epoch = epoch
                best = {k: v.detach().clone() for k, v in params.items()}

        if self.distill_select_best_val:
            with torch.no_grad():
                for key in _HEAD_KEYS:
                    params[key].copy_(best[key])
            selected_epoch, selected_val = best_epoch, best_val
        else:
            selected_epoch = self.distill_epochs
            final = kl_summary(validation_idx)
            selected_val = float("nan") if final is None else final["mean"]

        train_stats = kl_summary(train_idx)
        test_stats = kl_summary(test_idx)
        metrics = {
            "policy/distill_train_kl": None if train_stats is None else train_stats["mean"],
            "policy/distill_test_kl": None if test_stats is None else test_stats["mean"],
            "policy/distill_train_kl_p95": None if train_stats is None else train_stats["p95"],
            "policy/distill_test_kl_p95": None if test_stats is None else test_stats["p95"],
            "policy/distill_train_kl_max": None if train_stats is None else train_stats["max"],
            "policy/distill_test_kl_max": None if test_stats is None else test_stats["max"],
            "policy/distill_best_epoch": int(best_epoch),
            "policy/distill_best_val_kl": float(best_val),
            "policy/distill_selected_epoch": int(selected_epoch),
            "policy/distill_selected_val_kl": float(selected_val),
            "policy/distill_initial_val_kl": None if initial is None else float(initial["mean"]),
            "policy/distill_rows": int(len(obs)),
            "policy/distill_source_lineages": int(torch.unique(source_ids).numel()),
            "policy/distill_balance_source_lineages": float(self.balance_source_lineages),
        }
        return {k: v.detach().clone() for k, v in params.items()}, metrics

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
            buf1,
            buf2,
            self.max_distill_buffer,
            balance_source_lineages=self.balance_source_lineages,
        )
        merge_info.update(
            {
                "used_distillation": bool(used_distillation),
                "balance_source_lineages": bool(self.balance_source_lineages),
                "pool_size_before": int(self.policy_pool.pool_length()),
                "pool_size_after": int(self.policy_pool.pool_length() - 1),
                "parent_1_lineage": self._buffer_lineage(buf1, "task_ids"),
                "parent_2_lineage": self._buffer_lineage(buf2, "task_ids"),
                "merged_lineage": self._buffer_lineage(merged_buffer, "task_ids"),
                "parent_1_source_lineage": self._buffer_lineage(buf1, "source_ids"),
                "parent_2_source_lineage": self._buffer_lineage(buf2, "source_ids"),
                "merged_source_lineage": self._buffer_lineage(merged_buffer, "source_ids"),
            }
        )
        self.policy_pool.replace_pair(idx1, idx2, merged_params, merged_buffer, merge_info)
        self.last_merge_info = dict(merge_info)

        if self.policy_pool.pool_length() > self.pool_size:
            raise RuntimeError("pool remained over capacity after one merge")

    def get_distill_metrics(self):
        return dict(self.last_distill_metrics)

    def get_merge_info(self):
        return None if self.last_merge_info is None else dict(self.last_merge_info)

    # ------------------------------------------------------------------
    # Saving / snapshots
    # ------------------------------------------------------------------
    @staticmethod
    def _cpu_clone_dict(d):
        return {k: v.detach().cpu().clone() for k, v in d.items()}

    def export_effective_policy(self):
        """Exact pre-finalize policy, authoritative for evaluation.

        Finalization changes pool topology, which makes the pre-finalize alpha
        vector semantically stale; this snapshot is taken before that happens.
        """
        if self.composition_space == "policy":
            with torch.no_grad():
                snapshot = self.export_policy_ensemble()
                snapshot["encoder_format_version"] = int(
                    getattr(self.fc, "format_version", ENCODER_FORMAT_VERSION)
                )
                return snapshot

        with torch.no_grad():
            w0, b0, w2, b2 = self.policy_pool._effective()
            return {
                "format_version": _POLICY_SNAPSHOT_FORMAT_VERSION,
                "encoder_format_version": int(
                    getattr(self.fc, "format_version", ENCODER_FORMAT_VERSION)
                ),
                "composition_space": "parameter",
                "policy_type": "squashed_gaussian",
                "obs_dim": int(self.obs_dim),
                "act_dim": int(self.act_dim),
                "shared_dim": int(self.shared_dim),
                "hidden_dim": int(self.hidden_dim),
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
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.fc, os.path.join(dirname, "fc.pt"))
        torch.save(self.policy_pool, os.path.join(dirname, "policy_pool.pt"))

    @staticmethod
    def load(dirname, map_location=None, **_ignored):
        del _ignored
        return FrozenCkaPolicy.load(dirname, map_location=map_location)


class FrozenCkaPolicy(nn.Module):
    """Inference-only policy reconstructed from ``policy_snapshot.pt``."""

    def __init__(self, snapshot):
        super().__init__()
        if int(snapshot.get("format_version", -1)) != _POLICY_SNAPSHOT_FORMAT_VERSION:
            raise ValueError("unsupported PointMaze policy snapshot format")
        if snapshot.get("policy_type") != "squashed_gaussian":
            raise ValueError("snapshot is not a squashed-Gaussian policy")
        if int(snapshot.get("encoder_format_version", -1)) != ENCODER_FORMAT_VERSION:
            raise ValueError("snapshot encoder format does not match this code")

        self.composition_space = snapshot.get("composition_space", "parameter")
        self.obs_dim = int(snapshot["obs_dim"])
        self.act_dim = int(snapshot["act_dim"])
        self.shared_dim = int(snapshot.get("shared_dim", 256))
        self.hidden_dim = int(snapshot.get("hidden_dim", 256))

        self.fc = shared(self.obs_dim, output_dim=self.shared_dim)
        self.fc.load_state_dict(snapshot["fc_state_dict"])
        validate_shared_encoder(
            self.fc, self.obs_dim, self.shared_dim, source="policy_snapshot.pt encoder"
        )

        if self.composition_space == "policy":
            self.register_buffer("mixture_weights", snapshot["mixture_weights"].clone())
            for key, value in snapshot["policy_components"].items():
                self.register_buffer("policy_" + key, value.clone())
        else:
            for key, value in snapshot["policy"].items():
                self.register_buffer("policy_" + key, value.clone())

    def policy_components(self, obs):
        features = self.fc(obs)
        if self.composition_space == "policy":
            head = {k: getattr(self, "policy_" + k) for k in _HEAD_KEYS}
            return stacked_head_forward(features, head), self.mixture_weights
        raw = torch.nn.functional.linear(
            torch.relu(
                torch.nn.functional.linear(
                    features, self.policy_l0_weight, self.policy_l0_bias
                )
            ),
            self.policy_l2_weight,
            self.policy_l2_bias,
        )
        return raw[:, None, :], raw.new_ones(1)

    def forward(self, obs):
        return self.policy_components(obs)[0]

    def action_distribution(self, obs):
        raw, weights = self.policy_components(obs)
        return gaussian_mixture_distribution(raw, weights, self.act_dim)

    @staticmethod
    def load(dirname, map_location=None):
        path = _require_file(
            os.path.join(os.fspath(dirname), "policy_snapshot.pt"),
            "PointMaze policy snapshot",
        )
        return FrozenCkaPolicy(_torch_load(path, map_location=map_location))
