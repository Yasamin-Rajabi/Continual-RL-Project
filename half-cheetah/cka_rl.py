import torch


def _torch_load(path, map_location=None):
    """Load our own full-module checkpoints without globally patching torch.load."""
    kwargs = {} if map_location is None else {"map_location": map_location}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        return torch.load(path, **kwargs)

import os
from typing import Dict, Tuple

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from knowledge_pools import BASE_FUSION_MODE, HeadPool, balanced_lineage_indices
from policy_utils import bound_log_std, diagonal_gaussian_kl, symmetric_diagonal_gaussian_kl
from shared_arch import shared, validate_shared_encoder

_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


from policy_space import PolicySpaceMixin
from policy_composition import gaussian_summary, stacked_head_forward


class CkaRlAgent(PolicySpaceMixin, nn.Module):
    """Shared encoder + an aligned pool of Gaussian policy heads.

    Pool slots are policy-level objects: the mean and log-std tensors at index i
    always belong to the same knowledge item. A single shared alpha vector is
    used for both heads (matching the paper's one coefficient per knowledge
    vector), and a single behavioral-KL merge decision is applied to both heads.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        base_dir,
        latest_dir,
        pool_size=5,
        alpha_init="Randn",
        alpha_major=0.6,
        alpha_factor=1e-3,
        fix_alpha=False,
        use_alpha_scale=False,
        fix_alpha_scale=False,
        use_alpha_mass=False,
        constrain_alpha_mass=True,
        encoder_from_base=False,
        distillation=True,
        distill_observation_skip=False,
        fusion_mode=BASE_FUSION_MODE,
        max_distill_buffer=50_000,
        distill_test_frac=0.2,
        distill_select_best_val=True,
        distill_epochs=8,
        distill_lr=3e-4,
        distill_batch_size=256,
        distill_max_samples=20_000,
        similarity_samples=2048,
        balance_source_lineages=False,
        hidden_dim=128,
        shared_dim=256,
        train_shared=False,
        freeze_root_encoder=False,
        pretrained_encoder=None,
        encoder_linear_out=False,
        composition_space="parameter",
        projection_epochs=16,
        projection_max_samples=20000,
        policy_student_replay=False,
        merge_ablation="kl_merge",
    ):
        super().__init__()
        if composition_space not in ("parameter", "policy"):
            raise ValueError("composition_space must be 'parameter' or 'policy'")
        if composition_space == "policy" and use_alpha_mass and not constrain_alpha_mass:
            raise ValueError("A probability mixture requires a bounded sigmoid alpha-mass")
        self.composition_space = composition_space
        self.policy_student_replay = bool(policy_student_replay)
        if merge_ablation not in ("kl_merge", "random_merge", "kl_discard"):
            raise ValueError("invalid merge_ablation")
        self.merge_ablation = str(merge_ablation)
        if self.policy_student_replay and (composition_space != "policy" or not use_alpha_mass or fusion_mode != "weight_delta"):
            raise ValueError("policy_student_replay requires policy composition, weight_delta, and alpha-mass")
        self.projection_epochs = int(projection_epochs)
        self.projection_max_samples = int(projection_max_samples)
        if self.projection_epochs < 1 or self.projection_max_samples < 2:
            raise ValueError("projection_epochs >= 1 and projection_max_samples >= 2 are required")
        self.mixture_warmup = False
        self.pool_only = False
        self.last_projection_metrics = {}
        self.encoder_linear_out = bool(encoder_linear_out)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.pool_size = int(pool_size)
        self.distillation = bool(distillation)
        self.distill_observation_skip = bool(distill_observation_skip)
        self.use_alpha_scale = bool(use_alpha_scale)
        self.fix_alpha_scale = bool(fix_alpha_scale)
        if self.fix_alpha_scale and self.use_alpha_scale:
            raise ValueError("use_alpha_scale=True and fix_alpha_scale=True are mutually exclusive")
        self.fusion_mode = fusion_mode
        self.max_distill_buffer = int(max_distill_buffer)
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
        if self.train_shared and self.freeze_root_encoder:
            raise ValueError("train_shared=True and freeze_root_encoder=True are contradictory")
        self.last_merge_info = None
        self.last_distill_metrics = {}

        head_in_dim = (
            shared_dim + self.obs_dim
            if self.distillation and self.distill_observation_skip
            else shared_dim
        )
        self.mean_pool = HeadPool(
            "mean", head_in_dim, hidden_dim, act_dim,
            fusion_mode=fusion_mode, pool_size=pool_size,
            distillation=distillation, max_distill_buffer=max_distill_buffer,
            use_alpha_mass=use_alpha_mass, constrain_alpha_mass=constrain_alpha_mass,
            distill_test_frac=distill_test_frac,
        )
        self.logstd_pool = HeadPool(
            "logstd", head_in_dim, hidden_dim, act_dim,
            fusion_mode=fusion_mode, pool_size=pool_size,
            distillation=distillation, max_distill_buffer=max_distill_buffer,
            use_alpha_mass=use_alpha_mass, constrain_alpha_mass=constrain_alpha_mass,
            distill_test_frac=distill_test_frac,
        )

        self.mean_pool.composition_space = composition_space
        self.logstd_pool.composition_space = composition_space
        if latest_dir is not None:
            latest_mean_pool = _torch_load(f"{latest_dir}/mean_pool.pt", map_location="cpu")
            latest_logstd_pool = _torch_load(f"{latest_dir}/logstd_pool.pt", map_location="cpu")
            if getattr(latest_mean_pool, "composition_space", "parameter") != composition_space:
                raise ValueError("Cannot continue a chain with a different composition space; start a fresh run")
            self.mean_pool.inherit_pool_from(latest_mean_pool)
            self.logstd_pool.inherit_pool_from(latest_logstd_pool)
            self._assert_pool_alignment()
            self.mean_pool.reset_own_to_zero()
            self.logstd_pool.reset_own_to_zero()

        # One alpha vector controls each whole knowledge vector, across both heads.
        self.alpha, self.alpha_scale, self.alpha_mass = self._make_alpha(
            self.mean_pool.pool_length(), fix_alpha, alpha_init, alpha_major,
            alpha_factor, use_alpha_scale, fix_alpha_scale, use_alpha_mass,
        )
        self.mean_pool.set_alpha(self.alpha, self.alpha_scale, self.alpha_mass)
        self.logstd_pool.set_alpha(self.alpha, self.alpha_scale, self.alpha_mass)
        self.initialize_policy_space_own()
        logger.info(f"shared alpha: {self.alpha}")
        if use_alpha_mass:
            logger.info(f"shared alpha_mass: {self.alpha_mass}")

        # train_shared=False (default): the shared encoder is frozen after the
        # root task, exactly like theta_base in the pool heads -- later tasks
        # reuse the SAME encoder learned on task 1, never updating it further.
        # train_shared=True: the encoder is never frozen. Every task keeps
        # training it, starting from whatever the previous task left it at
        # (loaded from latest_dir, NOT re-initialized from scratch) -- i.e.
        # the encoder itself is carried forward and fine-tuned continually.
        #
        # NOTE: earlier in this project we tracked a `fuse_shared` flag that
        # kept the shared encoder OUTSIDE the theta_base/knowledge-vector
        # formula entirely (fuse_shared=False), implying it was meant to keep
        # training continually, not freeze. So train_shared=True is closer to
        # that original behavior; train_shared=False (current default) is a
        # deliberate later change, not a literal reading of the paper.
        # Encoder policy is explicit and ablatable.  The current friend-workflow
        # does NOT require encoder pretraining: with train_shared=True, task 0
        # starts from scratch (unless an optional pretrained encoder is supplied)
        # and every later task continues from the latest encoder.  The historical
        # pretrained-encoder path is retained only as an optional ablation /
        # backwards-compatible experiment mode.
        #   * train_shared=True: continually fine-tune from latest_dir;
        #   * no pretrained + train_shared=False: task 0 learns the root encoder,
        #     then later tasks freeze/reuse that basis;
        #   * pretrained + train_shared=False: optional frozen-pretraining ablation;
        #   * freeze_root_encoder=True: explicit random-frozen ablation.
        if latest_dir is not None and self.train_shared:
            logger.info(f"Loading latest trainable encoder from {latest_dir}")
            self.fc = _torch_load(f"{latest_dir}/fc.pt", map_location="cpu")
            source = f"latest encoder {latest_dir}/fc.pt"
        elif pretrained_encoder is not None:
            logger.info(f"Loading pretrained encoder from {pretrained_encoder}")
            self.fc = _torch_load(pretrained_encoder, map_location="cpu")
            source = f"pretrained encoder {pretrained_encoder}"
        elif encoder_from_base and base_dir is not None:
            logger.info(f"Loading frozen encoder from base {base_dir}")
            self.fc = _torch_load(f"{base_dir}/fc.pt", map_location="cpu")
            source = f"base encoder {base_dir}/fc.pt"
        elif latest_dir is not None:
            logger.info(f"Loading shared encoder from {latest_dir}")
            self.fc = _torch_load(f"{latest_dir}/fc.pt", map_location="cpu")
            source = f"latest encoder {latest_dir}/fc.pt"
        else:
            logger.info("Initializing root shared encoder from scratch")
            self.fc = shared(input_dim=obs_dim, linear_out=self.encoder_linear_out)
            source = "new root encoder"

        validate_shared_encoder(
            self.fc, input_dim=obs_dim, linear_out=self.encoder_linear_out, source=source
        )

        # Frozen TD-JEPA encoders are frozen from task 0.  Without pretraining,
        # the default lets task 0 learn a root basis and freezes it only on later
        # tasks. freeze_root_encoder=True exposes the random-frozen ablation.
        should_freeze = (
            not self.train_shared
            and (pretrained_encoder is not None or latest_dir is not None or self.freeze_root_encoder)
        )
        if should_freeze:
            logger.info("Shared encoder frozen")
            self.fc.requires_grad_(False)

    def _assert_pool_alignment(self):
        if self.mean_pool.pool_length() != self.logstd_pool.pool_length():
            raise RuntimeError(
                "Mean/logstd pools are misaligned. This checkpoint was likely produced "
                "by the old independent-head merge code; restart the continual chain "
                "from the root task with the new implementation."
            )

    def _make_alpha(
        self, num_vectors, fix_alpha, alpha_init, alpha_major, alpha_factor,
        use_alpha_scale, fix_alpha_scale, use_alpha_mass,
    ):
        if num_vectors <= 0:
            return None, None, None
        if fix_alpha:
            alpha = nn.Parameter(torch.zeros(num_vectors), requires_grad=False)
        elif alpha_init == "Uniform" or num_vectors == 1:
            alpha = nn.Parameter(torch.ones(num_vectors) * alpha_factor, requires_grad=True)
        elif alpha_init == "Randn":
            alpha = nn.Parameter(torch.randn(num_vectors) / max(num_vectors, 1), requires_grad=True)
        elif alpha_init == "Major" and num_vectors > 1:
            vals = [np.log((1 - alpha_major) / (num_vectors - 1)) for _ in range(num_vectors - 1)]
            vals.append(np.log(alpha_major))
            alpha = nn.Parameter(torch.tensor(vals, dtype=torch.float32), requires_grad=True)
        else:
            raise NotImplementedError(f"unknown alpha_init: {alpha_init}")

        # Three alpha-scale regimes are kept explicit for ablations:
        #   off     -> fixed multiplier 1 (closest to the paper's plain softmax)
        #   learned -> trainable multiplier initialized at 1
        #   fixed   -> friend's stabilized weight-delta setting, multiplier 5
        scale_init_val = 5.0 if fix_alpha_scale else 1.0
        alpha_scale = nn.Parameter(
            torch.tensor([scale_init_val], dtype=torch.float32),
            requires_grad=(use_alpha_scale and (not fix_alpha_scale) and (not fix_alpha)),
        )

        alpha_mass = (
            nn.Parameter(torch.full((1,), (float(np.log(0.95 / 0.05)) if self.constrain_alpha_mass else 1.0)), requires_grad=not fix_alpha)
            if use_alpha_mass else None
        )
        return alpha, alpha_scale, alpha_mass


    def forward(self, x):
        if self.composition_space == "policy":
            # Compatibility summary; inference and SAC use policy_components.
            return gaussian_summary(*self.policy_components(x))
        features = self.fc(x)
        z = (
            torch.cat([features, x], dim=-1)
            if self.distillation and self.distill_observation_skip
            else features
        )
        return self.mean_pool(z), self.logstd_pool(z)


    def set_own_buffer(self, buffer):
        """Attach raw rollout states to the new policy slot.

        Only the mean pool stores the physical buffer to avoid duplicating it in
        two checkpoint files; mean/logstd slots remain index-aligned.
        """
        self.mean_pool.set_own_buffer(buffer)
        self.logstd_pool.set_own_buffer(None)

    def set_base(self):
        self.mean_pool.set_base()
        self.logstd_pool.set_base()
        self._assert_pool_alignment()

    # ------------------------------------------------------------------
    # Behavioral similarity + joint policy distillation
    # ------------------------------------------------------------------
    def _sample_reference_observations(self):
        """Sample a balanced reference state subset for every pool slot.

        Pair (i, j) is compared on the union of states from *those two*
        lineages rather than on states belonging to unrelated pool entries.
        Samples are drawn once per slot and reused across all pair comparisons.
        """
        buffers = [entry.get("buffer") for entry in self.mean_pool.pool]
        if any(buf is None or "obs" not in buf or len(buf["obs"]) == 0 for buf in buffers):
            raise RuntimeError(
                "Behavioral KL merging needs an observation buffer for every pool slot. "
                "Old checkpoints without merge buffers are not compatible; rerun from task 0."
            )
        # similarity_samples is the approximate budget for ONE pair.
        per_slot = max(1, self.similarity_samples // 2)
        samples = []
        for buf in buffers:
            obs = buf["obs"]
            take = min(per_slot, len(obs))
            if self.balance_source_lineages:
                if "source_ids" not in buf:
                    raise RuntimeError(
                        "--balance-source-lineages requires source_ids in every retained buffer; "
                        "start the ablation from task 0 rather than an older checkpoint."
                    )
                idx = balanced_lineage_indices(buf["source_ids"], take)
            else:
                idx = np.random.choice(len(obs), size=take, replace=False)
            samples.append(obs[idx].astype(np.float32, copy=False))
        return samples

    def _encode_obs(self, obs: np.ndarray, batch_size: int = 4096) -> torch.Tensor:
        device = self.mean_pool.base_l0_weight.device
        chunks = []

        with torch.no_grad():
            for start in range(0, len(obs), batch_size):
                x = torch.as_tensor(obs[start:start + batch_size], dtype=torch.float32, device=device)
                features = self.fc(x)
                z_chunk = (
                    torch.cat([features, x], dim=-1)
                    if self.distillation and self.distill_observation_skip
                    else features
                )
                chunks.append(z_chunk)
        return torch.cat(chunks, dim=0)

    def _entry_outputs(self, z: torch.Tensor, index: int):
        mean = self.mean_pool.forward_entry(z, index)
        raw_logstd = self.logstd_pool.forward_entry(z, index)
        return mean, raw_logstd

    def _select_cosine_pair(self):
        """Select the most similar aligned pool pair in parameter space.

        This is the CKA-style selector used by the non-distillation conditions.
        A pool slot is a whole policy knowledge item, so mean and log-std
        parameters are concatenated and one pair is selected jointly for both
        heads.  We compare the STORED representation itself: classic_cka slots
        are task increments v_k; weight_delta slots are the reconstructed weights
        stored by that mode.
        """
        n = self.mean_pool.pool_length()
        if n < 2:
            raise RuntimeError("cannot select a merge pair from fewer than two pool entries")

        vectors = []
        for index in range(n):
            chunks = []
            for pool in (self.mean_pool, self.logstd_pool):
                entry = pool.pool[index]
                chunks.extend(entry[key].reshape(-1) for key in _HEAD_KEYS)
            vectors.append(torch.cat(chunks, dim=0))

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
                raise RuntimeError("cosine pair selection failed: no finite pairwise similarity")
            selected = float(matrix[idx1, idx2].item())

        stats = {
            "idx1": int(idx1),
            "idx2": int(idx2),
            "similarity_metric": "cosine",
            "cosine_similarity": selected,
            "pairwise_cosine_min": float(np.min(finite_values)),
            "pairwise_cosine_mean": float(np.mean(finite_values)),
            "pairwise_cosine_max": float(np.max(finite_values)),
            "pairwise_cosine_similarity": matrix.detach().cpu().numpy().tolist(),
        }
        logger.info(
            f"[cosine merge] pair=({idx1},{idx2}) cosine={selected:.6f}"
        )
        return idx1, idx2, stats

    def _select_random_pair(self):
        n = self.mean_pool.pool_length()
        if n < 2:
            raise RuntimeError("cannot select a merge pair from fewer than two pool entries")
        i, j = np.random.choice(n, size=2, replace=False)
        return int(i), int(j), {"idx1": int(i), "idx2": int(j), "similarity_metric": "random", "selected_random_pair": True}

    def _select_behavioral_pair(self):
        n = self.mean_pool.pool_length()
        if n < 2:
            raise RuntimeError("cannot select a merge pair from fewer than two pool entries")

        # One state subset per lineage.  For pair (i,j), evaluate BOTH policies
        # on states from i and states from j.  This makes the similarity score
        # behaviorally local to the pair and avoids unrelated tasks dominating
        # the merge decision.
        obs_by_slot = self._sample_reference_observations()
        z_by_slot = [self._encode_obs(obs) for obs in obs_by_slot]

        with torch.no_grad():
            # outputs[policy_index][state_source_index] = (mean, raw_logstd)
            outputs = [
                [self._entry_outputs(z_by_slot[source], policy) for source in range(n)]
                for policy in range(n)
            ]
            device = z_by_slot[0].device
            matrix = torch.full((n, n), float("inf"), device=device)
            pair_rows = torch.zeros((n, n), dtype=torch.int64, device=device)
            finite_values = []
            for i in range(n):
                for j in range(i + 1, n):
                    mean_i = torch.cat((outputs[i][i][0], outputs[i][j][0]), dim=0)
                    log_i = torch.cat((outputs[i][i][1], outputs[i][j][1]), dim=0)
                    mean_j = torch.cat((outputs[j][i][0], outputs[j][j][0]), dim=0)
                    log_j = torch.cat((outputs[j][i][1], outputs[j][j][1]), dim=0)
                    skl = symmetric_diagonal_gaussian_kl(mean_i, log_i, mean_j, log_j)
                    score = torch.nan_to_num(
                        skl, nan=float("inf"), posinf=float("inf")
                    ).mean()
                    matrix[i, j] = score
                    matrix[j, i] = score
                    pair_rows[i, j] = pair_rows[j, i] = int(mean_i.shape[0])
                    if torch.isfinite(score):
                        finite_values.append(float(score.item()))

            flat_idx = int(torch.argmin(matrix).item())
            idx1, idx2 = divmod(flat_idx, n)
            if idx1 == idx2 or not torch.isfinite(matrix[idx1, idx2]):
                raise RuntimeError("behavioral KL pair selection failed: no finite pairwise KL")
            selected = float(matrix[idx1, idx2].item())
            selected_rows = int(pair_rows[idx1, idx2].item())

            # The matrix stores one MEAN KL per candidate pair.  Keep tail
            # statistics for the selected pair too: a moderate mean can hide a
            # small set of states with extremely large KL.
            selected_mean_i = torch.cat((outputs[idx1][idx1][0], outputs[idx1][idx2][0]), dim=0)
            selected_log_i = torch.cat((outputs[idx1][idx1][1], outputs[idx1][idx2][1]), dim=0)
            selected_mean_j = torch.cat((outputs[idx2][idx1][0], outputs[idx2][idx2][0]), dim=0)
            selected_log_j = torch.cat((outputs[idx2][idx1][1], outputs[idx2][idx2][1]), dim=0)
            selected_state_kl = symmetric_diagonal_gaussian_kl(
                selected_mean_i, selected_log_i, selected_mean_j, selected_log_j
            )
            selected_state_kl = torch.nan_to_num(
                selected_state_kl, nan=1e12, posinf=1e12, neginf=0.0
            )
            selected_state_kl_p95 = float(torch.quantile(selected_state_kl, 0.95).item())
            selected_state_kl_max = float(selected_state_kl.max().item())

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
            "pairwise_symmetric_kl": matrix.detach().cpu().numpy().tolist(),
        }
        logger.info(
            f"[behavioral merge] pair=({idx1},{idx2}) symmetric_KL={selected:.6f} "
            f"p95={selected_state_kl_p95:.6f} max={selected_state_kl_max:.6f} "
            f"over {selected_rows} parent-reference states"
        )
        return idx1, idx2, stats

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
    def _head_forward_from_params(pool: HeadPool, z: torch.Tensor, params: Dict[str, torch.Tensor]):
        return pool._forward_with_weights(z, CkaRlAgent._params_to_effective(pool, params))

    @staticmethod
    def _buffer_lineage(buffer, key="task_ids"):
        if buffer is None or key not in buffer:
            return {}
        ids, counts = np.unique(np.asarray(buffer[key]).reshape(-1), return_counts=True)
        return {str(int(source_id)): int(count) for source_id, count in zip(ids, counts)}

    def _balanced_parent_data(self, idx1: int, idx2: int):
        buf1 = self.mean_pool.pool[idx1].get("buffer")
        buf2 = self.mean_pool.pool[idx2].get("buffer")
        if buf1 is None or buf2 is None:
            raise RuntimeError("distillation requested but a selected pool entry has no observation buffer")

        if self.balance_source_lineages:
            for buf in (buf1, buf2):
                if "source_ids" not in buf:
                    raise RuntimeError(
                        "--balance-source-lineages requires source_ids in every retained buffer; "
                        "start the ablation from task 0 rather than an older checkpoint."
                    )
            obs_all = np.concatenate([buf1["obs"], buf2["obs"]], axis=0)
            teacher_all = np.concatenate([
                np.zeros(len(buf1["obs"]), dtype=np.int64),
                np.ones(len(buf2["obs"]), dtype=np.int64),
            ])
            source_all = np.concatenate([
                np.asarray(buf1["source_ids"]).reshape(-1),
                np.asarray(buf2["source_ids"]).reshape(-1),
            ])
            take = min(len(obs_all), self.distill_max_samples)
            idx = balanced_lineage_indices(source_all, take)
            return (
                obs_all[idx].astype(np.float32, copy=False),
                teacher_all[idx],
                source_all[idx].astype(np.int64, copy=False),
            )

        # Legacy behavior: equal row budget for the two immediate parents.
        max_each = max(1, self.distill_max_samples // 2)
        obs_parts, teacher_ids, source_ids = [], [], []
        for teacher_id, buf in enumerate((buf1, buf2)):
            obs = buf["obs"]
            take = min(len(obs), max_each)
            idx = np.random.choice(len(obs), size=take, replace=False)
            obs_parts.append(obs[idx])
            teacher_ids.append(np.full(take, teacher_id, dtype=np.int64))
            if "source_ids" in buf:
                source_ids.append(np.asarray(buf["source_ids"])[idx].reshape(-1))
            else:
                source_ids.append(np.full(take, teacher_id, dtype=np.int64))
        return (
            np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False),
            np.concatenate(teacher_ids, axis=0),
            np.concatenate(source_ids, axis=0).astype(np.int64, copy=False),
        )

    def _distill_pair(self, idx1: int, idx2: int):
        """KL-distill two aligned Gaussian policy entries into one student.

        The student starts from the arithmetic parent average.  A stratified
        held-out split is used both for diagnostics and model selection: the
        returned parameters are the epoch with the lowest held-out KL (or, if
        no held-out rows exist, the lowest training KL), not blindly the last
        optimization epoch.
        """
        obs, teacher_ids_np, source_ids_np = self._balanced_parent_data(idx1, idx2)
        z = self._encode_obs(obs)
        device = z.device
        teacher_ids = torch.as_tensor(teacher_ids_np, dtype=torch.long, device=device)
        source_ids = torch.as_tensor(source_ids_np, dtype=torch.long, device=device)

        with torch.no_grad():
            m1, l1 = self._entry_outputs(z, idx1)
            m2, l2 = self._entry_outputs(z, idx2)
            mask = teacher_ids.unsqueeze(-1).bool()
            teacher_mean = torch.where(mask, m2, m1)
            teacher_raw_logstd = torch.where(mask, l2, l1)
            teacher_logstd = bound_log_std(teacher_raw_logstd)

        mean_init = self.mean_pool.average_pair_params(idx1, idx2)
        log_init = self.logstd_pool.average_pair_params(idx1, idx2)
        mean_params = {key: value.detach().clone().requires_grad_(True) for key, value in mean_init.items()}
        log_params = {key: value.detach().clone().requires_grad_(True) for key, value in log_init.items()}

        trainables = list(mean_params.values()) + list(log_params.values())
        optimizer = torch.optim.Adam(trainables, lr=self.distill_lr)

        n = len(obs)
        # Legacy mode stratifies by immediate parent.  Lineage-balanced mode
        # stratifies by original source occurrence so every retained lineage is
        # represented in training and, when possible, held-out validation.
        train_parts, test_parts = [], []
        split_groups = (
            torch.unique(source_ids).tolist()
            if self.balance_source_lineages
            else [0, 1]
        )
        split_labels = source_ids if self.balance_source_lineages else teacher_ids
        for group_id in split_groups:
            group_idx = torch.nonzero(split_labels == int(group_id), as_tuple=False).flatten()
            group_idx = group_idx[torch.randperm(group_idx.numel(), device=device)]
            n_group_test = (
                int(group_idx.numel() * self.distill_test_frac)
                if self.distill_test_frac > 0 else 0
            )
            # Keep at least one training row for every non-empty lineage/parent.
            n_group_test = min(n_group_test, max(group_idx.numel() - 1, 0))
            test_parts.append(group_idx[:n_group_test])
            train_parts.append(group_idx[n_group_test:])
        train_idx = torch.cat(train_parts)
        test_idx = torch.cat(test_parts)
        train_idx = train_idx[torch.randperm(train_idx.numel(), device=device)]
        if test_idx.numel() > 0:
            test_idx = test_idx[torch.randperm(test_idx.numel(), device=device)]

        def student_outputs(batch_z):
            mean = self._head_forward_from_params(self.mean_pool, batch_z, mean_params)
            raw_logstd = self._head_forward_from_params(self.logstd_pool, batch_z, log_params)
            return mean, raw_logstd

        @torch.no_grad()
        def kl_summary(indices):
            if indices.numel() == 0:
                return None
            sm, sl_raw = student_outputs(z[indices])
            sl = bound_log_std(sl_raw)
            values = diagonal_gaussian_kl(
                teacher_mean[indices], teacher_logstd[indices], sm, sl
            )
            values = torch.nan_to_num(values, nan=1e12, posinf=1e12, neginf=0.0)
            return {
                "mean": float(values.mean().item()),
                "p95": float(torch.quantile(values, 0.95).item()),
                "max": float(values.max().item()),
            }

        def clone_student_params():
            return (
                {key: value.detach().clone() for key, value in mean_params.items()},
                {key: value.detach().clone() for key, value in log_params.items()},
            )

        # Epoch 0 is the arithmetic-average initialization and is a legitimate
        # candidate.  Distillation is allowed to keep it if every optimization
        # epoch makes held-out behavior worse.
        validation_idx = test_idx if test_idx.numel() > 0 else train_idx
        initial_val = kl_summary(validation_idx)
        best_val_kl = float("inf") if initial_val is None else initial_val["mean"]
        best_epoch = 0
        best_mean_params, best_log_params = clone_student_params()

        for epoch in range(1, self.distill_epochs + 1):
            shuffled = train_idx[torch.randperm(train_idx.numel(), device=device)]
            for start in range(0, shuffled.numel(), self.distill_batch_size):
                idx = shuffled[start:start + self.distill_batch_size]
                student_mean, student_raw_logstd = student_outputs(z[idx])
                student_logstd = bound_log_std(student_raw_logstd)
                loss = diagonal_gaussian_kl(
                    teacher_mean[idx], teacher_logstd[idx], student_mean, student_logstd
                ).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainables, max_norm=10.0)
                optimizer.step()

            val = kl_summary(validation_idx)
            if val is not None and val["mean"] < best_val_kl:
                best_val_kl = val["mean"]
                best_epoch = epoch
                best_mean_params, best_log_params = clone_student_params()

        # Validation-best selection is an optimization, not part of the core
        # distillation definition. Keep the legacy "last epoch" behaviour behind
        # a flag for controlled ablations.
        if self.distill_select_best_val:
            with torch.no_grad():
                for key in _HEAD_KEYS:
                    mean_params[key].copy_(best_mean_params[key])
                    log_params[key].copy_(best_log_params[key])
            selected_epoch = int(best_epoch)
            selected_val_kl = float(best_val_kl)
        else:
            selected_epoch = int(self.distill_epochs)
            final_val = kl_summary(validation_idx)
            selected_val_kl = float("nan") if final_val is None else float(final_val["mean"])

        with torch.no_grad():
            def metrics(indices):
                if indices.numel() == 0:
                    return None, None, None, None, None
                sm, sl_raw = student_outputs(z[indices])
                sl = bound_log_std(sl_raw)
                kl_values = diagonal_gaussian_kl(
                    teacher_mean[indices], teacher_logstd[indices], sm, sl
                )
                kl_values = torch.nan_to_num(
                    kl_values, nan=1e12, posinf=1e12, neginf=0.0
                )
                kl_mean = kl_values.mean().item()
                kl_p95 = torch.quantile(kl_values, 0.95).item()
                kl_max = kl_values.max().item()
                mean_mse = F.mse_loss(sm, teacher_mean[indices]).item()
                logstd_mse = F.mse_loss(sl, teacher_logstd[indices]).item()
                return kl_mean, kl_p95, kl_max, mean_mse, logstd_mse

            train_kl, train_kl_p95, train_kl_max, train_mean_mse, train_logstd_mse = metrics(train_idx)
            test_kl, test_kl_p95, test_kl_max, test_mean_mse, test_logstd_mse = metrics(test_idx)

        out_mean = {key: value.detach().clone() for key, value in mean_params.items()}
        out_log = {key: value.detach().clone() for key, value in log_params.items()}
        metrics_out = {
            "policy/distill_train_kl": train_kl,
            "policy/distill_test_kl": test_kl,
            "policy/distill_train_kl_p95": train_kl_p95,
            "policy/distill_test_kl_p95": test_kl_p95,
            "policy/distill_train_kl_max": train_kl_max,
            "policy/distill_test_kl_max": test_kl_max,
            "policy/distill_train_mean_mse": train_mean_mse,
            "policy/distill_test_mean_mse": test_mean_mse,
            "policy/distill_train_logstd_mse": train_logstd_mse,
            "policy/distill_test_logstd_mse": test_logstd_mse,
            "policy/distill_best_epoch": int(best_epoch),
            "policy/distill_best_val_kl": float(best_val_kl),
            "policy/distill_selected_epoch": selected_epoch,
            "policy/distill_selected_val_kl": selected_val_kl,
            "policy/distill_select_best_val": float(self.distill_select_best_val),
            "policy/distill_initial_val_kl": None if initial_val is None else float(initial_val["mean"]),
            "policy/distill_rows": int(n),
            "policy/distill_source_lineages": int(torch.unique(source_ids).numel()),
            "policy/distill_balance_source_lineages": float(self.balance_source_lineages),
        }
        logger.info(
            f"[policy distill] rows={n} best_epoch={best_epoch} selected_epoch={selected_epoch} "
            f"train_KL={train_kl:.6f} test_KL={test_kl if test_kl is not None else 'n/a'} "
            f"train_p95={train_kl_p95:.6f} "
            f"test_p95={test_kl_p95 if test_kl_p95 is not None else 'n/a'}"
        )
        return out_mean, out_log, metrics_out

    def finalize(self):
        """Insert the new slot, then (if needed) merge one policy-level pair."""
        self.last_merge_info = None
        self.last_distill_metrics = {}
        if self.composition_space == "policy":
            if self.policy_student_replay:
                self.store_novel_policy_for_storage()
            else:
                self.project_policy_for_storage()
        else:
            self.mean_pool.finalize_own_contribution()
            self.logstd_pool.finalize_own_contribution()
        self._assert_pool_alignment()

        if not self.mean_pool.needs_merge():
            return

        if self.merge_ablation == "random_merge":
            idx1, idx2, merge_info = self._select_random_pair()
        elif self.merge_ablation == "kl_merge" and self.distillation:
            idx1, idx2, merge_info = self._select_behavioral_pair()
        elif self.merge_ablation == "kl_discard":
            idx1, idx2, merge_info = self._select_behavioral_pair()
        elif self.distillation:
            idx1, idx2, merge_info = self._select_behavioral_pair()
        else:
            idx1, idx2, merge_info = self._select_cosine_pair()

        if self.merge_ablation == "kl_discard":
            # Unbiased, seed-reproducible discard among the minimum-KL pair.
            # The survivor and its buffer are retained without modification.
            # Do not relabel discarded source states as belonging to the survivor.
            remove = int(np.random.choice([idx1, idx2]))
            survivor = idx2 if remove == idx1 else idx1
            self.mean_pool.pool.pop(remove)
            self.logstd_pool.pool.pop(remove)
            merge_info.update({
                "idx1": int(idx1), "idx2": int(idx2),
                "used_distillation": False, "discard_only": True,
                "merge_ablation": "kl_discard", "discarded_index": int(remove),
                "surviving_index_before_removal": int(survivor),
                "pool_size_before": int(self.mean_pool.pool_length() + 1),
                "pool_size_after": int(self.mean_pool.pool_length()),
            })
            self.last_merge_info = merge_info
            self._assert_pool_alignment()
            return

        if self.distillation:
            mean_params, log_params, distill_metrics = self._distill_pair(idx1, idx2)
            self.last_distill_metrics = distill_metrics
            used_distillation = True
        else:
            mean_params = self.mean_pool.average_pair_params(idx1, idx2)
            log_params = self.logstd_pool.average_pair_params(idx1, idx2)
            used_distillation = False

        buf1 = self.mean_pool.pool[idx1].get("buffer")
        buf2 = self.mean_pool.pool[idx2].get("buffer")
        merged_buffer = HeadPool.merge_buffers(
            buf1, buf2, self.max_distill_buffer,
            balance_source_lineages=self.balance_source_lineages,
        )
        merge_info.update({
            "used_distillation": used_distillation,
            "balance_source_lineages": bool(self.balance_source_lineages),
            "pool_size_before": int(self.mean_pool.pool_length()),
            "pool_size_after": int(self.mean_pool.pool_length() - 1),
            # task-level lineage is useful for semantic task composition; source
            # lineage uses unique sequence positions so repeated task IDs remain
            # distinguishable in decay analyses.
            "parent_1_lineage": self._buffer_lineage(buf1, "task_ids"),
            "parent_2_lineage": self._buffer_lineage(buf2, "task_ids"),
            "merged_lineage": self._buffer_lineage(merged_buffer, "task_ids"),
            "parent_1_source_lineage": self._buffer_lineage(buf1, "source_ids"),
            "parent_2_source_lineage": self._buffer_lineage(buf2, "source_ids"),
            "merged_source_lineage": self._buffer_lineage(merged_buffer, "source_ids"),
        })
        self.mean_pool.replace_pair(idx1, idx2, mean_params, merged_buffer, merge_info)
        self.logstd_pool.replace_pair(idx1, idx2, log_params, None, merge_info)
        self.last_merge_info = merge_info
        self._assert_pool_alignment()

        if "policy/distill_train_kl" in self.last_distill_metrics:
            self.mean_pool.last_distill_train_kl = self.last_distill_metrics["policy/distill_train_kl"]
            self.mean_pool.last_distill_test_kl = self.last_distill_metrics["policy/distill_test_kl"]
            self.logstd_pool.last_distill_train_kl = self.last_distill_metrics["policy/distill_train_kl"]
            self.logstd_pool.last_distill_test_kl = self.last_distill_metrics["policy/distill_test_kl"]

    def get_distill_metrics(self):
        return dict(self.last_distill_metrics)

    def get_merge_info(self):
        return self.last_merge_info

    # ------------------------------------------------------------------
    # Saving / inference snapshots
    # ------------------------------------------------------------------
    @staticmethod
    def _cpu_clone_dict(d):
        return {key: value.detach().cpu().clone() for key, value in d.items()}

    def export_effective_policy(self):
        if self.composition_space == "policy":
            with torch.no_grad():
                return self.export_policy_ensemble()
        with torch.no_grad():
            mean_w0, mean_b0, mean_w2, mean_b2 = self.mean_pool._effective()
            log_w0, log_b0, log_w2, log_b2 = self.logstd_pool._effective()
            return {
                "composition_space": self.composition_space,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "distillation": self.distillation,
                "distill_observation_skip": self.distill_observation_skip,
                # FrozenCkaPolicy rebuilds the encoder with shared(...) and then
                # load_state_dict's into it. shared(linear_out=True) has the SAME
                # parameters as shared(linear_out=False) but a different forward,
                # so without this flag the load would silently succeed and then
                # evaluate a different function than the one that was trained.
                "encoder_linear_out": self.encoder_linear_out,
                "fc_state_dict": self._cpu_clone_dict(self.fc.state_dict()),
                "mean": {
                    "l0_weight": mean_w0.detach().cpu().clone(),
                    "l0_bias": mean_b0.detach().cpu().clone(),
                    "l2_weight": mean_w2.detach().cpu().clone(),
                    "l2_bias": mean_b2.detach().cpu().clone(),
                },
                "logstd": {
                    "l0_weight": log_w0.detach().cpu().clone(),
                    "l0_bias": log_b0.detach().cpu().clone(),
                    "l2_weight": log_w2.detach().cpu().clone(),
                    "l2_bias": log_b2.detach().cpu().clone(),
                },
            }

    def save_policy_snapshot(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.export_effective_policy(), f"{dirname}/policy_snapshot.pt")

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.fc, f"{dirname}/fc.pt")
        torch.save(self.mean_pool, f"{dirname}/mean_pool.pt")
        torch.save(self.logstd_pool, f"{dirname}/logstd_pool.pt")

    @staticmethod
    def load(dirname, obs_dim=None, act_dim=None, map_location=None):
        """Load the exact policy saved for evaluation.

        Finalizing a task changes the pool topology, so the pre-finalize alpha
        vector saved inside a full HeadPool checkpoint no longer describes the
        exact just-trained policy.  Older code silently returned that ambiguous
        object here.  The compact policy_snapshot.pt is the authoritative
        inference checkpoint; continuation should use CkaRlAgent(...,
        base_dir=..., latest_dir=...) so a fresh alpha is built for the current
        pool length.
        """
        snapshot_path = f"{dirname}/policy_snapshot.pt"
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(
                f"{snapshot_path} is missing; this checkpoint predates exact policy snapshots."
            )
        return FrozenCkaPolicy.load(dirname, map_location=map_location)


class FrozenCkaPolicy(nn.Module):
    """Compact inference-only policy loaded from policy_snapshot.pt."""

    def __init__(self, snapshot):
        super().__init__()
        self.composition_space = snapshot.get("composition_space", "parameter")
        self.obs_dim = int(snapshot["obs_dim"])
        self.act_dim = int(snapshot["act_dim"])
        self.distillation = bool(snapshot.get("distillation", False))
        self.distill_observation_skip = bool(snapshot.get("distill_observation_skip", True))
        # .get() so snapshots written before this flag existed still load, with
        # the original ReLU-terminated encoder.
        self.encoder_linear_out = bool(snapshot.get("encoder_linear_out", False))
        self.fc = shared(input_dim=self.obs_dim, linear_out=self.encoder_linear_out)
        self.fc.load_state_dict(snapshot["fc_state_dict"])
        if self.composition_space == "policy":
            self.register_buffer("mixture_weights", snapshot["mixture_weights"].clone())
            for head_name in ("mean", "logstd"):
                for tensor_name, tensor in snapshot[head_name + "_components"].items():
                    self.register_buffer(f"{head_name}_{tensor_name}", tensor.clone())
        else:
            for head_name in ("mean", "logstd"):
                for tensor_name, tensor in snapshot[head_name].items():
                    self.register_buffer(f"{head_name}_{tensor_name}", tensor.clone())

    def _head(self, x, head_name):
        w0 = getattr(self, f"{head_name}_l0_weight")
        b0 = getattr(self, f"{head_name}_l0_bias")
        w2 = getattr(self, f"{head_name}_l2_weight")
        b2 = getattr(self, f"{head_name}_l2_bias")
        h = F.relu(F.linear(x, w0, b0))
        return F.linear(h, w2, b2)

    def policy_components(self, obs):
        features = self.fc(obs)
        z = torch.cat((features, obs), -1) if self.distillation and self.distill_observation_skip else features
        if self.composition_space == "policy":
            heads = [{k: getattr(self, head + "_" + k) for k in _HEAD_KEYS} for head in ("mean", "logstd")]
            return (stacked_head_forward(z, heads[0]),
                    bound_log_std(stacked_head_forward(z, heads[1])), self.mixture_weights)
        mean, raw = self._head(z, "mean"), self._head(z, "logstd")
        return mean[:, None, :], bound_log_std(raw)[:, None, :], mean.new_ones(1)

    def forward(self, obs):
        if self.composition_space == "policy":
            return gaussian_summary(*self.policy_components(obs))
        features = self.fc(obs)
        z = (
            torch.cat([features, obs], dim=-1)
            if self.distillation and self.distill_observation_skip
            else features
        )
        return self._head(z, "mean"), self._head(z, "logstd")

    @staticmethod
    def load(dirname, map_location=None):
        snapshot = _torch_load(f"{dirname}/policy_snapshot.pt", map_location=map_location)
        return FrozenCkaPolicy(snapshot)
