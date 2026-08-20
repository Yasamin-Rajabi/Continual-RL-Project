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

from knowledge_pools import BASE_FUSION_MODE, HeadPool
from policy_utils import bound_log_std, diagonal_gaussian_kl, symmetric_diagonal_gaussian_kl
from shared_arch import shared

_HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


class CkaRlAgent(nn.Module):
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
        use_alpha_mass=False,
        encoder_from_base=False,
        distillation=True,
        fusion_mode=BASE_FUSION_MODE,
        max_distill_buffer=50_000,
        distill_test_frac=0.2,
        distill_epochs=8,
        distill_lr=3e-4,
        distill_batch_size=256,
        distill_max_samples=20_000,
        similarity_samples=2048,
        hidden_dim=128,
        shared_dim=256,
        train_shared=False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.pool_size = int(pool_size)
        self.distillation = bool(distillation)
        self.fusion_mode = fusion_mode
        self.max_distill_buffer = int(max_distill_buffer)
        self.use_alpha_mass = bool(use_alpha_mass)
        self.distill_test_frac = float(distill_test_frac)
        self.distill_epochs = int(distill_epochs)
        self.distill_lr = float(distill_lr)
        self.distill_batch_size = int(distill_batch_size)
        self.distill_max_samples = int(distill_max_samples)
        self.similarity_samples = int(similarity_samples)
        self.train_shared = bool(train_shared)
        self.last_merge_info = None
        self.last_distill_metrics = {}

        self.mean_pool = HeadPool(
            "mean", shared_dim, hidden_dim, act_dim,
            fusion_mode=fusion_mode, pool_size=pool_size,
            distillation=distillation, max_distill_buffer=max_distill_buffer,
            use_alpha_mass=use_alpha_mass, distill_test_frac=distill_test_frac,
        )
        self.logstd_pool = HeadPool(
            "logstd", shared_dim, hidden_dim, act_dim,
            fusion_mode=fusion_mode, pool_size=pool_size,
            distillation=distillation, max_distill_buffer=max_distill_buffer,
            use_alpha_mass=use_alpha_mass, distill_test_frac=distill_test_frac,
        )

        if latest_dir is not None:
            latest_mean_pool = _torch_load(f"{latest_dir}/mean_pool.pt", map_location="cpu")
            latest_logstd_pool = _torch_load(f"{latest_dir}/logstd_pool.pt", map_location="cpu")
            self.mean_pool.inherit_pool_from(latest_mean_pool)
            self.logstd_pool.inherit_pool_from(latest_logstd_pool)
            self._assert_pool_alignment()
            self.mean_pool.reset_own_to_zero()
            self.logstd_pool.reset_own_to_zero()

        # One alpha vector controls each whole knowledge vector, across both heads.
        self.alpha, self.alpha_scale, self.alpha_mass = self._make_alpha(
            self.mean_pool.pool_length(), fix_alpha, alpha_init, alpha_major,
            alpha_factor, use_alpha_scale, use_alpha_mass,
        )
        self.mean_pool.set_alpha(self.alpha, self.alpha_scale, self.alpha_mass)
        self.logstd_pool.set_alpha(self.alpha, self.alpha_scale, self.alpha_mass)
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
        if encoder_from_base and base_dir is not None:
            logger.info(f"Loading encoder from base {base_dir}")
            self.fc = _torch_load(f"{base_dir}/fc.pt", map_location="cpu")
        elif latest_dir is not None:
            logger.info(f"Loading shared encoder from {latest_dir}")
            self.fc = _torch_load(f"{latest_dir}/fc.pt", map_location="cpu")
        else:
            logger.info("Training root shared encoder from scratch")
            self.fc = shared(input_dim=obs_dim)

        if latest_dir is not None and not self.train_shared:
            logger.info("Shared encoder frozen (train_shared=False)")
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
        use_alpha_scale, use_alpha_mass,
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
        alpha_scale = nn.Parameter(torch.ones(1), requires_grad=(use_alpha_scale and not fix_alpha))
        alpha_mass = (
            nn.Parameter(torch.ones(1), requires_grad=not fix_alpha)
            if use_alpha_mass else None
        )
        return alpha, alpha_scale, alpha_mass

    def forward(self, x):
        z = self.fc(x)
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
            idx = np.random.choice(len(obs), size=take, replace=False)
            samples.append(obs[idx].astype(np.float32, copy=False))
        return samples

    def _encode_obs(self, obs: np.ndarray, batch_size: int = 4096) -> torch.Tensor:
        device = self.mean_pool.base_l0_weight.device
        chunks = []
        with torch.no_grad():
            for start in range(0, len(obs), batch_size):
                x = torch.as_tensor(obs[start:start + batch_size], dtype=torch.float32, device=device)
                chunks.append(self.fc(x))
        return torch.cat(chunks, dim=0)

    def _entry_outputs(self, z: torch.Tensor, index: int):
        mean = self.mean_pool.forward_entry(z, index)
        raw_logstd = self.logstd_pool.forward_entry(z, index)
        return mean, raw_logstd

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

        stats = {
            "idx1": int(idx1),
            "idx2": int(idx2),
            "symmetric_kl": selected,
            "pairwise_kl_min": float(np.min(finite_values)),
            "pairwise_kl_mean": float(np.mean(finite_values)),
            "pairwise_kl_max": float(np.max(finite_values)),
            "similarity_states": selected_rows,
            "reference_rows_per_slot": [int(len(x)) for x in obs_by_slot],
            "pairwise_symmetric_kl": matrix.detach().cpu().numpy().tolist(),
        }
        logger.info(
            f"[behavioral merge] pair=({idx1},{idx2}) symmetric_KL={selected:.6f} "
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
    def _buffer_lineage(buffer):
        if buffer is None or "task_ids" not in buffer:
            return {}
        ids, counts = np.unique(np.asarray(buffer["task_ids"]).reshape(-1), return_counts=True)
        return {str(int(task_id)): int(count) for task_id, count in zip(ids, counts)}

    def _balanced_parent_data(self, idx1: int, idx2: int):
        buf1 = self.mean_pool.pool[idx1].get("buffer")
        buf2 = self.mean_pool.pool[idx2].get("buffer")
        if buf1 is None or buf2 is None:
            raise RuntimeError("distillation requested but a selected pool entry has no observation buffer")
        max_each = max(1, self.distill_max_samples // 2)
        obs_parts, teacher_ids = [], []
        for teacher_id, buf in enumerate((buf1, buf2)):
            obs = buf["obs"]
            take = min(len(obs), max_each)
            idx = np.random.choice(len(obs), size=take, replace=False)
            obs_parts.append(obs[idx])
            teacher_ids.append(np.full(take, teacher_id, dtype=np.int64))
        return (
            np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False),
            np.concatenate(teacher_ids, axis=0),
        )

    def _distill_pair(self, idx1: int, idx2: int):
        """KL-distill two aligned Gaussian policy entries into one student."""
        obs, teacher_ids_np = self._balanced_parent_data(idx1, idx2)
        z = self._encode_obs(obs)
        device = z.device
        teacher_ids = torch.as_tensor(teacher_ids_np, dtype=torch.long, device=device)

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
        # Stratify the held-out split by parent so train/test diagnostics do not
        # accidentally contain only one teacher when buffers are small.
        train_parts, test_parts = [], []
        for teacher_id in (0, 1):
            parent_idx = torch.nonzero(teacher_ids == teacher_id, as_tuple=False).flatten()
            parent_idx = parent_idx[torch.randperm(parent_idx.numel(), device=device)]
            n_parent_test = (
                int(parent_idx.numel() * self.distill_test_frac)
                if self.distill_test_frac > 0 else 0
            )
            # Keep at least one training row for every non-empty parent.
            n_parent_test = min(n_parent_test, max(parent_idx.numel() - 1, 0))
            test_parts.append(parent_idx[:n_parent_test])
            train_parts.append(parent_idx[n_parent_test:])
        train_idx = torch.cat(train_parts)
        test_idx = torch.cat(test_parts)
        train_idx = train_idx[torch.randperm(train_idx.numel(), device=device)]
        if test_idx.numel() > 0:
            test_idx = test_idx[torch.randperm(test_idx.numel(), device=device)]

        def student_outputs(batch_z):
            mean = self._head_forward_from_params(self.mean_pool, batch_z, mean_params)
            raw_logstd = self._head_forward_from_params(self.logstd_pool, batch_z, log_params)
            return mean, raw_logstd

        for _ in range(self.distill_epochs):
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

        with torch.no_grad():
            def metrics(indices):
                if indices.numel() == 0:
                    return None, None, None
                sm, sl_raw = student_outputs(z[indices])
                sl = bound_log_std(sl_raw)
                kl = diagonal_gaussian_kl(
                    teacher_mean[indices], teacher_logstd[indices], sm, sl
                ).mean().item()
                mean_mse = F.mse_loss(sm, teacher_mean[indices]).item()
                logstd_mse = F.mse_loss(sl, teacher_logstd[indices]).item()
                return kl, mean_mse, logstd_mse

            train_kl, train_mean_mse, train_logstd_mse = metrics(train_idx)
            test_kl, test_mean_mse, test_logstd_mse = metrics(test_idx)

        out_mean = {key: value.detach().clone() for key, value in mean_params.items()}
        out_log = {key: value.detach().clone() for key, value in log_params.items()}
        metrics_out = {
            "policy/distill_train_kl": train_kl,
            "policy/distill_test_kl": test_kl,
            "policy/distill_train_mean_mse": train_mean_mse,
            "policy/distill_test_mean_mse": test_mean_mse,
            "policy/distill_train_logstd_mse": train_logstd_mse,
            "policy/distill_test_logstd_mse": test_logstd_mse,
            "policy/distill_rows": int(n),
        }
        logger.info(
            f"[policy distill] rows={n} train_KL={train_kl:.6f} "
            f"test_KL={test_kl if test_kl is not None else 'n/a'}"
        )
        return out_mean, out_log, metrics_out

    def finalize(self):
        """Insert the new slot, then (if needed) merge one policy-level pair."""
        self.last_merge_info = None
        self.last_distill_metrics = {}
        self.mean_pool.finalize_own_contribution()
        self.logstd_pool.finalize_own_contribution()
        self._assert_pool_alignment()

        if not self.mean_pool.needs_merge():
            return

        idx1, idx2, merge_info = self._select_behavioral_pair()
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
        merged_buffer = HeadPool.merge_buffers(buf1, buf2, self.max_distill_buffer)
        merge_info.update({
            "used_distillation": used_distillation,
            "pool_size_before": int(self.mean_pool.pool_length()),
            "pool_size_after": int(self.mean_pool.pool_length() - 1),
            "parent_1_lineage": self._buffer_lineage(buf1),
            "parent_2_lineage": self._buffer_lineage(buf2),
            "merged_lineage": self._buffer_lineage(merged_buffer),
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
        with torch.no_grad():
            mean_w0, mean_b0, mean_w2, mean_b2 = self.mean_pool._effective()
            log_w0, log_b0, log_w2, log_b2 = self.logstd_pool._effective()
            return {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
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
        self.obs_dim = int(snapshot["obs_dim"])
        self.act_dim = int(snapshot["act_dim"])
        self.fc = shared(input_dim=self.obs_dim)
        self.fc.load_state_dict(snapshot["fc_state_dict"])
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

    def forward(self, obs):
        z = self.fc(obs)
        return self._head(z, "mean"), self._head(z, "logstd")

    @staticmethod
    def load(dirname, map_location=None):
        snapshot = _torch_load(f"{dirname}/policy_snapshot.pt", map_location=map_location)
        return FrozenCkaPolicy(snapshot)
