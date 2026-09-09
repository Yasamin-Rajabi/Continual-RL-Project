"""Policy-space adaptation and bounded-pool insertion support.

This module intentionally does NOT alter the existing pair selector or pair
merger. A policy ensemble cannot generally be serialized as one averaged MLP.
It is therefore projected into a single Gaussian head before pool insertion,
using only the frozen tail states already collected inside the task budget.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from policy_utils import bound_log_std, diagonal_gaussian_kl
from policy_composition import gaussian_summary, stacked_head_forward

KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


class PolicySpaceMixin:
    def _head_features(self, obs):
        features = self.fc(obs)
        if self.distillation and self.distill_observation_skip:
            return torch.cat((features, obs), dim=-1)
        return features

    def set_mixture_warmup(self, enabled):
        """Exact historical-only phase; sigmoid parameters remain finite."""
        self.mixture_warmup = bool(enabled)
        self.mean_pool.force_unit_mass = bool(enabled)
        self.logstd_pool.force_unit_mass = bool(enabled)

    def initialize_policy_space_own(self):
        # A gated novel expert needs an actual MLP, not an all-zero ReLU net.
        # Copy a prior complete head only as initialization; all its own
        # parameters can then adapt, while historical entries stay untouched.
        if (self.composition_space == "policy" and self.use_alpha_mass
                and self.mean_pool.pool):
            with torch.no_grad():
                for pool in (self.mean_pool, self.logstd_pool):
                    for key, value in zip(KEYS, pool.entry_effective_weights(pool.pool[0])):
                        getattr(pool, "own_" + key).copy_(value)

    def _ensemble_heads(self):
        """Return two stacked COMPLETE-head dictionaries and simplex weights.

        With alpha-mass: m sum_i alpha_i pi_i + (1-m) pi_new.
        Without alpha-mass: sum_i alpha_i pi_(entry_i + current_delta).
        The latter retains trainable task residuals for the baseline and the
        no-mass ablations; a residual by itself is never treated as a policy.
        """
        pools = (self.mean_pool, self.logstd_pool)
        historical_only = bool(getattr(self, "pool_only", False))
        n = self.mean_pool.pool_length()
        if self.composition_space == "parameter" or not n:
            return [dict(zip(KEYS, (x.unsqueeze(0) for x in p._effective()))) for p in pools], self.mean_pool.own_l0_weight.new_ones(1)
        scale = self.alpha_scale if self.alpha_scale is not None else 1.0
        weights = F.softmax(self.alpha * scale, dim=0)
        warmup = bool(getattr(self, "mixture_warmup", False))
        include_novel = self.use_alpha_mass and not historical_only and not warmup
        if include_novel:
            mass = self.mean_pool.effective_alpha_mass().reshape(1)
            weights = torch.cat((weights * mass, 1.0 - mass))
        heads = []
        for pool in pools:
            head_values = [pool.entry_effective_weights(e) for e in pool.pool]
            if not self.use_alpha_mass and not historical_only:
                own = tuple(getattr(pool, "own_" + key) for key in KEYS)
                head_values = [tuple(a + b for a, b in zip(head, own)) for head in head_values]
            if include_novel:
                head_values.append(tuple(getattr(pool, "own_" + key) for key in KEYS))
            heads.append({key: torch.stack([h[idx] for h in head_values]) for idx, key in enumerate(KEYS)})
        return heads, weights

    def _components_at_features(self, z):
        heads, weights = self._ensemble_heads()
        means = stacked_head_forward(z, heads[0])
        logs = bound_log_std(stacked_head_forward(z, heads[1]))
        return means, logs, weights

    def policy_components(self, obs):
        return self._components_at_features(self._head_features(obs))

    def novel_policy_components(self, obs):
        """Return the standalone current expert, independent of alpha/gating.

        In the policy-student mode this is the policy optimized from replay and
        ultimately inserted into the bounded pool.  Historical slots are not
        included in these logits.
        """
        z = self._head_features(obs)
        values = []
        for pool in (self.mean_pool, self.logstd_pool):
            own = tuple(getattr(pool, "own_" + key) for key in KEYS)
            values.append(pool._forward_with_weights(z, own))
        return values[0], bound_log_std(values[1])

    def export_policy_ensemble(self):
        heads, weights = self._ensemble_heads()
        return {
            "snapshot_format_version": 2,
            "composition_space": "policy",
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "distillation": self.distillation,
            "distill_observation_skip": self.distill_observation_skip,
            "encoder_linear_out": self.encoder_linear_out,
            "fc_state_dict": self._cpu_clone_dict(self.fc.state_dict()),
            "mixture_weights": weights.detach().cpu().clone(),
            "mean_components": {k: v.detach().cpu().clone() for k, v in heads[0].items()},
            "logstd_components": {k: v.detach().cpu().clone() for k, v in heads[1].items()},
        }

    def store_novel_policy_for_storage(self):
        """Insert the learned standalone novel expert without mixture projection.

        This is used only by the replay-trained policy-student variant.  It is
        intentionally different from ``project_policy_for_storage``: the point
        of the variant is to test whether the current expert itself becomes
        sufficient (alpha-mass -> 0), rather than hiding residual dependence on
        history by projecting the final execution ensemble.
        """
        if self.composition_space != "policy" or not self.use_alpha_mass:
            raise RuntimeError("novel-policy storage requires gated policy-space composition")
        if self.fusion_mode != "weight_delta":
            raise RuntimeError("novel-policy storage is defined for full-weight/weight_delta entries")
        buffer = self.mean_pool.own_buffer
        for pool in (self.mean_pool, self.logstd_pool):
            entry = {key: getattr(pool, "own_" + key).detach().clone() for key in KEYS}
            entry["buffer"] = buffer if pool is self.mean_pool else None
            pool.pool = [entry] + pool.pool
            pool.reset_own_to_zero()
        self.last_projection_metrics = {
            "policy/storage_used_novel_expert": 1.0,
            "policy/projection_components": 0,
            "policy/projection_rows": 0 if buffer is None else int(len(buffer.get("obs", ()))),
        }

    def project_policy_for_storage(self):
        """Train one complete Gaussian head to represent the current mixture.

        sum_k w_k KL(pi_k || student) has the SAME student-dependent term
        as KL(sum_k w_k pi_k || student). Its value includes a constant
        component-disagreement term, so diagnostics are explicitly named
        component_KL, not exact mixture_KL. A validation split selects epochs;
        it is never called an independent test set.
        """
        buffer = self.mean_pool.own_buffer
        if buffer is None or len(buffer.get("obs", ())) < 2:
            raise ValueError("Policy-space insertion requires at least two frozen tail states (B >= 2).")
        obs = np.asarray(buffer["obs"], dtype=np.float32)
        if len(obs) > self.projection_max_samples:
            obs = obs[np.random.choice(len(obs), self.projection_max_samples, replace=False)]
        z = self._encode_obs(obs)
        with torch.no_grad():
            target_mean, target_log, weights = self._components_at_features(z)
            target_mean, target_log, weights = target_mean.detach(), target_log.detach(), weights.detach()
            heads, _ = self._ensemble_heads()
        n = len(z)
        order = torch.randperm(n, device=z.device)
        nv = min(max(1, int(n * 0.2)), n - 1)
        validation, train = order[:nv], order[nv:]

        def values(params, indices):
            mean = self.mean_pool._forward_with_weights(z[indices], tuple(params[0][k] for k in KEYS))
            log = bound_log_std(self.logstd_pool._forward_with_weights(z[indices], tuple(params[1][k] for k in KEYS)))
            loss = diagonal_gaussian_kl(target_mean[indices], target_log[indices], mean[:, None, :], log[:, None, :])
            return (loss * weights[None, :]).sum(-1)

        def evaluate(params, indices):
            with torch.no_grad():
                total = 0.0
                for start in range(0, len(indices), self.distill_batch_size):
                    idx = indices[start:start + self.distill_batch_size]
                    total += float(values(params, idx).sum())
                return total / len(indices)

        # Try a weighted-parameter initializer AND every complete component.
        # Choosing among them is not the policy definition; it only avoids a
        # catastrophically bad initialization before function-space fitting.
        candidates = [[{key: (value * weights.reshape((-1,) + (1,) * (value.ndim - 1))).sum(0).detach().clone()
                         for key, value in head.items()} for head in heads]]
        candidates += [[{key: value[i].detach().clone() for key, value in head.items()} for head in heads]
                       for i in range(len(weights))]
        scores = [evaluate(p, validation) for p in candidates]
        chosen = int(np.argmin(scores))
        params = [{k: v.clone().requires_grad_(True) for k, v in h.items()} for h in candidates[chosen]]
        optimizer = torch.optim.Adam([v for h in params for v in h.values()], lr=self.distill_lr)
        best_score = initial = scores[chosen]
        best_epoch = 0
        best = [{k: v.detach().clone() for k, v in h.items()} for h in params]
        for epoch in range(self.projection_epochs):
            perm = train[torch.randperm(len(train), device=z.device)]
            for start in range(0, len(perm), self.distill_batch_size):
                loss = values(params, perm[start:start + self.distill_batch_size]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite policy-space projection loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            score = evaluate(params, validation)
            if np.isfinite(score) and score < best_score:
                best_score, best_epoch = score, epoch + 1
                best = [{k: v.detach().clone() for k, v in h.items()} for h in params]
        self.last_projection_metrics = {
            "policy/projection_initial_val_component_kl": initial,
            "policy/projection_val_component_kl": best_score,
            "policy/projection_train_component_kl": evaluate(best, train),
            "policy/projection_best_epoch": best_epoch,
            "policy/projection_rows": n,
            "policy/projection_components": len(weights),
        }
        # In classic mode a standalone entry is base + delta. Save the
        # projected full head relative to that base, not the bare task delta.
        for pool, head in zip((self.mean_pool, self.logstd_pool), best):
            if self.fusion_mode == "classic_cka":
                head = {k: v - getattr(pool, "base_" + k) for k, v in head.items()}
            entry = {k: v.detach().clone() for k, v in head.items()}
            entry["buffer"] = buffer if pool is self.mean_pool else None
            pool.pool = [entry] + pool.pool
            pool.reset_own_to_zero()
