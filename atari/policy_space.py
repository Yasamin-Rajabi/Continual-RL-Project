"""Policy-space composition and bounded-pool insertion for Atari categorical policies."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from policy_composition import (
    categorical_mixture_distribution,
    categorical_mixture_probs,
    stacked_head_forward,
)
from policy_utils import categorical_kl_from_probs

KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


class PolicySpaceMixin:
    def _head_features(self, obs):
        return self.fc(obs)

    def set_mixture_warmup(self, enabled):
        self.mixture_warmup = bool(enabled)
        self.policy_pool.force_unit_mass = bool(enabled)

    def initialize_policy_space_own(self):
        # As in HalfCheetah, a gated novel expert must be a real MLP rather than
        # an all-zero ReLU network.
        if (
            self.composition_space == "policy"
            and self.use_alpha_mass
            and self.policy_pool.pool
        ):
            with torch.no_grad():
                for key, value in zip(
                    KEYS,
                    self.policy_pool.entry_effective_weights(self.policy_pool.pool[0]),
                ):
                    getattr(self.policy_pool, "own_" + key).copy_(value)

    def _ensemble_head(self):
        historical_only = bool(getattr(self, "pool_only", False))
        n = self.policy_pool.pool_length()

        if self.composition_space == "parameter" or not n:
            head = {
                key: value.unsqueeze(0)
                for key, value in zip(KEYS, self.policy_pool._effective())
            }
            return head, self.policy_pool.own_l0_weight.new_ones(1)

        scale = self.alpha_scale if self.alpha_scale is not None else 1.0
        weights = F.softmax(self.alpha * scale, dim=0)
        warmup = bool(getattr(self, "mixture_warmup", False))
        include_novel = self.use_alpha_mass and not historical_only and not warmup

        if include_novel:
            mass = self.policy_pool.effective_alpha_mass().reshape(1)
            weights = torch.cat((weights * mass, 1.0 - mass))

        head_values = [
            self.policy_pool.entry_effective_weights(entry)
            for entry in self.policy_pool.pool
        ]

        # Same semantics as HalfCheetah policy-space without alpha-mass:
        # each historical complete policy receives the trainable task residual.
        if not self.use_alpha_mass and not historical_only:
            own = tuple(getattr(self.policy_pool, "own_" + key) for key in KEYS)
            head_values = [
                tuple(a + b for a, b in zip(head, own))
                for head in head_values
            ]

        if include_novel:
            head_values.append(
                tuple(getattr(self.policy_pool, "own_" + key) for key in KEYS)
            )

        head = {
            key: torch.stack([values[idx] for values in head_values])
            for idx, key in enumerate(KEYS)
        }
        return head, weights

    def _components_at_features(self, features):
        head, weights = self._ensemble_head()
        return stacked_head_forward(features, head), weights

    def policy_components(self, obs):
        return self._components_at_features(self._head_features(obs))

    def novel_policy_logits(self, obs):
        features = self._head_features(obs)
        own = tuple(getattr(self.policy_pool, "own_" + key) for key in KEYS)
        return self.policy_pool._forward_with_weights(features, own)

    def novel_action_distribution(self, obs):
        """Standalone novel categorical expert used by policy-student PPO."""
        return torch.distributions.Categorical(logits=self.novel_policy_logits(obs))

    def routing_action_distribution(self, obs):
        """Execution mixture with expert functions detached.

        Gradients flow only through routing weights (alpha/alpha-mass).  This is
        the categorical Atari counterpart of the HalfCheetah routing-only
        objective used by the policy-student variant.
        """
        features = self._head_features(obs)
        logits, weights = self._components_at_features(features)
        return categorical_mixture_distribution(logits.detach(), weights)

    def export_policy_ensemble(self):
        head, weights = self._ensemble_head()
        return {
            "format_version": 3,
            "composition_space": "policy",
            "policy_type": "categorical",
            "obs_shape": tuple(self.obs_shape),
            "act_dim": int(self.act_dim),
            "shared_dim": int(self.shared_dim),
            "hidden_dim": int(self.hidden_dim),
            "distillation": bool(self.distillation),
            "fc_state_dict": self._cpu_clone_dict(self.fc.state_dict()),
            "mixture_weights": weights.detach().cpu().clone(),
            "policy_components": {
                key: value.detach().cpu().clone()
                for key, value in head.items()
            },
        }

    def store_novel_policy_for_storage(self):
        if self.composition_space != "policy" or not self.use_alpha_mass:
            raise RuntimeError(
                "novel-policy storage requires gated policy-space composition"
            )
        if self.fusion_mode != "weight_delta":
            raise RuntimeError(
                "novel-policy storage is defined for full-weight/weight_delta entries"
            )
        buffer = self.policy_pool.own_buffer
        entry = {
            key: getattr(self.policy_pool, "own_" + key).detach().clone()
            for key in KEYS
        }
        entry["buffer"] = buffer
        self.policy_pool.pool = [entry] + self.policy_pool.pool
        self.policy_pool.reset_own_to_zero()
        self.last_projection_metrics = {
            "policy/storage_used_novel_expert": 1.0,
            "policy/projection_components": 0,
            "policy/projection_rows": 0 if buffer is None else int(len(buffer.get("obs", ()))),
        }

    def project_policy_for_storage(self):
        """Fit one categorical head to the exact current policy mixture."""
        buffer = self.policy_pool.own_buffer
        if buffer is None or len(buffer.get("obs", ())) < 2:
            raise ValueError(
                "Policy-space insertion requires at least two frozen tail states (B >= 2)."
            )

        obs = np.asarray(buffer["obs"])
        if len(obs) > self.projection_max_samples:
            obs = obs[
                np.random.choice(len(obs), self.projection_max_samples, replace=False)
            ]
        features = self._encode_obs(obs)

        with torch.no_grad():
            target_logits, weights = self._components_at_features(features)
            target_probs = categorical_mixture_probs(
                target_logits.detach(), weights.detach()
            )
            heads, _ = self._ensemble_head()

        n = len(features)
        order = torch.randperm(n, device=features.device)
        n_val = min(max(1, int(n * 0.2)), n - 1)
        validation, train = order[:n_val], order[n_val:]

        def values(params, indices):
            logits = self.policy_pool._forward_with_weights(
                features[indices], tuple(params[k] for k in KEYS)
            )
            return categorical_kl_from_probs(target_probs[indices], logits)

        def evaluate(params, indices):
            with torch.no_grad():
                total = 0.0
                for start in range(0, len(indices), self.distill_batch_size):
                    idx = indices[start:start + self.distill_batch_size]
                    total += float(values(params, idx).sum())
                return total / len(indices)

        # Same robust initializer strategy as HalfCheetah: weighted parameters
        # plus each complete component, then function-space fitting.
        weighted = {
            key: (
                value
                * weights.reshape((-1,) + (1,) * (value.ndim - 1))
            ).sum(0).detach().clone()
            for key, value in heads.items()
        }
        candidates = [weighted]
        candidates += [
            {key: value[i].detach().clone() for key, value in heads.items()}
            for i in range(len(weights))
        ]
        scores = [evaluate(candidate, validation) for candidate in candidates]
        chosen = int(np.argmin(scores))

        params = {
            key: value.clone().requires_grad_(True)
            for key, value in candidates[chosen].items()
        }
        optimizer = torch.optim.Adam(list(params.values()), lr=self.distill_lr)

        best_score = initial = scores[chosen]
        best_epoch = 0
        best = {key: value.detach().clone() for key, value in params.items()}

        for epoch in range(self.projection_epochs):
            perm = train[torch.randperm(len(train), device=features.device)]
            for start in range(0, len(perm), self.distill_batch_size):
                loss = values(
                    params, perm[start:start + self.distill_batch_size]
                ).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite policy-space projection loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            score = evaluate(params, validation)
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_epoch = epoch + 1
                best = {
                    key: value.detach().clone()
                    for key, value in params.items()
                }

        self.last_projection_metrics = {
            "policy/projection_initial_val_mixture_kl": initial,
            "policy/projection_val_mixture_kl": best_score,
            "policy/projection_train_mixture_kl": evaluate(best, train),
            "policy/projection_best_epoch": best_epoch,
            "policy/projection_rows": n,
            "policy/projection_components": len(weights),
        }

        if self.fusion_mode == "classic_cka":
            best = {
                key: value - getattr(self.policy_pool, "base_" + key)
                for key, value in best.items()
            }

        entry = {key: value.detach().clone() for key, value in best.items()}
        entry["buffer"] = buffer
        self.policy_pool.pool = [entry] + self.policy_pool.pool
        self.policy_pool.reset_own_to_zero()
