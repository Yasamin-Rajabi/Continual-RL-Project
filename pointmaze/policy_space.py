"""Policy-space composition and bounded-pool insertion for PointMaze.

Parameter-space composition averages *weights* and then runs one head.
Policy-space composition runs every stored head and mixes the resulting
*distributions*.  The second is what condition 4 ("combined") uses, and it is
the setting in which a knowledge pool can express "use the south-west expert
here and the north-east expert there" rather than being forced to blend two
unrelated behaviors into one set of weights.

Because a mixture of Gaussians cannot be stored as a single head, inserting the
current task into the pool requires a *projection*: fit one head to the exact
mixture the agent has been executing.  That fit is a Monte-Carlo cross-entropy
against samples drawn from the mixture, which has the same minimizer as
``KL(mixture || head)``.
"""
from __future__ import annotations

import numpy as np
import torch

from policy_composition import split_stacked, stacked_head_forward
from policy_utils import SquashedGaussianMixture, mixture_cross_entropy

KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


class PolicySpaceMixin:
    """Mixed into ``CkaRlAgent``; assumes the host defines ``policy_pool`` etc."""

    def _head_features(self, obs):
        return self.fc(obs)

    def set_mixture_warmup(self, enabled: bool):
        self.mixture_warmup = bool(enabled)
        self.policy_pool.force_unit_mass = bool(enabled)

    def initialize_policy_space_own(self):
        """Seed the novel expert with a real network rather than all zeros.

        A gated novel expert starts at zero weights, which for a ReLU head is a
        constant function; leaving it there gives the mixture a dead component
        and a useless gradient at the start of the task.  Copying the first
        pool entry gives it a working policy to move away from.
        """
        if (
            self.composition_space == "policy"
            and self.use_alpha_mass
            and self.policy_pool.pool
        ):
            with torch.no_grad():
                weights = self.policy_pool.entry_effective_weights(
                    self.policy_pool.pool[0]
                )
                for key, value in zip(KEYS, weights):
                    getattr(self.policy_pool, "own_" + key).copy_(value)

    # ------------------------------------------------------------------
    # Ensemble construction
    # ------------------------------------------------------------------
    def _ensemble_head(self):
        """Return stacked head parameters [K, ...] and mixture weights [K]."""
        historical_only = bool(getattr(self, "pool_only", False))
        n = self.policy_pool.pool_length()

        if self.composition_space == "parameter" or not n:
            head = {
                key: value.unsqueeze(0)
                for key, value in zip(KEYS, self.policy_pool._effective())
            }
            return head, self.policy_pool.own_l0_weight.new_ones(1)

        scale = self.alpha_scale if self.alpha_scale is not None else 1.0
        weights = torch.softmax(self.alpha * scale, dim=0)
        warmup = bool(getattr(self, "mixture_warmup", False))
        include_novel = self.use_alpha_mass and not historical_only and not warmup

        if include_novel:
            mass = self.policy_pool.effective_alpha_mass().reshape(1)
            weights = torch.cat((weights * mass, 1.0 - mass))

        head_values = [
            self.policy_pool.entry_effective_weights(entry)
            for entry in self.policy_pool.pool
        ]

        # Without alpha-mass there is no separate novel component, so the
        # trainable task residual is added to every historical head instead.
        if not self.use_alpha_mass and not historical_only:
            own = tuple(getattr(self.policy_pool, "own_" + key) for key in KEYS)
            head_values = [
                tuple(a + b for a, b in zip(head, own)) for head in head_values
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

    def novel_policy_raw(self, obs):
        """Raw output of the standalone novel expert (policy-student variant)."""
        features = self._head_features(obs)
        own = tuple(getattr(self.policy_pool, "own_" + key) for key in KEYS)
        return self.policy_pool._forward_with_weights(features, own)

    def routing_distribution(self, obs):
        """Execution mixture with the expert functions detached.

        Gradients reach only the routing parameters (alpha, alpha-mass), which
        is what lets the policy-student variant train "where to route" and
        "what the new expert does" with two separate objectives.
        """
        raw, weights = self._components_at_features(self._head_features(obs))
        means, log_stds = split_stacked(raw.detach(), self.act_dim)
        return SquashedGaussianMixture(means, log_stds, weights)

    def export_policy_ensemble(self):
        head, weights = self._ensemble_head()
        return {
            "format_version": self.SNAPSHOT_FORMAT_VERSION,
            "composition_space": "policy",
            "policy_type": "squashed_gaussian",
            "obs_dim": int(self.obs_dim),
            "act_dim": int(self.act_dim),
            "shared_dim": int(self.shared_dim),
            "hidden_dim": int(self.hidden_dim),
            "distillation": bool(self.distillation),
            "fc_state_dict": self._cpu_clone_dict(self.fc.state_dict()),
            "mixture_weights": weights.detach().cpu().clone(),
            "policy_components": {
                key: value.detach().cpu().clone() for key, value in head.items()
            },
        }

    # ------------------------------------------------------------------
    # Insertion into the bounded pool
    # ------------------------------------------------------------------
    def store_novel_policy_for_storage(self):
        """Store the standalone novel expert verbatim (policy-student variant)."""
        if self.composition_space != "policy" or not self.use_alpha_mass:
            raise RuntimeError("novel-policy storage requires gated policy composition")
        if self.fusion_mode != "weight_delta":
            raise RuntimeError("novel-policy storage requires full-weight entries")
        buffer = self.policy_pool.own_buffer
        entry = {
            key: getattr(self.policy_pool, "own_" + key).detach().clone() for key in KEYS
        }
        entry["buffer"] = buffer
        self.policy_pool.pool = [entry] + self.policy_pool.pool
        self.policy_pool.reset_own_to_zero()
        self.last_projection_metrics = {
            "policy/storage_used_novel_expert": 1.0,
            "policy/projection_components": 0,
            "policy/projection_rows": 0 if buffer is None else int(len(buffer["obs"])),
        }

    def project_policy_for_storage(self):
        """Fit one Gaussian head to the exact mixture currently being executed."""
        buffer = self.policy_pool.own_buffer
        if buffer is None or len(buffer.get("obs", ())) < 2:
            raise ValueError(
                "policy-space insertion requires at least two frozen tail states"
            )

        obs = np.asarray(buffer["obs"])
        if len(obs) > self.projection_max_samples:
            obs = obs[
                np.random.choice(len(obs), self.projection_max_samples, replace=False)
            ]
        features = self._encode_obs(obs)
        device = features.device

        with torch.no_grad():
            raw, weights = self._components_at_features(features)
            means, log_stds = split_stacked(raw, self.act_dim)
            target = SquashedGaussianMixture(means, log_stds, weights)
            # Draw the projection dataset once so the objective is a fixed
            # target rather than a fresh stochastic one at every step.
            samples_u = target.sample_pre_squash(self.projection_samples)
            heads, _ = self._ensemble_head()

        n = len(features)
        order = torch.randperm(n, device=device)
        n_val = min(max(1, int(n * 0.2)), n - 1)
        validation, train = order[:n_val], order[n_val:]

        def objective(params, indices):
            raw_out = self.policy_pool._forward_with_weights(
                features[indices], tuple(params[k] for k in KEYS)
            )
            mean_q, log_std_q = torch.split(raw_out, self.act_dim, dim=-1)
            from policy_utils import clamp_log_std

            return mixture_cross_entropy(
                samples_u[:, indices, :], mean_q, clamp_log_std(log_std_q)
            )

        def evaluate(params, indices):
            with torch.no_grad():
                total = 0.0
                for start in range(0, len(indices), self.distill_batch_size):
                    idx = indices[start:start + self.distill_batch_size]
                    total += float(objective(params, idx).sum())
                return total / max(len(indices), 1)

        # Robust initialization: the weight-space blend plus each complete
        # component, scored on held-out states before any gradient step.
        weighted = {
            key: (value * weights.reshape((-1,) + (1,) * (value.ndim - 1)))
            .sum(0)
            .detach()
            .clone()
            for key, value in heads.items()
        }
        candidates = [weighted] + [
            {key: value[i].detach().clone() for key, value in heads.items()}
            for i in range(len(weights))
        ]
        scores = [evaluate(c, validation) for c in candidates]
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
            perm = train[torch.randperm(len(train), device=device)]
            for start in range(0, len(perm), self.distill_batch_size):
                loss = objective(params, perm[start:start + self.distill_batch_size]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite policy-space projection loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            score = evaluate(params, validation)
            if np.isfinite(score) and score < best_score:
                best_score = score
                best_epoch = epoch + 1
                best = {key: value.detach().clone() for key, value in params.items()}

        self.last_projection_metrics = {
            "policy/projection_initial_val_ce": float(initial),
            "policy/projection_val_ce": float(best_score),
            "policy/projection_train_ce": float(evaluate(best, train)),
            "policy/projection_best_epoch": int(best_epoch),
            "policy/projection_rows": int(n),
            "policy/projection_components": int(len(weights)),
            "policy/projection_init_candidate": int(chosen),
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
