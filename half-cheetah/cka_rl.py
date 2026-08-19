# ==========================================
# TORCH SECURITY PATCH FOR KAGGLE COMPATIBILITY
# ==========================================
import torch
orig_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)
torch.load = patched_load
# ==========================================

import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from knowledge_pools import HeadPool
from shared_arch import shared


class CkaRlAgent(nn.Module):
    """
    Wraps a shared encoder (`self.fc`) + two independent HeadPool instances
    (`self.mean_pool`, `self.logstd_pool`). mean and logstd share no weights
    with each other, so they merge independently and each gets its OWN
    alpha -- unlike l0/l2 WITHIN one head, which must move together (see
    knowledge_pools.py's module docstring for the reasoning).
    """

    def __init__(self,
                 obs_dim,
                 act_dim,
                 base_dir,
                 latest_dir,
                 pool_size=9,
                 alpha_init="Randn",
                 alpha_major=0.6,
                 alpha_factor=1e-3,
                 fix_alpha=False,
                 use_alpha_scale=True,
                 use_alpha_mass=True,
                 encoder_from_base=False,
                 distillation=True,
                 fusion_mode="classic_cka",
                 max_distill_buffer=50_000,
                 distill_test_frac=0.2,
                 hidden_dim=128,
                 shared_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pool_size = pool_size
        self.distillation = distillation
        self.fusion_mode = fusion_mode
        self.max_distill_buffer = max_distill_buffer
        self.use_alpha_mass = use_alpha_mass

        # 1. Load the previous task's pools (if any) FIRST -- same reasoning
        #    as before: merge() doesn't need alpha, and alpha's size depends
        #    on the pool's size AFTER merging.
        #
        #    Only inherit when latest_dir is a genuine intermediate task
        #    beyond the root (latest_dir != base_dir) -- the root task IS
        #    theta_base (loaded separately via load_base below); it must not
        #    also become a pool entry.

        self.mean_pool = HeadPool("mean", shared_dim, hidden_dim, act_dim,
                                           fusion_mode=fusion_mode, pool_size=pool_size,
                                           distillation=distillation, max_distill_buffer=max_distill_buffer,
                                           use_alpha_mass=use_alpha_mass, distill_test_frac=distill_test_frac)
        self.logstd_pool = HeadPool("logstd", shared_dim, hidden_dim, act_dim,
                                            fusion_mode=fusion_mode, pool_size=pool_size,
                                            distillation=distillation, max_distill_buffer=max_distill_buffer,
                                            use_alpha_mass=use_alpha_mass, distill_test_frac=distill_test_frac)
        
        latest_mean_pool = None
        latest_logstd_pool = None

        if latest_dir is not None:
            latest_mean_pool = torch.load(f"{latest_dir}/mean_pool.pt")
            latest_logstd_pool = torch.load(f"{latest_dir}/logstd_pool.pt")
            assert (latest_mean_pool is not None) and (latest_logstd_pool is not None), (
                    f"latest_dir is not None and latest_mean_pool or latest_logstd_pool is None."
                )
            
            self.mean_pool.inherit_pool_from(latest_mean_pool)
            self.logstd_pool.inherit_pool_from(latest_logstd_pool)

            # CKA-RL initializes the NEW task vector v_k at zero.  The HeadPool
            # constructor uses normal neural-network initialization so the root
            # task can train from scratch; every non-root task must override
            # that initialization after history has been inherited.
            self.mean_pool.reset_own_to_zero()
            self.logstd_pool.reset_own_to_zero()
        

        # 2. Now each pool's final size is known -- set up its OWN alpha
        #    (independent of the other head's), plus its own alpha_mass if
        #    requested.
        self.mean_alpha, self.mean_alpha_scale, self.mean_alpha_mass = self._make_alpha(
            self.mean_pool.pool_length(), fix_alpha, alpha_init, alpha_major, alpha_factor,
            use_alpha_scale, use_alpha_mass)
        self.logstd_alpha, self.logstd_alpha_scale, self.logstd_alpha_mass = self._make_alpha(
            self.logstd_pool.pool_length(), fix_alpha, alpha_init, alpha_major, alpha_factor,
            use_alpha_scale, use_alpha_mass)
        self.mean_pool.set_alpha(self.mean_alpha, self.mean_alpha_scale, self.mean_alpha_mass)
        self.logstd_pool.set_alpha(self.logstd_alpha, self.logstd_alpha_scale, self.logstd_alpha_mass)
        logger.info(f"mean alpha: {self.mean_alpha}")
        logger.info(f"logstd alpha: {self.logstd_alpha}")
        if use_alpha_mass:
            logger.info(f"mean alpha_mass: {self.mean_alpha_mass}")
            logger.info(f"logstd alpha_mass: {self.logstd_alpha_mass}")

        # 3. Shared encoder.
        if encoder_from_base and base_dir is not None:
            logger.info(f"Loading encoder from {base_dir}")
            self.fc = torch.load(f"{base_dir}/fc.pt")
        elif latest_dir is not None:
            logger.info(f"Loading latest shared from {latest_dir}")
            self.fc = torch.load(f"{latest_dir}/fc.pt")
        else:
            logger.info("Train shared from scratch")
            self.fc = shared(input_dim=obs_dim)

    def _make_alpha(self, num_vectors, fix_alpha, alpha_init, alpha_major, alpha_factor,
                     use_alpha_scale, use_alpha_mass):
        if num_vectors <= 0:
            return None, None, None
        if fix_alpha:
            alpha = nn.Parameter(torch.zeros(num_vectors), requires_grad=False)
        else:
            if alpha_init == "Uniform" or num_vectors == 1:
                alpha = nn.Parameter(torch.ones(num_vectors) * alpha_factor, requires_grad=True)
            elif alpha_init == "Randn":
                alpha = nn.Parameter(torch.randn(num_vectors) / num_vectors, requires_grad=True)
            elif alpha_init == "Major" and num_vectors > 1:
                a = [np.log((1 - alpha_major) / (num_vectors - 1)) for _ in range(num_vectors - 1)]
                a.append(np.log(alpha_major))
                alpha = nn.Parameter(torch.tensor(a, dtype=torch.float), requires_grad=True)
            else:
                raise NotImplementedError(f"unknown alpha_init: {alpha_init}")
        alpha_scale = nn.Parameter(torch.ones(1), requires_grad=(use_alpha_scale and not fix_alpha))
        alpha_mass = nn.Parameter(torch.ones(1), requires_grad=(use_alpha_mass and not fix_alpha))
        return alpha, alpha_scale, alpha_mass

    def get_distill_metrics(self):
        """Distillation train/test MSE from the most recent merge (if any
        happened during this construction). None for a metric that never had
        a merge use distillation (e.g. first task, or a round that fell back
        to simple averaging)."""
        return {
            "mean/distill_train_mse": self.mean_pool.last_distill_train_mse,
            "mean/distill_test_mse": self.mean_pool.last_distill_test_mse,
            "logstd/distill_train_mse": self.logstd_pool.last_distill_train_mse,
            "logstd/distill_test_mse": self.logstd_pool.last_distill_test_mse,
        }

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_pool(x)
        log_std = self.logstd_pool(x)
        return mean, log_std

    def set_own_buffer(self, buffer):
        """Attach this task's own freshly-collected distillation buffer to
        BOTH heads (each slices out its own half of `targets`). Call before
        save()."""
        self.mean_pool.set_own_buffer(buffer)
        self.logstd_pool.set_own_buffer(buffer)

    def set_base(self):
        """Call this INSTEAD of finalize() for the very first task in a
        chain (base_dir is None AND latest_dir is None). See
        HeadPool.set_base() in knowledge_pools.py for what it actually does."""
        self.mean_pool.set_base()
        self.logstd_pool.set_base()

    def finalize(self):
        """Call this ONCE, after training (and evaluation, and everything
        else) is completely finished for this task -- right before save().
        Folds this task's own contribution into each head's pool and merges
        if needed. Order matters: call set_own_buffer() first if you have a
        distillation buffer to attach, then finalize(), then save()."""
        # Pass the CURRENT encoder into merging/distillation so historical raw
        # observations are re-encoded in the feature space used at inference.
        self.mean_pool.finalize_own_contribution(feature_fn=self.fc)
        self.logstd_pool.finalize_own_contribution(feature_fn=self.fc)

    @staticmethod
    def _cpu_clone_dict(d):
        return {k: v.detach().cpu().clone() for k, v in d.items()}

    def export_effective_policy(self):
        """Return the exact PRE-finalize actor used for this task.

        The finalized knowledge pool is a data structure for FUTURE tasks and
        does not, in general, preserve the just-trained alpha/own composition.
        This compact snapshot stores only the current encoder and the two
        effective 2-layer heads, with no replay/distillation buffers.
        """
        with torch.no_grad():
            mean_w0, mean_b0, mean_w2, mean_b2 = self.mean_pool._effective()
            log_w0, log_b0, log_w2, log_b2 = self.logstd_pool._effective()
            return {
                "obs_dim": int(self.obs_dim),
                "act_dim": int(self.act_dim),
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

    def get_merge_info(self):
        return {
            "mean": self.mean_pool.last_merge_info,
            "logstd": self.logstd_pool.last_merge_info,
        }

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.fc, f"{dirname}/fc.pt")
        torch.save(self.mean_pool, f"{dirname}/mean_pool.pt")
        torch.save(self.logstd_pool, f"{dirname}/logstd_pool.pt")

    @staticmethod
    def load(dirname, obs_dim, act_dim, map_location=None):
        model = CkaRlAgent(obs_dim, act_dim, None, None)
        model.fc = torch.load(f"{dirname}/fc.pt", map_location=map_location)
        model.mean_pool = torch.load(f"{dirname}/mean_pool.pt", map_location=map_location)
        model.logstd_pool = torch.load(f"{dirname}/logstd_pool.pt", map_location=map_location)
        return model


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
        snapshot = torch.load(f"{dirname}/policy_snapshot.pt", map_location=map_location)
        return FrozenCkaPolicy(snapshot)
