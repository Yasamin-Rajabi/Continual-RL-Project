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
from loguru import logger

from fuse_module import HeadPool
from shared_arch import shared


class CkaRlAgent(nn.Module):
    """
    Wraps a shared encoder (`self.fc`) + two independent HeadPool instances
    (`self.mean_pool`, `self.logstd_pool`). mean and logstd share no weights
    with each other, so they merge independently and each gets its OWN
    alpha -- unlike l0/l2 WITHIN one head, which must move together (see
    fuse_module.py's module docstring for the reasoning).
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
                 encoder_from_base=False,
                 distillation=True,
                 fusion_mode="classic_cka",
                 max_distill_buffer=50_000,
                 use_alpha_mass=False,
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
        latest_mean_pool = None
        latest_logstd_pool = None
        if latest_dir is not None and latest_dir != base_dir:
            latest_mean_pool = torch.load(f"{latest_dir}/mean_pool.pt")
            latest_logstd_pool = torch.load(f"{latest_dir}/logstd_pool.pt")

        self.mean_pool = HeadPool("mean", shared_dim, hidden_dim, act_dim,
                                   fusion_mode=fusion_mode, pool_size=pool_size,
                                   distillation=distillation, max_distill_buffer=max_distill_buffer,
                                   use_alpha_mass=use_alpha_mass, distill_test_frac=distill_test_frac)
        self.logstd_pool = HeadPool("logstd", shared_dim, hidden_dim, act_dim,
                                     fusion_mode=fusion_mode, pool_size=pool_size,
                                     distillation=distillation, max_distill_buffer=max_distill_buffer,
                                     use_alpha_mass=use_alpha_mass, distill_test_frac=distill_test_frac)

        if base_dir is not None:
            self.mean_pool.load_base(base_dir)
            self.logstd_pool.load_base(base_dir)
        if latest_mean_pool is not None:
            self.mean_pool.inherit_pool_from(latest_mean_pool)
            self.mean_pool.merge()
        if latest_logstd_pool is not None:
            self.logstd_pool.inherit_pool_from(latest_logstd_pool)
            self.logstd_pool.merge()

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
        alpha_mass = nn.Parameter(torch.ones(1), requires_grad=(use_alpha_mass and not fix_alpha)) if use_alpha_mass else None
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
