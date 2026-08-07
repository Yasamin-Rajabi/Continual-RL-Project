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

from fuse_module import SeparatePoolHead, DistillPool
from shared_arch import shared


class CkaRlAgent(nn.Module):
    """
    Wraps a shared encoder (`self.fc`) + a pool (`self.pool`) that owns the
    mean/log_std heads and all continual-learning merge logic.

    Which pool CLASS gets used depends on `distillation`:
      - distillation=False -> SeparatePoolHead: faithful reproduction of the
        original base CKA-RL merge (every tensor merges independently).
      - distillation=True  -> DistillPool: bundled head-wide pool needed for
        the supervised-distillation merge.
    These are two different algorithms, not two configurations of one -- see
    fuse_module.py's module docstring for why.
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
                 hidden_dim=128,
                 shared_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pool_size = pool_size
        self.distillation = distillation
        self.fusion_mode = fusion_mode
        self.max_distill_buffer = max_distill_buffer

        # 1. Load the previous task's pool (if any) FIRST, before creating
        #    alpha or our own pool -- we need to know the final (possibly
        #    already-merged) pool size before alpha can be sized, and
        #    merge() itself doesn't need alpha at all.
        #
        #    Only inherit when latest_dir is a genuine intermediate task
        #    beyond the root (latest_dir != base_dir). When they're equal --
        #    true for the very first continual task, built directly on the
        #    root -- the root task itself already IS theta_base (loaded
        #    separately via load_base() below); it must NOT also become a
        #    pool entry, or its weight gets counted twice.
        latest_pool = None
        if latest_dir is not None and latest_dir != base_dir:
            latest_pool = torch.load(f"{latest_dir}/head_pool.pt")

        if distillation:
            self.pool = DistillPool(
                shared_dim=shared_dim, hidden_dim=hidden_dim, act_dim=act_dim,
                alpha=None, alpha_scale=None,
                fusion_mode=fusion_mode, pool_size=pool_size,
                max_distill_buffer=max_distill_buffer,
            )
        else:
            self.pool = SeparatePoolHead(
                shared_dim=shared_dim, hidden_dim=hidden_dim, act_dim=act_dim,
                fusion_mode=fusion_mode, pool_size=pool_size,
            )

        if base_dir is not None:
            self.pool.load_base(base_dir)
        if latest_pool is not None:
            self.pool.inherit_pool_from(latest_pool)
            self.pool.merge()

        # 2. Now the pool's final size is known -- set up alpha to match, and
        #    attach it to the pool.
        num_vectors = self.pool.pool_length()
        self.setup_alpha(num_vectors, fix_alpha, alpha_init, alpha_major, alpha_factor, use_alpha_scale)
        self.pool.set_alpha(self.alpha, self.alpha_scale)
        self.log_alpha()

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

    def setup_alpha(self, num_vectors, fix_alpha, alpha_init, alpha_major, alpha_factor, use_alpha_scale):
        if num_vectors > 0:
            if fix_alpha:
                self.alpha = nn.Parameter(torch.zeros(num_vectors), requires_grad=False)
                logger.info("Fix alpha to all 0")
            else:
                logger.info(f"alpha_init, {alpha_init}")
                logger.info(f"alpha_major, {alpha_major}")
                if alpha_init == "Uniform" or num_vectors == 1:
                    self.alpha = nn.Parameter(torch.ones(num_vectors) * alpha_factor, requires_grad=True)
                elif alpha_init == "Randn":
                    self.alpha = nn.Parameter(torch.randn(num_vectors) / num_vectors, requires_grad=True)
                elif alpha_init == "Major" and num_vectors > 1:
                    alpha = [np.log((1 - alpha_major) / (num_vectors - 1)) for _ in range(num_vectors - 1)]
                    alpha.append(np.log(alpha_major))
                    self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float), requires_grad=True)
                elif alpha_init not in ["Uniform", "Randn", "Major"]:
                    raise NotImplementedError
                logger.info("Train alpha")
            self.alpha_scale = nn.Parameter(torch.ones(1), requires_grad=(use_alpha_scale and not fix_alpha))
        else:
            self.alpha = None
            self.alpha_scale = None

    def log_alpha(self):
        logger.info(self.alpha)

    def forward(self, x):
        x = self.fc(x)
        return self.pool(x)

    def set_own_buffer(self, buffer):
        """Attach this task's own freshly-collected distillation buffer. No-op
        when the active pool is a SeparatePoolHead (which never uses
        buffers). Call before save()."""
        self.pool.set_own_buffer(buffer)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.fc, f"{dirname}/fc.pt")
        torch.save(self.pool, f"{dirname}/head_pool.pt")

    @staticmethod
    def load(dirname, obs_dim, act_dim, map_location=None):
        model = CkaRlAgent(obs_dim, act_dim, None, None)
        model.fc = torch.load(f"{dirname}/fc.pt", map_location=map_location)
        model.pool = torch.load(f"{dirname}/head_pool.pt", map_location=map_location)
        return model
