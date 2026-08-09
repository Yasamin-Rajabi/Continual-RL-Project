"""
HeadPool: owns ONE 2-layer sub-network (hidden layer + output layer) for a
SINGLE output head (either "mean" or "logstd"), plus everything about its
continual-learning pool.

Why one HeadPool per output head, not one per single layer and not one
bundling both heads together:
  - l0 and l2 WITHIN one head are one coherent 2-layer sub-network -- they
    must merge together (one cosine-similarity decision, one alpha), or a
    merge could recombine l0 from one historical task with l2 from another,
    which never corresponds to any network that was ever actually trained.
  - mean and logstd, by contrast, share no weights with each other at all --
    only the same shared_features input. There's no structural reason to
    force their merge decisions (or their alpha) to move together, and doing
    so was itself the earlier design's problem (one alpha, but independent
    merge decisions per component -- see the original merge_vectors(), which
    called merge(mean_vectors) and merge(logstd_vectors) as two independent
    calls despite both reading the SAME shared self.alpha).

distillation is a plain bool here, not a separate class, because both
branches (simple average vs. supervised distillation) now operate at the
exact same granularity (one merge decision covering l0+l2 together) -- the
only difference is HOW the replacement entry gets computed.

fusion_mode ("classic_cka" stores each task's pure increment v_k;
"weight_delta" stores the full reconstructed offset theta_k - theta_base) is
an orthogonal axis, same as before.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
from loguru import logger

# The base/original method's fusion_mode -- alpha_mass is deliberately
# restricted away from this one specifically (see the assert in HeadPool
# below), not tied to distillation on/off.
BASE_FUSION_MODE = "classic_cka"


class HeadPool(nn.Module):
    def __init__(self,
                 head_type: str,
                 shared_dim: int,
                 hidden_dim: int,
                 act_dim: int,
                 fusion_mode: str = "classic_cka",
                 pool_size: int = 9,
                 distillation: bool = True,
                 max_distill_buffer: int = 50_000,
                 use_alpha_mass: bool = False,
                 distill_test_frac: float = 0.2):
        super().__init__()
        assert head_type in ("mean", "logstd"), head_type
        assert fusion_mode in ("classic_cka", "weight_delta"), fusion_mode
        # alpha_mass is deliberately restricted away from the BASE method: the
        # point is to let a NEW method's alpha distribution carry less (or
        # more) than 100% total mass; the base method should always match
        # the original paper's formula (alpha always sums to exactly 1), so
        # baseline comparisons stay a faithful reproduction. Independent of
        # distillation on/off -- fusion_mode is the only thing that matters.
        assert not (use_alpha_mass and fusion_mode == BASE_FUSION_MODE and distillation == False), (
            f"use_alpha_mass is not available with fusion_mode='{BASE_FUSION_MODE}' "
            "(the base method) -- not exposed to it on purpose."
        )

        self.head_type = head_type
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.pool_size = pool_size
        self.distillation = distillation
        self.max_distill_buffer = max_distill_buffer
        self.use_alpha_mass = use_alpha_mass
        self.distill_test_frac = distill_test_frac

        # theta_base (frozen), l0 and l2.
        self.register_buffer("base_l0_weight", torch.zeros(hidden_dim, shared_dim))
        self.register_buffer("base_l0_bias", torch.zeros(hidden_dim))
        self.register_buffer("base_l2_weight", torch.zeros(act_dim, hidden_dim))
        self.register_buffer("base_l2_bias", torch.zeros(act_dim))

        # this round's own trainable delta, l0 and l2.
        self.own_l0_weight = Parameter(torch.empty(hidden_dim, shared_dim))
        self.own_l0_bias = Parameter(torch.empty(hidden_dim))
        self.own_l2_weight = Parameter(torch.empty(act_dim, hidden_dim))
        self.own_l2_bias = Parameter(torch.empty(act_dim))
        init.kaiming_uniform_(self.own_l0_weight, a=math.sqrt(5))
        fan_in0, _ = init._calculate_fan_in_and_fan_out(self.own_l0_weight)
        bound0 = 1 / math.sqrt(fan_in0) if fan_in0 > 0 else 0
        init.uniform_(self.own_l0_bias, -bound0, bound0)
        init.kaiming_uniform_(self.own_l2_weight, a=math.sqrt(5))
        fan_in2, _ = init._calculate_fan_in_and_fan_out(self.own_l2_weight)
        bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
        init.uniform_(self.own_l2_bias, -bound2, bound2)

        # pool: list of {"l0_weight":T, "l0_bias":T, "l2_weight":T, "l2_bias":T,
        #                "buffer": {"obs","shared","targets"} | None}
        self.pool = []
        self.own_buffer = None  # this task's own freshly-collected buffer

        # attached from outside once its size is known -- merge() itself
        # doesn't need alpha, only the forward pass does.
        self.alpha = None
        self.alpha_scale = None
        self.alpha_mass = None  # only used when use_alpha_mass=True

        # populated by merge() whenever it actually used distillation (stays
        # None otherwise -- e.g. first task, or a round that fell back to
        # simple averaging). Read from outside via CkaRlAgent.get_distill_metrics().
        self.last_distill_train_mse = None
        self.last_distill_test_mse = None

    # ------------------------------------------------------------------
    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        for e in self.pool:
            for k in ("l0_weight", "l0_bias", "l2_weight", "l2_bias"):
                e[k] = fn(e[k])
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _historical(self):
        if not self.pool:
            return {"l0_weight": 0.0, "l0_bias": 0.0, "l2_weight": 0.0, "l2_bias": 0.0}
        
        weights = F.softmax(self.alpha * self.alpha_scale, dim=0)
        if self.use_alpha_mass and self.alpha_mass is not None:
            # decouples "how mass is distributed across old vectors" (the
            # softmax's relative proportions, unaffected) from "how much
            # total weight history gets overall" (this scalar -- normally
            # always exactly 1.0, since softmax sums to 1 by construction).
            weights = self.alpha_mass * weights
        out = {}
        for name, ndim in (("l0_weight", 2), ("l0_bias", 1), ("l2_weight", 2), ("l2_bias", 1)):
            stacked = torch.stack([e[name] for e in self.pool], dim=0)
            view_shape = (-1,) + (1,) * ndim
            out[name] = (weights.view(*view_shape) * stacked).sum(dim=0)
        return out

    def _effective(self):
        hist = self._historical()
        w0 = self.own_l0_weight + hist["l0_weight"]
        b0 = self.own_l0_bias + hist["l0_bias"]
        w2 = self.own_l2_weight + hist["l2_weight"]
        b2 = self.own_l2_bias + hist["l2_bias"]

        if self.fusion_mode == BASE_FUSION_MODE:
            w0 = self.base_l0_weight + w0
            b0 = self.base_l0_bias + b0
            w2 = self.base_l2_weight + w2
            b2 = self.base_l2_bias + b2
            
        return w0, b0, w2, b2

    def _base_only_forward(self, shared_features):
        """Runs shared_features through JUST the frozen base sub-network
        (ignoring own/hist). Used to compute a residual distillation target
        -- see merge()/_distill()."""
        h = F.relu(F.linear(shared_features, self.base_l0_weight, self.base_l0_bias))
        return F.linear(h, self.base_l2_weight, self.base_l2_bias)

    def forward(self, shared_features):
        w0, b0, w2, b2 = self._effective()
        h = F.relu(F.linear(shared_features, w0, b0))
        return F.linear(h, w2, b2)

    # ------------------------------------------------------------------
    # Loading from disk
    # ------------------------------------------------------------------
    # def load_base(self, base_dir):
    #     """theta_base = the root task's own effective (fully-formed) weight.
    #     After finalize_own_contribution(), the root's own contribution lives
    #     in pool[0] (own_weight itself is zeroed at that point), so that's
    #     where we read it from -- not own_weight directly."""
    #     base_pool = torch.load(f"{base_dir}/{self.head_type}_pool.pt")
    #     root_entry = base_pool.pool[0]
    #     self.base_l0_weight.copy_(base_pool.base_l0_weight + root_entry["l0_weight"])
    #     self.base_l0_bias.copy_(base_pool.base_l0_bias + root_entry["l0_bias"])
    #     self.base_l2_weight.copy_(base_pool.base_l2_weight + root_entry["l2_weight"])
    #     self.base_l2_bias.copy_(base_pool.base_l2_bias + root_entry["l2_bias"])

    def inherit_pool_from(self, latest_pool: "HeadPool"):
        """Copy latest_pool's pool directly. latest_pool already includes its
        own contribution as its first entry (folded in by its OWN
        finalize_own_contribution() call, already merged down to pool_size
        if it needed to be) -- there's nothing left to compute here anymore,
        just copy."""
        self.pool = [dict(e) for e in latest_pool.pool]

        self.base_l0_weight.copy_(latest_pool.base_l0_weight)
        self.base_l0_bias.copy_(latest_pool.base_l0_bias)
        self.base_l2_weight.copy_(latest_pool.base_l2_weight)
        self.base_l2_bias.copy_(latest_pool.base_l2_bias)


    def set_base(self):
        """Call this INSTEAD of finalize_own_contribution() for the very
        first task in a chain (base_dir is None AND latest_dir is None --
        there's no existing base or pool to load/inherit, this task IS the
        root). Makes this task's own just-trained weight into theta_base
        going forward, and seeds pool[0] according to fusion_mode:
          - classic_cka (BASE_FUSION_MODE): pool[0] = 0, matching the paper
            exactly (v_1 = theta_1 - theta_base = 0, included in V).
          - weight_delta: pool[0] = the same value as base (not zero) -- for
            this method, the root's own contribution stays revisable via
            alpha/merging like any other pool entry.
        """
        self.base_l0_weight.copy_(self.own_l0_weight.data)
        self.base_l0_bias.copy_(self.own_l0_bias.data)
        self.base_l2_weight.copy_(self.own_l2_weight.data)
        self.base_l2_bias.copy_(self.own_l2_bias.data)
 
        if self.fusion_mode == BASE_FUSION_MODE:
            entry = {
                "l0_weight": torch.zeros_like(self.own_l0_weight.data),
                "l0_bias": torch.zeros_like(self.own_l0_bias.data),
                "l2_weight": torch.zeros_like(self.own_l2_weight.data),
                "l2_bias": torch.zeros_like(self.own_l2_bias.data),
            }
        else:
            entry = {
                "l0_weight": self.own_l0_weight.data.clone(),
                "l0_bias": self.own_l0_bias.data.clone(),
                "l2_weight": self.own_l2_weight.data.clone(),
                "l2_bias": self.own_l2_bias.data.clone(),
            }
        entry["buffer"] = self.own_buffer
        self.pool = [entry]
 
        # own_weight's contribution is now captured in base (and, for
        # weight_delta, in pool[0] too) -- zero it so forward() doesn't
        # double/triple-count it if this object is used again.
        self.own_l0_weight.data.zero_()
        self.own_l0_bias.data.zero_()
        self.own_l2_weight.data.zero_()
        self.own_l2_bias.data.zero_()
 
        
    def finalize_own_contribution(self):
        """Call this ONCE, right after this task's training (and evaluation,
        and anything else) is completely finished -- right before save().
        Folds this task's own trainable delta into the pool as a new entry
        (per fusion_mode: classic_cka stores the pure increment; weight_delta
        stores the full reconstructed offset, own + whatever was already
        inherited), then zeros out own_weight/own_bias so they can never
        double-count if this same saved object is loaded again later (as a
        base_dir/latest_dir source, or directly via CkaRlAgent.load()), then
        merges if the pool now exceeds pool_size.
 
        This used to happen lazily, deferred to the NEXT task's construction
        (inside the old inherit_pool_from). Moved here so a task's own
        contribution and its merge are finalized immediately, as a normal
        part of finishing this task -- not postponed to whenever, if ever,
        another task happens to load this one."""
        #CHECK change if you want in delta_weight we start new task with initial weight = base weight
        if self.fusion_mode == "weight_delta":
            hist = self._historical()
            entry = {
                "l0_weight": (self.own_l0_weight.data + hist["l0_weight"]).clone(),
                "l0_bias": (self.own_l0_bias.data + hist["l0_bias"]).clone(),
                "l2_weight": (self.own_l2_weight.data + hist["l2_weight"]).clone(),
                "l2_bias": (self.own_l2_bias.data + hist["l2_bias"]).clone(),
            }
        else:
            entry = {
                "l0_weight": self.own_l0_weight.data.clone(),
                "l0_bias": self.own_l0_bias.data.clone(),
                "l2_weight": self.own_l2_weight.data.clone(),
                "l2_bias": self.own_l2_bias.data.clone(),
            }

        entry["buffer"] = self.own_buffer
        self.pool = [entry] + self.pool
 
        # own_weight's contribution is now captured in the pool entry above --
        # zero it so forward() stays correct if this object is ever used
        # again (it would otherwise double-count: once via own_weight, once
        # via the new pool entry).
        self.own_l0_weight.data.zero_()
        self.own_l0_bias.data.zero_()
        self.own_l2_weight.data.zero_()
        self.own_l2_bias.data.zero_()
 
        self.merge()


    def set_own_buffer(self, buffer):
        self.own_buffer = buffer

    def pool_length(self):
        return len(self.pool)

    def set_alpha(self, alpha, alpha_scale, alpha_mass=None):
        self.alpha = alpha
        self.alpha_scale = alpha_scale
        self.alpha_mass = alpha_mass

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------
    def needs_merge(self):
        return len(self.pool) > self.pool_size

    def merge(self):
        """Collapse the two most-similar pool entries into one -- ONE
        decision covering l0+l2 together (they're one coherent sub-network),
        completely independent of whatever the OTHER head (mean vs logstd)
        does."""
        if not self.needs_merge():
            return
        n = len(self.pool)
        flat = torch.stack([
            torch.cat([self.pool[i][k].flatten() for k in ("l0_weight", "l0_bias", "l2_weight", "l2_bias")])
            for i in range(n)
        ], dim=0)
        similarities = torch.ones((n, n)) * -1
        for i in range(n):
            for j in range(i + 1, n):
                similarities[i, j] = F.cosine_similarity(flat[i], flat[j], dim=0)
        idx1, idx2 = divmod(torch.argmax(similarities).item(), n)
        logger.info(f"[HeadPool:{self.head_type}:{self.fusion_mode}] "
                    f"merging idx1={idx1}, idx2={idx2} (l0+l2 together, independent of the other head)")

        buf1 = self.pool[idx1]["buffer"]
        buf2 = self.pool[idx2]["buffer"]
        use_distillation = self.distillation and buf1 is not None and buf2 is not None
        if self.distillation and not use_distillation:
            logger.warning(f"[HeadPool:{self.head_type}] distillation requested but one or both "
                            "merging entries have no buffer; falling back to simple averaging.")

        if use_distillation:
            new_l0w, new_l0b, new_l2w, new_l2b = self._distill(buf1, buf2)
        else:
            new_l0w = (self.pool[idx1]["l0_weight"] + self.pool[idx2]["l0_weight"]) / 2
            new_l0b = (self.pool[idx1]["l0_bias"] + self.pool[idx2]["l0_bias"]) / 2
            new_l2w = (self.pool[idx1]["l2_weight"] + self.pool[idx2]["l2_weight"]) / 2
            new_l2b = (self.pool[idx1]["l2_bias"] + self.pool[idx2]["l2_bias"]) / 2

        merged_buffer = self._merge_buffers(buf1, buf2) if use_distillation else (buf1 or buf2)
        entry = {"l0_weight": new_l0w, "l0_bias": new_l0b, "l2_weight": new_l2w, "l2_bias": new_l2b,
                 "buffer": merged_buffer}
        keep = [self.pool[i] for i in range(n) if i not in (idx1, idx2)]
        keep.append(entry)
        self.pool = keep

    def _distill(self, buffer1, buffer2, epochs=5, lr=1e-3, batch_size=128):
        """Directly optimizes a candidate pool entry (v_l0_w, v_l0_b, v_l2_w,
        v_l2_b) through the SAME composition _effective() actually uses --
        base + candidate (classic_cka) or just candidate (weight_delta),
        through the real ReLU -- instead of training a separate scaffold
        network and subtracting a separately-computed base output. That
        subtraction doesn't hold once there's a ReLU between two linear layers
        (ReLU(a)+ReLU(c) != ReLU(a+c) in general), so this optimizes the
        candidate exactly the way it will actually be used -- no approximation
        from that source.

        A different, unavoidable approximation remains: combining multiple
        tasks' contributions in WEIGHT space (before the ReLU) still isn't the
        same as combining their OUTPUT behavior -- that's inherent to
        weight-space pooling with a nonlinear network, not something any one
        merge step can fully remove.
        """
        logger.info(f"[HeadPool:{self.head_type}] distillation training (direct, through real composition)")
        inputs = np.concatenate([buffer1["shared"], buffer2["shared"]], axis=0)
        raw_targets = np.concatenate([buffer1["targets"], buffer2["targets"]], axis=0)
        target_slice = slice(0, self.act_dim) if self.head_type == "mean" else slice(self.act_dim, 2 * self.act_dim)
        head_targets = raw_targets[:, target_slice]

        n = inputs.shape[0]
        perm = np.random.permutation(n)
        n_test = int(n * self.distill_test_frac) if self.distill_test_frac > 0 else 0
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        if len(train_idx) == 0:
            train_idx, test_idx = perm, np.array([], dtype=int)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs_t = torch.tensor(inputs, dtype=torch.float32).to(device)
        targets_t = torch.tensor(head_targets, dtype=torch.float32).to(device)

        base_l0_w = self.base_l0_weight.detach().to(device)
        base_l0_b = self.base_l0_bias.detach().to(device)
        base_l2_w = self.base_l2_weight.detach().to(device)
        base_l2_b = self.base_l2_bias.detach().to(device)

        v_l0_w = torch.zeros_like(base_l0_w, requires_grad=True)
        v_l0_b = torch.zeros_like(base_l0_b, requires_grad=True)
        v_l2_w = torch.zeros_like(base_l2_w, requires_grad=True)
        v_l2_b = torch.zeros_like(base_l2_b, requires_grad=True)

        def compute(batch_in):
            if self.fusion_mode == BASE_FUSION_MODE:
                eff_l0_w, eff_l0_b = base_l0_w + v_l0_w, base_l0_b + v_l0_b
                eff_l2_w, eff_l2_b = base_l2_w + v_l2_w, base_l2_b + v_l2_b
            else:
                eff_l0_w, eff_l0_b = v_l0_w, v_l0_b
                eff_l2_w, eff_l2_b = v_l2_w, v_l2_b
            h = F.relu(F.linear(batch_in, eff_l0_w, eff_l0_b))
            return F.linear(h, eff_l2_w, eff_l2_b)

        optimizer = torch.optim.Adam([v_l0_w, v_l0_b, v_l2_w, v_l2_b], lr=lr)
        train_inputs, train_targets = inputs_t[train_idx], targets_t[train_idx]
        n_train = train_inputs.shape[0]
        for _ in range(epochs):
            perm_epoch = torch.randperm(n_train)
            for start in range(0, n_train, batch_size):
                idx = perm_epoch[start:start + batch_size]
                preds = compute(train_inputs[idx])
                loss = F.mse_loss(preds, train_targets[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            preds_train = compute(train_inputs)
            self.last_distill_train_mse = F.mse_loss(preds_train, train_targets).item()
            if len(test_idx) > 0:
                preds_test = compute(inputs_t[test_idx])
                self.last_distill_test_mse = F.mse_loss(preds_test, targets_t[test_idx]).item()
            else:
                self.last_distill_test_mse = None
        logger.info(f"[HeadPool:{self.head_type}] distill train_mse={self.last_distill_train_mse:.5f} "
                    f"test_mse={self.last_distill_test_mse}")

        out_device = self.base_l0_weight.device
        return (v_l0_w.detach().clone().to(out_device), v_l0_b.detach().clone().to(out_device),
                v_l2_w.detach().clone().to(out_device), v_l2_b.detach().clone().to(out_device))

    def _merge_buffers(self, buf1, buf2):
        merged = {k: np.concatenate([buf1[k], buf2[k]], axis=0) for k in ("obs", "shared", "targets")}
        n_rows = merged["obs"].shape[0]
        if n_rows > self.max_distill_buffer:
            keep_idx = np.random.choice(n_rows, size=self.max_distill_buffer, replace=False)
            merged = {k: v[keep_idx] for k, v in merged.items()}
        return merged
