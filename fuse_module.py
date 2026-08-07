"""
Two genuinely different pool strategies, kept as two separate classes on
purpose -- they represent two different algorithms being compared
experimentally, not two variations of the same thing:

  SeparatePool  -- faithful reproduction of the ORIGINAL base CKA-RL merge:
                   every tensor (one weight matrix or one bias vector, for
                   one specific layer) keeps its own independent pool and
                   picks its own most-similar pair via cosine similarity,
                   completely independent of every other tensor. No buffers,
                   no distillation, no cross-tensor coordination -- this is
                   deliberate fidelity to the original design, verified
                   against the first commit's merge_vectors(), not a "fixed"
                   version of it.

  DistillPool   -- the supervised-distillation merge needs a batch of
                   (shared_features -> mean/log_std) pairs, which only makes
                   sense computed across the WHOLE head at once, and (since
                   all four sub-layers share one alpha) the merge decision
                   -- which two historical entries are most similar -- must
                   be made ONCE and applied identically everywhere. So this
                   class bundles all four sub-layers + each entry's own
                   distillation buffer together, by necessity, not by
                   convenience.

Both support fusion_mode ("classic_cka" stores each task's pure increment
v_k; "weight_delta" stores the full reconstructed offset theta_k - theta_base)
-- that's an orthogonal axis to which of these two classes is in use.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, init
from loguru import logger


# ==========================================================================
# SeparatePool -- one independent pool per tensor. The "base CKA-RL" method.
# ==========================================================================
class SeparatePool(nn.Module):
    def __init__(self, shape, fusion_mode="classic_cka", pool_size=9, init_bound=None):
        super().__init__()
        assert fusion_mode in ("classic_cka", "weight_delta"), fusion_mode
        self.shape = tuple(shape)
        self.fusion_mode = fusion_mode
        self.pool_size = pool_size

        self.register_buffer("base", torch.zeros(*self.shape))
        self.own = Parameter(torch.empty(*self.shape))
        if len(self.shape) == 2:
            init.kaiming_uniform_(self.own, a=math.sqrt(5))
        else:
            bound = init_bound if init_bound is not None else 1.0
            init.uniform_(self.own, -bound, bound)

        # frozen historical entries: plain list of Tensors, not
        # Parameters/buffers (variable length, never trained) -- see
        # _apply() for how .to(device) still reaches them.
        self.pool = []

        # attached from outside once its size is known (see cka_rl.py) --
        # merge() doesn't need alpha at all, only the forward pass does.
        self.alpha = None
        self.alpha_scale = None

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        self.pool = [fn(t) for t in self.pool]
        return self

    def historical_contribution(self):
        if not self.pool:
            return 0.0
        alphas = F.softmax(self.alpha * self.alpha_scale, dim=0)
        stacked = torch.stack(self.pool, dim=0)
        view_shape = (-1,) + (1,) * len(self.shape)
        return (alphas.view(*view_shape) * stacked).sum(dim=0)

    def effective(self):
        """theta_base + this task's own delta + alpha-weighted historical pool."""
        return self.base + self.own + self.historical_contribution()

    def load_base(self, base_pool: "SeparatePool"):
        """theta_base = the root task's own effective value (root's own base
        is always zero by construction, included anyway for correctness)."""
        self.base.copy_(base_pool.base + base_pool.own.data)

    def inherit_pool_from(self, latest_pool: "SeparatePool"):
        """Prepend latest_pool's own contribution as one new pool entry, onto
        whatever it itself already inherited (already merged if it needed to
        be). See fusion_mode docstring at module top for what "own
        contribution" means in each mode."""
        if self.fusion_mode == "weight_delta":
            new_entry = (latest_pool.own.data + latest_pool.historical_contribution()).clone()
        else:
            new_entry = latest_pool.own.data.clone()
        self.pool = [new_entry] + [t.clone() for t in latest_pool.pool]

    def needs_merge(self):
        return len(self.pool) > self.pool_size

    def merge(self):
        """Independent cosine-similarity merge for THIS tensor only -- no
        coordination with any other SeparatePool. This is deliberate: it's
        exactly how the original base CKA-RL merge worked."""
        if not self.needs_merge():
            return
        n = len(self.pool)
        flat = torch.stack([t.flatten() for t in self.pool], dim=0)
        similarities = torch.ones((n, n)) * -1
        for i in range(n):
            for j in range(i + 1, n):
                similarities[i, j] = F.cosine_similarity(flat[i], flat[j], dim=0)
        idx1, idx2 = divmod(torch.argmax(similarities).item(), n)
        logger.info(f"[SeparatePool:{self.fusion_mode}, shape={self.shape}] "
                    f"merging idx1={idx1}, idx2={idx2} (independent of any other tensor)")
        new_entry = (self.pool[idx1] + self.pool[idx2]) / 2
        keep = [self.pool[i] for i in range(n) if i not in (idx1, idx2)]
        keep.append(new_entry)
        self.pool = keep


class SeparatePoolHead(nn.Module):
    """Thin container: holds 8 SeparatePool instances (weight+bias for each of
    mean_l0/mean_l2/logstd_l0/logstd_l2) and exposes the SAME call surface as
    DistillPool (load_base/inherit_pool_from/merge/forward/pool-length), so
    CkaRlAgent doesn't need to know which pool strategy is active. This is
    organizational glue, not a second pool implementation -- SeparatePool
    above is still the one class doing the actual work, just instantiated 8
    times."""

    SUB_LAYERS = ("mean_l0", "mean_l2", "logstd_l0", "logstd_l2")

    def __init__(self, shared_dim, hidden_dim, act_dim, fusion_mode="classic_cka", pool_size=9):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.pool_size = pool_size
        self.act_dim = act_dim
        shapes = {
            "mean_l0":   (hidden_dim, shared_dim),
            "mean_l2":   (act_dim, hidden_dim),
            "logstd_l0": (hidden_dim, shared_dim),
            "logstd_l2": (act_dim, hidden_dim),
        }
        self.weight_pools = nn.ModuleDict()
        self.bias_pools = nn.ModuleDict()
        for name, shape in shapes.items():
            wp = SeparatePool(shape, fusion_mode=fusion_mode, pool_size=pool_size)
            fan_in = shape[1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            bp = SeparatePool((shape[0],), fusion_mode=fusion_mode, pool_size=pool_size, init_bound=bound)
            self.weight_pools[name] = wp
            self.bias_pools[name] = bp

    # ---- uniform interface, matching DistillPool ----
    def load_base(self, base_dir):
        base_head = torch.load(f"{base_dir}/head_pool.pt")
        for name in self.SUB_LAYERS:
            self.weight_pools[name].load_base(base_head.weight_pools[name])
            self.bias_pools[name].load_base(base_head.bias_pools[name])

    def inherit_pool_from(self, latest_head: "SeparatePoolHead"):
        for name in self.SUB_LAYERS:
            self.weight_pools[name].inherit_pool_from(latest_head.weight_pools[name])
            self.bias_pools[name].inherit_pool_from(latest_head.bias_pools[name])

    def merge(self):
        for name in self.SUB_LAYERS:
            self.weight_pools[name].merge()
            self.bias_pools[name].merge()

    def pool_length(self):
        """All 8 sub-pools end up the same length after a merge round (they
        all inherit the same count and all merge in the same round), so any
        one of them tells you the current size."""
        return len(self.weight_pools["mean_l0"].pool)

    def set_alpha(self, alpha, alpha_scale):
        for name in self.SUB_LAYERS:
            self.weight_pools[name].alpha = alpha
            self.weight_pools[name].alpha_scale = alpha_scale
            self.bias_pools[name].alpha = alpha
            self.bias_pools[name].alpha_scale = alpha_scale

    def set_own_buffer(self, buffer):
        pass  # SeparatePool never uses distillation buffers -- no-op for interface parity

    def forward(self, shared_features):
        w = self.weight_pools["mean_l0"].effective()
        b = self.bias_pools["mean_l0"].effective()
        h = F.relu(F.linear(shared_features, w, b))
        w = self.weight_pools["mean_l2"].effective()
        b = self.bias_pools["mean_l2"].effective()
        mean = F.linear(h, w, b)

        w = self.weight_pools["logstd_l0"].effective()
        b = self.bias_pools["logstd_l0"].effective()
        h = F.relu(F.linear(shared_features, w, b))
        w = self.weight_pools["logstd_l2"].effective()
        b = self.bias_pools["logstd_l2"].effective()
        log_std = F.linear(h, w, b)
        return mean, log_std


# ==========================================================================
# DistillPool -- bundled head-wide pool for the supervised-distillation merge.
# ==========================================================================
class DistillPool(nn.Module):
    SUB_LAYERS = ("mean_l0", "mean_l2", "logstd_l0", "logstd_l2")

    def __init__(self,
                 shared_dim: int,
                 hidden_dim: int,
                 act_dim: int,
                 bias: bool = True,
                 alpha: nn.Parameter = None,
                 alpha_scale: nn.Parameter = None,
                 fusion_mode: str = "classic_cka",
                 pool_size: int = 9,
                 max_distill_buffer: int = 50_000):
        super().__init__()
        assert fusion_mode in ("classic_cka", "weight_delta"), fusion_mode

        self.shapes = {
            "mean_l0":   (hidden_dim, shared_dim),
            "mean_l2":   (act_dim, hidden_dim),
            "logstd_l0": (hidden_dim, shared_dim),
            "logstd_l2": (act_dim, hidden_dim),
        }
        self._bias = bias
        self.act_dim = act_dim
        self.fusion_mode = fusion_mode
        self.pool_size = pool_size
        self.max_distill_buffer = max_distill_buffer

        self.alpha = alpha
        self.alpha_scale = alpha_scale

        for name, (out_f, in_f) in self.shapes.items():
            self.register_buffer(f"base_{name}_weight", torch.zeros(out_f, in_f))
            if bias:
                self.register_buffer(f"base_{name}_bias", torch.zeros(out_f))

        self.own_weight = nn.ParameterDict()
        self.own_bias = nn.ParameterDict()
        for name, (out_f, in_f) in self.shapes.items():
            w = Parameter(torch.empty(out_f, in_f))
            init.kaiming_uniform_(w, a=math.sqrt(5))
            self.own_weight[name] = w
            if bias:
                b = Parameter(torch.empty(out_f))
                fan_in, _ = init._calculate_fan_in_and_fan_out(w)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(b, -bound, bound)
                self.own_bias[name] = b

        # list of {"weights": {name: {"weight": T, "bias": T|None}, ...}, "buffer": dict|None}
        self.pool = []
        self.own_buffer = None

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        for entry in self.pool:
            for name in self.SUB_LAYERS:
                w = entry["weights"][name]
                w["weight"] = fn(w["weight"])
                if w["bias"] is not None:
                    w["bias"] = fn(w["bias"])
        return self

    def _historical_contribution(self, name):
        if not self.pool:
            return 0.0, 0.0
        alphas = F.softmax(self.alpha * self.alpha_scale, dim=0)
        stacked_w = torch.stack([e["weights"][name]["weight"] for e in self.pool], dim=0)
        hist_w = (alphas.view(-1, 1, 1) * stacked_w).sum(dim=0)
        hist_b = 0.0
        if self._bias:
            stacked_b = torch.stack([e["weights"][name]["bias"] for e in self.pool], dim=0)
            hist_b = (alphas.view(-1, 1) * stacked_b).sum(dim=0)
        return hist_w, hist_b

    def _effective(self, name):
        base_w = getattr(self, f"base_{name}_weight")
        hist_w, hist_b = self._historical_contribution(name)
        w = base_w + self.own_weight[name] + hist_w
        b = None
        if self._bias:
            base_b = getattr(self, f"base_{name}_bias")
            b = base_b + self.own_bias[name] + hist_b
        return w, b

    def pool_length(self):
        return len(self.pool)

    def set_alpha(self, alpha, alpha_scale):
        self.alpha = alpha
        self.alpha_scale = alpha_scale

    def forward(self, shared_features):
        w, b = self._effective("mean_l0")
        h = F.relu(F.linear(shared_features, w, b))
        w, b = self._effective("mean_l2")
        mean = F.linear(h, w, b)

        w, b = self._effective("logstd_l0")
        h = F.relu(F.linear(shared_features, w, b))
        w, b = self._effective("logstd_l2")
        log_std = F.linear(h, w, b)
        return mean, log_std

    def load_base(self, base_dir):
        base_pool = torch.load(f"{base_dir}/head_pool.pt")
        for name in self.SUB_LAYERS:
            w = base_pool.own_weight[name].data + getattr(base_pool, f"base_{name}_weight")
            getattr(self, f"base_{name}_weight").copy_(w)
            if self._bias:
                b = base_pool.own_bias[name].data + getattr(base_pool, f"base_{name}_bias")
                getattr(self, f"base_{name}_bias").copy_(b)

    def inherit_pool_from(self, latest_pool: "DistillPool"):
        new_weights = {}
        for name in self.SUB_LAYERS:
            own_w = latest_pool.own_weight[name].data
            own_b = latest_pool.own_bias[name].data if self._bias else None
            if self.fusion_mode == "weight_delta":
                hist_w, hist_b = latest_pool._historical_contribution(name)
                w = own_w + hist_w
                b = (own_b + hist_b) if self._bias else None
            else:
                w = own_w.clone()
                b = own_b.clone() if self._bias else None
            new_weights[name] = {"weight": w, "bias": b}

        inherited = [
            {"weights": {n: dict(v) for n, v in e["weights"].items()}, "buffer": e["buffer"]}
            for e in latest_pool.pool
        ]
        new_entry = {"weights": new_weights, "buffer": latest_pool.own_buffer}
        self.pool = [new_entry] + inherited

    def set_own_buffer(self, buffer):
        self.own_buffer = buffer

    def needs_merge(self):
        return len(self.pool) > self.pool_size

    def merge(self):
        if not self.needs_merge():
            return
        n = len(self.pool)
        flat_per_slot = torch.stack([
            torch.cat([
                torch.cat([self.pool[i]["weights"][name]["weight"].flatten(),
                           self.pool[i]["weights"][name]["bias"].flatten()])
                for name in self.SUB_LAYERS
            ])
            for i in range(n)
        ], dim=0)

        similarities = torch.ones((n, n)) * -1
        for i in range(n):
            for j in range(i + 1, n):
                similarities[i, j] = F.cosine_similarity(flat_per_slot[i], flat_per_slot[j], dim=0)
        idx1, idx2 = divmod(torch.argmax(similarities).item(), n)
        logger.info(f"[DistillPool:{self.fusion_mode}] merging pool slots idx1={idx1}, idx2={idx2} "
                    f"(one shared pair for all 4 sub-layers + buffers)")

        buf1 = self.pool[idx1]["buffer"]
        buf2 = self.pool[idx2]["buffer"]
        use_distillation = buf1 is not None and buf2 is not None
        if not use_distillation:
            logger.warning("[DistillPool] one or both merging entries have no distillation "
                            "buffer; falling back to simple averaging for this round.")

        distilled = None
        if use_distillation:
            distilled = {
                "mean": self._distill(buf1, buf2, "mean"),
                "logstd": self._distill(buf1, buf2, "logstd"),
            }

        merged_weights = {}
        for name in self.SUB_LAYERS:
            w1 = self.pool[idx1]["weights"][name]["weight"]
            w2 = self.pool[idx2]["weights"][name]["weight"]
            b1 = self.pool[idx1]["weights"][name]["bias"]
            b2 = self.pool[idx2]["weights"][name]["bias"]

            head = "mean" if name.startswith("mean") else "logstd"
            if distilled is not None and name.endswith("_l2"):
                new_w, new_b = distilled[head]
                new_w = new_w.to(w1.device)
                new_b = new_b.to(b1.device) if b1 is not None else None
            else:
                new_w = (w1 + w2) / 2
                new_b = (b1 + b2) / 2 if b1 is not None else None
            merged_weights[name] = {"weight": new_w, "bias": new_b}

        merged_buffer = self._merge_buffers(buf1, buf2) if use_distillation else (buf1 or buf2)

        keep = [self.pool[i] for i in range(n) if i not in (idx1, idx2)]
        keep.append({"weights": merged_weights, "buffer": merged_buffer})
        self.pool = keep

    def _distill(self, buffer1, buffer2, head_name, epochs=5, lr=1e-3, batch_size=128):
        logger.info(f"[DistillPool] distillation training for {head_name}_l2")
        inputs = np.concatenate([buffer1["shared"], buffer2["shared"]], axis=0)
        targets = np.concatenate([buffer1["targets"], buffer2["targets"]], axis=0)
        inputs_t = torch.tensor(inputs, dtype=torch.float32)
        targets_t = torch.tensor(targets, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(inputs_t, targets_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        student = nn.Sequential(
            nn.Linear(inputs_t.shape[-1], 128), nn.ReLU(), nn.Linear(128, self.act_dim)
        ).to(device)
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        student.train()
        for _ in range(epochs):
            for batch_in, batch_target in loader:
                batch_in, batch_target = batch_in.to(device), batch_target.to(device)
                final_target = batch_target[:, :self.act_dim] if head_name == "mean" else batch_target[:, self.act_dim:]
                preds = student(batch_in)
                loss = F.mse_loss(preds, final_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return student[2].weight.data.clone().cpu(), student[2].bias.data.clone().cpu()

    def _merge_buffers(self, buf1, buf2):
        merged = {k: np.concatenate([buf1[k], buf2[k]], axis=0) for k in ("obs", "shared", "targets")}
        n_rows = merged["obs"].shape[0]
        if n_rows > self.max_distill_buffer:
            keep_idx = np.random.choice(n_rows, size=self.max_distill_buffer, replace=False)
            merged = {k: v[keep_idx] for k, v in merged.items()}
        return merged
