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


class HeadPool(nn.Module):
    def __init__(self,
                 head_type: str,
                 shared_dim: int,
                 hidden_dim: int,
                 act_dim: int,
                 fusion_mode: str = "classic_cka",
                 pool_size: int = 9,
                 distillation: bool = True,
                 max_distill_buffer: int = 50_000):
        super().__init__()
        assert head_type in ("mean", "logstd"), head_type
        assert fusion_mode in ("classic_cka", "weight_delta"), fusion_mode

        self.head_type = head_type
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode
        self.pool_size = pool_size
        self.distillation = distillation
        self.max_distill_buffer = max_distill_buffer

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
        alphas = F.softmax(self.alpha * self.alpha_scale, dim=0)
        out = {}
        for name, ndim in (("l0_weight", 2), ("l0_bias", 1), ("l2_weight", 2), ("l2_bias", 1)):
            stacked = torch.stack([e[name] for e in self.pool], dim=0)
            view_shape = (-1,) + (1,) * ndim
            out[name] = (alphas.view(*view_shape) * stacked).sum(dim=0)
        return out

    def _effective(self):
        hist = self._historical()
        w0 = self.base_l0_weight + self.own_l0_weight + hist["l0_weight"]
        b0 = self.base_l0_bias + self.own_l0_bias + hist["l0_bias"]
        w2 = self.base_l2_weight + self.own_l2_weight + hist["l2_weight"]
        b2 = self.base_l2_bias + self.own_l2_bias + hist["l2_bias"]
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
    def load_base(self, base_dir):
        """theta_base = the root task's own effective (fully-formed) weight.
        The root task always has zero base of its own, included anyway for
        correctness."""
        base_pool = torch.load(f"{base_dir}/{self.head_type}_pool.pt")
        self.base_l0_weight.copy_(base_pool.base_l0_weight + base_pool.own_l0_weight.data)
        self.base_l0_bias.copy_(base_pool.base_l0_bias + base_pool.own_l0_bias.data)
        self.base_l2_weight.copy_(base_pool.base_l2_weight + base_pool.own_l2_weight.data)
        self.base_l2_bias.copy_(base_pool.base_l2_bias + base_pool.own_l2_bias.data)

    def inherit_pool_from(self, latest_pool: "HeadPool"):
        """Prepend latest_pool's own contribution as one new pool entry, onto
        whatever it itself already inherited (already merged if it needed to
        be). classic_cka stores the pure increment; weight_delta stores the
        full reconstructed offset (own + whatever it itself inherited)."""
        if self.fusion_mode == "weight_delta":
            hist = latest_pool._historical()
            entry = {
                "l0_weight": (latest_pool.own_l0_weight.data + hist["l0_weight"]).clone(),
                "l0_bias": (latest_pool.own_l0_bias.data + hist["l0_bias"]).clone(),
                "l2_weight": (latest_pool.own_l2_weight.data + hist["l2_weight"]).clone(),
                "l2_bias": (latest_pool.own_l2_bias.data + hist["l2_bias"]).clone(),
            }
        else:
            entry = {
                "l0_weight": latest_pool.own_l0_weight.data.clone(),
                "l0_bias": latest_pool.own_l0_bias.data.clone(),
                "l2_weight": latest_pool.own_l2_weight.data.clone(),
                "l2_bias": latest_pool.own_l2_bias.data.clone(),
            }
        entry["buffer"] = latest_pool.own_buffer
        inherited = [dict(e) for e in latest_pool.pool]
        self.pool = [entry] + inherited

    def set_own_buffer(self, buffer):
        self.own_buffer = buffer

    def pool_length(self):
        return len(self.pool)

    def set_alpha(self, alpha, alpha_scale):
        self.alpha = alpha
        self.alpha_scale = alpha_scale

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
        """Trains a fresh 2-layer student (same shape as l0+l2: shared_dim ->
        hidden_dim -> act_dim) on the RESIDUAL target -- targets minus what
        the frozen base sub-network ALONE would already produce -- so the
        student's weights directly represent the quantity that gets ADDED to
        base (v_k for classic_cka, theta_k - theta_base for weight_delta),
        not the raw absolute output. This is exact for weight_delta (the
        additive term this replaces IS exactly "output - base_output" in
        composition); it's an approximation for classic_cka (doesn't isolate
        out what other pool entries' hist would have contributed at the time
        each buffer was recorded) and, for both modes, an approximation
        across the ReLU nonlinearity between l0 and l2.
        """
        logger.info(f"[HeadPool:{self.head_type}] distillation training (residual vs. base)")
        inputs = np.concatenate([buffer1["shared"], buffer2["shared"]], axis=0)
        raw_targets = np.concatenate([buffer1["targets"], buffer2["targets"]], axis=0)
        target_slice = slice(0, self.act_dim) if self.head_type == "mean" else slice(self.act_dim, 2 * self.act_dim)
        head_targets = raw_targets[:, target_slice]

        with torch.no_grad():
            inputs_for_base = torch.tensor(inputs, dtype=torch.float32, device=self.base_l0_weight.device)
            base_out = self._base_only_forward(inputs_for_base).cpu().numpy()
        residual = head_targets - base_out

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs_t = torch.tensor(inputs, dtype=torch.float32)
        residual_t = torch.tensor(residual, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(inputs_t, residual_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        student = nn.Sequential(
            nn.Linear(inputs_t.shape[-1], self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.act_dim)
        ).to(device)
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        student.train()
        for _ in range(epochs):
            for batch_in, batch_target in loader:
                batch_in, batch_target = batch_in.to(device), batch_target.to(device)
                preds = student(batch_in)
                loss = F.mse_loss(preds, batch_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        out_device = self.base_l0_weight.device
        l0w = student[0].weight.data.clone().to(out_device)
        l0b = student[0].bias.data.clone().to(out_device)
        l2w = student[2].weight.data.clone().to(out_device)
        l2b = student[2].bias.data.clone().to(out_device)
        return l0w, l0b, l2w, l2b

    def _merge_buffers(self, buf1, buf2):
        merged = {k: np.concatenate([buf1[k], buf2[k]], axis=0) for k in ("obs", "shared", "targets")}
        n_rows = merged["obs"].shape[0]
        if n_rows > self.max_distill_buffer:
            keep_idx = np.random.choice(n_rows, size=self.max_distill_buffer, replace=False)
            merged = {k: v[keep_idx] for k, v in merged.items()}
        return merged
