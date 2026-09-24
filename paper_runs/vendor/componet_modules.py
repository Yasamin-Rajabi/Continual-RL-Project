"""CompoNet: attention-based composition of frozen previous policies.

Ported from the project's Atari implementation (Malagon et al.).  The module
is architecture agnostic -- it operates on output vectors -- so the only change
for continuous control is that ``out_dim`` is ``2 * act_dim`` (mean and
log-std) and the outputs are raw vectors rather than probabilities.

Two attention heads:

* the *output* head attends over the previous modules' outputs and returns a
  convex combination of them;
* the *input* head attends over those outputs plus the output head's result,
  and feeds the summary to an internal policy together with the state.

The final output is (output head) + (internal policy), so the module can copy
a previous policy, correct it, or ignore it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def get_position_encoding(seq_len: int, d: int, n: float = 10_000.0) -> np.ndarray:
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P


class Identity:
    """Picklable stand-in for ``lambda x: x`` (torch.save cannot pickle lambdas)."""

    def __call__(self, arg, **kwargs):
        return arg


class FirstModuleWrapper(nn.Module):
    """Wraps the task-0 policy so it presents the CompoNet module interface."""

    def __init__(self, model: nn.Module, encoder: nn.Module = None):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.is_prev = False

    def forward(self, x, ret_encoder_out: bool = False):
        h = x if self.encoder is None else self.encoder(x)
        out = self.model(h)
        phi = out[:, None, :]
        if self.is_prev:
            return phi, x
        if ret_encoder_out:
            return out, phi, h
        return out, phi


class CompoNet(nn.Module):
    def __init__(
        self,
        previous_units,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        internal_policy: nn.Module,
        encoder: nn.Module = None,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.internal_policy = internal_policy
        self.encoder = encoder
        self.att_temp = float(np.sqrt(hidden_dim))
        self.is_prev = False

        self.headout_wq = nn.Linear(input_dim, hidden_dim, bias=proj_bias)
        self.headout_wk = nn.Linear(out_dim, hidden_dim, bias=proj_bias)

        self.headin_wq = nn.Linear(input_dim, hidden_dim, bias=proj_bias)
        self.headin_wk = nn.Linear(out_dim, hidden_dim, bias=proj_bias)
        self.headin_wv = nn.Linear(out_dim, hidden_dim, bias=proj_bias)

        n_prev = len(previous_units)
        pe1 = torch.tensor(
            get_position_encoding(seq_len=n_prev + 1, d=out_dim), dtype=torch.float32
        )
        self.register_buffer("pe1", pe1[None, :, :])
        if n_prev >= 2:
            self.register_buffer("pe0", pe1[None, :-1, :])
        else:
            self.pe0 = None

        for unit in previous_units:
            if hasattr(unit, "previous_units"):
                del unit.previous_units
            unit.is_prev = True
            unit.eval()
            for param in unit.parameters():
                param.requires_grad = False
        self.previous_units = nn.Sequential(*previous_units)

    def _forward_headout(self, s, phi):
        query = self.headout_wq(s)
        keys = self.headout_wk(phi + self.pe0 if self.pe0 is not None else phi)
        w = torch.matmul(query[:, None, :], keys.permute(0, 2, 1))
        att = torch.softmax(w / self.att_temp, dim=-1)
        return torch.matmul(att, phi), att

    def _forward_internal(self, s, phi):
        query = self.headin_wq(s)
        values = self.headin_wv(phi)
        keys = self.headin_wk(phi + self.pe1)
        w = torch.matmul(query[:, None, :], keys.permute(0, 2, 1))
        att = torch.softmax(w / self.att_temp, dim=-1)
        summary = torch.matmul(att, values)[:, 0, :]
        return self.internal_policy(torch.hstack([summary, s])), att

    def forward(self, s, ret_encoder_out: bool = False, return_atts: bool = False):
        if not self.is_prev:
            with torch.no_grad():
                phi, _ = self.previous_units(s)
        else:
            if not isinstance(s, tuple):
                raise TypeError("input to a previous CompoNet unit must be (phi, s)")
            phi, s = s

        hs = s if self.encoder is None else self.encoder(s)
        out_head, att_out = self._forward_headout(hs, phi)
        internal, att_in = self._forward_internal(
            hs, torch.cat([phi, out_head], dim=1)
        )
        out = out_head[:, 0, :] + internal
        phi = torch.cat([phi, out[:, None, :]], dim=1)

        if self.is_prev:
            return phi, s
        results = [out, phi]
        if ret_encoder_out:
            results.append(hs)
        if return_atts:
            results += [att_in, att_out]
        return results
