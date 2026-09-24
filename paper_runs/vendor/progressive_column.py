"""ProgressiveColumn port from donor pointmaze/src/baselines/agents.py."""
import torch
from torch import nn
from paper_runs.layers import layer_init

class ProgressiveColumn(nn.Module):
    """One column: two hidden layers plus lateral adapters from earlier columns."""

    def __init__(self, obs_dim, hidden_dim, out_dim, n_prev: int):
        super().__init__()
        self.l1 = layer_init(nn.Linear(int(obs_dim), int(hidden_dim)))
        self.l2 = layer_init(nn.Linear(int(hidden_dim), int(hidden_dim)))
        self.u2 = nn.ModuleList(
            [layer_init(nn.Linear(hidden_dim, hidden_dim, bias=False)) for _ in range(n_prev)]
        )
        self.out = layer_init(nn.Linear(int(hidden_dim), int(out_dim)), std=0.01)
        self.uo = nn.ModuleList(
            [layer_init(nn.Linear(hidden_dim, out_dim, bias=False)) for _ in range(n_prev)]
        )
        self.act = nn.ReLU()

    def forward_first(self, x):
        h1 = self.act(self.l1(x))
        h2 = self.act(self.l2(h1))
        return self.out(h2), h1, h2

    def forward_other(self, x, h1s, h2s):
        if len(h1s) != len(self.u2):
            raise AssertionError(
                "number of previous activations does not match the adapter count"
            )
        h1 = self.act(self.l1(x))
        h2 = self.act(self.l2(h1) + sum(u(prev) for u, prev in zip(self.u2, h1s)))
        out = self.out(h2) + sum(u(prev) for u, prev in zip(self.uo, h2s))
        return out, h1, h2

