"""Multitask supermask layers (MaskNet).

Ported from the project's Atari implementation, which itself adapts
https://github.com/RAIVNLab/supsup.  The layer is already ``nn.Linear`` based,
so the port to a continuous-control MLP is direct: only the surrounding
network changes.

The weights are fixed signed constants and are never trained; learning selects
a *subnetwork* per task by thresholding a per-task score matrix, with a
straight-through estimator on the backward pass.  With
``NEW_MASK_LINEAR_COMB`` a new task's mask starts as a learned convex
combination of the previous tasks' masks, which is the transfer mechanism.
"""
from __future__ import annotations

import math

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

NEW_MASK_RANDOM = "random"
NEW_MASK_LINEAR_COMB = "linear_comb"


class GetSubnetDiscrete(autograd.Function):
    @staticmethod
    def forward(ctx, scores, a=0):
        return (scores >= a).float()

    @staticmethod
    def backward(ctx, g):
        # Straight-through: the threshold has zero gradient everywhere, so the
        # incoming gradient is passed to the scores unchanged.
        return g, None


class GetSubnetContinuous(autograd.Function):
    @staticmethod
    def forward(ctx, scores, a=0):
        return (scores >= a).float() * scores

    @staticmethod
    def backward(ctx, g):
        return g, None


def mask_init(module: nn.Linear) -> torch.Tensor:
    scores = torch.empty_like(module.weight)
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores


def signed_constant(module: nn.Linear) -> None:
    fan = nn.init._calculate_correct_fan(module.weight, "fan_in")
    gain = nn.init.calculate_gain("relu")
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std


class MultitaskMaskLinear(nn.Linear):
    def __init__(
        self,
        *args,
        discrete: bool = True,
        num_tasks: int = 1,
        new_mask_type: str = NEW_MASK_RANDOM,
        **kwargs,
    ):
        kwargs.pop("bias", None)
        super().__init__(*args, bias=False, **kwargs)
        self.num_tasks = int(num_tasks)
        self.scores = nn.ParameterList(
            [nn.Parameter(mask_init(self)) for _ in range(self.num_tasks)]
        )
        self.weight.requires_grad = False
        signed_constant(self)

        self.task = -1
        self.num_tasks_learned = 0
        self.new_mask_type = new_mask_type
        if new_mask_type == NEW_MASK_LINEAR_COMB:
            self.betas = nn.Parameter(torch.zeros(self.num_tasks, self.num_tasks))
        else:
            self.betas = None
        self._subnet_class = GetSubnetDiscrete if discrete else GetSubnetContinuous

    def _mask(self) -> torch.Tensor:
        score = self.scores[self.task]
        if self.new_mask_type != NEW_MASK_LINEAR_COMB:
            return self._subnet_class.apply(score)
        if self.task < self.num_tasks_learned or self.task == 0:
            return self._subnet_class.apply(score)
        # New task: combine previous masks (frozen) with the new score, and
        # learn only the combination coefficients plus the new score.
        previous = [self.scores[i].detach() for i in range(self.task)]
        betas = torch.softmax(self.betas[self.task, : self.task + 1], dim=-1)
        stacked = previous + [score]
        combined = torch.stack([b * s for b, s in zip(betas, stacked)], dim=0).sum(dim=0)
        return self._subnet_class.apply(combined)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task < 0:
            raise ValueError("MultitaskMaskLinear.task must be set to >= 0")
        return F.linear(x, self.weight * self._mask(), self.bias)

    @torch.no_grad()
    def consolidate_mask(self):
        if self.new_mask_type != NEW_MASK_LINEAR_COMB:
            return
        if self.task <= 0 or self.task < self.num_tasks_learned:
            return
        previous = [self.scores[i].detach() for i in range(self.task)]
        betas = torch.softmax(self.betas[self.task, : self.task + 1], dim=-1)
        stacked = previous + [self.scores[self.task]]
        combined = torch.stack([b * s for b, s in zip(betas, stacked)], dim=0).sum(dim=0)
        self.scores[self.task].data = combined.data

    @torch.no_grad()
    def set_task(self, task: int, new_task: bool = False):
        self.task = int(task)
        if self.new_mask_type == NEW_MASK_LINEAR_COMB and new_task and self.task > 0:
            k = self.task + 1
            self.betas.data[self.task, :k] = 1.0 / k


def set_model_task(model: nn.Module, task: int, new_task: bool = False):
    for module in model.modules():
        if isinstance(module, MultitaskMaskLinear):
            module.set_task(task, new_task)


def set_num_tasks_learned(model: nn.Module, num_tasks_learned: int):
    for module in model.modules():
        if isinstance(module, MultitaskMaskLinear):
            module.num_tasks_learned = int(num_tasks_learned)


def consolidate_mask(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MultitaskMaskLinear):
            module.consolidate_mask()
