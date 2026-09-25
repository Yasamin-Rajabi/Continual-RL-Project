"""Exact categorical policy mixtures for MiniGrid.

The continual method stores two policy heads for checkpoint/layout compatibility
with the continuous-control implementation.  For MiniGrid the two head outputs
jointly parameterize one categorical component via

    logits_k(s) = head_a_k(s) + head_b_k(s).

Historical reuse is performed in *policy distribution space*: component
categorical probabilities are mixed with the learned simplex weights.  Because
the action space is discrete, the resulting mixture is itself a categorical
policy and all SAC expectations can be evaluated exactly over the seven actions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def components(model, obs):
    """Return component logits ``[B,K,A]`` and simplex weights ``[K]``."""
    if hasattr(model, "policy_components"):
        head_a, head_b, weights = model.policy_components(obs)
        return head_a + head_b, weights
    head_a, head_b = model(obs)
    logits = head_a + head_b
    return logits[:, None, :], logits.new_ones(1)


def mixture_distribution(model, obs):
    """Return component probs/log-probs and exact mixture probs/log-probs."""
    logits, weights = components(model, obs)
    component_log_probs = torch.log_softmax(logits, dim=-1)
    component_probs = component_log_probs.exp()
    mixture_probs = (component_probs * weights[None, :, None]).sum(dim=1)
    tiny = torch.finfo(mixture_probs.dtype).tiny
    mixture_log_probs = mixture_probs.clamp_min(tiny).log()
    return component_probs, component_log_probs, mixture_probs, mixture_log_probs, weights


def representative_action(model, obs, action_scale=None, action_bias=None):
    """Greedy categorical action under the executed mixture policy."""
    _, _, probs, _, _ = mixture_distribution(model, obs)
    return probs.argmax(dim=-1)


def sample_action(model, obs, action_scale=None, action_bias=None, *, score_function=False):
    """Sample the exact categorical mixture.

    ``score_function`` is accepted for API parity with continuous control.  A
    categorical sample is non-reparameterized in either case; gradients for
    routing/test adaptation flow through the returned log probability.
    """
    _, _, probs, log_probs, _ = mixture_distribution(model, obs)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    selected_log_prob = log_probs.gather(1, action[:, None])
    representative = probs.argmax(dim=-1)
    return action, selected_log_prob, representative


def sac_actor_objective(model, obs, q1, q2, temperature, action_scale=None, action_bias=None):
    """Exact SAC-Discrete actor objective for a categorical policy mixture."""
    _, _, probs, log_probs, _ = mixture_distribution(model, obs)
    q = torch.minimum(q1(obs), q2(obs))
    return (probs * (temperature * log_probs - q)).sum(dim=-1).mean()


def novel_sac_actor_objective(model, obs, q1, q2, temperature, action_scale=None, action_bias=None):
    """Exact SAC-Discrete objective for the standalone current expert."""
    if not hasattr(model, "novel_policy_components"):
        raise TypeError("model does not expose a standalone novel policy")
    head_a, head_b = model.novel_policy_components(obs)
    logits = head_a + head_b
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    q = torch.minimum(q1(obs), q2(obs))
    return (probs * (temperature * log_probs - q)).sum(dim=-1).mean()


def mixture_weight_sac_actor_objective(model, obs, q1, q2, temperature,
                                       action_scale=None, action_bias=None):
    """Routing-only SAC objective with component policies held fixed."""
    logits, weights = components(model, obs)
    component_probs = torch.softmax(logits.detach(), dim=-1)
    probs = (component_probs * weights[None, :, None]).sum(dim=1)
    log_probs = probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
    with torch.no_grad():
        q = torch.minimum(q1(obs), q2(obs))
    return (probs * (temperature * log_probs - q)).sum(dim=-1).mean()


def categorical_summary(head_a, head_b, weights):
    """Compatibility summary of a mixture as one categorical logits pair.

    The mixture distribution is exact.  We return ``log p(a|s)`` in the first
    head and zeros in the second so their sum reproduces the same categorical
    distribution.  This is used only by code paths that expect ``forward`` to
    return two tensors; inference itself uses ``policy_components`` directly.
    """
    logits = head_a + head_b
    probs = torch.softmax(logits, dim=-1)
    mixed = (probs * weights[None, :, None]).sum(dim=1)
    out = mixed.clamp_min(torch.finfo(mixed.dtype).tiny).log()
    return out, torch.zeros_like(out)


def stacked_head_forward(z, head):
    """Evaluate K complete MLP heads represented as four stacked tensors."""
    h = torch.einsum("bd,khd->bkh", z, head["l0_weight"]) + head["l0_bias"][None, :, :]
    return torch.einsum("bkh,kah->bka", F.relu(h), head["l2_weight"]) + head["l2_bias"][None, :, :]
