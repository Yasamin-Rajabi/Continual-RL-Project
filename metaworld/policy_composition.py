"""Exact mixtures of diagonal, tanh-squashed Gaussian policies.

The component dimension is [batch, component, action]. A single categorical
choice selects the WHOLE action vector, not separate components per actuator.
SAC's actor objective enumerates the small component set and reparameterizes
within each Gaussian. This keeps gradients to mixing logits without pretending
that a categorical sample is reparameterizable. No moment-matched approximation
is used for sampling, log probabilities, or SAC updates.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from policy_utils import bound_log_std, LOG_STD_MIN, LOG_STD_MAX


def components(model, obs):
    """Return means, bounded log stds, and state-independent simplex weights."""
    if hasattr(model, "policy_components"):
        return model.policy_components(obs)
    mean, raw = model(obs)
    return mean[:, None, :], bound_log_std(raw)[:, None, :], mean.new_ones(1)


def normal_mixture_log_prob(value, means, log_stds, weights):
    """Log density before tanh; value has shape [B,A] or [B,L,A]."""
    single = value.ndim == 2
    if single:
        value = value[:, None, :]
    log_weights = weights.clamp_min(torch.finfo(weights.dtype).tiny).log()
    log_weights = log_weights.masked_fill(weights <= 0, -torch.inf)
    terms = torch.distributions.Normal(
        means[:, None, :, :], log_stds.exp()[:, None, :, :]
    ).log_prob(value[:, :, None, :]).sum(-1)
    result = torch.logsumexp(terms + log_weights[None, None, :], dim=-1)
    return result[:, 0] if single else result


def squash_log_det(pre_tanh, action_scale):
    # Stable even when tanh(pre_tanh) rounds to +/-1 in floating point.
    tanh_log_det = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
    return (tanh_log_det + action_scale.log()).sum(-1)


def representative_action(model, obs, action_scale, action_bias):
    """Weighted squashed component means (not the exact mixture mean or mode)."""
    means, _, weights = components(model, obs)
    return (means.tanh() * weights[None, :, None]).sum(1) * action_scale + action_bias


def sample_action(model, obs, action_scale, action_bias, *, score_function=False):
    """Sample the exact mixture; optionally detach the score-function sample."""
    means, log_stds, weights = components(model, obs)
    choice = torch.distributions.Categorical(probs=weights).sample((len(obs),))
    batch = torch.arange(len(obs), device=obs.device)
    normal = torch.distributions.Normal(means[batch, choice], log_stds[batch, choice].exp())
    pre = normal.sample() if score_function else normal.rsample()
    action = pre.tanh() * action_scale + action_bias
    lp = normal_mixture_log_prob(pre, means, log_stds, weights) - squash_log_det(pre, action_scale)
    representative = (means.tanh() * weights[None, :, None]).sum(1) * action_scale + action_bias
    return action, lp[:, None], representative


def sac_actor_objective(model, obs, q1, q2, temperature, action_scale, action_bias):
    """Unbiased stratified mixture expectation for the SAC actor objective.

    Sum_k w_k E_{a~pi_k}[temperature * log(sum_j w_j pi_j(a|s)) - Q(s,a)].
    Each component contributes one reparameterized Gaussian sample per state.
    Crucially both the outer weights and the mixture density remain in graph.
    """
    means, log_stds, weights = components(model, obs)
    pre = torch.distributions.Normal(means, log_stds.exp()).rsample()
    actions = pre.tanh() * action_scale + action_bias
    lp = normal_mixture_log_prob(pre, means, log_stds, weights) - squash_log_det(pre, action_scale)
    batch, count, act_dim = actions.shape
    repeated_obs = obs[:, None, :].expand(-1, count, -1).reshape(batch * count, -1)
    flat_actions = actions.reshape(batch * count, act_dim)
    q = torch.minimum(q1(repeated_obs, flat_actions), q2(repeated_obs, flat_actions)).reshape(batch, count)
    return ((temperature * lp - q) * weights[None, :]).sum(-1).mean()


def novel_sac_actor_objective(model, obs, q1, q2, temperature, action_scale, action_bias):
    """Standard SAC actor loss for the standalone current/novel Gaussian expert.

    Replay states may have been collected by the execution mixture.  Only the
    novel expert is sampled here, so gradients do not update historical experts
    or the mixture weights/gate.
    """
    if not hasattr(model, "novel_policy_components"):
        raise TypeError("model does not expose a standalone novel policy")
    mean, log_std = model.novel_policy_components(obs)
    normal = torch.distributions.Normal(mean, log_std.exp())
    pre = normal.rsample()
    action = pre.tanh() * action_scale + action_bias
    log_prob = normal.log_prob(pre).sum(-1) - squash_log_det(pre, action_scale)
    q = torch.minimum(q1(obs, action), q2(obs, action)).view(-1)
    return (temperature * log_prob - q).mean()


def mixture_weight_sac_actor_objective(model, obs, q1, q2, temperature, action_scale, action_bias):
    """SAC objective for alpha/alpha-mass with expert functions held fixed.

    This is the same stratified expectation used by ``sac_actor_objective``,
    except component Gaussian parameters are detached.  The outer mixture
    weights and the mixture log density remain differentiable, so this update
    changes only the routing coefficients when paired with an alpha-only
    optimizer.
    """
    means, log_stds, weights = components(model, obs)
    means = means.detach()
    log_stds = log_stds.detach()
    # No pathwise expert gradient is wanted in the routing step.
    pre = torch.distributions.Normal(means, log_stds.exp()).sample()
    actions = pre.tanh() * action_scale + action_bias
    lp = normal_mixture_log_prob(pre, means, log_stds, weights) - squash_log_det(pre, action_scale)
    batch, count, act_dim = actions.shape
    repeated_obs = obs[:, None, :].expand(-1, count, -1).reshape(batch * count, -1)
    flat_actions = actions.reshape(batch * count, act_dim)
    with torch.no_grad():
        q = torch.minimum(q1(repeated_obs, flat_actions), q2(repeated_obs, flat_actions)).reshape(batch, count)
    return ((temperature * lp - q) * weights[None, :]).sum(-1).mean()


def gaussian_summary(means, log_stds, weights):
    """Compatibility/diagnostic moment summary ONLY, never an inference policy."""
    mean = (means * weights[None, :, None]).sum(1)
    variance = ((log_stds.mul(2).exp() + (means - mean[:, None, :]).square())
                * weights[None, :, None]).sum(1)
    ls = (0.5 * variance.clamp_min(1e-12).log()).clamp(LOG_STD_MIN, LOG_STD_MAX)
    unit = (2.0 * (ls - LOG_STD_MIN) / (LOG_STD_MAX - LOG_STD_MIN) - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
    return mean, torch.atanh(unit)


def stacked_head_forward(z, head):
    """Evaluate K complete heads, represented as four stacked tensors."""
    h = torch.einsum("bd,khd->bkh", z, head["l0_weight"]) + head["l0_bias"][None, :, :]
    return torch.einsum("bkh,kah->bka", F.relu(h), head["l2_weight"]) + head["l2_bias"][None, :, :]
