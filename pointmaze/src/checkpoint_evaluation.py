"""Load a finished checkpoint and score it, with optional test-time adaptation.

Test-time adaptation budget
---------------------------
``--test-adapt-steps`` is a number of *environment interactions* spent, at
evaluation time, adapting only the routing weights of a composed policy.  The
stored experts never move, so this measures "can the method retrieve the right
piece of what it already knows", not "can it relearn the task".

The default is 6000 = 20 episodes at the 300-step horizon.  That number is
chosen to be defensible rather than convenient:

* MAML's RL protocol adapts with 20 rollouts per gradient step for its
  locomotion tasks, and its 2D-navigation task -- which is what PointMaze is --
  uses 40 samples per step with up to 4 updates.  20 rollouts sits at the
  low end of that range.
* PEARL-style few-shot evaluation adapts from a handful of trajectories.
* It is 6% of the 100k per-task training budget, far too little to be mistaken
  for retraining, and it is applied identically to every method that has
  routing weights to adapt.

Methods without routing weights get no adaptation, which is reported rather
than hidden; for them the adaptation phase is skipped entirely and their score
is the zero-adaptation score.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import torch

from pointmaze_env import MAX_EPISODE_STEPS
from policy_composition import gaussian_mixture_distribution
from tasks import get_task

DEFAULT_TEST_ADAPT_STEPS = 6_000
DEFAULT_TEST_ADAPT_LR = 1e-2


def evaluation_seed(task_id: int, episode: int) -> int:
    return 10_000 + 10_000 * int(task_id) + int(episode)


def load_policy(run_dir, method: str, device):
    """Load whatever the given method saved, as an object with an action dist."""
    from baselines import canonical_method

    method = canonical_method(method)
    run_dir = pathlib.Path(run_dir)

    if method in ("Ours", "Ours-parameter", "CKA-RL"):
        from cka_rl import FrozenCkaPolicy

        return FrozenCkaPolicy.load(str(run_dir), map_location=device).to(device).eval()

    path = run_dir / "agent.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {path}")
    try:
        agent = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        agent = torch.load(path, map_location=device)
    return agent.to(device).eval()


def _adaptation_parameters(policy):
    """Routing parameters only: mixture weights, never the experts.

    Only a policy-space snapshot has a mixture to route over.  A
    parameter-space snapshot (CKA-RL, Ours-parameter) has already collapsed
    its pool into one head, so there is nothing to adapt and it is scored
    without adaptation; the result records ``adapted: False`` rather than
    pretending otherwise.
    """
    weights = getattr(policy, "mixture_weights", None)
    if weights is None or weights.numel() < 2:
        return []
    # Adapt logits rather than the weights themselves so the mixture can never
    # leave the simplex during adaptation.
    logits = torch.log(weights.clamp_min(1e-8)).clone().requires_grad_(True)
    policy._adapt_logits = logits
    return [logits]


def _apply_adapted_weights(policy):
    logits = getattr(policy, "_adapt_logits", None)
    if logits is not None:
        with torch.no_grad():
            policy.mixture_weights.copy_(torch.softmax(logits, dim=0))


def evaluate_checkpoint(
    run_dir,
    method: str,
    suite: str,
    task_id: int,
    episodes: int,
    seed: int,
    device,
    *,
    adapt_steps: int = 0,
    adapt_lr: float = DEFAULT_TEST_ADAPT_LR,
    action_mode: str = "deterministic",
    success_threshold: Optional[float] = None,
):
    if episodes < 1 or adapt_steps < 0 or adapt_lr <= 0:
        raise ValueError("invalid evaluation budget")
    if action_mode not in ("deterministic", "stochastic"):
        raise ValueError("action_mode must be 'deterministic' or 'stochastic'")

    policy = load_policy(run_dir, method, device)
    params = _adaptation_parameters(policy) if adapt_steps else []
    optimizer = torch.optim.Adam(params, lr=adapt_lr) if params else None

    adaptation_interactions = 0
    evaluation_interactions = 0

    def _obs_tensor(obs_np):
        return torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)

    def _dist(obs_np):
        """Scoring-time distribution; weights are whatever the policy holds."""
        return policy.action_distribution(_obs_tensor(obs_np))

    def _adapt_dist(obs_np):
        """Adaptation-time distribution whose weights carry a gradient.

        The mixture weights are rebuilt from the trainable logits on every
        call.  Writing softmax(logits) into the registered buffer instead
        would detach them, and the routing update would silently be a no-op.
        """
        logits = policy._adapt_logits
        raw, _ = policy.policy_components(_obs_tensor(obs_np))
        return gaussian_mixture_distribution(
            raw, torch.softmax(logits, dim=0), policy.act_dim
        )

    # ---- optional routing adaptation ----
    if adapt_steps and optimizer is not None and hasattr(policy, "_adapt_logits"):
        env = get_task(task_id, suite)
        obs, _ = env.reset(seed=evaluation_seed(task_id, 0))
        try:
            for step in range(int(adapt_steps)):
                dist = _adapt_dist(obs)
                action = dist.sample()
                logp = dist.log_prob(action)
                obs, reward, terminated, truncated, _ = env.step(
                    action.squeeze(0).detach().cpu().numpy()
                )
                adaptation_interactions += 1
                # One-step score-function update on routing weights only.
                loss = -(logp.mean() * float(reward))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if terminated or truncated:
                    obs, _ = env.reset(
                        seed=evaluation_seed(task_id, 0) + step + 1
                    )
        finally:
            env.close()
        _apply_adapted_weights(policy)

    # ---- scoring ----
    returns, successes = [], []
    env = get_task(task_id, suite)
    try:
        with torch.no_grad():
            for ep in range(int(episodes)):
                obs, _ = env.reset(seed=evaluation_seed(task_id, ep))
                total, success = 0.0, 0.0
                for _ in range(MAX_EPISODE_STEPS + 1):
                    dist = _dist(obs)
                    action = (
                        dist.sample()
                        if action_mode == "stochastic"
                        else dist.deterministic_action()
                    )
                    obs, reward, terminated, truncated, info = env.step(
                        action.squeeze(0).cpu().numpy()
                    )
                    evaluation_interactions += 1
                    total += float(reward)
                    # Success latches, so take a max over the episode.
                    success = max(success, float(info.get("is_success", False)))
                    if terminated or truncated:
                        break
                returns.append(total)
                successes.append(success)
    finally:
        env.close()

    result = {
        "return": float(np.mean(returns)),
        "success": float(np.mean(successes)),
        "return_std": float(np.std(returns)),
        "episodes": int(episodes),
        "adaptation_interactions": int(adaptation_interactions),
        "evaluation_interactions": int(evaluation_interactions),
        "adapted": bool(params),
    }
    if success_threshold is not None:
        result["threshold_success"] = float(
            np.mean([r >= float(success_threshold) for r in returns])
        )
    return result
