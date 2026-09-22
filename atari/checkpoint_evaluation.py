"""Shared checkpoint semantics for continual Atari CKA-RL.

pool: a single task-agnostic policy from the FINALIZED pool.
snapshot: the exact active policy saved before insertion/consolidation.

Atari-specific evaluation rules:
- categorical actions;
- observations are uint8 frame stacks and are normalized by /255 before policy use;
- evaluation uses full-game raw reward: clip_reward=False, episodic_life=False;
- optional success is 1[raw full-game return >= success_threshold].

Test-time adaptation is optional and explicitly counted.  As in the current
HalfCheetah protocol, it changes only mixture logits and an originally-learnable
softmax scale.  There is no novel expert at test time, so alpha-mass is forced
to one through pool-only / mixture-warmup semantics.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch

from atari_tasks import get_task
from cka_rl import CkaRlAgent, FrozenCkaPolicy


def _torch_load(path, map_location=None):
    kwargs = {} if map_location is None else {"map_location": map_location}
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _evaluation_seed(task_id: int, episode: int) -> int:
    """Fixed benchmark seed, independent of the training seed."""
    return 10_000 + 10_000 * int(task_id) + int(episode)


def _eval_env(suite: str, task_id: int):
    return get_task(
        task_id,
        task_suite=suite,
        clip_reward=False,
        episodic_life=False,
    )


def _normalized_obs(obs, device):
    return (
        torch.as_tensor(obs, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .div_(255.0)
    )


def _distribution(policy, obs_t):
    if not hasattr(policy, "action_distribution"):
        raise TypeError(
            f"{type(policy).__name__} does not expose action_distribution(); "
            "the Atari policy must provide a categorical distribution interface."
        )
    return policy.action_distribution(obs_t)


def load_finalized_policy(run_dir, device):
    """Reconstruct the finalized task-agnostic pool policy."""
    run_dir = pathlib.Path(run_dir)
    snapshot = _torch_load(run_dir / "policy_snapshot.pt", map_location="cpu")
    data = _torch_load(run_dir / "policy_pool.pt", map_location="cpu")

    scale = getattr(data, "alpha_scale", None)
    learned_scale = bool(scale is not None and scale.requires_grad)
    fixed_scale = bool(
        scale is not None
        and not learned_scale
        and abs(float(scale.detach().reshape(-1)[0]) - 5.0) < 1e-6
    )

    agent = CkaRlAgent(
        obs_shape=tuple(int(x) for x in snapshot["obs_shape"]),
        act_dim=int(snapshot["act_dim"]),
        shared_dim=int(snapshot.get("shared_dim", getattr(data, "shared_dim", 512))),
        hidden_dim=int(snapshot.get("hidden_dim", getattr(data, "hidden_dim", 128))),
        pool_size=int(getattr(data, "pool_size", 5)),
        base_dir=None,
        latest_dir=str(run_dir),
        alpha_init="Uniform",
        train_shared=False,
        fusion_mode=getattr(data, "fusion_mode", "classic_cka"),
        composition_space=snapshot.get("composition_space", "parameter"),
        use_alpha_mass=bool(getattr(data, "use_alpha_mass", False)),
        constrain_alpha_mass=bool(getattr(data, "constrain_alpha_mass", True)),
        use_alpha_scale=learned_scale,
        fix_alpha_scale=fixed_scale,
        distillation=bool(snapshot.get("distillation", getattr(data, "distillation", False))),
    ).to(device)

    # Finalized-pool evaluation has historical knowledge only.  There is no
    # untrained/new expert at test time, so any alpha-mass gate must be exactly 1.
    agent.pool_only = True
    if hasattr(agent, "set_mixture_warmup"):
        agent.set_mixture_warmup(True)
    else:
        # Kept as a loud compatibility guard while the new Atari cka_rl.py is
        # being ported to the updated HalfCheetah policy-space semantics.
        if getattr(agent, "use_alpha_mass", False):
            raise RuntimeError(
                "Updated checkpoint evaluation requires CkaRlAgent.set_mixture_warmup() "
                "for finalized alpha-mass semantics. Update cka_rl.py first."
            )

    agent.requires_grad_(False)
    agent.eval()
    return agent, learned_scale


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def evaluate(
    run_dir,
    suite,
    task_id,
    episodes,
    seed,
    device,
    *,
    adapt_steps=0,
    adapt_lr=1e-2,
    frozen_policy="pool",
    action_mode="deterministic",
    success_threshold=None,
):
    """Evaluate one Atari checkpoint using raw full-game score."""
    if frozen_policy not in ("pool", "snapshot"):
        raise ValueError("frozen_policy must be 'pool' or 'snapshot'")
    if action_mode not in ("deterministic", "stochastic"):
        raise ValueError("action_mode must be 'deterministic' or 'stochastic'")
    if adapt_steps < 0 or adapt_lr <= 0 or episodes < 1:
        raise ValueError("Invalid evaluation budget, learning rate, or episode count")
    if adapt_steps and frozen_policy != "pool":
        raise ValueError("Test-time mixture adaptation requires --frozen-eval-policy pool")

    if frozen_policy == "snapshot":
        policy = FrozenCkaPolicy.load(str(run_dir), map_location=device).to(device).eval()
        learned_scale = False
    else:
        policy, learned_scale = load_finalized_policy(run_dir, device)

    env = _eval_env(suite, task_id)
    expected_shape = tuple(int(x) for x in env.observation_space.shape)
    policy_shape = tuple(int(x) for x in getattr(policy, "obs_shape", expected_shape))
    if expected_shape != policy_shape:
        env.close()
        raise ValueError(
            f"Checkpoint observation shape {policy_shape} is incompatible with "
            f"Atari task observation shape {expected_shape}"
        )

    adapt_params = []
    if adapt_steps and getattr(policy, "alpha", None) is not None and policy.alpha.numel() > 1:
        policy.alpha.requires_grad_(True)
        adapt_params.append(policy.alpha)
        if learned_scale and getattr(policy, "alpha_scale", None) is not None:
            policy.alpha_scale.requires_grad_(True)
            adapt_params.append(policy.alpha_scale)

    optimizer = torch.optim.Adam(adapt_params, lr=adapt_lr) if adapt_params else None
    adaptation_interactions = 0
    evaluation_interactions = 0

    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )

    try:
        with torch.random.fork_rng(devices=cuda_devices):
            # `seed` controls stochastic policy sampling only. Environment
            # evaluation seeds are fixed across training seeds by _evaluation_seed.
            torch.manual_seed(int(seed) + 10_000 * int(task_id))

            if adapt_steps:
                obs, _ = env.reset(seed=_evaluation_seed(task_id, 0))
                for adapt_idx in range(int(adapt_steps)):
                    obs_t = _normalized_obs(obs, device)
                    dist = _distribution(policy, obs_t)
                    action = dist.sample()

                    obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                    adaptation_interactions += 1

                    if optimizer is not None:
                        # Same one-step score-function heuristic used by the
                        # HalfCheetah evaluator; only mixture routing parameters
                        # are trainable here.
                        loss = -dist.log_prob(action).mean() * float(reward)
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

                    if terminated or truncated:
                        obs, _ = env.reset(
                            seed=_evaluation_seed(task_id, 1 + adapt_idx)
                        )

            policy.requires_grad_(False)
            returns = []
            successes = []

            for ep in range(int(episodes)):
                obs, _ = env.reset(seed=_evaluation_seed(task_id, ep))
                ep_return = 0.0

                while True:
                    obs_t = _normalized_obs(obs, device)
                    with torch.no_grad():
                        dist = _distribution(policy, obs_t)
                        if action_mode == "stochastic":
                            action = dist.sample()
                        else:
                            action = torch.argmax(dist.probs, dim=-1)

                    obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                    evaluation_interactions += 1
                    ep_return += float(reward)

                    if terminated or truncated:
                        break

                returns.append(ep_return)
                if success_threshold is not None:
                    successes.append(
                        float(ep_return >= float(success_threshold))
                    )
    finally:
        env.close()

    reward = _finite_mean(returns)
    success = _finite_mean(successes) if successes else float("nan")
    return {
        "reward": reward,
        "return": reward,
        "success": success,
        "adaptation_interactions": int(adaptation_interactions),
        "evaluation_interactions": int(evaluation_interactions),
    }
