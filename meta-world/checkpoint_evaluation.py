"""Shared checkpoint semantics across all requested continuous-control suites.

pool: a single uniform policy over the FINALIZED pool, without a task oracle.
snapshot: the exact active policy saved before insertion/consolidation.
Test-time adaptation is optional, explicitly counted, and only changes mixture
logits/an originally-learnable softmax scale. There is no new expert at test time,
so alpha-mass is fixed to one for both composition spaces.
"""
from __future__ import annotations
import pathlib
import numpy as np
import torch
from cka_rl import CkaRlAgent, FrozenCkaPolicy
from policy_composition import representative_action, sample_action
from tasks import get_task


def load_finalized_policy(run_dir, device):
    run_dir = pathlib.Path(run_dir)
    snapshot = torch.load(run_dir / "policy_snapshot.pt", map_location="cpu", weights_only=False)
    data = torch.load(run_dir / "mean_pool.pt", map_location="cpu", weights_only=False)
    scale = getattr(data, "alpha_scale", None)
    learned_scale = bool(scale is not None and scale.requires_grad)
    fixed_scale = bool(scale is not None and not learned_scale
                       and abs(float(scale.detach().reshape(-1)[0]) - 5.0) < 1e-6)
    agent = CkaRlAgent(
        obs_dim=int(snapshot["obs_dim"]), act_dim=int(snapshot["act_dim"]),
        hidden_dim=int(data.hidden_dim), pool_size=int(data.pool_size),
        base_dir=None, latest_dir=str(run_dir), alpha_init="Uniform", train_shared=False,
        fusion_mode=data.fusion_mode, composition_space=snapshot.get("composition_space", "parameter"),
        use_alpha_mass=data.use_alpha_mass, constrain_alpha_mass=data.constrain_alpha_mass,
        use_alpha_scale=learned_scale, fix_alpha_scale=fixed_scale,
        distillation=bool(snapshot.get("distillation", False)),
        distill_observation_skip=bool(snapshot.get("distill_observation_skip", True)),
        encoder_linear_out=bool(snapshot.get("encoder_linear_out", False)),
    ).to(device)
    agent.pool_only = True
    agent.set_mixture_warmup(True)  # exact unit history mass; never an untrained novel expert
    agent.requires_grad_(False)
    agent.eval()
    return agent, learned_scale


def _finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def evaluate(run_dir, suite, task_id, episodes, seed, device, *,
             adapt_steps=0, adapt_lr=1e-2, frozen_policy="pool",
             action_mode="deterministic", error_key="velocity_error", episodic_success=False):
    if frozen_policy not in ("pool", "snapshot") or action_mode not in ("deterministic", "stochastic"):
        raise ValueError("Invalid checkpoint/evaluation action mode")
    if adapt_steps < 0 or adapt_lr <= 0 or episodes < 1:
        raise ValueError("Invalid evaluation budget, learning rate or episode count")
    if adapt_steps and frozen_policy != "pool":
        raise ValueError("Test-time mixture adaptation requires --frozen-eval-policy pool")
    if frozen_policy == "snapshot":
        policy = FrozenCkaPolicy.load(str(run_dir), map_location=device).to(device).eval()
        learned_scale = False
    else:
        policy, learned_scale = load_finalized_policy(run_dir, device)
    env = get_task(task_id, task_suite=suite)
    if int(np.prod(env.observation_space.shape)) != policy.obs_dim:
        env.close()
        raise ValueError("Checkpoint observation dimension is incompatible with task-blind observations; retrain")
    scale = torch.as_tensor((env.action_space.high - env.action_space.low) / 2, device=device, dtype=torch.float32)
    bias = torch.as_tensor((env.action_space.high + env.action_space.low) / 2, device=device, dtype=torch.float32)
    adapt_params = []
    if adapt_steps and policy.alpha is not None and policy.alpha.numel() > 1:
        policy.alpha.requires_grad_(True)
        adapt_params.append(policy.alpha)
        if learned_scale and policy.alpha_scale is not None:
            policy.alpha_scale.requires_grad_(True)
            adapt_params.append(policy.alpha_scale)
    optimizer = torch.optim.Adam(adapt_params, lr=adapt_lr) if adapt_params else None
    adaptation_interactions = evaluation_interactions = 0
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed) + 10000 * int(task_id))
            if adapt_steps:
                obs, _ = env.reset(seed=int(seed))
                for _ in range(int(adapt_steps)):
                    x = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                    action, log_prob, _ = sample_action(policy, x, scale, bias, score_function=True)
                    obs, reward, terminated, truncated, _ = env.step(action[0].detach().cpu().numpy())
                    adaptation_interactions += 1
                    if optimizer is not None:
                        # Existing immediate-reward REINFORCE heuristic; not a
                        # return-to-go policy-gradient or a new SAC training run.
                        loss = -log_prob.mean() * float(reward)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    if terminated or truncated:
                        obs, _ = env.reset()
            policy.requires_grad_(False)
            returns, successes, errors = [], [], []
            for ep in range(episodes):
                obs, _ = env.reset(seed=int(seed) + 10000 * int(task_id) + ep)
                ret, ep_success, ep_error = 0.0, [], []
                while True:
                    x = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action = (sample_action(policy, x, scale, bias)[0] if action_mode == "stochastic"
                                  else representative_action(policy, x, scale, bias))
                    obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                    evaluation_interactions += 1
                    ret += float(reward)
                    ep_success.append(float(info.get("success", np.nan)))
                    ep_error.append(float(info.get(error_key, np.nan)))
                    if terminated or truncated:
                        break
                returns.append(ret)
                finite_success = np.asarray(ep_success)[np.isfinite(ep_success)]
                successes.append(float(finite_success.max()) if episodic_success and len(finite_success)
                                 else _finite_mean(ep_success))
                errors.append(_finite_mean(ep_error))
    finally:
        env.close()
    return {"return": _finite_mean(returns), "success": _finite_mean(successes),
            error_key: _finite_mean(errors), "adaptation_interactions": adaptation_interactions,
            "evaluation_interactions": evaluation_interactions}
