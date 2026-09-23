"""The ``policy_snapshot.pt`` contract, shared with the CKA-RL method.

WHY THIS IS WORTH GETTING EXACTLY RIGHT
---------------------------------------
``cka_rl.FrozenCkaPolicy`` already knows how to load this format. Its
``composition_space == "parameter"`` branch registers four buffers per head and
evaluates

    h = relu(linear(z, l0_weight, l0_bias))
    out = linear(h, l2_weight, l2_bias)

with ``z = fc(obs)`` when ``distillation`` is False. A baseline that emits this
dict is therefore evaluated by the UNMODIFIED
``checkpoint_evaluation.evaluate(run_dir, ..., frozen_policy="snapshot")``,
using the same episode seeds, the same deterministic/stochastic action mode and
the same success and error keys as the method it is being compared against.

That is the strongest available guarantee of evaluation parity: the baselines
are not evaluated by baseline code at all. They go through the method's own
evaluator.

THE ONE THING A BASELINE MUST GUARANTEE
---------------------------------------
``effective_head_parameters()`` must describe a two-layer head that is
FUNCTIONALLY IDENTICAL to what the agent actually executed during training for
the active task. For FT-N, PackNet and MaskNet the executed head already is two
linear layers (PackNet and MaskNet fold their masks into the weights), so this
is exact. ProgNet's lateral adapters are not expressible that way, so it
overrides ``export_snapshot`` and stores its columns instead; see prognet.py.
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict

import torch

# The parameter names cka_rl._HEAD_KEYS uses. Keep in this order.
HEAD_KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")
SNAPSHOT_NAME = "policy_snapshot.pt"


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def _validate_head(name: str, head: Dict[str, torch.Tensor], *,
                   in_dim: int, hidden_dim: int, act_dim: int) -> None:
    missing = [key for key in HEAD_KEYS if key not in head]
    if missing:
        raise ValueError(f"{name} head is missing {missing}")
    expected = {
        "l0_weight": (hidden_dim, in_dim),
        "l0_bias": (hidden_dim,),
        "l2_weight": (act_dim, hidden_dim),
        "l2_bias": (act_dim,),
    }
    for key, shape in expected.items():
        actual = tuple(head[key].shape)
        if actual != shape:
            raise ValueError(
                f"{name} head parameter {key} has shape {actual}, expected "
                f"{shape}. FrozenCkaPolicy would load this without complaint "
                "and then evaluate a different function than was trained."
            )


def export_snapshot(agent) -> Dict[str, Any]:
    """Build the FrozenCkaPolicy-compatible payload for ``agent``.

    ``agent`` supplies ``shared_encoder()`` and ``effective_head_parameters()``;
    see baselines/common/lifecycle.py.
    """
    with torch.no_grad():
        heads = agent.effective_head_parameters()
        for head_name in ("mean", "logstd"):
            if head_name not in heads:
                raise ValueError(
                    f"effective_head_parameters() did not return a {head_name!r} head"
                )
            _validate_head(
                head_name, heads[head_name],
                in_dim=256, hidden_dim=agent.effective_hidden_dim(),
                act_dim=agent.act_dim,
            )
        encoder = agent.shared_encoder()
        return {
            "composition_space": "parameter",
            "obs_dim": int(agent.obs_dim),
            "act_dim": int(agent.act_dim),
            # False so FrozenCkaPolicy.forward feeds fc(obs) straight to the
            # head with no raw-observation skip connection. Every baseline here
            # uses the plain encoder output.
            "distillation": False,
            "distill_observation_skip": False,
            # Identical parameter shapes, different forward function. Without
            # this flag the load silently succeeds and evaluates the wrong
            # network.
            "encoder_linear_out": bool(agent.encoder_linear_out),
            "fc_state_dict": {k: _cpu_clone(v)
                              for k, v in encoder.state_dict().items()},
            "mean": {k: _cpu_clone(heads["mean"][k]) for k in HEAD_KEYS},
            "logstd": {k: _cpu_clone(heads["logstd"][k]) for k in HEAD_KEYS},
        }


def save_snapshot(agent, run_dir) -> pathlib.Path:
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / SNAPSHOT_NAME
    torch.save(agent.export_policy_snapshot(), path)
    return path


def verify_snapshot(agent, run_dir, device, *, atol: float = 1e-5) -> float:
    """Re-load the snapshot and assert it matches the live agent.

    A snapshot that silently disagrees with the network that was trained
    produces a plausible-looking retention matrix built on the wrong policy,
    and nothing downstream can detect it. This is cheap, so it runs on every
    task rather than only in tests.

    Returns the maximum absolute deviation over a batch of random observations.
    """
    from cka_rl import FrozenCkaPolicy

    run_dir = pathlib.Path(run_dir)
    loaded = FrozenCkaPolicy.load(str(run_dir), map_location=device).to(device).eval()
    probe = torch.randn(64, agent.obs_dim, device=device)
    was_training = agent.training
    agent.eval()
    try:
        with torch.no_grad():
            live_mean, live_raw = agent.policy(probe)
            snap_mean, snap_raw = loaded(probe)
            deviation = max(
                float((live_mean - snap_mean).abs().max()),
                float((live_raw - snap_raw).abs().max()),
            )
    finally:
        if was_training:
            agent.train()
    if deviation > atol:
        raise RuntimeError(
            f"policy_snapshot.pt disagrees with the trained network by "
            f"{deviation:.3e} (tolerance {atol:.1e}). The saved checkpoint "
            "would be evaluated as a different policy than the one trained."
        )
    return deviation
