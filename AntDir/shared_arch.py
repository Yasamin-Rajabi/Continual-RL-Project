import torch.nn as nn


def shared(input_dim, linear_out=False):
    """The shared 2-layer encoder used by BOTH the actor (CkaRlAgent.fc) and the
    critic (SoftQNetwork.fc).

    linear_out=False (default, original behaviour): the encoder ends in ReLU, so
    every feature is non-negative.

    linear_out=True: the trailing ReLU is dropped. Required by the TD-JEPA-style
    orthonormality regulariser, which pushes phi(s_i)^T phi(s_j) -> 0 for i != j.
    With non-negative features that inner product can never be negative, so the
    only way to satisfy the penalty is disjoint sparse supports -- a much harder
    and qualitatively different constraint than the one the objective intends.

    Dropping the ReLU costs nothing representationally here: HeadPool applies its
    own ReLU immediately after (see _forward_with_weights), so the original
    architecture had two ReLUs separated by one linear map.

    IMPORTANT: this flag changes the critic too, so a run with linear_out=True is
    NOT comparable to a baseline run with linear_out=False. Re-run baselines under
    the same setting.
    """
    layers = [
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
    ]
    if not linear_out:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def inspect_shared_encoder(module):
    """Return the architecture facts that are invisible to a state_dict.

    The linear-out and ReLU-out variants have identical parameter tensors, so
    load_state_dict cannot detect a mismatch.  We therefore validate the actual
    serialized module whenever an encoder checkpoint is loaded.
    """
    if not isinstance(module, nn.Sequential):
        raise TypeError(
            "shared encoder checkpoint must be nn.Sequential produced by shared(); "
            f"got {type(module).__name__}"
        )
    layers = list(module.children())
    if len(layers) not in (3, 4):
        raise ValueError(f"unexpected shared encoder depth: {len(layers)} layers")
    if not isinstance(layers[0], nn.Linear) or not isinstance(layers[1], nn.ReLU):
        raise ValueError("unexpected shared encoder prefix; expected Linear -> ReLU")
    if not isinstance(layers[2], nn.Linear):
        raise ValueError("unexpected shared encoder third layer; expected Linear")
    if len(layers) == 4 and not isinstance(layers[3], nn.ReLU):
        raise ValueError("unexpected shared encoder final layer; expected ReLU")
    if layers[0].out_features != 256 or layers[2].in_features != 256 or layers[2].out_features != 256:
        raise ValueError("unexpected shared encoder hidden/output dimensions")
    return {
        "input_dim": int(layers[0].in_features),
        "linear_out": len(layers) == 3,
    }


def validate_shared_encoder(module, *, input_dim, linear_out, source="encoder checkpoint"):
    info = inspect_shared_encoder(module)
    if info["input_dim"] != int(input_dim):
        raise ValueError(
            f"{source} expects obs_dim={info['input_dim']}, but this run has obs_dim={int(input_dim)}"
        )
    if info["linear_out"] != bool(linear_out):
        raise ValueError(
            f"{source} has linear_out={info['linear_out']}, but the run was configured with "
            f"encoder_linear_out={bool(linear_out)}. Use the matching flag; the two variants "
            "have identical parameter shapes but different forward functions."
        )
    return module
