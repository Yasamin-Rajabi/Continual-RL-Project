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
