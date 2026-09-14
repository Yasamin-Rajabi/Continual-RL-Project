"""Shared CNN architecture for Atari CKA-RL PPO.

This module is the discrete/Atari counterpart of the HalfCheetah ``shared_arch``
module.  The continual-learning mechanism is not changed here; this file only
provides the visual encoder used by the PPO actor/policy-pool and value head.

Expected preprocessing (performed outside this module):
    uint8 stacked Atari frames, shape (4, 84, 84)
        -> convert to float32
        -> divide by 255
        -> AtariSharedEncoder
        -> 512-D feature vector (default)

The policy head consumes ONLY the encoder feature vector.  Raw pixels are never
concatenated to the head input.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

ATARI_INPUT_SHAPE: Tuple[int, int, int] = (4, 84, 84)
ATARI_FEATURE_DIM = 512
ENCODER_FORMAT_VERSION = 1


def layer_init(
    layer: nn.Module,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
):
    """CleanRL-style orthogonal initialization used by the Atari PPO network."""
    if not hasattr(layer, "weight") or layer.weight is None:
        raise TypeError(
            "layer_init expects a module with a weight tensor, such as "
            f"nn.Conv2d or nn.Linear; got {type(layer).__name__}"
        )
    torch.nn.init.orthogonal_(layer.weight, std)
    if getattr(layer, "bias", None) is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AtariSharedEncoder(nn.Module):
    """CleanRL-style CNN encoder for 4-frame 84x84 Atari observations.

    Input must already be floating point and normalized by the caller.  Keeping
    normalization outside the encoder is intentional: training, behavioral-KL
    replay, distillation, and checkpoint evaluation all use the same explicit
    ``obs.float() / 255.0`` preprocessing.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = ATARI_INPUT_SHAPE,
        output_dim: int = ATARI_FEATURE_DIM,
    ):
        super().__init__()

        self.format_version = ENCODER_FORMAT_VERSION
        self.input_shape = tuple(int(x) for x in input_shape)
        self.output_dim = int(output_dim)

        if len(self.input_shape) != 3:
            raise ValueError(
                f"Atari encoder expects CHW input with three dimensions, got {self.input_shape}"
            )
        if self.input_shape[0] != 4:
            raise ValueError(
                "Atari encoder expects four stacked grayscale frames in CHW order; "
                f"got input_shape={self.input_shape}"
            )
        if self.input_shape[1:] != (84, 84):
            raise ValueError(
                "This project uses the legacy Atari PPO architecture for 84x84 frames. "
                f"Expected input_shape=(4, 84, 84), got {self.input_shape}. "
                "Changing screen size changes the encoder architecture and requires a "
                "separate controlled experiment."
            )
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {self.output_dim}")

        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # For 84x84 input this is 64*7*7 = 3136.  Compute it rather than
        # hard-coding so the architecture is self-checking.
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape, dtype=torch.float32)
            flat_dim = int(self.conv(dummy).shape[-1])
        self.flat_dim = flat_dim

        self.fc = nn.Sequential(
            layer_init(nn.Linear(self.flat_dim, self.output_dim)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"AtariSharedEncoder expects [B,C,H,W], got shape={tuple(x.shape)}"
            )
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"AtariSharedEncoder expects per-sample shape {self.input_shape}, "
                f"got {tuple(x.shape[1:])}"
            )
        if not x.is_floating_point():
            raise TypeError(
                "AtariSharedEncoder expects floating-point normalized observations. "
                "Convert uint8 frames with obs.float() / 255.0 before calling it."
            )
        return self.fc(self.conv(x))


def shared(
    input_shape: Tuple[int, int, int] = ATARI_INPUT_SHAPE,
    output_dim: int = ATARI_FEATURE_DIM,
):
    """Build the shared Atari CNN under the historical ``shared`` API name.

    The signature is deliberately strict.  HalfCheetah-only arguments such as
    ``input_dim`` or ``linear_out`` should fail loudly instead of being silently
    ignored and producing the wrong Atari architecture.
    """
    return AtariSharedEncoder(input_shape=input_shape, output_dim=output_dim)


def _pair(value):
    if isinstance(value, tuple):
        return tuple(int(x) for x in value)
    return (int(value), int(value))


def inspect_shared_encoder(module: nn.Module):
    """Return and validate architecture facts that a state_dict alone cannot.

    Checking only ``input_shape``/``output_dim`` attributes is not sufficient:
    a serialized module with modified convolution widths, kernels, strides, or
    final activation could otherwise pass validation.  This mirrors the strict
    architecture validation used by the HalfCheetah encoder.
    """
    if not isinstance(module, AtariSharedEncoder):
        raise TypeError(
            "shared encoder checkpoint must be AtariSharedEncoder produced by "
            f"shared(); got {type(module).__name__}"
        )

    input_shape = tuple(int(x) for x in module.input_shape)
    output_dim = int(module.output_dim)
    format_version = int(getattr(module, "format_version", 1))

    if input_shape != ATARI_INPUT_SHAPE:
        raise ValueError(
            f"unexpected Atari encoder input_shape={input_shape}; "
            f"expected {ATARI_INPUT_SHAPE}"
        )
    if output_dim <= 0:
        raise ValueError(f"unexpected Atari encoder output_dim={output_dim}")
    if format_version != ENCODER_FORMAT_VERSION:
        raise ValueError(
            f"unsupported Atari encoder format_version={format_version}; "
            f"expected {ENCODER_FORMAT_VERSION}"
        )

    conv_layers = list(module.conv.children())
    if len(conv_layers) != 7:
        raise ValueError(
            f"unexpected Atari convolutional encoder depth: {len(conv_layers)} layers"
        )

    expected_types = (
        nn.Conv2d,
        nn.ReLU,
        nn.Conv2d,
        nn.ReLU,
        nn.Conv2d,
        nn.ReLU,
        nn.Flatten,
    )
    for idx, (layer, expected_type) in enumerate(zip(conv_layers, expected_types)):
        if not isinstance(layer, expected_type):
            raise ValueError(
                f"unexpected conv layer {idx}: expected {expected_type.__name__}, "
                f"got {type(layer).__name__}"
            )

    conv0, _, conv1, _, conv2, _, _ = conv_layers
    conv_specs = (
        (conv0, 4, 32, (8, 8), (4, 4)),
        (conv1, 32, 64, (4, 4), (2, 2)),
        (conv2, 64, 64, (3, 3), (1, 1)),
    )
    for idx, (layer, in_ch, out_ch, kernel, stride) in enumerate(conv_specs):
        if layer.in_channels != in_ch or layer.out_channels != out_ch:
            raise ValueError(
                f"unexpected Conv2d[{idx}] channels: "
                f"{layer.in_channels}->{layer.out_channels}, expected {in_ch}->{out_ch}"
            )
        if _pair(layer.kernel_size) != kernel or _pair(layer.stride) != stride:
            raise ValueError(
                f"unexpected Conv2d[{idx}] kernel/stride: "
                f"kernel={_pair(layer.kernel_size)}, stride={_pair(layer.stride)}, "
                f"expected kernel={kernel}, stride={stride}"
            )

    fc_layers = list(module.fc.children())
    if len(fc_layers) != 2 or not isinstance(fc_layers[0], nn.Linear) or not isinstance(fc_layers[1], nn.ReLU):
        raise ValueError(
            "unexpected Atari encoder output block; expected Linear -> ReLU"
        )

    linear = fc_layers[0]
    expected_flat_dim = 64 * 7 * 7
    if linear.in_features != expected_flat_dim:
        raise ValueError(
            f"unexpected Atari encoder flattened size {linear.in_features}; "
            f"expected {expected_flat_dim} for 84x84 input"
        )
    if linear.out_features != output_dim:
        raise ValueError(
            f"encoder attribute output_dim={output_dim} but final Linear outputs "
            f"{linear.out_features}"
        )

    stored_flat_dim = int(getattr(module, "flat_dim", linear.in_features))
    if stored_flat_dim != linear.in_features:
        raise ValueError(
            f"encoder flat_dim metadata={stored_flat_dim} disagrees with "
            f"Linear.in_features={linear.in_features}"
        )

    return {
        "format_version": format_version,
        "input_shape": input_shape,
        "output_dim": output_dim,
        "flat_dim": int(linear.in_features),
    }


def validate_shared_encoder(
    module: nn.Module,
    *,
    input_shape,
    output_dim: int,
    source: str = "encoder checkpoint",
):
    """Validate that a loaded encoder exactly matches the current Atari run."""
    info = inspect_shared_encoder(module)
    expected_shape = tuple(int(x) for x in input_shape)
    expected_dim = int(output_dim)

    if info["input_shape"] != expected_shape:
        raise ValueError(
            f"{source} expects input_shape={info['input_shape']}, "
            f"but this run has input_shape={expected_shape}"
        )
    if info["output_dim"] != expected_dim:
        raise ValueError(
            f"{source} outputs {info['output_dim']} features, "
            f"but this run expects {expected_dim}"
        )
    return module
