"""Continual-RL baselines for the PointMaze benchmark."""
from __future__ import annotations

METHODS = (
    "FT-N",
    "ProgNet",
    "PackNet",
    "MaskNet",
    "CReLUs",
    "CompoNet",
    "CbpNet",
    "CKA-RL",
)

# Our method's two reported configurations.  "Ours" is condition 4 (combined)
# in policy-composition space; "Ours-parameter" is the same condition in
# parameter space and exists so that the composition-space axis can be read
# off directly rather than inferred.
OURS_METHODS = ("Ours", "Ours-parameter")

ALL_METHODS = METHODS + OURS_METHODS

# Methods whose past tasks are served by a dedicated frozen module, so an old
# task must be evaluated with that task's module rather than the final one.
TASK_MODULE_METHODS = frozenset({"ProgNet", "CompoNet"})

# Methods that continue from a single latest checkpoint.
LATEST_ONLY_METHODS = frozenset({"FT-N", "PackNet", "MaskNet", "CReLUs", "CbpNet"})

# Methods that need the full list of prior checkpoints when constructed.
ALL_PREVIOUS_METHODS = frozenset({"ProgNet", "CompoNet", "CKA-RL", "Ours", "Ours-parameter"})

# Methods that are handed the task index by construction.
TASK_CONDITIONED_METHODS = frozenset(
    {"FT-N", "PackNet", "MaskNet", "ProgNet", "CompoNet"}
)

_ALIASES = {
    "Finetune": "FT-N",
    "FTN": "FT-N",
    "ft_n": "FT-N",
    "ft-n": "FT-N",
    "FT-N": "FT-N",
    "prognet": "ProgNet",
    "ProgressiveNet": "ProgNet",
    "ProgNet": "ProgNet",
    "packnet": "PackNet",
    "PackNet": "PackNet",
    "masknet": "MaskNet",
    "MaskNet": "MaskNet",
    "crelus": "CReLUs",
    "CReLUs": "CReLUs",
    "componet": "CompoNet",
    "CompoNet": "CompoNet",
    "cbpnet": "CbpNet",
    "CbpNet": "CbpNet",
    "cka": "CKA-RL",
    "cka_rl": "CKA-RL",
    "CKA-RL": "CKA-RL",
    "ours": "Ours",
    "Ours": "Ours",
    "ours_combined": "Ours",
    "ours_parameter": "Ours-parameter",
    "Ours-parameter": "Ours-parameter",
}


def canonical_method(name: str) -> str:
    try:
        return _ALIASES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown method {name!r}; valid names: {sorted(set(_ALIASES.values()))}"
        ) from exc


def is_ours(name: str) -> bool:
    return canonical_method(name) in OURS_METHODS
