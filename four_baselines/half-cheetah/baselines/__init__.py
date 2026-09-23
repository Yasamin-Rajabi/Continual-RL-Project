"""Continual-RL baselines evaluated against the CKA-RL method.

This package is BYTE-IDENTICAL between the HalfCheetah and Meta-World suite
folders. Everything suite-specific is reached through three module-level
constants that ``metrics.py`` already defines per folder (``ERROR_KEY``,
``EPISODIC_SUCCESS``, ``RETURN_UPPER_BOUND``) and through ``tasks.get_task``,
so no baseline file ever names an environment.

WHY THIS PACKAGE EXISTS SEPARATELY FROM run_sac.py
--------------------------------------------------
``experiment_identity.SOURCE_CANDIDATES`` fingerprints run_sac.py, cka_rl.py,
policy_composition.py, policy_space.py, training_protocol.py,
knowledge_pools.py, shared_arch.py, policy_utils.py, tasks.py,
<suite>_envs.py and analysis_logging.py. Editing ANY of them invalidates every
CKA-RL checkpoint already trained and forces a full retrain. The baselines
therefore live in new files and IMPORT those modules without modifying them.

THE FOUR BASELINES
------------------
ft_n     Sequential fine-tuning across N tasks. Task-blind, no capacity
         isolation, no regularization. The naive lower bound.
prognet  Progressive Neural Networks. One frozen actor column per task with
         lateral adapters from all previous columns. Shared critic.
packnet  Iterative magnitude pruning with parameter isolation and binary
         masks. Prune/retrain is carved out of the same (Delta - B) budget.
masknet  Task-conditioned gating over a single shared backbone.

Per the approved experiment design, ProgNet, PackNet and MaskNet receive an
explicit task oracle at both train and eval time and are reported as
Task-Aware Upper Bounds. FT-N stays task-blind, as standard naive fine-tuning.
Recurring task ids REUSE the capacity slice allocated on first encounter
(column, mask or gate) rather than allocating a new one, so the second pass
through a sequence measures retention rather than fresh capacity.
"""
from __future__ import annotations

# Populated at the bottom of this module. Keys are the --method CLI values and
# the condition labels that appear in output paths and plot legends.
REGISTRY: dict[str, type] = {}

TASK_AWARE_METHODS = ("prognet", "packnet", "masknet")
TASK_BLIND_METHODS = ("ft_n",)


def get(method: str):
    """Return the agent class registered under ``method``."""
    if method not in REGISTRY:
        raise KeyError(
            f"unknown baseline {method!r}; available: {sorted(REGISTRY)}"
        )
    return REGISTRY[method]


def available() -> tuple[str, ...]:
    return tuple(sorted(REGISTRY))


def is_task_aware(method: str) -> bool:
    """Whether this baseline consumes the task oracle.

    Used by run_baseline.py to reject a task-blind method that was handed a
    task id, and by the manifest so a run can never be silently reinterpreted.
    """
    if method not in REGISTRY:
        raise KeyError(f"unknown baseline {method!r}")
    return method in TASK_AWARE_METHODS


def _register():
    """Import the baseline modules and populate REGISTRY.

    Deferred to the bottom of the module so each baseline can import
    ``baselines.common`` without a circular import at package-init time.
    """
    from baselines.ft_n import FineTuneAgent

    REGISTRY["ft_n"] = FineTuneAgent

    # The remaining three are registered as they are implemented; a missing
    # module must not break the baselines that do exist.
    try:
        from baselines.prognet import ProgNetAgent

        REGISTRY["prognet"] = ProgNetAgent
    except ImportError:
        pass
    try:
        from baselines.packnet import PackNetAgent

        REGISTRY["packnet"] = PackNetAgent
    except ImportError:
        pass
    try:
        from baselines.masknet import MaskNetAgent

        REGISTRY["masknet"] = MaskNetAgent
    except ImportError:
        pass


_register()
