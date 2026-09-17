"""Shared infrastructure for every continual-RL baseline.

Four modules, each with one job:

lifecycle.py  The ``ContinualAgent`` protocol every baseline implements, and
              the ``TaskContext`` record describing one position in the
              sequence. This is the entire surface a baseline has to fill in.
sac_core.py   The SAC training loop, lifted from run_sac.py and shared by all
              four baselines so environment, budget and evaluation parity is
              structural rather than promised.
masks.py      Magnitude pruning, binary mask bookkeeping and task-conditioned
              gating. Shared by PackNet and MaskNet.
snapshot.py   Reads and writes the ``policy_snapshot.pt`` contract that
              ``cka_rl.FrozenCkaPolicy`` already understands, so baseline
              checkpoints evaluate through the unmodified
              ``checkpoint_evaluation.evaluate(..., frozen_policy="snapshot")``.
"""
