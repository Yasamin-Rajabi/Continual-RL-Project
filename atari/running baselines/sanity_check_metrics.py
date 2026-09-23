"""CPU-only checks for repeat-safe metric bookkeeping and formulas."""
from __future__ import annotations

import math
import tempfile
from types import SimpleNamespace

from benchmark_protocol import checkpoint_dir, task_slot_map
from metrics import compute_A_N, compute_fg_bwt, _first_unseen_positions


def main():
    sequence = [0, 1, 2, 0, 1, 3]
    slots = task_slot_map(sequence)
    assert slots == {0: 0, 1: 1, 2: 2, 3: 3}

    p0 = checkpoint_dir("agents", "Freeway", "main", "FT-N", 1, 0, 0)
    p3 = checkpoint_dir("agents", "Freeway", "main", "FT-N", 1, 3, 0)
    assert p0 != p3, "repeated semantic tasks must not overwrite checkpoints"

    assert _first_unseen_positions(sequence) == [(1, 1), (2, 2), (5, 3)]

    diagonal = {
        "0": 0.8,
        "1": 0.7,
        "2": 0.6,
        "3": 0.9,
        "4": 0.75,
        "5": 0.5,
    }
    final = {"0": 0.7, "1": 0.8, "2": 0.4, "3": 0.5}
    result = compute_fg_bwt(diagonal, final, sequence)
    expected_fg = (0.1 + 0.0 + 0.2 + 0.2 + 0.0) / 5
    expected_bwt = (-0.1 + 0.1 - 0.2 - 0.2 + 0.05) / 5
    assert abs(result["FG"] - expected_fg) < 1e-12
    assert abs(result["BWT"] - expected_bwt) < 1e-12
    assert abs(compute_A_N(final) - 0.6) < 1e-12

    print("*** METRIC BOOKKEEPING CHECKS PASSED ***")


if __name__ == "__main__":
    main()
