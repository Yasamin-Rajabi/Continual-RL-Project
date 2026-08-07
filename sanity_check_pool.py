"""
Fast standalone sanity check for the SeparatePool/DistillPool/CkaRlAgent
design. No MetaWorld, no GPU, no run_sac.py needed -- just torch/numpy.
Should run in well under a minute. Run this BEFORE spending real training
time on the new design: `python3 sanity_check_pool.py`

What it checks, for BOTH fusion_mode in ("classic_cka", "weight_delta") AND
BOTH distillation in (False, True) -- i.e. all 4 Mode1-4 combinations,
exercising both SeparatePoolHead (distillation=False) and DistillPool
(distillation=True):
  - Task 0 (root): constructs, forward doesn't crash, save works.
  - Task 1 (base=root, latest=root): the degenerate case -- pool should stay
    EMPTY (0 entries), not get a wasted "v_0" entry. This is the specific bug
    that got caught and fixed during design.
  - Task 2 (base=root, latest=task1): pool should have exactly 1 entry.
  - Task 3 (base=root, latest=task2, pool_size=1): pool should trigger a
    merge and end up back at exactly 1 entry (not 2).
  - Round-trip save/load via CkaRlAgent.load() reproduces the same forward
    output (sanity check that nothing is silently dropped by pickling).
"""
import os
import shutil
import numpy as np
import torch

from cka_rl import CkaRlAgent

OBS_DIM = 6
ACT_DIM = 3
TMP_ROOT = "/tmp/pool_sanity_check"


def make_fake_buffer(n=200, shared_dim=256, act_dim=ACT_DIM):
    return {
        "obs": np.random.randn(n, OBS_DIM).astype(np.float32),
        "shared": np.random.randn(n, shared_dim).astype(np.float32),
        "targets": np.random.randn(n, 2 * act_dim).astype(np.float32),
    }


def train_a_bit(model, steps=5):
    """Fake 'training': a few gradient steps on a random regression target,
    just to make own weights move away from their random init so merges/loads
    are exercised on non-trivial values."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        x = torch.randn(8, OBS_DIM)
        mean, log_std = model(x)
        loss = (mean ** 2).sum() + (log_std ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()


def pool_len(model):
    return model.pool.pool_length()


def run_chain(fusion_mode, distillation, pool_size):
    print(f"\n=== fusion_mode={fusion_mode}, distillation={distillation}, pool_size={pool_size} "
          f"({'DistillPool' if distillation else 'SeparatePoolHead'}) ===")
    root = f"{TMP_ROOT}/{fusion_mode}_{distillation}_p{pool_size}"
    if os.path.exists(root):
        shutil.rmtree(root)

    dirs = []

    # Task 0: root, no base/latest
    m0 = CkaRlAgent(OBS_DIM, ACT_DIM, None, None, pool_size=pool_size,
                     distillation=distillation, fusion_mode=fusion_mode)
    train_a_bit(m0)
    assert pool_len(m0) == 0, f"root should have empty pool, got {pool_len(m0)}"
    d0 = f"{root}/task0"
    if distillation:
        m0.set_own_buffer(make_fake_buffer())
    m0.save(d0)
    dirs.append(d0)
    print(f"  task0: pool={pool_len(m0)} (expect 0) OK")

    # Task 1: base=root, latest=root (degenerate case -- must stay empty)
    m1 = CkaRlAgent(OBS_DIM, ACT_DIM, dirs[0], dirs[0], pool_size=pool_size,
                     distillation=distillation, fusion_mode=fusion_mode)
    assert pool_len(m1) == 0, f"task1 (base==latest) should have empty pool, got {pool_len(m1)}"
    train_a_bit(m1)
    d1 = f"{root}/task1"
    if distillation:
        m1.set_own_buffer(make_fake_buffer())
    m1.save(d1)
    dirs.append(d1)
    print(f"  task1 (degenerate base==latest): pool={pool_len(m1)} (expect 0) OK")

    # Task 2: base=root, latest=task1 -- should now have exactly 1 entry
    m2 = CkaRlAgent(OBS_DIM, ACT_DIM, dirs[0], dirs[1], pool_size=pool_size,
                     distillation=distillation, fusion_mode=fusion_mode)
    assert pool_len(m2) == 1, f"task2 should have 1 pool entry, got {pool_len(m2)}"
    train_a_bit(m2)
    d2 = f"{root}/task2"
    if distillation:
        m2.set_own_buffer(make_fake_buffer())
    m2.save(d2)
    dirs.append(d2)
    print(f"  task2: pool={pool_len(m2)} (expect 1) OK")

    # Task 3: base=root, latest=task2 -- inherits 2 entries; if pool_size=1,
    # a merge must trigger and bring it back down to 1.
    m3 = CkaRlAgent(OBS_DIM, ACT_DIM, dirs[0], dirs[2], pool_size=pool_size,
                     distillation=distillation, fusion_mode=fusion_mode)
    expected3 = min(2, pool_size)
    assert pool_len(m3) == expected3, \
        f"task3 should have {expected3} pool entries (pool_size={pool_size}), got {pool_len(m3)}"
    print(f"  task3: pool={pool_len(m3)} (expect {expected3}) OK")

    # forward doesn't crash
    x = torch.randn(4, OBS_DIM)
    mean, log_std = m3(x)
    assert mean.shape == (4, ACT_DIM) and log_std.shape == (4, ACT_DIM)
    print(f"  forward shapes OK: mean={tuple(mean.shape)}, log_std={tuple(log_std.shape)}")

    # save/load round-trip preserves forward output exactly
    d3 = f"{root}/task3"
    m3.save(d3)
    m3_reloaded = CkaRlAgent.load(d3, OBS_DIM, ACT_DIM)
    m3_reloaded.eval()
    m3.eval()
    with torch.no_grad():
        mean_a, _ = m3(x)
        mean_b, _ = m3_reloaded(x)
    assert torch.allclose(mean_a, mean_b, atol=1e-6), "save/load round-trip changed forward output!"
    print("  save/load round-trip: forward output identical OK")

    shutil.rmtree(root)
    return True


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    ok = True
    for fusion_mode in ("classic_cka", "weight_delta"):
        for distillation in (False, True):
            try:
                run_chain(fusion_mode, distillation, pool_size=2)
            except AssertionError as e:
                ok = False
                print(f"  FAILED: {e}")

    # pool_size=1 forces a merge to trigger at task3 (2 inherited > 1) -- extra check
    for fusion_mode in ("classic_cka", "weight_delta"):
        for distillation in (False, True):
            try:
                run_chain(fusion_mode, distillation, pool_size=1)
            except AssertionError as e:
                ok = False
                print(f"  FAILED: {e}")

    print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED -- see above ***"))

