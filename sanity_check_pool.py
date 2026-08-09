"""
Fast standalone sanity check for the HeadPool/CkaRlAgent design (two
independent pools: mean_pool, logstd_pool, each covering l0+l2 together).
No MetaWorld, no GPU, no run_sac.py needed -- just torch/numpy. Should run in
well under a minute. Run this BEFORE spending real training time on the new
design: `python3 sanity_check_pool.py`

IMPORTANT: a task's own contribution only enters its pool via finalize(),
called once training is completely done (right before save()) -- NOT
automatically. Call order is always: [set_own_buffer() if using
distillation] -> finalize() -> save().

What it checks, for BOTH fusion_mode in ("classic_cka", "weight_delta") AND
BOTH distillation in (False, True):
  - Task 0 (root): constructs empty, finalize() gives it exactly 1 pool
    entry (itself), forward doesn't crash, save works.
  - Task 1 (base=root, latest=root): the degenerate case -- pools stay EMPTY
    at construction (not a wasted "v_0" entry), then finalize() gives 1.
  - Task 2 (base=root, latest=task1): 1 entry at construction (inherited
    from task1's already-finalized pool); after task2's OWN finalize, either
    2 entries (pool_size=2, no merge needed) or 1 (pool_size=1, merge
    triggers -- this is where distillation, if enabled, actually happens now).
  - Task 3 (base=root, latest=task2): inherits exactly what task2's finalize
    produced, unchanged (construction is a pure copy, no merging there
    anymore).
  - Round-trip save/load via CkaRlAgent.load() reproduces the same forward
    output.
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
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        x = torch.randn(8, OBS_DIM)
        mean, log_std = model(x)
        loss = (mean ** 2).sum() + (log_std ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()


def pool_lens(model):
    return model.mean_pool.pool_length(), model.logstd_pool.pool_length()


def run_chain(fusion_mode, distillation, pool_size):
    print(f"\n=== fusion_mode={fusion_mode}, distillation={distillation}, pool_size={pool_size} ===")
    root = f"{TMP_ROOT}/{fusion_mode}_{distillation}_p{pool_size}"
    if os.path.exists(root):
        shutil.rmtree(root)

    dirs = []

    m0 = CkaRlAgent(OBS_DIM, ACT_DIM, None, None, pool_size=pool_size,
                     distillation=distillation, fusion_mode=fusion_mode)
    train_a_bit(m0)
    assert pool_lens(m0) == (0, 0), f"root should have empty pools before finalize, got {pool_lens(m0)}"
    d0 = f"{root}/task0"
    if distillation:
        m0.set_own_buffer(make_fake_buffer())
    m0.finalize()  # required now: this is where a task's own contribution actually enters the pool
    assert pool_lens(m0) == (1, 1), f"root should have 1 entry each after finalize, got {pool_lens(m0)}"
    m0.save(d0)
    dirs.append(d0)
    print(f"  task0: pools after finalize={pool_lens(m0)} (expect (1,1)) OK")

    # degenerate case: base==latest -- must stay empty (at construction, before its own finalize)
    m1 = CkaRlAgent(OBS_DIM, ACT_DIM, dirs[0], dirs[0], pool_size=pool_size,
                     distillation=distillation, fusion_mode=fusion_mode)
    assert pool_lens(m1) == (0, 0), f"task1 (base==latest) should have empty pools, got {pool_lens(m1)}"
    train_a_bit(m1)
    d1 = f"{root}/task1"
    if distillation:
        m1.set_own_buffer(make_fake_buffer())
    m1.finalize()
    assert pool_lens(m1) == (1, 1), f"task1 should have 1 entry each after finalize, got {pool_lens(m1)}"
    m1.save(d1)
    dirs.append(d1)
    print(f"  task1 (degenerate base==latest): pools after finalize={pool_lens(m1)} (expect (1,1)) OK")

    m2 = CkaRlAgent(OBS_DIM, ACT_DIM, dirs[0], dirs[1], pool_size=pool_size,
                     distillation=distillation, fusion_mode=fusion_mode)
    assert pool_lens(m2) == (1, 1), f"task2 should have 1 entry each at construction, got {pool_lens(m2)}"
    train_a_bit(m2)
    d2 = f"{root}/task2"
    if distillation:
        m2.set_own_buffer(make_fake_buffer())
    m2.finalize()  # this is where the pool_size=1 merge (task1_entry + task2's own) actually triggers
    expected2 = (min(2, pool_size), min(2, pool_size))
    assert pool_lens(m2) == expected2, f"task2 after finalize expected {expected2}, got {pool_lens(m2)}"
    if distillation and pool_size < 2:
        metrics = m2.get_distill_metrics()
        assert metrics["mean/distill_test_mse"] is not None, \
            "expected task2's finalize (pool_size exceeded) to have used distillation -- no test MSE recorded"
        print(f"  distill metrics at task2 finalize OK: {metrics}")
    m2.save(d2)
    dirs.append(d2)
    print(f"  task2: pools after finalize={pool_lens(m2)} (expect {expected2}) OK")

    m3 = CkaRlAgent(OBS_DIM, ACT_DIM, dirs[0], dirs[2], pool_size=pool_size,
                     distillation=distillation, fusion_mode=fusion_mode)
    assert pool_lens(m3) == expected2, f"task3 at construction expected {expected2}, got {pool_lens(m3)}"
    print(f"  task3: pools at construction={pool_lens(m3)} (expect {expected2}) OK")

    x = torch.randn(4, OBS_DIM)
    mean, log_std = m3(x)
    assert mean.shape == (4, ACT_DIM) and log_std.shape == (4, ACT_DIM)
    print(f"  forward shapes OK: mean={tuple(mean.shape)}, log_std={tuple(log_std.shape)}")

    train_a_bit(m3)
    d3 = f"{root}/task3"
    if distillation:
        m3.set_own_buffer(make_fake_buffer())
    m3.finalize()
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


def check_alpha_mass_restricted_to_new_method():
    """use_alpha_mass must be rejected ONLY for the pure baseline
    (classic_cka + no distillation) -- the other 3 combinations (classic_cka
    +distillation, weight_delta+no-distillation, weight_delta+distillation)
    are all allowed."""
    print("\n=== alpha_mass restriction check ===")

    # ONLY this exact combination must raise
    raised = False
    try:
        CkaRlAgent(OBS_DIM, ACT_DIM, None, None, fusion_mode="classic_cka",
                   use_alpha_mass=True, distillation=False)
    except AssertionError:
        raised = True
    assert raised, "expected classic_cka+distillation=False+use_alpha_mass=True to raise, it didn't"
    print("  classic_cka + distillation=False + use_alpha_mass=True correctly raised OK")

    # the other 3 combinations must all be allowed
    allowed_combos = [
        ("classic_cka", True),
        ("weight_delta", False),
        ("weight_delta", True),
    ]
    for fusion_mode, distillation in allowed_combos:
        m = CkaRlAgent(OBS_DIM, ACT_DIM, None, None, fusion_mode=fusion_mode,
                        use_alpha_mass=True, distillation=distillation)
        x = torch.randn(2, OBS_DIM)
        m(x)  # just needs to not crash
        print(f"  {fusion_mode} + distillation={distillation} + use_alpha_mass=True: constructed & forward OK")


def check_fusion_mode_base_handling_in_distill():
    """Confirms _distill's fusion-mode branch does what it's supposed to:
    for classic_cka, optimization goes through base+v_k (so a different base
    must change what gets learned); for weight_delta, base is excluded
    entirely (matching _effective()), so changing base must NOT change the
    distilled result at all."""
    from fuse_module import HeadPool
    print("\n=== fusion_mode base-handling check ===")

    buf1 = make_fake_buffer()
    buf2 = make_fake_buffer()

    def make_pool(fusion_mode, base_scale):
        p = HeadPool("mean", shared_dim=256, hidden_dim=128, act_dim=ACT_DIM,
                      fusion_mode=fusion_mode, pool_size=1, distillation=True)
        if base_scale > 0:
            with torch.no_grad():
                p.base_l0_weight.normal_(0, base_scale)
                p.base_l0_bias.normal_(0, base_scale)
                p.base_l2_weight.normal_(0, base_scale)
                p.base_l2_bias.normal_(0, base_scale)
        return p

    # weight_delta: base must NOT affect the result
    torch.manual_seed(3)
    result_a = make_pool("weight_delta", 0.0)._distill(buf1, buf2, epochs=2)
    torch.manual_seed(3)
    result_b = make_pool("weight_delta", 5.0)._distill(buf1, buf2, epochs=2)
    for t_a, t_b in zip(result_a, result_b):
        assert torch.allclose(t_a, t_b, atol=1e-5), \
            "weight_delta distillation result changed when base changed -- base should be excluded entirely"
    print("  weight_delta: base correctly excluded from distillation OK")

    # classic_cka: base MUST affect the result
    torch.manual_seed(3)
    result_c = make_pool("classic_cka", 0.0)._distill(buf1, buf2, epochs=2)
    torch.manual_seed(3)
    result_d = make_pool("classic_cka", 5.0)._distill(buf1, buf2, epochs=2)
    differs = any(not torch.allclose(t_c, t_d, atol=1e-5) for t_c, t_d in zip(result_c, result_d))
    assert differs, "classic_cka distillation result did NOT change when base changed -- base should matter"
    print("  classic_cka: base correctly affects distillation result OK")


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

    for fusion_mode in ("classic_cka", "weight_delta"):
        for distillation in (False, True):
            try:
                run_chain(fusion_mode, distillation, pool_size=1)
            except AssertionError as e:
                ok = False
                print(f"  FAILED: {e}")

    try:
        check_alpha_mass_restricted_to_new_method()
    except AssertionError as e:
        ok = False
        print(f"  FAILED: {e}")

    try:
        check_fusion_mode_base_handling_in_distill()
    except AssertionError as e:
        ok = False
        print(f"  FAILED: {e}")

    print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED -- see above ***"))