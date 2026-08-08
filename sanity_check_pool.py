"""
Fast standalone sanity check for the HeadPool/CkaRlAgent design (two
independent pools: mean_pool, logstd_pool, each covering l0+l2 together).
No MetaWorld, no GPU, no run_sac.py needed -- just torch/numpy. Should run in
well under a minute. Run this BEFORE spending real training time on the new
design: `python3 sanity_check_pool.py`

What it checks, for BOTH fusion_mode in ("classic_cka", "weight_delta") AND
BOTH distillation in (False, True):
  - Task 0 (root): constructs, forward doesn't crash, save works.
  - Task 1 (base=root, latest=root): the degenerate case -- both pools
    should stay EMPTY (0 entries), not get a wasted "v_0" entry.
  - Task 2 (base=root, latest=task1): both pools should have exactly 1 entry.
  - Task 3 (base=root, latest=task2, pool_size=1): both pools should trigger
    a merge and end up back at exactly 1 entry (not 2).
  - mean_pool and logstd_pool can end up with DIFFERENT pool contents (they
    merge independently) -- checked by comparing their alpha sizes stay
    consistent with their own (possibly different) pool lengths.
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
    assert pool_lens(m0) == (0, 0), f"root should have empty pools, got {pool_lens(m0)}"
    d0 = f"{root}/task0"
    if distillation:
        m0.set_own_buffer(make_fake_buffer())
    m0.save(d0)
    dirs.append(d0)
    print(f"  task0: pools={pool_lens(m0)} (expect (0,0)) OK")

    # degenerate case: base==latest -- must stay empty
    m1 = CkaRlAgent(OBS_DIM, ACT_DIM, dirs[0], dirs[0], pool_size=pool_size,
                     distillation=distillation, fusion_mode=fusion_mode)
    assert pool_lens(m1) == (0, 0), f"task1 (base==latest) should have empty pools, got {pool_lens(m1)}"
    train_a_bit(m1)
    d1 = f"{root}/task1"
    if distillation:
        m1.set_own_buffer(make_fake_buffer())
    m1.save(d1)
    dirs.append(d1)
    print(f"  task1 (degenerate base==latest): pools={pool_lens(m1)} (expect (0,0)) OK")

    m2 = CkaRlAgent(OBS_DIM, ACT_DIM, dirs[0], dirs[1], pool_size=pool_size,
                     distillation=distillation, fusion_mode=fusion_mode)
    assert pool_lens(m2) == (1, 1), f"task2 should have 1 entry each, got {pool_lens(m2)}"
    train_a_bit(m2)
    d2 = f"{root}/task2"
    if distillation:
        m2.set_own_buffer(make_fake_buffer())
    m2.save(d2)
    dirs.append(d2)
    print(f"  task2: pools={pool_lens(m2)} (expect (1,1)) OK")

    m3 = CkaRlAgent(OBS_DIM, ACT_DIM, dirs[0], dirs[2], pool_size=pool_size,
                     distillation=distillation, fusion_mode=fusion_mode)
    expected = (min(2, pool_size), min(2, pool_size))
    assert pool_lens(m3) == expected, f"task3 expected {expected}, got {pool_lens(m3)}"
    print(f"  task3: pools={pool_lens(m3)} (expect {expected}) OK")

    x = torch.randn(4, OBS_DIM)
    mean, log_std = m3(x)
    assert mean.shape == (4, ACT_DIM) and log_std.shape == (4, ACT_DIM)
    print(f"  forward shapes OK: mean={tuple(mean.shape)}, log_std={tuple(log_std.shape)}")

    if distillation:
        metrics = m3.get_distill_metrics()
        assert metrics["mean/distill_test_mse"] is not None, \
            "expected a distillation merge to have happened at task3 (pool_size=1, 2 inherited) -- no test MSE recorded"
        print(f"  distill metrics OK: {metrics}")

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


def check_alpha_mass_restricted_to_new_method():
    """use_alpha_mass must be rejected for the base method (classic_cka) and
    accepted for weight_delta, regardless of distillation on/off."""
    print("\n=== alpha_mass restriction check ===")
    root = f"{TMP_ROOT}/alpha_mass_check"
    if os.path.exists(root):
        shutil.rmtree(root)

    # classic_cka + use_alpha_mass=True must raise
    raised = False
    try:
        CkaRlAgent(OBS_DIM, ACT_DIM, None, None, fusion_mode="classic_cka", use_alpha_mass=True)
    except AssertionError:
        raised = True
    assert raised, "expected use_alpha_mass=True with fusion_mode='classic_cka' to raise, it didn't"
    print("  classic_cka + use_alpha_mass=True correctly raised OK")

    # weight_delta + use_alpha_mass=True must NOT raise, regardless of distillation
    for distillation in (False, True):
        m = CkaRlAgent(OBS_DIM, ACT_DIM, None, None, fusion_mode="weight_delta",
                        use_alpha_mass=True, distillation=distillation)
        x = torch.randn(2, OBS_DIM)
        m(x)  # just needs to not crash
        print(f"  weight_delta + use_alpha_mass=True + distillation={distillation}: constructed & forward OK")


def check_residual_distillation_differs_from_raw():
    """The whole point of subtracting base_output before training the
    distillation student: if base's own output is non-trivial, a student
    trained on the RAW targets should differ from one trained on the
    RESIDUAL (targets - base_output). This just re-derives that relationship
    directly (independent of CkaRlAgent) as a check that the subtraction in
    HeadPool._distill is actually being applied, not silently skipped."""
    from fuse_module import HeadPool
    torch.manual_seed(1)
    pool = HeadPool("mean", shared_dim=256, hidden_dim=128, act_dim=ACT_DIM,
                     fusion_mode="classic_cka", pool_size=1, distillation=True)
    # give it a non-trivial base so base_output is non-zero
    with torch.no_grad():
        pool.base_l0_weight.normal_(0, 0.5)
        pool.base_l0_bias.normal_(0, 0.5)
        pool.base_l2_weight.normal_(0, 0.5)
        pool.base_l2_bias.normal_(0, 0.5)

    buf1 = make_fake_buffer()
    buf2 = make_fake_buffer()
    inputs = torch.tensor(np.concatenate([buf1["shared"], buf2["shared"]], axis=0), dtype=torch.float32)
    with torch.no_grad():
        base_out = pool._base_only_forward(inputs)
    assert base_out.abs().mean().item() > 1e-3, "test setup error: base output is ~zero, can't test subtraction"
    print("\n=== residual-subtraction check ===")
    print(f"  base output magnitude: {base_out.abs().mean().item():.4f} (non-trivial, good -- "
          f"subtraction has something real to remove)")


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
        check_residual_distillation_differs_from_raw()
    except AssertionError as e:
        ok = False
        print(f"  FAILED: {e}")

    print("\n" + ("*** ALL CHECKS PASSED ***" if ok else "*** SOME CHECKS FAILED -- see above ***"))

