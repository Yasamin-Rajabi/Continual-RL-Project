"""Fast CPU-only sanity checks for the Atari categorical CKA-RL pool.

No Gymnasium/ALE installation is required.  The tests exercise the shared CNN,
root/continuation lifecycle, bounded merging, categorical behavioral KL,
distillation, snapshot loading, alpha-mass, and buffer lineage.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import torch

from cka_rl import CkaRlAgent, FrozenCkaPolicy, categorical_kl
from knowledge_pools import HeadPool

OBS_SHAPE = (4, 84, 84)
ACT_DIM = 4
SHARED_DIM = 32
HIDDEN_DIM = 16


def fake_buffer(n=16, task_id=0, source_id=None):
    if source_id is None:
        source_id = task_id
    return {
        "obs": np.random.randint(0, 256, size=(n,) + OBS_SHAPE, dtype=np.uint8),
        "actions": np.random.randint(0, ACT_DIM, size=n, dtype=np.int16),
        "task_ids": np.full(n, task_id, dtype=np.int16),
        "source_ids": np.full(n, source_id, dtype=np.int16),
    }


def make_agent(**kwargs):
    defaults = dict(
        obs_shape=OBS_SHAPE,
        act_dim=ACT_DIM,
        pool_size=2,
        shared_dim=SHARED_DIM,
        hidden_dim=HIDDEN_DIM,
        similarity_samples=8,
        distill_max_samples=8,
        distill_epochs=1,
        distill_batch_size=4,
        max_distill_buffer=16,
    )
    defaults.update(kwargs)
    return CkaRlAgent(**defaults)


def train_a_bit(model, steps=1):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)
    for _ in range(steps):
        x = torch.rand(2, *OBS_SHAPE)
        action, logprob, entropy, value = model.get_action_and_value(x)
        # Synthetic differentiable objective; the goal is only to move weights.
        loss = -(logprob.mean() + 0.01 * entropy.mean()) + 0.1 * value.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def save_root(path, fusion_mode, distillation, use_alpha_mass=False):
    model = make_agent(
        base_dir=None,
        latest_dir=None,
        fusion_mode=fusion_mode,
        distillation=distillation,
        use_alpha_mass=use_alpha_mass,
    )
    train_a_bit(model)
    model.set_own_buffer(fake_buffer(task_id=0, source_id=0))

    probe = torch.rand(2, *OBS_SHAPE)
    with torch.no_grad():
        pre_logits = model(probe)
        features = model.fc(probe)
    model.save_policy_snapshot(path)
    model.set_base()
    assert model.policy_pool.pool_length() == 1
    with torch.no_grad():
        entry_logits = model.policy_pool.forward_entry(features, 0)
    assert torch.allclose(pre_logits, entry_logits, atol=1e-5), (
        "root standalone pool entry must represent the trained root policy"
    )
    if fusion_mode == "classic_cka":
        entry = model.policy_pool.pool[0]
        assert torch.count_nonzero(entry["l0_weight"]) == 0
        assert torch.count_nonzero(entry["l2_weight"]) == 0
    model.save(path)
    return model


def run_chain(fusion_mode, distillation, use_alpha_mass):
    print(
        f"\n=== chain fusion={fusion_mode} distillation={distillation} "
        f"alpha_mass={use_alpha_mass} ==="
    )
    root = tempfile.mkdtemp(prefix="cka_atari_pool_")
    try:
        d0 = os.path.join(root, "task0")
        save_root(d0, fusion_mode, distillation, use_alpha_mass)

        m1 = make_agent(
            base_dir=d0,
            latest_dir=d0,
            fusion_mode=fusion_mode,
            distillation=distillation,
            use_alpha_mass=use_alpha_mass,
            encoder_from_base=True,
        )
        assert m1.policy_pool.pool_length() == 1
        assert m1.alpha is m1.policy_pool.alpha
        assert m1.alpha.numel() == 1
        assert not any(p.requires_grad for p in m1.fc.parameters())
        train_a_bit(m1)
        m1.set_own_buffer(fake_buffer(task_id=1, source_id=1))
        m1.finalize()
        assert m1.policy_pool.pool_length() == 2
        assert m1.get_merge_info() is None
        d1 = os.path.join(root, "task1")
        m1.save(d1)

        m2 = make_agent(
            base_dir=d0,
            latest_dir=d1,
            fusion_mode=fusion_mode,
            distillation=distillation,
            use_alpha_mass=use_alpha_mass,
            encoder_from_base=True,
            distill_test_frac=0.25,
        )
        assert m2.policy_pool.pool_length() == 2
        assert m2.alpha.numel() == 2
        train_a_bit(m2)
        m2.set_own_buffer(fake_buffer(task_id=2, source_id=2))

        probe = torch.rand(2, *OBS_SHAPE)
        with torch.no_grad():
            expected = m2(probe)
        snapshot_dir = os.path.join(root, "snapshot")
        m2.save_policy_snapshot(snapshot_dir)
        frozen = FrozenCkaPolicy.load(snapshot_dir)
        with torch.no_grad():
            got = frozen(probe)
        assert torch.allclose(expected, got, atol=1e-5)

        m2.finalize()
        assert m2.policy_pool.pool_length() == 2
        info = m2.get_merge_info()
        assert info is not None
        assert info["pool_size_before"] == 3
        assert info["pool_size_after"] == 2
        assert info["idx1"] != info["idx2"]
        assert info["merged_source_lineage"]
        if distillation:
            assert info["similarity_metric"] == "symmetric_kl"
            assert np.isfinite(info["symmetric_kl"])
            metrics = m2.get_distill_metrics()
            assert np.isfinite(metrics["policy/distill_train_kl"])
            assert metrics["policy/distill_selected_epoch"] >= 0
        else:
            assert info["similarity_metric"] == "cosine"
            assert np.isfinite(info["cosine_similarity"])
            assert m2.get_distill_metrics() == {}

        d2 = os.path.join(root, "task2")
        m2.save(d2)
        m3 = make_agent(
            base_dir=d0,
            latest_dir=d2,
            fusion_mode=fusion_mode,
            distillation=distillation,
            use_alpha_mass=use_alpha_mass,
            encoder_from_base=True,
        )
        assert m3.policy_pool.pool_length() == 2
        assert m3.alpha.numel() == 2
        logits = m3(torch.rand(2, *OBS_SHAPE))
        assert logits.shape == (2, ACT_DIM)
        print("  root -> inheritance -> bounded merge -> reload OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_behavioral_pair_is_output_space():
    print("\n=== behavioral pair selection is policy-output based ===")
    m = make_agent(pool_size=3, distillation=True, similarity_samples=8)
    with torch.no_grad():
        m.policy_pool.base_l0_weight.zero_()
        m.policy_pool.base_l0_bias.zero_()
        m.policy_pool.base_l2_weight.zero_()
        m.policy_pool.base_l2_bias.zero_()
    m.policy_pool.pool = []

    for i in range(3):
        entry = {
            "l0_weight": torch.randn_like(m.policy_pool.own_l0_weight) * (20.0 if i == 1 else 1.0),
            "l0_bias": torch.randn_like(m.policy_pool.own_l0_bias),
            "l2_weight": torch.zeros_like(m.policy_pool.own_l2_weight),
            "l2_bias": (
                torch.zeros_like(m.policy_pool.own_l2_bias)
                if i < 2
                else torch.tensor([3.0, 0.0, 0.0, 0.0])
            ),
            "buffer": fake_buffer(8, task_id=i, source_id=i),
        }
        m.policy_pool.pool.append(entry)

    i, j, info = m._select_behavioral_pair()
    assert {i, j} == {0, 1}, f"expected identical-output pair 0/1, got {i}/{j}"
    assert abs(info["symmetric_kl"]) < 1e-7
    print("  identical categorical behavior wins despite very different hidden weights OK")


def check_categorical_kl():
    print("\n=== categorical KL numeric check ===")
    p = torch.randn(7, ACT_DIM)
    q = torch.randn(7, ACT_DIM)
    ours = categorical_kl(p, q)
    ref = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=p),
        torch.distributions.Categorical(logits=q),
    )
    assert torch.allclose(ours, ref, atol=1e-6)
    print("  categorical_kl matches torch.distributions OK")


def check_alpha_mass():
    print("\n=== alpha-mass restriction ===")
    try:
        make_agent(fusion_mode="classic_cka", use_alpha_mass=True)
    except ValueError:
        pass
    else:
        raise AssertionError("classic_cka must reject use_alpha_mass")

    root = tempfile.mkdtemp(prefix="cka_atari_mass_")
    try:
        d0 = os.path.join(root, "task0")
        save_root(d0, "weight_delta", False, True)
        model = make_agent(
            base_dir=d0,
            latest_dir=d0,
            fusion_mode="weight_delta",
            distillation=False,
            use_alpha_mass=True,
            constrain_alpha_mass=True,
        )
        with torch.no_grad():
            model.alpha_mass.fill_(-10.0)
        assert model.policy_pool.effective_alpha_mass().item() > 0.0
        print("  constrained mass stays positive OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_buffer_bound():
    print("\n=== bounded buffer merge ===")
    a = fake_buffer(20, 0, 0)
    b = fake_buffer(3, 1, 1)
    merged = HeadPool.merge_buffers(a, b, max_rows=7)
    assert len(merged["obs"]) == 7
    assert all(len(v) == 7 for v in merged.values() if isinstance(v, np.ndarray))
    only = HeadPool.merge_buffers(a, None, max_rows=5)
    assert len(only["obs"]) == 5
    print("  two-parent and single-parent merges respect hard row budget OK")


def check_trainable_encoder_loads_latest():
    print("\n=== train_shared continuation uses latest CNN ===")
    root = tempfile.mkdtemp(prefix="cka_atari_encoder_")
    try:
        d0, d1 = os.path.join(root, "task0"), os.path.join(root, "task1")
        save_root(d0, "classic_cka", False, False)
        m1 = make_agent(
            base_dir=d0,
            latest_dir=d0,
            fusion_mode="classic_cka",
            distillation=False,
            encoder_from_base=True,
            train_shared=True,
        )
        with torch.no_grad():
            first = next(m1.fc.parameters())
            first.add_(0.01)
        m1.save(d1)
        expected = next(m1.fc.parameters()).detach().clone()
        m2 = make_agent(
            base_dir=d0,
            latest_dir=d1,
            fusion_mode="classic_cka",
            distillation=False,
            encoder_from_base=True,
            train_shared=True,
        )
        got = next(m2.fc.parameters()).detach()
        assert torch.allclose(got, expected)
        assert all(p.requires_grad for p in m2.fc.parameters())
        print("  latest trainable CNN is continued, not reset to root OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    check_categorical_kl()
    check_buffer_bound()
    check_behavioral_pair_is_output_space()
    check_alpha_mass()
    check_trainable_encoder_loads_latest()
    run_chain("classic_cka", False, False)
    run_chain("classic_cka", True, False)
    run_chain("weight_delta", False, True)
    run_chain("weight_delta", True, True)
    print("\nAll Atari categorical CKA-RL sanity checks passed.")


if __name__ == "__main__":
    main()
