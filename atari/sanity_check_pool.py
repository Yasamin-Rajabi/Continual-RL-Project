"""Fast CPU-only sanity checks for the Atari categorical CKA-RL stack.

No Gymnasium/ALE installation is required. The tests cover the shared continual
learning semantics that also exist in the HalfCheetah implementation, adapted
only where Atari is inherently different: CNN observations, categorical
policies, and a task-local PPO value head.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from cka_rl import CkaRlAgent, FrozenCkaPolicy
from knowledge_pools import HeadPool, balanced_lineage_indices
from policy_composition import categorical_mixture_probs
from policy_utils import categorical_kl
from shared_arch import shared
from training_protocol import TaskBudget, bounded_buffer, mixture_warmup_active

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
        "task_ids": np.full(n, task_id, dtype=np.int32),
        "source_ids": np.full(n, source_id, dtype=np.int32),
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
        loss = -(logprob.mean() + 0.01 * entropy.mean()) + 0.1 * value.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def policy_probs(model, obs):
    with torch.no_grad():
        return model.action_distribution(obs).probs


def save_root(
    path,
    fusion_mode,
    distillation,
    use_alpha_mass=False,
    composition_space="parameter",
    policy_student_replay=False,
):
    model = make_agent(
        base_dir=None,
        latest_dir=None,
        fusion_mode=fusion_mode,
        distillation=distillation,
        use_alpha_mass=use_alpha_mass,
        composition_space=composition_space,
        policy_student_replay=policy_student_replay,
    )
    train_a_bit(model)
    model.set_own_buffer(fake_buffer(task_id=0, source_id=0))

    probe = torch.rand(2, *OBS_SHAPE)
    expected_probs = policy_probs(model, probe)
    model.save_policy_snapshot(path)

    frozen = FrozenCkaPolicy.load(path)
    with torch.no_grad():
        frozen_probs = frozen.action_distribution(probe).probs
    assert torch.allclose(expected_probs, frozen_probs, atol=1e-6), (
        "pre-finalize snapshot must preserve the exact categorical behavior"
    )

    model.set_base()
    assert model.policy_pool.pool_length() == 1

    if composition_space == "parameter":
        with torch.no_grad():
            features = model.fc(probe)
            entry_logits = model.policy_pool.forward_entry(features, 0)
            entry_probs = F.softmax(entry_logits, dim=-1)
        assert torch.allclose(expected_probs, entry_probs, atol=1e-5), (
            "root pool entry must preserve the trained root categorical policy"
        )

    if fusion_mode == "classic_cka":
        entry = model.policy_pool.pool[0]
        assert torch.count_nonzero(entry["l0_weight"]) == 0
        assert torch.count_nonzero(entry["l2_weight"]) == 0

    model.save(path)
    return model


def run_parameter_chain(fusion_mode, distillation, use_alpha_mass):
    print(
        f"\n=== parameter chain fusion={fusion_mode} "
        f"distillation={distillation} alpha_mass={use_alpha_mass} ==="
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
        expected_probs = policy_probs(m2, probe)
        snapshot_dir = os.path.join(root, "snapshot")
        m2.save_policy_snapshot(snapshot_dir)
        frozen = FrozenCkaPolicy.load(snapshot_dir)
        with torch.no_grad():
            got_probs = frozen.action_distribution(probe).probs
        assert torch.allclose(expected_probs, got_probs, atol=1e-6)

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
            assert np.isfinite(metrics["policy/distill_best_val_kl"])
            assert np.isfinite(metrics["policy/distill_train_kl_p95"])
            assert np.isfinite(metrics["policy/distill_train_kl_max"])
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


def check_protocol_budget():
    print("\n=== Delta/B task-budget protocol ===")
    budget = TaskBudget(100, 20)
    assert budget.total == 100 and budget.training == 80 and budget.frozen_tail == 20
    for total, tail in ((0, 0), (10, -1), (10, 10), (10, 11)):
        try:
            TaskBudget(total, tail)
        except ValueError:
            pass
        else:
            raise AssertionError("TaskBudget must enforce 0 <= B < Delta")

    assert mixture_warmup_active(3, 2, 5, "weight_delta", 1)
    assert not mixture_warmup_active(8, 2, 5, "weight_delta", 1)
    assert not mixture_warmup_active(3, 2, 5, "classic_cka", 1)

    buf = fake_buffer(20, 0, 0)
    trimmed = bounded_buffer(buf, 7)
    assert len(trimmed["obs"]) == 7
    assert all(
        len(value) == 7
        for value in trimmed.values()
        if isinstance(value, np.ndarray)
    )
    print("  B is inside Delta + aligned bounded-buffer trimming OK")


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
            "l0_weight": torch.randn_like(m.policy_pool.own_l0_weight)
            * (20.0 if i == 1 else 1.0),
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
    assert {i, j} == {0, 1}, (
        f"expected functionally identical entries 0/1, got {i}/{j}"
    )
    assert abs(info["symmetric_kl"]) < 1e-7
    print("  identical categorical behavior wins despite different hidden weights OK")


def check_lineage_balancing():
    print("\n=== source-lineage balancing ===")
    source_ids = np.asarray([0] * 30 + [1] * 30 + [2] * 30, dtype=np.int32)
    idx = balanced_lineage_indices(source_ids, 12)
    _, counts = np.unique(source_ids[idx], return_counts=True)
    assert len(idx) == 12
    assert counts.max() - counts.min() <= 1

    a = fake_buffer(20, task_id=0, source_id=0)
    b = fake_buffer(20, task_id=1, source_id=1)
    merged = HeadPool.merge_buffers(
        a, b, max_rows=10, balance_source_lineages=True
    )
    ids, counts = np.unique(merged["source_ids"], return_counts=True)
    assert set(ids.tolist()) == {0, 1}
    assert counts.tolist() == [5, 5]
    print("  behavioral/distillation/retained-buffer lineage balancing primitive OK")


def check_alpha_mass():
    print("\n=== alpha-mass restriction and sigmoid semantics ===")
    for distillation in (False, True):
        try:
            make_agent(
                fusion_mode="classic_cka",
                distillation=distillation,
                use_alpha_mass=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "classic_cka must reject use_alpha_mass in both distillation settings"
            )

    root = tempfile.mkdtemp(prefix="cka_atari_mass_")
    try:
        for distillation in (False, True):
            d0 = os.path.join(root, f"task0_{distillation}")
            save_root(d0, "weight_delta", distillation, True)

            model = make_agent(
                base_dir=d0,
                latest_dir=d0,
                fusion_mode="weight_delta",
                distillation=distillation,
                use_alpha_mass=True,
                constrain_alpha_mass=True,
            )
            assert model.alpha_mass is not None
            assert abs(float(model.policy_pool.effective_alpha_mass().detach()) - 0.95) < 1e-5
            with torch.no_grad():
                model.alpha_mass.zero_()
            assert abs(float(model.policy_pool.effective_alpha_mass().detach()) - 0.5) < 1e-6

            legacy = make_agent(
                base_dir=d0,
                latest_dir=d0,
                fusion_mode="weight_delta",
                distillation=distillation,
                use_alpha_mass=True,
                constrain_alpha_mass=False,
            )
            with torch.no_grad():
                legacy.alpha_mass.fill_(-2.0)
            assert float(legacy.policy_pool.effective_alpha_mass().detach()) == -2.0
        print("  sigmoid mass + legacy unconstrained ablation OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


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


def check_encoder_policy_flags():
    print("\n=== shared-CNN policy/architecture checks ===")
    root = tempfile.mkdtemp(prefix="cka_atari_encoder_flags_")
    try:
        learned_root = make_agent(distillation=False)
        assert all(p.requires_grad for p in learned_root.fc.parameters())

        frozen_root = make_agent(
            distillation=False, freeze_root_encoder=True
        )
        assert not any(p.requires_grad for p in frozen_root.fc.parameters())

        good = os.path.join(root, "good_fc.pt")
        bad = os.path.join(root, "bad_fc.pt")
        torch.save(shared(input_shape=OBS_SHAPE, output_dim=SHARED_DIM), good)
        torch.save(shared(input_shape=OBS_SHAPE, output_dim=SHARED_DIM * 2), bad)

        frozen_pre = make_agent(
            distillation=False,
            pretrained_encoder=good,
            train_shared=False,
        )
        assert not any(p.requires_grad for p in frozen_pre.fc.parameters())

        try:
            make_agent(
                distillation=False,
                pretrained_encoder=bad,
                train_shared=False,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("pretrained CNN output-dimension mismatch must fail fast")

        try:
            make_agent(
                distillation=True,
                distill_observation_skip=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Atari must reject the HalfCheetah raw-observation skip architecture"
            )

        print("  root/frozen/pretrained CNN policies + Atari architecture guard OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_alpha_scale_controls():
    print("\n=== learned/fixed alpha-scale controls ===")
    root = tempfile.mkdtemp(prefix="cka_atari_alpha_scale_")
    try:
        d0 = os.path.join(root, "task0")
        save_root(d0, "classic_cka", False, False)

        learned = make_agent(
            base_dir=d0,
            latest_dir=d0,
            distillation=False,
            fusion_mode="classic_cka",
            use_alpha_scale=True,
            fix_alpha_scale=False,
        )
        assert learned.alpha_scale is not None and learned.alpha_scale.requires_grad
        assert abs(float(learned.alpha_scale) - 1.0) < 1e-8

        fixed = make_agent(
            base_dir=d0,
            latest_dir=d0,
            distillation=False,
            fusion_mode="weight_delta",
            use_alpha_scale=False,
            fix_alpha_scale=True,
        )
        assert fixed.alpha_scale is not None and not fixed.alpha_scale.requires_grad
        assert abs(float(fixed.alpha_scale) - 5.0) < 1e-8

        try:
            make_agent(
                base_dir=d0,
                latest_dir=d0,
                distillation=False,
                use_alpha_scale=True,
                fix_alpha_scale=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "learned and fixed alpha-scale modes must be mutually exclusive"
            )
        print("  learned/fixed alpha-scale modes OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_distill_selection_ablation():
    print("\n=== distillation model-selection ablation ===")
    root = tempfile.mkdtemp(prefix="cka_atari_distill_last_")
    try:
        d0 = os.path.join(root, "task0")
        d1 = os.path.join(root, "task1")
        save_root(d0, "classic_cka", True, False)

        m1 = make_agent(
            base_dir=d0,
            latest_dir=d0,
            pool_size=2,
            distillation=True,
            fusion_mode="classic_cka",
            encoder_from_base=True,
            similarity_samples=8,
            distill_max_samples=8,
            distill_epochs=2,
            distill_batch_size=4,
            distill_select_best_val=False,
        )
        train_a_bit(m1)
        m1.set_own_buffer(fake_buffer(8, task_id=1, source_id=1))
        m1.finalize()
        m1.save(d1)

        m2 = make_agent(
            base_dir=d0,
            latest_dir=d1,
            pool_size=2,
            distillation=True,
            fusion_mode="classic_cka",
            encoder_from_base=True,
            similarity_samples=8,
            distill_max_samples=8,
            distill_epochs=2,
            distill_batch_size=4,
            distill_test_frac=0.25,
            distill_select_best_val=False,
        )
        train_a_bit(m2)
        m2.set_own_buffer(fake_buffer(8, task_id=2, source_id=2))
        m2.finalize()
        metrics = m2.get_distill_metrics()
        assert metrics["policy/distill_selected_epoch"] == 2
        assert metrics["policy/distill_select_best_val"] == 0.0
        print("  --no-distill-select-best-val keeps final epoch OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_policy_space_exact_mixture_and_projection():
    print("\n=== exact categorical policy-space mixture + projection ===")
    root = tempfile.mkdtemp(prefix="cka_atari_policy_space_")
    try:
        d0 = os.path.join(root, "task0")
        save_root(
            d0,
            "weight_delta",
            True,
            use_alpha_mass=True,
            composition_space="policy",
        )

        m1 = make_agent(
            base_dir=d0,
            latest_dir=d0,
            pool_size=3,
            distillation=True,
            fusion_mode="weight_delta",
            use_alpha_mass=True,
            composition_space="policy",
            projection_epochs=2,
            projection_max_samples=8,
        )
        train_a_bit(m1)
        probe = torch.rand(3, *OBS_SHAPE)
        with torch.no_grad():
            component_logits, weights = m1.policy_components(probe)
            manual = categorical_mixture_probs(component_logits, weights)
            exact = m1.action_distribution(probe).probs
        assert torch.allclose(manual, exact, atol=1e-7)
        assert component_logits.shape[1] == 2, (
            "historical policy + gated novel expert expected"
        )

        m1.set_mixture_warmup(True)
        with torch.no_grad():
            warm_logits, warm_weights = m1.policy_components(probe)
        assert warm_logits.shape[1] == 1
        assert torch.allclose(warm_weights.sum(), torch.tensor(1.0), atol=1e-7)
        m1.set_mixture_warmup(False)

        before = policy_probs(m1, probe)
        snap = os.path.join(root, "snapshot")
        m1.save_policy_snapshot(snap)
        frozen = FrozenCkaPolicy.load(snap)
        with torch.no_grad():
            after = frozen.action_distribution(probe).probs
        assert torch.allclose(before, after, atol=1e-6)

        m1.set_own_buffer(fake_buffer(8, task_id=1, source_id=1))
        m1.finalize()
        assert m1.policy_pool.pool_length() == 2
        metrics = m1.last_projection_metrics
        assert metrics["policy/projection_rows"] >= 2
        assert metrics["policy/projection_components"] == 2
        assert np.isfinite(metrics["policy/projection_val_mixture_kl"])
        print("  exact probability mixture + warmup + snapshot + projection OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check_policy_student_replay_mode():
    print("\n=== categorical policy-student routing/storage ===")
    root = tempfile.mkdtemp(prefix="cka_atari_policy_student_")
    try:
        d0 = os.path.join(root, "task0")
        save_root(
            d0,
            "weight_delta",
            True,
            use_alpha_mass=True,
            composition_space="policy",
            policy_student_replay=True,
        )

        m1 = make_agent(
            base_dir=d0,
            latest_dir=d0,
            pool_size=3,
            distillation=True,
            fusion_mode="weight_delta",
            use_alpha_mass=True,
            composition_space="policy",
            policy_student_replay=True,
        )

        x = torch.rand(4, *OBS_SHAPE)
        actions = torch.randint(0, ACT_DIM, (4,))

        for p in m1.parameters():
            p.grad = None
        novel = m1.novel_action_distribution(x)
        novel_loss = -novel.log_prob(actions).mean()
        novel_loss.backward()

        own = [
            getattr(m1.policy_pool, "own_" + key)
            for key in ("l0_weight", "l0_bias", "l2_weight", "l2_bias")
        ]
        assert any(
            p.grad is not None and torch.count_nonzero(p.grad)
            for p in own
        )
        assert m1.alpha.grad is None
        assert m1.alpha_mass.grad is None

        for p in m1.parameters():
            p.grad = None
        routing = m1.routing_action_distribution(x)
        routing_loss = -routing.log_prob(actions).mean()
        routing_loss.backward()
        assert m1.alpha.grad is not None
        assert m1.alpha_mass.grad is not None
        assert all(p.grad is None for p in own), (
            "routing-only objective leaked gradients into novel expert"
        )
        assert all(p.grad is None for p in m1.fc.parameters()), (
            "routing-only objective leaked gradients into shared encoder"
        )

        before = {
            key: getattr(m1.policy_pool, "own_" + key).detach().clone()
            for key in ("l0_weight", "l0_bias", "l2_weight", "l2_bias")
        }
        m1.set_own_buffer(fake_buffer(8, task_id=1, source_id=1))
        m1.finalize()
        assert (
            m1.last_projection_metrics.get("policy/storage_used_novel_expert")
            == 1.0
        )
        for key, value in before.items():
            assert torch.equal(m1.policy_pool.pool[0][key], value)
        print("  novel-only gradients + routing-only gradients + direct novel storage OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    check_protocol_budget()
    check_categorical_kl()
    check_lineage_balancing()
    check_behavioral_pair_is_output_space()
    check_alpha_mass()
    check_trainable_encoder_loads_latest()
    check_encoder_policy_flags()
    check_alpha_scale_controls()
    check_distill_selection_ablation()
    check_policy_space_exact_mixture_and_projection()
    check_policy_student_replay_mode()

    run_parameter_chain("classic_cka", False, False)
    run_parameter_chain("classic_cka", True, False)
    run_parameter_chain("weight_delta", False, True)
    run_parameter_chain("weight_delta", True, True)

    print("\n*** ALL ATARI CATEGORICAL CKA-RL SANITY CHECKS PASSED ***")


if __name__ == "__main__":
    main()
