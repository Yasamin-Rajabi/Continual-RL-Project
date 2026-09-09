"""Fast CPU-only tests for the aligned policy pool and behavioral-KL merge.

Run before expensive MuJoCo experiments:
    python3 sanity_check_pool.py

No Gymnasium/MuJoCo installation is required for these tests.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import torch

from cka_rl import CkaRlAgent, FrozenCkaPolicy
from shared_arch import shared

OBS_DIM = 6
ACT_DIM = 3
TMP_ROOT = "/tmp/cka_pool_sanity"


def fake_buffer(n=128, task_id=0, source_id=None):
    if source_id is None:
        source_id = task_id
    return {
        "obs": np.random.randn(n, OBS_DIM).astype(np.float32),
        "actions": np.tanh(np.random.randn(n, ACT_DIM)).astype(np.float32),
        "task_ids": np.full(n, task_id, dtype=np.int32),
        "source_ids": np.full(n, source_id, dtype=np.int32),
        "x_velocity": np.random.randn(n, 1).astype(np.float32),
        "task_error": np.abs(np.random.randn(n, 1)).astype(np.float32),
    }


def train_a_bit(model, steps=3):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-3)
    for _ in range(steps):
        x = torch.randn(16, OBS_DIM)
        mean, raw_logstd = model(x)
        loss = mean.square().mean() + 0.01 * raw_logstd.square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def pool_lens(model):
    return model.mean_pool.pool_length(), model.logstd_pool.pool_length()


def save_root(root_dir, fusion_mode, distillation, use_alpha_mass=False):
    model = CkaRlAgent(
        OBS_DIM, ACT_DIM, None, None,
        pool_size=2,
        distillation=distillation,
        fusion_mode=fusion_mode,
        use_alpha_mass=use_alpha_mass,
    )
    train_a_bit(model)
    model.set_own_buffer(fake_buffer(task_id=0))
    # Root is special: trained weights become theta_base and v1 is seeded.
    model.set_base()
    assert pool_lens(model) == (1, 1)
    if fusion_mode == "classic_cka":
        entry = model.mean_pool.pool[0]
        assert torch.count_nonzero(entry["l0_weight"]) == 0
        assert torch.count_nonzero(entry["l2_weight"]) == 0
    model.save(root_dir)
    return model


def run_chain(fusion_mode, distillation, use_alpha_mass):
    print(
        f"\n=== chain fusion={fusion_mode} distillation={distillation} "
        f"alpha_mass={use_alpha_mass} ==="
    )
    root = f"{TMP_ROOT}/{fusion_mode}_{distillation}_{use_alpha_mass}"
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)

    d0 = f"{root}/task0"
    m0 = save_root(d0, fusion_mode, distillation, use_alpha_mass)
    print("  root base + v1 seeded OK")

    # Task 1 inherits the root slot. Mean/logstd must share exactly one alpha.
    m1 = CkaRlAgent(
        OBS_DIM, ACT_DIM, d0, d0,
        pool_size=2,
        distillation=distillation,
        fusion_mode=fusion_mode,
        use_alpha_mass=use_alpha_mass,
        encoder_from_base=True,
        similarity_samples=64,
        distill_max_samples=128,
        distill_epochs=2,
        distill_batch_size=32,
    )
    assert pool_lens(m1) == (1, 1)
    assert m1.mean_pool.alpha is m1.logstd_pool.alpha
    assert m1.alpha is m1.mean_pool.alpha
    assert m1.alpha.numel() == 1
    assert not any(p.requires_grad for p in m1.fc.parameters()), "later-task encoder must be frozen"
    train_a_bit(m1)
    m1.set_own_buffer(fake_buffer(task_id=1, source_id=1))
    m1.finalize()
    assert pool_lens(m1) == (2, 2)
    assert m1.get_merge_info() is None
    d1 = f"{root}/task1"
    m1.save(d1)
    print("  shared alpha + frozen encoder + no-merge finalize OK")

    # Task 2 pushes pool length to 3, so pool_size=2 forces one aligned merge.
    m2 = CkaRlAgent(
        OBS_DIM, ACT_DIM, d0, d1,
        pool_size=2,
        distillation=distillation,
        fusion_mode=fusion_mode,
        use_alpha_mass=use_alpha_mass,
        encoder_from_base=True,
        similarity_samples=64,
        distill_max_samples=128,
        distill_epochs=2,
        distill_batch_size=32,
        distill_test_frac=0.25,
    )
    assert pool_lens(m2) == (2, 2)
    train_a_bit(m2)
    m2.set_own_buffer(fake_buffer(task_id=2, source_id=2))
    # Save exact pre-finalize policy and verify the compact inference snapshot.
    probe = torch.randn(8, OBS_DIM)
    with torch.no_grad():
        expected_mean, expected_log = m2(probe)
    snap_dir = f"{root}/snapshot"
    m2.save_policy_snapshot(snap_dir)
    frozen = FrozenCkaPolicy.load(snap_dir)
    with torch.no_grad():
        got_mean, got_log = frozen(probe)
    assert torch.allclose(expected_mean, got_mean, atol=1e-6)
    assert torch.allclose(expected_log, got_log, atol=1e-6)

    m2.finalize()
    assert pool_lens(m2) == (2, 2)
    info = m2.get_merge_info()
    assert info is not None
    assert 0 <= info["idx1"] < 3 and 0 <= info["idx2"] < 3
    assert info["idx1"] != info["idx2"]
    if distillation:
        assert info["similarity_metric"] == "symmetric_kl"
        assert np.isfinite(info["symmetric_kl"])
    else:
        assert info["similarity_metric"] == "cosine"
        assert np.isfinite(info["cosine_similarity"])
    assert info["pool_size_before"] == 3 and info["pool_size_after"] == 2
    assert info["merged_source_lineage"], "source-occurrence lineage must be preserved"
    assert m2.mean_pool.last_merge_info["idx1"] == m2.logstd_pool.last_merge_info["idx1"]
    assert m2.mean_pool.last_merge_info["idx2"] == m2.logstd_pool.last_merge_info["idx2"]
    if distillation:
        metrics = m2.get_distill_metrics()
        assert metrics["policy/distill_train_kl"] is not None
        assert metrics["policy/distill_test_kl"] is not None
        assert np.isfinite(metrics["policy/distill_test_kl"])
        assert metrics["policy/distill_best_epoch"] >= 0
        assert np.isfinite(metrics["policy/distill_best_val_kl"])
        assert np.isfinite(metrics["policy/distill_train_kl_p95"])
        assert np.isfinite(metrics["policy/distill_train_kl_max"])
    else:
        assert m2.get_distill_metrics() == {}
    if distillation:
        detail = f"SKL={info['symmetric_kl']:.6g}"
    else:
        detail = f"cosine={info['cosine_similarity']:.6g}"
    print(f"  aligned merge OK: pair=({info['idx1']},{info['idx2']}), {detail}")

    d2 = f"{root}/task2"
    m2.save(d2)
    m3 = CkaRlAgent(
        OBS_DIM, ACT_DIM, d0, d2,
        pool_size=2,
        distillation=distillation,
        fusion_mode=fusion_mode,
        use_alpha_mass=use_alpha_mass,
        encoder_from_base=True,
    )
    assert pool_lens(m3) == (2, 2)
    assert m3.alpha.numel() == 2
    mean, logstd = m3(torch.randn(4, OBS_DIM))
    assert mean.shape == (4, ACT_DIM) and logstd.shape == (4, ACT_DIM)
    print("  post-merge inheritance and forward shapes OK")

    shutil.rmtree(root)


def check_behavioral_pair_not_weight_cosine():
    """Two functionally identical entries must be selected even with different weights."""
    print("\n=== explicit output-space pair-selection check ===")
    m = CkaRlAgent(
        OBS_DIM, ACT_DIM, None, None,
        pool_size=3,
        distillation=True,
        fusion_mode="classic_cka",
        similarity_samples=96,
    )
    # Make theta_base zero so standalone entry behavior is easy to control.
    for pool in (m.mean_pool, m.logstd_pool):
        with torch.no_grad():
            pool.base_l0_weight.zero_()
            pool.base_l0_bias.zero_()
            pool.base_l2_weight.zero_()
            pool.base_l2_bias.zero_()
        pool.pool = []

    buffers = [fake_buffer(64, i) for i in range(3)]
    for i in range(3):
        mean_entry = {
            "l0_weight": torch.randn_like(m.mean_pool.own_l0_weight) * (20.0 if i == 1 else 1.0),
            "l0_bias": torch.randn_like(m.mean_pool.own_l0_bias),
            # Entries 0 and 1 have wildly different hidden weights but l2=0,
            # so both output exactly mean=0. Entry 2 outputs mean=3.
            "l2_weight": torch.zeros_like(m.mean_pool.own_l2_weight),
            "l2_bias": torch.zeros_like(m.mean_pool.own_l2_bias) if i < 2 else torch.full_like(m.mean_pool.own_l2_bias, 3.0),
            "buffer": buffers[i],
        }
        log_entry = {
            "l0_weight": torch.randn_like(m.logstd_pool.own_l0_weight) * (15.0 if i == 1 else 1.0),
            "l0_bias": torch.randn_like(m.logstd_pool.own_l0_bias),
            "l2_weight": torch.zeros_like(m.logstd_pool.own_l2_weight),
            "l2_bias": torch.zeros_like(m.logstd_pool.own_l2_bias),
            "buffer": None,
        }
        m.mean_pool.pool.append(mean_entry)
        m.logstd_pool.pool.append(log_entry)

    i, j, info = m._select_behavioral_pair()
    assert {i, j} == {0, 1}, f"expected functionally identical entries 0/1, got {i}/{j}"
    assert abs(info["symmetric_kl"]) < 1e-8
    print("  lowest output KL selects functionally identical pair despite different weights OK")


def check_alpha_mass_restriction():
    print("\n=== four-case alpha-mass restriction check ===")
    for distillation in (False, True):
        try:
            CkaRlAgent(
                OBS_DIM, ACT_DIM, None, None,
                fusion_mode="classic_cka", distillation=distillation, use_alpha_mass=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("classic_cka must reject use_alpha_mass in both distillation settings")

    # alpha_mass exists only when there is historical knowledge to weight. Build
    # a one-entry root checkpoint, then inspect the next-task agents.
    for distillation in (False, True):
        case_root = f"{TMP_ROOT}/alpha_mass_{distillation}"
        shutil.rmtree(case_root, ignore_errors=True)
        base_dir = f"{case_root}/task0"
        save_root(base_dir, "weight_delta", distillation, use_alpha_mass=True)

        model = CkaRlAgent(
            OBS_DIM, ACT_DIM, base_dir, base_dir,
            fusion_mode="weight_delta", distillation=distillation, use_alpha_mass=True,
            constrain_alpha_mass=True,
        )
        assert model.alpha_mass is not None
        # Raw is unconstrained; the default effective mass must stay positive.
        with torch.no_grad():
            model.alpha_mass.fill_(-10.0)
        assert model.mean_pool.effective_alpha_mass().item() > 0.0

        legacy = CkaRlAgent(
            OBS_DIM, ACT_DIM, base_dir, base_dir,
            fusion_mode="weight_delta", distillation=distillation, use_alpha_mass=True,
            constrain_alpha_mass=False,
        )
        assert legacy.alpha_mass is not None
        with torch.no_grad():
            legacy.alpha_mass.fill_(-2.0)
        assert legacy.mean_pool.effective_alpha_mass().item() == -2.0
    print("  alpha_mass restriction + positive/default and legacy/unconstrained ablations OK")


def check_trainable_encoder_loads_latest():
    print("\n=== explicit trainable-encoder continuation check ===")
    root = f"{TMP_ROOT}/encoder_latest_check"
    shutil.rmtree(root, ignore_errors=True)
    d0, d1 = f"{root}/task0", f"{root}/task1"
    os.makedirs(root, exist_ok=True)

    m0 = save_root(d0, "classic_cka", False, False)
    m1 = CkaRlAgent(
        OBS_DIM, ACT_DIM, d0, d0,
        pool_size=2, distillation=False, fusion_mode="classic_cka",
        encoder_from_base=True, train_shared=True,
    )
    with torch.no_grad():
        first_param = next(m1.fc.parameters())
        first_param.add_(0.12345)
    m1.save(d1)

    expected = next(m1.fc.parameters()).detach().clone()
    m2 = CkaRlAgent(
        OBS_DIM, ACT_DIM, d0, d1,
        pool_size=2, distillation=False, fusion_mode="classic_cka",
        encoder_from_base=True, train_shared=True,
    )
    got = next(m2.fc.parameters()).detach()
    assert torch.allclose(got, expected), "train_shared=True must continue from latest encoder"
    assert all(p.requires_grad for p in m2.fc.parameters())
    shutil.rmtree(root, ignore_errors=True)
    print("  train_shared=True loads latest encoder instead of resetting to root OK")


def check_encoder_policy_flags():
    print("\n=== shared-encoder policy/architecture checks ===")
    root = f"{TMP_ROOT}/encoder_flags"
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)

    # Normal scratch root learns; explicit random-frozen ablation does not.
    learned_root = CkaRlAgent(OBS_DIM, ACT_DIM, None, None, distillation=False)
    assert all(p.requires_grad for p in learned_root.fc.parameters())
    frozen_root = CkaRlAgent(
        OBS_DIM, ACT_DIM, None, None, distillation=False, freeze_root_encoder=True
    )
    assert not any(p.requires_grad for p in frozen_root.fc.parameters())

    # Serialized architecture is validated even though ReLU-vs-linear has the
    # same state_dict shapes and would otherwise fail silently.
    pretrained = f"{root}/linear_fc.pt"
    torch.save(shared(OBS_DIM, linear_out=True), pretrained)
    frozen_pre = CkaRlAgent(
        OBS_DIM, ACT_DIM, None, None, distillation=False,
        pretrained_encoder=pretrained, encoder_linear_out=True, train_shared=False,
    )
    assert not any(p.requires_grad for p in frozen_pre.fc.parameters())
    try:
        CkaRlAgent(
            OBS_DIM, ACT_DIM, None, None, distillation=False,
            pretrained_encoder=pretrained, encoder_linear_out=False, train_shared=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("pretrained encoder ReLU/linear-out mismatch must fail fast")

    shutil.rmtree(root, ignore_errors=True)
    print("  root/frozen/pretrained policies + architecture mismatch guard OK")


def check_distill_selection_ablation():
    print("\n=== distillation model-selection ablation check ===")
    root = f"{TMP_ROOT}/distill_last_epoch"
    shutil.rmtree(root, ignore_errors=True)
    d0 = f"{root}/task0"
    d1 = f"{root}/task1"
    save_root(d0, "classic_cka", True, False)
    m1 = CkaRlAgent(
        OBS_DIM, ACT_DIM, d0, d0, pool_size=2, distillation=True,
        fusion_mode="classic_cka", encoder_from_base=True, similarity_samples=32,
        distill_max_samples=64, distill_epochs=2, distill_batch_size=32,
        distill_select_best_val=False,
    )
    train_a_bit(m1, steps=1)
    m1.set_own_buffer(fake_buffer(48, task_id=1, source_id=1))
    m1.finalize(); m1.save(d1)
    m2 = CkaRlAgent(
        OBS_DIM, ACT_DIM, d0, d1, pool_size=2, distillation=True,
        fusion_mode="classic_cka", encoder_from_base=True, similarity_samples=32,
        distill_max_samples=64, distill_epochs=2, distill_batch_size=32,
        distill_test_frac=0.25, distill_select_best_val=False,
    )
    train_a_bit(m2, steps=1)
    m2.set_own_buffer(fake_buffer(48, task_id=2, source_id=2))
    m2.finalize()
    metrics = m2.get_distill_metrics()
    assert metrics["policy/distill_selected_epoch"] == 2
    assert metrics["policy/distill_select_best_val"] == 0.0
    shutil.rmtree(root, ignore_errors=True)
    print("  --no-distill-select-best-val keeps final epoch OK")



def check_friend_policy_controls():
    print("\n=== friend-method policy-control checks ===")

    # Distillation observation skip changes the first head's input dimension,
    # but remains an explicit ablation switch.
    with_skip = CkaRlAgent(
        OBS_DIM, ACT_DIM, None, None, distillation=True,
        distill_observation_skip=True,
    )
    without_skip = CkaRlAgent(
        OBS_DIM, ACT_DIM, None, None, distillation=True,
        distill_observation_skip=False,
    )
    assert with_skip.mean_pool.own_l0_weight.shape[1] == 256 + OBS_DIM
    assert without_skip.mean_pool.own_l0_weight.shape[1] == 256
    assert with_skip(torch.zeros(2, OBS_DIM))[0].shape == (2, ACT_DIM)

    root = f"{TMP_ROOT}/friend_scale_modes"
    shutil.rmtree(root, ignore_errors=True)
    d0 = f"{root}/task0"
    save_root(d0, "classic_cka", False, False)

    learned = CkaRlAgent(
        OBS_DIM, ACT_DIM, d0, d0, distillation=False,
        fusion_mode="classic_cka", use_alpha_scale=True, fix_alpha_scale=False,
    )
    assert learned.alpha_scale is not None and learned.alpha_scale.requires_grad
    assert abs(float(learned.alpha_scale.item()) - 1.0) < 1e-8

    fixed = CkaRlAgent(
        OBS_DIM, ACT_DIM, d0, d0, distillation=False,
        fusion_mode="weight_delta", use_alpha_scale=False, fix_alpha_scale=True,
    )
    assert fixed.alpha_scale is not None and not fixed.alpha_scale.requires_grad
    assert abs(float(fixed.alpha_scale.item()) - 5.0) < 1e-8

    try:
        CkaRlAgent(
            OBS_DIM, ACT_DIM, d0, d0, distillation=False,
            use_alpha_scale=True, fix_alpha_scale=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("learned and fixed alpha-scale modes must be mutually exclusive")

    shutil.rmtree(root, ignore_errors=True)
    print("  observation-skip + learned/fixed alpha-scale modes OK")


def check_policy_student_replay_mode():
    print("\n=== replay-trained standalone policy-student check ===")
    from policy_composition import novel_sac_actor_objective, mixture_weight_sac_actor_objective

    root = f"{TMP_ROOT}/policy_student"
    shutil.rmtree(root, ignore_errors=True)
    d0 = f"{root}/task0"
    m0 = CkaRlAgent(
        OBS_DIM, ACT_DIM, None, None, pool_size=3, distillation=True,
        fusion_mode="weight_delta", use_alpha_mass=True,
        composition_space="policy", policy_student_replay=True,
    )
    train_a_bit(m0, steps=1)
    m0.set_own_buffer(fake_buffer(32, task_id=0, source_id=0))
    m0.set_base(); m0.save(d0)

    m1 = CkaRlAgent(
        OBS_DIM, ACT_DIM, d0, d0, pool_size=3, distillation=True,
        fusion_mode="weight_delta", use_alpha_mass=True,
        composition_space="policy", policy_student_replay=True,
        encoder_from_base=True,
    )
    x = torch.randn(16, OBS_DIM)
    action_scale = torch.ones(ACT_DIM)
    action_bias = torch.zeros(ACT_DIM)

    class Q(torch.nn.Module):
        def forward(self, obs, action):
            return -(action.square().sum(-1, keepdim=True))
    q1 = Q(); q2 = Q()

    for p in m1.parameters():
        p.grad = None
    loss = novel_sac_actor_objective(m1, x, q1, q2, 0.2, action_scale, action_bias)
    loss.backward()
    own = [getattr(pool, "own_" + key) for pool in (m1.mean_pool, m1.logstd_pool) for key in
           ("l0_weight", "l0_bias", "l2_weight", "l2_bias")]
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in own)
    assert m1.alpha.grad is None and m1.alpha_mass.grad is None

    for p in m1.parameters():
        p.grad = None
    routing = mixture_weight_sac_actor_objective(m1, x, q1, q2, 0.2, action_scale, action_bias)
    routing.backward()
    assert m1.alpha.grad is not None
    assert m1.alpha_mass.grad is not None
    assert all(p.grad is None for p in own), "routing loss leaked gradient into novel expert"

    before = {key: getattr(m1.mean_pool, "own_" + key).detach().clone()
              for key in ("l0_weight", "l0_bias", "l2_weight", "l2_bias")}
    m1.set_own_buffer(fake_buffer(32, task_id=1, source_id=1))
    m1.finalize()
    assert m1.last_projection_metrics.get("policy/storage_used_novel_expert") == 1.0
    for key, value in before.items():
        assert torch.equal(m1.mean_pool.pool[0][key], value)
    shutil.rmtree(root, ignore_errors=True)
    print("  novel SAC gradient / alpha-only gradient / direct novel storage OK")

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    shutil.rmtree(TMP_ROOT, ignore_errors=True)

    run_chain("classic_cka", False, False)   # baseline
    run_chain("classic_cka", True, False)    # distil_only
    run_chain("weight_delta", False, True)   # weight_only
    run_chain("weight_delta", True, True)    # combined
    check_behavioral_pair_not_weight_cosine()
    check_alpha_mass_restriction()
    check_trainable_encoder_loads_latest()
    check_encoder_policy_flags()
    check_distill_selection_ablation()
    check_friend_policy_controls()
    check_policy_student_replay_mode()

    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    print("\n*** ALL CHECKS PASSED ***")


if __name__ == "__main__":
    main()
