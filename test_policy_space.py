"""CPU regression tests; no Gymnasium, MuJoCo or MetaWorld needed.

CRL_FAMILY=Walker2D python -m pytest -q tests/test_policy_space.py
Run separately per family because these original folders use unqualified imports.
"""
from __future__ import annotations
import ast
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FAMILY = os.environ.get("CRL_FAMILY", "half-cheetah")
sys.path.insert(0, str(ROOT / FAMILY))
from cka_rl import CkaRlAgent, FrozenCkaPolicy
from knowledge_pools import HeadPool, balanced_lineage_indices
from policy_composition import (normal_mixture_log_prob, sample_action,
    representative_action, sac_actor_objective, components, squash_log_det)
from policy_utils import bound_log_std
from training_protocol import TaskBudget, mixture_warmup_active, bounded_buffer
from analysis_logging import effective_theta_vector
from checkpoint_evaluation import load_finalized_policy

torch.set_num_threads(1)
KEYS = ("l0_weight", "l0_bias", "l2_weight", "l2_bias")


class ToyMixture(torch.nn.Module):
    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.4, -0.7], dtype=dtype))
        self.mean = torch.nn.Parameter(torch.tensor([[-0.8, 0.3], [0.7, -0.4]], dtype=dtype))
        self.log = torch.nn.Parameter(torch.full((2, 2), -0.9, dtype=dtype))
    def policy_components(self, obs):
        return self.mean.expand(len(obs), -1, -1), self.log.expand(len(obs), -1, -1), self.logits.softmax(0)


def test_density_matches_torch_mixture():
    torch.manual_seed(0)
    means = torch.randn(7, 3, 2, dtype=torch.double)
    logs = torch.randn_like(means) * 0.3
    weights = torch.tensor([0.1, 0.3, 0.6], dtype=torch.double)
    value = torch.randn(7, 2, dtype=torch.double)
    reference = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(weights),
        torch.distributions.Independent(torch.distributions.Normal(means, logs.exp()), 1))
    assert torch.allclose(normal_mixture_log_prob(value, means, logs, weights), reference.log_prob(value), atol=1e-12)
    values = torch.randn(7, 4, 2, dtype=torch.double)
    lp = normal_mixture_log_prob(values, means, logs, weights)
    for k in range(4):
        assert torch.allclose(lp[:, k], reference.log_prob(values[:, k]), atol=1e-12)


def test_squash_jacobian_and_zero_weight():
    value = torch.tensor([[0.2, -0.8], [15., -15.]], dtype=torch.double)
    scale = torch.tensor([2., 0.5], dtype=torch.double)
    normal = torch.distributions.Normal(torch.zeros(2, dtype=torch.double), torch.ones(2, dtype=torch.double))
    transform = torch.distributions.transforms.TanhTransform()
    reference = (transform.log_abs_det_jacobian(value, value.tanh()) + scale.log()).sum(-1)
    assert torch.allclose(squash_log_det(value, scale), reference)
    lp = normal_mixture_log_prob(value, torch.zeros(2, 2, 2, dtype=torch.double),
        torch.zeros(2, 2, 2, dtype=torch.double), torch.tensor([1., 0.], dtype=torch.double))
    assert torch.allclose(lp, normal.log_prob(value).sum(-1))
    assert torch.isfinite(lp).all()


def test_actor_mixing_gradient_matches_finite_difference():
    model = ToyMixture()
    x = torch.zeros(8, 3, dtype=torch.double)
    scale, bias = torch.ones(2, dtype=torch.double), torch.zeros(2, dtype=torch.double)
    def q(obs, act):
        return -(act - 0.5).square().sum(-1, keepdim=True)
    def loss():
        torch.manual_seed(33)  # common random numbers for central differences
        return sac_actor_objective(model, x, q, q, 0.2, scale, bias)
    automatic = torch.autograd.grad(loss(), model.logits)[0]
    assert automatic.abs().max() > 1e-5
    numerical = []
    for i in range(2):
        original = model.logits[i].item()
        with torch.no_grad(): model.logits[i] = original + 1e-5
        plus = loss().item()
        with torch.no_grad(): model.logits[i] = original - 1e-5
        minus = loss().item()
        numerical.append((plus - minus) / 2e-5)
        with torch.no_grad(): model.logits[i] = original
    assert torch.allclose(automatic, torch.tensor(numerical, dtype=torch.double), atol=1e-7, rtol=1e-5)


def test_sampling_uses_one_component_for_whole_action():
    model = ToyMixture(dtype=torch.float32)
    with torch.no_grad():
        model.logits.copy_(torch.tensor([np.log(0.25), np.log(0.75)]))
        model.mean.copy_(torch.tensor([[-2., -2.], [2., 2.]]))
        model.log.fill_(-7)
    torch.manual_seed(1)
    x = torch.zeros(16000, 3)
    act, logp, deterministic = sample_action(model, x, torch.ones(2), torch.zeros(2))
    assert ((act[:, 0] > 0) == (act[:, 1] > 0)).all()
    assert abs(float((act[:, 0] > 0).float().mean()) - 0.75) < 0.02
    assert torch.isfinite(logp).all()
    assert torch.allclose(deterministic, representative_action(model, x, torch.ones(2), torch.zeros(2)))


def buffer(n=48, task_id=0):
    return dict(obs=np.random.randn(n, 6).astype(np.float32),
        actions=np.zeros((n, 3), np.float32), task_ids=np.full(n, task_id, np.int32),
        source_ids=np.full(n, task_id, np.int32))


def make_agent(mode, distill, mass, root=None, latest=None, space="policy"):
    return CkaRlAgent(6, 3, root, latest, pool_size=2, distillation=distill,
        fusion_mode=mode, use_alpha_mass=mass, composition_space=space,
        similarity_samples=16, distill_max_samples=32, max_distill_buffer=48,
        projection_epochs=2, projection_max_samples=48, distill_epochs=2, distill_batch_size=16)


@pytest.mark.parametrize("mode,distill,mass", [
    ("classic_cka", False, False), ("classic_cka", True, False),
    ("weight_delta", False, True), ("weight_delta", True, True)])
def test_policy_lifecycle_projection_merge_and_snapshots(tmp_path, mode, distill, mass):
    torch.manual_seed(2); np.random.seed(2)
    root = tmp_path / "root"
    model = make_agent(mode, distill, mass)
    model.save_policy_snapshot(str(root))
    model.set_own_buffer(buffer(task_id=0)); model.set_base(); model.save(str(root))
    previous = root
    for task in range(1, 4):
        model = make_agent(mode, distill, mass, str(root), str(previous))
        x = torch.randn(9, 6)
        before = effective_theta_vector(model)
        model.set_mixture_warmup(True)
        assert len(model.policy_components(x)[2]) == len(model.mean_pool.pool)
        if mass:
            assert float(model.mean_pool.effective_alpha_mass().detach()) == 1.0
        model.set_mixture_warmup(False)
        assert before.shape == effective_theta_vector(model).shape
        _, _, w = model.policy_components(x)
        assert torch.all(w >= 0) and torch.allclose(w.sum(), torch.tensor(1.))
        if mass:
            assert abs(float(model.mean_pool.effective_alpha_mass().detach()) - 0.95) < 1e-6
        history = [[e[k].clone() for k in KEYS] for e in model.mean_pool.pool]
        def q(obs, action): return -action.square().sum(-1, keepdim=True)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        loss = sac_actor_objective(model, x, q, q, 0.1, torch.ones(3), torch.zeros(3))
        opt.zero_grad(); loss.backward(); opt.step()
        for saved, entry in zip(history, model.mean_pool.pool):
            for v, k in zip(saved, KEYS): assert torch.equal(v, entry[k])
        expected = model.policy_components(x)
        checkpoint = tmp_path / str(task)
        model.save_policy_snapshot(str(checkpoint))
        frozen = FrozenCkaPolicy.load(str(checkpoint))
        for expected_t, actual_t in zip(expected, frozen.policy_components(x)):
            assert torch.allclose(expected_t, actual_t, atol=1e-6)
        model.set_own_buffer(buffer(task_id=task))
        model.finalize(); model.save(str(checkpoint))
        assert len(model.mean_pool.pool) == len(model.logstd_pool.pool) == min(task + 1, 2)
        assert model.last_projection_metrics["policy/projection_val_component_kl"] <= model.last_projection_metrics["policy/projection_initial_val_component_kl"] + 1e-6
        assert all(len(e['buffer']['obs']) <= 48 for e in model.mean_pool.pool)
        loaded, _ = load_finalized_policy(str(checkpoint), torch.device("cpu"))
        assert len(loaded.policy_components(x)[2]) == len(model.mean_pool.pool)
        assert torch.isfinite(sample_action(loaded, x, torch.ones(3), torch.zeros(3))[1]).all()
        previous = checkpoint


def test_parameter_mode_unchanged_distribution_and_mass_bounds(tmp_path):
    model = make_agent("weight_delta", True, True, space="parameter")
    x = torch.randn(4, 6)
    mu, raw = model(x)
    means, logs, weights = components(model, x)
    assert torch.allclose(mu, means[:, 0], atol=1e-7)
    assert torch.allclose(bound_log_std(raw), logs[:, 0], atol=1e-6)
    assert torch.equal(weights, torch.ones(1))
    model.set_own_buffer(buffer()); model.set_base(); model.save(str(tmp_path))
    model = make_agent("weight_delta", True, True, str(tmp_path), str(tmp_path), space="parameter")
    for v in (-100., -4., 0., 4., 100.):
        with torch.no_grad(): model.alpha_mass.fill_(v)
        mass = float(model.mean_pool.effective_alpha_mass().detach())
        assert 0.0 <= mass <= 1.0
    model.set_mixture_warmup(True)
    assert float(model.mean_pool.effective_alpha_mass().detach()) == 1.0


def test_budget_and_warmup_including_singleton():
    budget = TaskBudget(20000, 1000)
    assert budget.training == 19000 and budget.training + budget.frozen_tail == budget.total
    assert mixture_warmup_active(5001, 5000, 5000, "weight_delta", 1)
    assert not mixture_warmup_active(10000, 5000, 5000, "weight_delta", 1)
    assert not mixture_warmup_active(5001, 5000, 5000, "weight_delta", 0)
    for total, tail in ((100, 100), (100, -1), (0, 0)):
        with pytest.raises(ValueError): TaskBudget(total, tail)
    b = buffer(20)
    b['source_ids'] = np.arange(20)
    b['obs'][:, 0] = b['source_ids']
    clipped = bounded_buffer(b, 7)
    assert len(clipped['obs']) == 7
    assert np.all(clipped['obs'][:, 0] == clipped['source_ids'])


@pytest.mark.parametrize("family,envfile", [('half-cheetah','halfcheetah_envs.py'), ('Walker2D','walker2d_envs.py')])
def test_context_wrapper_is_identity_and_not_installed(family, envfile):
    source = (ROOT/family/envfile).read_text()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_task_specific_observation')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(envfile), 'exec'), namespace)
    obs = np.arange(17, dtype=np.float32)
    assert np.array_equal(namespace['make_task_specific_observation'](object(), obs), obs)
    tasks = ast.parse((ROOT/family/'tasks.py').read_text())
    active_calls = [n.func.id for n in ast.walk(tasks) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert 'TaskConditionedObservationWrapper' not in active_calls


def test_source_lineage_balancing_ablation():
    np.random.seed(7)
    source_ids = np.array([0] * 100 + [1] * 100 + [2] * 10, dtype=np.int32)
    idx = balanced_lineage_indices(source_ids, 90)
    ids, counts = np.unique(source_ids[idx], return_counts=True)
    assert dict(zip(ids.tolist(), counts.tolist())) == {0: 40, 1: 40, 2: 10}

    b1 = {
        "obs": np.arange(100, dtype=np.float32)[:, None],
        "task_ids": np.zeros(100, dtype=np.int32),
        "source_ids": np.array([0] * 90 + [1] * 10, dtype=np.int32),
    }
    b2 = {
        "obs": np.arange(100, 200, dtype=np.float32)[:, None],
        "task_ids": np.ones(100, dtype=np.int32),
        "source_ids": np.full(100, 2, dtype=np.int32),
    }
    merged = HeadPool.merge_buffers(
        b1, b2, 90, balance_source_lineages=True
    )
    ids, counts = np.unique(merged["source_ids"], return_counts=True)
    # Source 1 only has ten available rows; its unused quota is redistributed
    # evenly to the other original source lineages.
    assert dict(zip(ids.tolist(), counts.tolist())) == {0: 40, 1: 10, 2: 40}

    with pytest.raises(RuntimeError):
        HeadPool.merge_buffers(
            {"obs": b1["obs"], "task_ids": b1["task_ids"]},
            {"obs": b2["obs"], "task_ids": b2["task_ids"]},
            90,
            balance_source_lineages=True,
        )
