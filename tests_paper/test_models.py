"""Real PyTorch regression tests; no Gymnasium or simulator required."""
import copy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'half-cheetah')]
import pytest,torch,numpy as np
from paper_runs.baselines import BaselineAgent,TaskContext,METHODS
from policy_composition import sac_actor_objective,representative_action
from training_protocol import TaskBudget

torch.set_num_threads(1)

@pytest.mark.parametrize('method',METHODS)
def test_three_occurrences_save_reload_and_gradients(method,tmp_path):
    torch.manual_seed(31);np.random.seed(31)
    agent=BaselineAgent(method,6,3,3,hidden_dim=16,cbp_maturity_threshold=1,cbp_replacement_rate=.3)
    x=torch.randn(7,6);scale=torch.ones(3);bias=torch.zeros(3)
    old=None
    for i,t in enumerate([0,1,0]):
        ctx=TaskContext(t,i,'synthetic',31,i<2)
        agent.on_task_start(ctx)
        opt=torch.optim.Adam(agent.trainable_actor_parameters(),lr=1e-3);agent.attach_optimizer(opt,'cpu')
        for step in range(10):
            agent.on_phase_boundary(step,TaskBudget(10,0))
            def q(obs,a):return -(a-(i%2)*.4).square().sum(-1,keepdim=True)
            loss=sac_actor_objective(agent.policy,x,q,q,.2,scale,bias)
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward()
            assert any(p.grad is not None for p in agent.trainable_actor_parameters())
            agent.before_optimizer_step();opt.step();agent.after_optimizer_step()
        agent.on_task_end(ctx);agent.detach_optimizer()
        expected=representative_action(agent.policy,x,scale,bias).detach()
        file=tmp_path/f'{i}.pt';torch.save(agent,file)
        agent=torch.load(file,map_location='cpu',weights_only=False)
        assert torch.equal(expected,representative_action(agent.policy,x,scale,bias))
        policies=agent.candidates()
        assert len(policies)==(1 if method in ('crelus','cbpnet') else i+1)
        oldnow=representative_action(policies[0],x,scale,bias)
        if old is not None and method not in ('crelus','cbpnet'):
            assert torch.allclose(old,oldnow,atol=1e-7),f'{method} changed its old policy'
        old=oldnow.detach().clone()
        if method=='cbpnet':assert agent._gnt_state['ages'].sum()>0


def test_masknet_base_weights_frozen_and_scores_update():
    a=BaselineAgent('masknet',6,3,3,hidden_dim=16);a.on_task_start(TaskContext(0,0,'x',1))
    weights={n:p.clone() for n,p in a.policy.head.named_parameters() if n.endswith('weight')}
    opt=torch.optim.Adam(a.trainable_actor_parameters(),lr=.01)
    x=torch.randn(9,6);loss=sum(t.square().mean() for t in a.policy(x));loss.backward();opt.step()
    for n,p in a.policy.head.named_parameters():
        if n in weights:assert torch.equal(weights[n],p) and not p.requires_grad


def test_componet_flat_storage():
    a=BaselineAgent('componet',6,3,4)
    for i in range(4):a.on_task_start(TaskContext(i,i,'x',1));a.on_task_end(TaskContext(i,i,'x',1))
    assert not any(hasattr(u,'previous_units') for u in a.policy.units)
    assert len(a.policy.units)==4

@pytest.mark.parametrize('method',METHODS)
def test_ant_dimensions(method):
    a=BaselineAgent(method,111,8,2,hidden_dim=16)
    for i in range(2):
        ctx=TaskContext(i,i,'ant_dir',1);a.on_task_start(ctx)
        out=a.policy(torch.randn(5,111))
        assert out[0].shape==out[1].shape==(5,8)
        loss=sum(v.square().mean() for v in out)
        loss.backward()
        assert any(p.grad is not None for p in a.trainable_actor_parameters())
        a.on_task_end(ctx)
