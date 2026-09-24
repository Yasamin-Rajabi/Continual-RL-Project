import copy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'half-cheetah'),str(ROOT)]
import pytest,torch,numpy as np
from cka_rl import CkaRlAgent

torch.set_num_threads(1)
KEYS=('l0_weight','l0_bias','l2_weight','l2_bias')
def buffer(t):
    return dict(obs=np.random.randn(20,6).astype('float32'),actions=np.zeros((20,3),'float32'),
                task_ids=np.full(20,t),source_ids=np.full(20,t))
def agent(root=None,prev=None,mode='kl_merge'):
    return CkaRlAgent(6,3,root,prev,pool_size=2,fusion_mode='weight_delta',
        composition_space='policy',use_alpha_mass=True,distillation=True,
        merge_ablation=mode,similarity_samples=8,distill_max_samples=24,
        distill_epochs=1,distill_batch_size=8,projection_epochs=1,projection_max_samples=20,
        max_distill_buffer=32)

@pytest.mark.parametrize('mode',['kl_merge','random_merge','kl_discard'])
def test_merge_mode_lifecycle(tmp_path,mode):
    torch.manual_seed(5);np.random.seed(5)
    root=str(tmp_path/'0');prev=root
    a=agent();a.set_own_buffer(buffer(0));a.set_base();a.save(root)
    for t in (1,2):
        a=agent(root,prev,mode);a.set_own_buffer(buffer(t))
        if mode=='kl_discard' and t==2:
            a._distill_pair=lambda *_:(_ for _ in ()).throw(AssertionError('Discard must not distill'))
        a.finalize();prev=str(tmp_path/str(t));a.save(prev)
        assert len(a.mean_pool.pool)==len(a.logstd_pool.pool)==min(t+1,2)
    info=a.last_merge_info
    assert info is not None
    if mode=='kl_discard':assert info['discard_only'] and not info['used_distillation']
    else:assert info['used_distillation']
    if mode=='random_merge':assert info['selected_random_pair']


def test_discard_choice_uses_both_parents():
    # An explicit assertion on the source prevents reverting to remove=idx2.
    import inspect
    text=inspect.getsource(CkaRlAgent.finalize)
    assert 'np.random.choice([idx1, idx2])' in text
    np.random.seed(47)
    assert set(int(np.random.choice([1,3])) for _ in range(100))=={1,3}


def test_cli_is_forwarded():
    for env in ['half-cheetah','Walker2D','AntDir']:
        sac=(ROOT/env/'run_sac.py').read_text();runner=(ROOT/env/'run_continual_benchmark.py').read_text()
        assert 'merge_ablation=args.merge_ablation' in sac
        assert 'f"--merge-ablation={args.merge_ablation}"' in runner
