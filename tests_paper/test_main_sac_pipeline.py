"""Exercise the actual existing run_sac.py main body (test environment only).

Tyro parsing is stubbed to a dataclass instance; the real Torch training loop,
projection, consolidation, post-finalize logging, serialization and manifests
all run. This catches logging paths the lower-level model tests cannot cover.
"""
import sys,types,runpy,dataclasses,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'half-cheetah'),str(Path(__file__).parent)]
import pytest,torch
from fake_dependencies import install,TinyEnv

torch.set_num_threads(1)

@pytest.mark.parametrize('mode',['baseline','kl_merge','random_merge','kl_discard'])
def test_main_sac_three_tasks_including_post_finalize_logger(monkeypatch,tmp_path,mode):
    install(monkeypatch)
    import tasks
    monkeypatch.setattr(tasks,'get_task',lambda task_id,task_suite=None,**kw:TinyEnv(task_id))
    monkeypatch.delitem(sys.modules,'csv_summary_writer',raising=False)
    tyro=types.ModuleType('tyro');monkeypatch.setitem(sys.modules,'tyro',tyro)
    prior=[]
    for i,t in enumerate([0,1,0]):
        tag=f'halfcheetah_wind_vel/{mode}/seed_1/seq_{i}'
        cfg=dict(task_suite='halfcheetah_wind_vel',task_id=t,seq_idx=i,seed=1,cuda=False,
                 fusion_mode='classic_cka' if mode=='baseline' else 'weight_delta',
                 composition_space='parameter' if mode=='baseline' else 'policy',
                 merge_ablation='kl_merge' if mode=='baseline' else mode,
                 use_alpha_mass=mode!='baseline',distillation=mode!='baseline',
                 alpha_mass_reg=0.,pool_size=2,train_shared=False,encoder_from_base=True,
                 prev_units=tuple(prior if len(prior)<2 else [prior[0],prior[-1]]),
                 total_timesteps=32,distill_extra_steps=8,learning_starts=4,random_actions_end=4,
                 alpha_warmup_steps=4,batch_size=8,eval_every=8,num_evals=1,
                 similarity_samples=8,max_distill_buffer=32,distill_max_samples=32,
                 distill_epochs=1,distill_batch_size=8,projection_epochs=1,projection_max_samples=16,
                 save_analysis_snapshots=False,analysis_log_every=8,
                 runs_root=str(tmp_path/'runs'),analysis_root=str(tmp_path/'analysis'),
                 save_dir=str(tmp_path/'agents'/f'seq_{i}'),tag=tag)
        tyro.cli=lambda cls:dataclasses.replace(cls(),**cfg)
        runpy.run_path(str(ROOT/'half-cheetah'/'run_sac.py'),run_name='__main__')
        leaf=f'halfcheetah_wind_vel__task_{t}__cka-rl__run_sac__1'
        path=Path(cfg['save_dir'])/leaf;prior.append(path)
        assert (path/'run_manifest.json').is_file()
    rows=list(csv.DictReader((tmp_path/'runs'/tag/leaf/'scalars.csv').open()))
    tags={r['tag'] for r in rows}
    if mode=='random_merge':assert 'analysis/merge/random_pair' in tags
    if mode=='kl_discard':
        assert 'analysis/merge/discarded_index' in tags
        assert any(r['tag']=='analysis/merge/used_distillation' and float(r['value'])==0 for r in rows)
