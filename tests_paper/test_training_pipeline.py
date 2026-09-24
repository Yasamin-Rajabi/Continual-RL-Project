import sys,csv,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'half-cheetah'),str(Path(__file__).parent)]
import pytest,torch
from fake_dependencies import install,TinyEnv
from paper_runs.config import read_job,replace_cli
from paper_runs.io import atomic_json

torch.set_num_threads(1)

@pytest.mark.parametrize('method',['ft_n','prognet','packnet','masknet','crelus','componet','cbpnet'])
def test_train_reload_csv_and_evaluate_real_torch_fake_env(monkeypatch,tmp_path,method):
    install(monkeypatch)
    import tasks
    monkeypatch.setattr(tasks,'get_task',lambda task_id,task_suite=None,**kw:TinyEnv(task_id))
    # Reload modules with the test doubles, never modify the production source.
    for name in ['metrics','csv_summary_writer','run_continual_benchmark','paper_runs.baseline_sac','paper_runs.evaluation']:
        monkeypatch.delitem(sys.modules,name,raising=False)
    from paper_runs.worker import benchmark_args,reference_root
    from paper_runs.baseline_runner import train_one,valid,path_for
    from paper_runs.evaluation import evaluate_baseline
    # Plot rendering has its own collector tests. Keep this mechanics fixture
    # focused on real Torch training, metrics and CSV/cell-cache persistence.
    import plots
    monkeypatch.setattr(plots, 'plot_retention', lambda *a, **kw: None)
    s=read_job(ROOT,'half-cheetah')
    s.update(project_root=str(ROOT),suite='halfcheetah_wind_vel',sequence=[0,1,0],num_tasks=2,
             method=method,seed=1,seeds=[1,2],mode='deterministic',phase='train',comment='smoke',
             baseline_eval_protocol='reward_route',EXPERIMENT_ROOT=str(tmp_path),
             run_root=str(tmp_path/'main'/f'{method}_deterministic'))
    argv=s['COMMON_ARGS']
    for k,v in dict(total_timesteps=32,distill_buffer_steps=4,learning_starts=4,random_actions_end=4,
                    batch_size=8,eval_every=8,num_evals=1,retention_eval_episodes=1,test_adapt_steps=4).items():
        argv=replace_cli(argv,k,v)
    s['arguments']=argv
    args=benchmark_args(s);args.cpu=True
    parent=None
    for i in range(3):
        train_one(s,args,i)
        ok,why,m=valid(s,args,1,i,parent,True);assert ok,why;parent=m['signature']
        _,events,_=path_for(s,1,i)
        tags={r['tag'] for r in csv.DictReader((events/'scalars.csv').open())}
        assert {'charts/test_success','charts/test_episodic_return','charts/final_success','charts/final_return'}<=tags
    # Scratch has saved curves; FT doesn't need to execute/retrain scratch.
    ref=reference_root(s,args)
    import scratch_baselines as scratch
    for seed in [101,102,103]:
        d=scratch.scratch_event_dir(ref/'runs',s['suite'],1,32,seed,'plain');d.mkdir(parents=True)
        (d/'scalars.csv').write_text('wall_time,step,tag,value\n0,0,charts/test_success,0\n0,32,charts/test_success,0.5\n0,0,charts/test_episodic_return,-8\n0,32,charts/test_episodic_return,-4\n')
    evaluate_baseline(s,args)
    output=Path(args.plots_root)/s['suite']
    rows=list(csv.DictReader((output/'survey_metrics.csv').open()))
    assert len(rows)==1 and rows[0]['seed']=='1'
    status=json.loads((output/'seed_status.json').read_text());assert status['used']==[1] and '2' in status['skipped']
    cells=output/'cell_cache'/f'{method}_seed_1.json';before=cells.read_bytes()
    # Completed cell reuse must not call the environment evaluator a second time.
    import paper_runs.evaluation as ev
    monkeypatch.setattr(ev,'eval_cell',lambda *a:(_ for _ in ()).throw(AssertionError('cache not reused')))
    evaluate_baseline(s,args)
    assert cells.read_bytes()==before
