import sys, json, types, copy, subprocess, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'half-cheetah')]
from paper_runs.config import read_job,cli_dict,make_cases,suite_info
from paper_runs.io import read_json
import pytest

def test_scratch_reference_reservation_separates_core_settings(monkeypatch,tmp_path):
    from paper_runs import worker
    def scratch_args(spec,args,root):return types.SimpleNamespace(save_root=str(root/'models'))
    fake=types.SimpleNamespace(
        _expected_training_config=lambda suite,task,steps,seed,args,variant:dict(task_suite=suite,total_timesteps=steps,task_id=task,seed=seed),
        scratch_checkpoint_dir=lambda root,suite,task,steps,seed,variant:Path(root)/suite/str(task)/str(seed),
        checkpoint_complete=lambda run:False)
    monkeypatch.setitem(sys.modules,'scratch_baselines',fake)
    monkeypatch.setattr(worker,'scratch_args',scratch_args)
    spec=dict(EXPERIMENT_ROOT=str(tmp_path),mode='deterministic',suite='x',sequence=[0,1,0],SCRATCH_SEEDS=[101,102,103])
    a=types.SimpleNamespace(total_timesteps=100)
    first=worker.reference_root(spec,a)
    assert first==tmp_path/'scratch'/'deterministic'
    second=worker.reference_root(spec,types.SimpleNamespace(total_timesteps=200))
    assert second.parent==first/'references' and second!=first
    assert worker.reference_root(spec,a)==first
    assert read_json(first/'.paper_reference_settings.json')['core_settings']['total_timesteps']==100

def test_launch_preview_has_no_side_effects_or_slurm_calls(tmp_path):
    env=os.environ.copy();env['EXPERIMENT_ROOT']=str(tmp_path/'new_experiment')
    command=[sys.executable,str(ROOT/'paper_experiments.py'),'--environments','half-cheetah',
             '--groups','main','kl','--methods','combined_policy']
    proc=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
    assert 'Planned workers: 12' in proc.stdout # 3 reference + 3 modes * 3 seeds
    assert not (tmp_path/'new_experiment').exists()

def test_supplied_shell_files_are_not_changed():
    # Hashes are recorded from the uploaded main archive, not generated defaults.
    report=ROOT/'tests_paper'/'original_shell_sha256.json'
    import hashlib
    hashes=json.loads(report.read_text())
    for name,expected in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
