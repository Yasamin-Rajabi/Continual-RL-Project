#!/usr/bin/env python3
"""Real dependency smoke tests. No mocks and no production output directories.

Default: reset and randomly step every task in the chosen family.
--train: additionally run a tiny 3-occurrence chain for all requested methods,
including both nondefault compression ablations for combined_policy.
--evaluate: also generate one matched scratch reference and retention/FT metrics.
The short-run hyperparameters below are TEST SETTINGS, not experiment defaults.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, tempfile
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from paper_runs.config import ENVIRONMENTS,METHODS,read_job,cli_dict,suite_info,replace_cli
from paper_runs.io import atomic_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environments',nargs='+',choices=ENVIRONMENTS,default=list(ENVIRONMENTS))
    p.add_argument('--train',action='store_true')
    p.add_argument('--evaluate',action='store_true')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--output',type=Path,help='Empty TEST directory; default is a fresh temporary directory')
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.evaluate and not a.train:p.error('--evaluate requires --train')
    if not a.child:
        out=a.output or Path(tempfile.mkdtemp(prefix='crl_real_smoke_'))
        if a.output and out.exists() and any(out.iterdir()):p.error('--output must be empty; production folders are never modified')
        out.mkdir(parents=True,exist_ok=True)
        for env in a.environments:
            command=[sys.executable,__file__,'--child','--environments',env,'--output',str(out/env),'--methods',*a.methods]
            if a.train:command+=['--train']
            if a.evaluate:command+=['--evaluate']
            subprocess.run(command,check=True)
        print(f'REAL SMOKE PASS: output at {out}');return
    env=a.environments[0];sys.path.insert(0,str(PROJECT/env));os.chdir(PROJECT/env)
    import numpy as np,torch,gymnasium,mujoco
    print(f'Runtime: Python={sys.version.split()[0]} Torch={torch.__version__} Gymnasium={gymnasium.__version__} MuJoCo={mujoco.__version__}',flush=True)
    from tasks import get_task
    base=read_job(PROJECT,env);suite=cli_dict(base['COMMON_ARGS'])['task_suites']
    _,n=suite_info(PROJECT,env,suite)
    for task in range(n):
        e=get_task(task,task_suite=suite)
        try:
            obs,_=e.reset(seed=1);e.action_space.seed(1)
            assert np.all(np.isfinite(obs))
            for _ in range(32):
                obs,r,d,t,info=e.step(e.action_space.sample())
                assert np.all(np.isfinite(obs)) and np.isfinite(r)
                assert 'success' in info
                if d or t:obs,_=e.reset()
            print(f'ENV PASS {env}/{suite} task={task} obs={e.observation_space.shape} action={e.action_space.shape}',flush=True)
        finally:e.close()
    if not a.train:return
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    argv=list(base['COMMON_ARGS'])
    options=dict(total_timesteps=128,distill_buffer_steps=8,learning_starts=16,random_actions_end=16,
        alpha_warmup_steps=8,batch_size=8,pool_size=2,eval_every=64,num_evals=1,
        retention_eval_episodes=1,test_adapt_steps=8,similarity_samples=16,max_distill_buffer=32,
        distill_max_samples=32,distill_epochs=1,distill_batch_size=8,projection_epochs=1,
        projection_max_samples=32,analysis_log_every=64)
    for key,val in options.items():argv=replace_cli(argv,key,val)
    common=dict(base,project_root=str(PROJECT),suite=suite,sequence=[0,1,0],num_tasks=n,
        seed=1,seeds=[1],SCRATCH_SEEDS=[101],mode='deterministic',comment='smoke',
        phase='train',arguments=argv,EXPERIMENT_ROOT=str(out),reuse_equivalent=False,
        baseline_eval_protocol='reward_route')
    specfiles=[]
    if a.evaluate:
        spec=dict(common,method='scratch',phase='scratch',seed=101)
        path=out/'scratch_spec.json';atomic_json(path,spec)
        subprocess.run([sys.executable,str(PROJECT/'paper_runs/worker.py'),'--spec',str(path)],check=True)
    for method in a.methods:
        modes=['kl_merge','random_merge','kl_discard'] if method=='combined_policy' else ['kl_merge']
        for compression in modes:
            name=method+'_'+compression
            spec=dict(common,method=method,run_root=str(out/'main'/name))
            spec['arguments']=replace_cli(argv,'merge_ablation',compression)
            path=out/(name+'.json');atomic_json(path,spec);specfiles.append(path)
            command=[sys.executable,str(PROJECT/'paper_runs/worker.py'),'--spec',str(path)]
            subprocess.run(command,check=True)
            # A rerun must recognize the complete chain without training.
            result=subprocess.run(command+['--check'])
            if result.returncode!=10:raise RuntimeError(f'Completed-run reuse failed: {name} ({result.returncode})')
            if a.evaluate:
                spec['phase']='eval';path_eval=out/(name+'_eval.json');atomic_json(path_eval,spec)
                subprocess.run([sys.executable,str(PROJECT/'paper_runs/worker.py'),'--spec',str(path_eval)],check=True)
            print(f'TRAIN/RELOAD PASS {env}/{name}',flush=True)
    print(f'PASS {env}: {len(specfiles)} short method/ablation chains',flush=True)

if __name__=='__main__':main()
