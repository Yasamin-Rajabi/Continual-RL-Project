"""Task-process checkpointing for the seven new baselines."""
from __future__ import annotations
import dataclasses, json, os, random, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from paper_runs.io import digest,file_hash,atomic_json,read_json

TRAIN_FIELDS=('total_timesteps','gamma','tau','batch_size','learning_starts','random_actions_end',
              'policy_lr','q_lr','alpha','autotune','autotune_init_from_alpha','eval_every','num_evals',
              'encoder_linear_out','distill_extra_steps')

def settings(spec,args):
    c={k:getattr(args,k) for k in TRAIN_FIELDS}
    c.update(buffer_size=1000000,policy_frequency=2,target_network_frequency=1,
             frozen_tail_steps=args.distill_extra_steps,eval_action_mode=spec['mode'],
             hidden_dim=128,packnet_capacity='equal_share',packnet_keep=1.0,
             packnet_retrain_fraction=.3,cbp_replacement_rate=1e-4,
             cbp_maturity_threshold=100,cbp_decay_rate=.99)
    # Deliberate new-baseline architecture choices; never changes CKA/ETHOS settings.
    c.update(spec.get('baseline_settings',{}))
    return c

def path_for(spec,seed,index):
    task=spec['sequence'][index];method=spec['method'];suite=spec['suite']
    leaf=f'{suite}__task_{task}__{method}__run_sac__{seed}'
    tag=Path(suite)/method/f'seed_{seed}'/f'seq_{index}'/leaf
    root=Path(spec['run_root'])
    return root/'agents'/tag,root/'runs'/tag,root/'analysis'/tag

# These are the exact hashes of the shared baseline trainer in the uploaded
# pre-MiniGrid paper code. Keeping them for the continuous environments means
# adding a discrete branch does not invalidate already-completed
# HalfCheetah/Walker2D/AntDir baseline manifests. The MiniGrid source identity
# always hashes the current files normally.
_LEGACY_CONTINUOUS_SHARED_HASHES = {
    "paper_runs/baseline_runner.py": "3ff72ccbe4e2ded93f4296b757eec36e9c1d3ffbc1008953a3d4b017778dee61",
    "paper_runs/baseline_sac.py": "ea18a9a355aa91636e2220dd29e567c98a77f86e76e7ecde01a46b9155f333b5",
}


def source_signature(spec):
    root=Path(spec['project_root']);env=root/spec['environment']
    files=[root/'paper_runs'/name for name in
           ('baseline_runner.py','baseline_sac.py','baselines.py','layers.py','io.py')]
    files += sorted((root/'paper_runs'/'vendor').glob('*.py'))
    # Evaluator, launcher and smoke-test edits are not training changes.
    files += [env/x for x in ('tasks.py','shared_arch.py','policy_composition.py','policy_utils.py','training_protocol.py')]
    files += sorted(env.glob('*envs.py'))
    hashes={str(path.relative_to(root)):file_hash(path) for path in files}
    if spec['environment'] != 'minigrid':
        hashes.update(_LEGACY_CONTINUOUS_SHARED_HASHES)
    return digest(hashes)

def expected(spec,args,seed,index,parent):
    import experiment_identity as identity
    return dict(method=spec['method'],suite=spec['suite'],sequence=spec['sequence'],
                seq_idx=index,task_id=spec['sequence'][index],seed=seed,
                training_config=settings(spec,args),source=source_signature(spec),
                runtime=identity.runtime_versions(),parent_signature=parent)

def valid(spec,args,seed,index,parent,verify_bytes=False):
    run,_,_=path_for(spec,seed,index);m=read_json(run/'baseline_manifest.json')
    if m is None:return False,'missing baseline manifest',None
    want=expected(spec,args,seed,index,parent)
    if m.get('identity')!=want:return False,'baseline configuration/source/runtime/parent mismatch',m
    for name in ('agent_state.pt','policy_snapshot.pt','final_metrics.json'):
        p=run/name
        if not p.is_file():return False,f'missing {name}',m
        if verify_bytes and file_hash(p)!=m.get('files',{}).get(name):return False,f'corrupted {name}',m
    return True,'match',m

def check_chain(spec,args,seed,verify_bytes=False):
    parent=None;all_ok=True
    for i in range(len(spec['sequence'])):
        good,reason,m=valid(spec,args,seed,i,parent,verify_bytes)
        path,_,_=path_for(spec,seed,i)
        if not good:
            all_ok=False
            if m is not None and reason.endswith('mismatch'):
                raise RuntimeError(f'{path}: {reason}. Use a distinct --comment; existing results are not overwritten.')
        parent=None if m is None else m.get('signature')
    return all_ok

def train_chain(spec,args,spec_file):
    from paper_runs.worker import archive_partial
    parent=None
    for i in range(len(spec['sequence'])):
        good,reason,m=valid(spec,args,spec['seed'],i,parent,verify_bytes=True)
        run,event,analysis=path_for(spec,spec['seed'],i)
        if good:
            print(f'[baseline] SKIP {spec["method"]} seed={spec["seed"]} seq={i}: complete',flush=True)
            parent=m['signature'];continue
        if m is not None and reason.endswith('mismatch'):
            raise RuntimeError(f'{run}: {reason}; use a new --comment')
        archive_partial(spec,[run,event,analysis])
        cmd=[sys.executable,str(Path(spec['project_root'])/'paper_runs/worker.py'),
             '--spec',str(spec_file),'--baseline-task',str(i)]
        subprocess.run(cmd,check=True)
        good,reason,m=valid(spec,args,spec['seed'],i,parent,verify_bytes=True)
        if not good:raise RuntimeError(f'Baseline did not save a valid checkpoint: {run}: {reason}')
        parent=m['signature']

def train_one(spec,args,index):
    from tasks import get_task
    from csv_summary_writer import CsvSummaryWriter
    from paper_runs.baselines import BaselineAgent,TaskContext
    from paper_runs.baseline_sac import train_task
    seed=int(spec['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    task=spec['sequence'][index]
    cfg=settings(spec,args)
    prev=None
    if index:
        prev,_,_=path_for(spec,seed,index-1)
        agent=torch.load(prev/'agent_state.pt',map_location='cpu',weights_only=False)
        parent=read_json(prev/'baseline_manifest.json')['signature']
    else:
        probe=get_task(task,task_suite=spec['suite'])
        obs=int(np.prod(probe.observation_space.shape))
        act=(int(probe.action_space.n) if hasattr(probe.action_space, "n")
             else int(np.prod(probe.action_space.shape)))
        probe.close()
        agent=BaselineAgent(spec['method'],obs,act,len(spec['sequence']),
                 **{k:cfg[k] for k in ('hidden_dim','encoder_linear_out','packnet_capacity','packnet_keep',
                   'packnet_retrain_fraction','cbp_replacement_rate','cbp_maturity_threshold','cbp_decay_rate')})
        parent=None
    run,event,analysis=path_for(spec,seed,index)
    run.mkdir(parents=True,exist_ok=True);analysis.mkdir(parents=True,exist_ok=True)
    c=SimpleNamespace(**dict(cfg,seed=seed))
    ctx=TaskContext(task,index,spec['suite'],seed,task not in spec['sequence'][:index])
    writer=CsvSummaryWriter(str(event))
    try:
        writer.add_text('hyperparameters','|param|value|\n|-|-|\n'+'\n'.join(f'|{k}|{v}|' for k,v in vars(c).items()))
        results=train_task(agent,ctx,c,writer,device)
        # No live optimizer references (CBP statistics are retained separately).
        agent.detach_optimizer();agent.cpu()
        for name,obj in [('agent_state.pt',agent),('policy_snapshot.pt',agent.policy)]:
            tmp=run/('.'+name+'.tmp');torch.save(obj,tmp);os.replace(tmp,run/name)
        atomic_json(run/'final_metrics.json',results)
        atomic_json(analysis/'interaction_budget.json',dict(total=cfg['total_timesteps'],frozen_tail=cfg['frozen_tail_steps'],
                    training=cfg['total_timesteps']-cfg['frozen_tail_steps'],**results))
        ident=expected(spec,args,seed,index,parent)
        file_sums={n:file_hash(run/n) for n in ('agent_state.pt','policy_snapshot.pt','final_metrics.json')}
        atomic_json(run/'baseline_manifest.json',dict(identity=ident,files=file_sums,signature=digest([ident,file_sums])))
    finally:writer.close()
    print(f'[baseline] DONE {spec["method"]} seed={seed} seq={index}: {run}',flush=True)
