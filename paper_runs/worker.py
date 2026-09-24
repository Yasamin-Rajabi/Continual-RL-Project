"""Run one frozen Slurm spec inside the selected experiment container."""
from __future__ import annotations
import argparse,contextlib,copy,datetime,os,shutil,sys,tempfile
from pathlib import Path
# The script is called by absolute path while cwd is the environment directory.
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from paper_runs.io import atomic_json,read_json,digest,file_hash,lock

@contextlib.contextmanager
def arguments(argv):
    old=sys.argv;sys.argv=[old[0]]+list(argv)
    try:yield
    finally:sys.argv=old

def benchmark_args(spec):
    import run_continual_benchmark as benchmark
    root=Path(spec.get('run_root',Path(spec['EXPERIMENT_ROOT'])/'main'/'_scratch_preflight'))
    argv=list(spec['arguments'])+['--seeds',*map(str,spec['seeds']),'--task-sequence',*map(str,spec['sequence']),
        '--eval-action-mode',spec['mode'],'--save-root',str(root/'agents'),
        '--runs-root',str(root/'runs'),'--analysis-root',str(root/'analysis'),'--plots-root',str(root/'plots'),
        '--scratch-seeds',*map(str,spec['SCRATCH_SEEDS'])]
    with arguments(argv):args=benchmark.parse_args()
    args.task_sequence=list(spec['sequence'])
    return args

def archive_partial(spec,paths):
    """Preserve old attempt bytes instead of merging event files or deleting evidence."""
    existing=[Path(p) for p in paths if Path(p).exists()]
    if not existing:return
    root=Path(spec['EXPERIMENT_ROOT']); stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')
    archive=root/'.paper_suite'/'incomplete_attempts'/stamp
    for p in existing:
        try:rel=p.relative_to(root)
        except ValueError:raise ValueError(f'Refusing to archive outside experiment root: {p}')
        dst=archive/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(p),str(dst))
    print(f'[archive] Prior incomplete attempt preserved under {archive}',flush=True)

def method_config(spec,args):
    import run_continual_benchmark as benchmark
    name='baseline' if spec['method']=='baseline' else 'combined'
    cfg=dict(benchmark.CONDITIONS[name],composition_space='parameter' if name=='baseline' else 'policy')
    return benchmark._effective_condition_config(args,cfg)

def check_cka_chain(spec,args,seed,archive=False):
    import metrics,run_continual_benchmark as benchmark
    previous=[];all_good=True;gap=False;cfg=method_config(spec,args)
    for i,t in enumerate(spec['sequence']):
        run=metrics.checkpoint_dir(args.save_root,spec['suite'],spec['method'],seed,i,t)
        parents=[] if not previous else ([previous[0]] if len(previous)==1 else [previous[0],previous[-1]])
        if metrics.checkpoint_complete(run):
            expect=benchmark._expected_training_config(args,spec['suite'],t,i,seed,cfg)
            okay,reason=metrics.checkpoint_matches(run,expect,parent_dirs=parents,pretrained_encoder=args.pretrained_encoder)
            if not okay or gap:
                raise RuntimeError(f'{run}: {reason if not okay else "complete descendant after an incomplete parent"}. '
                                   'Existing completed results will NOT be overwritten. Use a different --comment.')
        else:
            all_good=False;gap=True
            if archive:
                archive_partial(spec,[run,metrics.event_dir(args.runs_root,spec['suite'],spec['method'],seed,i,t),
                  metrics.analysis_snapshot_path(args.analysis_root,spec['suite'],spec['method'],seed,i,t).parent])
        previous.append(run)
    return all_good

# Scratch root-task training does not use historical routing/consolidation knobs.
# Keep core training settings strict, but do not retrain scratch for an unused
# alpha_mass_lr, pool-size, lineage or pair-selection ablation.
SCRATCH_CORE=('model_type','task_suite','task_id','seq_idx','seed','fusion_mode','composition_space',
 'eval_action_mode','total_timesteps','buffer_size','gamma','tau','batch_size','learning_starts','random_actions_end',
 'policy_lr','q_lr','policy_frequency','target_network_frequency','alpha','autotune','autotune_init_from_alpha',
 'eval_every','num_evals','freeze_root_encoder','distillation','encoder_linear_out','distill_observation_skip','distill_extra_steps')

def scratch_args(spec,args,root):
    import scratch_baselines as scratch
    # Use parser defaults, then carry the main run's settings supported by scratch.
    with arguments(['--task-suites',spec['suite']]):a=scratch.parse_args()
    for k in vars(a):
        if hasattr(args,k):setattr(a,k,getattr(args,k))
    a.save_root=str(root/'models');a.runs_root=str(root/'runs');a.analysis_root=str(root/'analysis')
    a.force_retrain=False;a.variants=['plain'];a.eval_action_mode=spec['mode']
    return a

def scratch_match(run,expected,args):
    import experiment_identity as identity,scratch_baselines as scratch
    m=identity.load_manifest(run)
    if not scratch.checkpoint_complete(run):return False,'incomplete'
    actual=m['training_config'];compare={k:v for k,v in expected.items() if k in SCRATCH_CORE}
    # Ask the original guard to check source/runtime/parents as well. Irrelevant
    # fields come from the actual ROOT checkpoint, not the method ablation.
    merged=dict(actual,**compare)
    okay,reason=identity.checkpoint_matches(run,merged,pretrained_encoder=args.pretrained_encoder)
    return okay,reason

def reference_root(spec,args):
    import scratch_baselines as scratch
    canonical=Path(spec['EXPERIMENT_ROOT'])/'scratch'/spec['mode']
    a=scratch_args(spec,args,canonical)
    template=scratch._expected_training_config(spec['suite'],0,args.total_timesteps,101,a,'plain')
    core={k:v for k,v in template.items() if k in SCRATCH_CORE}
    refid=digest(core)[:16]
    record=Path(spec['EXPERIMENT_ROOT'])/'.paper_suite'/'references'/f'{spec["mode"]}_{refid}.json'
    # Different configurations can be submitted while canonical scratch is empty.
    # Reserve it under a mode-wide lock, not a separate lock per configuration.
    with lock(record.parent / f'{spec["mode"]}.allocation.lock'):
        saved=read_json(record)
        if saved:return Path(saved['root'])
        owner_path=canonical/'.paper_reference_settings.json'
        owner=read_json(owner_path,{})
        mismatches=[]
        if owner and owner.get('core_id')!=refid:
            mismatches.append('canonical scratch is reserved for another core training configuration')
        for task in sorted(set(spec['sequence'])):
            for seed in map(int,spec['SCRATCH_SEEDS']):
                run=scratch.scratch_checkpoint_dir(a.save_root,spec['suite'],task,args.total_timesteps,seed,'plain')
                if not scratch.checkpoint_complete(run):continue
                expect=scratch._expected_training_config(spec['suite'],task,args.total_timesteps,seed,a,'plain')
                good,why=scratch_match(run,expect,a)
                if not good:mismatches.append(f'task {task} seed {seed}: {why}')
        chosen=canonical if not mismatches else canonical/'references'/refid
        if chosen==canonical:
            atomic_json(owner_path,dict(core_id=refid,core_settings=core))
        atomic_json(record,dict(root=str(chosen),core_settings=core,old_reference_mismatches=mismatches))
        if mismatches:print('[scratch] Existing reference preserved; new matched reference:',chosen,mismatches,flush=True)
        return chosen

def check_scratch(spec,args,train=False):
    import scratch_baselines as scratch
    root=reference_root(spec,args);a=scratch_args(spec,args,root);all_good=True
    for task in sorted(set(spec['sequence'])):
        seed=int(spec['seed']);run=scratch.scratch_checkpoint_dir(a.save_root,spec['suite'],task,args.total_timesteps,seed,'plain')
        expect=scratch._expected_training_config(spec['suite'],task,args.total_timesteps,seed,a,'plain')
        good,reason=scratch_match(run,expect,a)
        event=scratch.scratch_event_dir(a.runs_root,spec['suite'],task,args.total_timesteps,seed,'plain')
        if good:
            # FT needs readable periodic curves as well, not merely a model.
            import metrics
            good=all(metrics.load_scalar(event,tag)[0].size>=2 for tag in ('charts/test_success','charts/test_episodic_return'))
            if not good:reason='complete checkpoint but missing/unreadable FT learning curves'
        if good:print(f'[scratch] SKIP task {task}, seed {seed}: compatible reference',flush=True);continue
        all_good=False
        if scratch.checkpoint_complete(run):
            raise RuntimeError(f'{run}: {reason}; existing reference preserved. Use a new reference root rather than overwrite it.')
        if train:
            with lock(root/'.locks'/f'{spec["suite"]}_{task}_{seed}.lock'):
                good,_=scratch_match(run,expect,a)
                if good:continue
                archive_partial(spec,[run,event,scratch.scratch_analysis_dir(a.analysis_root,spec['suite'],task,args.total_timesteps,seed,'plain')])
                scratch.train_one_baseline(spec['suite'],task,args.total_timesteps,seed,a,'plain')
    return all_good

def connect_reference(spec,args):
    root=reference_root(spec,args);target=root/'runs'/'scratch'
    if not target.is_dir():raise FileNotFoundError(f'Scratch jobs are not finished: {target}')
    link=Path(args.runs_root)/'scratch';link.parent.mkdir(parents=True,exist_ok=True)
    with lock(Path(spec['run_root'])/'.paper_reference.lock'):
        if link.is_symlink():
            if link.resolve()!=target.resolve():
                raise RuntimeError(f'{link} points to another reference ({link.resolve()}); refusing to silently change FT input')
        elif link.exists():raise RuntimeError(f'{link} is not a symlink')
        else:link.symlink_to(target.resolve(),target_is_directory=True)
    args.scratch_save_root=str(root/'models')
    return root

def execute_cka(spec,args,check):
    import run_continual_benchmark as benchmark
    if spec['phase']=='train':
        good=check_cka_chain(spec,args,spec['seed'])
        if good:print('[complete] All CKA/ETHOS checkpoints exist and match',flush=True);return 10
        if check:return 0
        with lock(Path(spec['run_root'])/'.paper_locks'/f'seed_{spec["seed"]}.lock',blocking=False):
            check_cka_chain(spec,args,spec['seed'],archive=True)
            # Call the original trainer directly: no shared plots/CSV writes.
            benchmark.train_chain(args,spec['suite'],spec['method'],method_config(spec,args),spec['seed'])
        return 0
    # Evaluation is the original protocol. Existing complete caches are retained.
    try:connect_reference(spec,args)
    except FileNotFoundError as exc:
        if check:
            print(f'[not ready] {exc}',flush=True);return 12
        raise
    valid=[];invalid={}
    for seed in spec['seeds']:
        try:
            if not check_cka_chain(spec,args,seed):raise FileNotFoundError('Incomplete seed chain')
            valid.append(seed)
        except (OSError,ValueError,RuntimeError) as exc:invalid[str(seed)]=str(exc)
    print('[eval] Valid seeds:',valid,'Skipped:',invalid,flush=True)
    if not valid:
        if check:return 12
        raise RuntimeError('No complete matching seed remains; do not average an empty set')
    if check:return 0
    root=Path(spec['run_root']);atomic_json(root/'plots'/'paper_seed_status.json',dict(requested=spec['seeds'],valid=valid,skipped=invalid))
    # All settings come from the same immutable training spec. No job_eval drift.
    argv=list(spec['arguments'])+['--skip-training','--skip-invalid-seeds','--seeds',*map(str,valid),
      '--task-sequence',*map(str,spec['sequence']),'--eval-action-mode',spec['mode'],
      '--condition-index','1' if spec['method']=='baseline' else '4',
      '--composition-spaces','parameter' if spec['method']=='baseline' else 'policy',
      '--scratch-seeds',*map(str,spec['SCRATCH_SEEDS']),'--scratch-save-root',args.scratch_save_root,
      '--save-root',args.save_root,'--runs-root',args.runs_root,'--analysis-root',args.analysis_root,'--plots-root',args.plots_root]
    with lock(root/'.paper_locks'/'evaluation.lock',blocking=False):
        with arguments(argv):benchmark.main()
    return 0

def reuse_equivalent_folder(spec, args, spec_file):
    """Resolve older --comment names by verified TRAINING identity, not names.

    A candidate must contain at least one completed occurrence and every
    completed checkpoint in the requested seed set must match. Corrupt/partial
    seeds are not mistaken for a completed run. No files are moved or renamed.
    """
    if not spec.get('reuse_equivalent', True) or spec['method']=='scratch':return spec,args
    import metrics
    requested=Path(spec['run_root'])
    # Respect an occupied requested path; never evade a mismatch by silently
    # falling back to a different experiment with different settings.
    if (requested/'agents').exists() and any((requested/'agents').rglob('*.pt')):return spec,args
    prefix=f"{spec['method']}_{spec['mode']}"
    for directory in sorted((Path(spec['EXPERIMENT_ROOT'])/'main').glob(prefix+'*')):
        if directory==requested or not directory.is_dir():continue
        if directory.name!=prefix and not directory.name.startswith(prefix+'_'):continue
        trial=copy.deepcopy(spec);trial['run_root']=str(directory)
        trial_args=benchmark_args(trial)
        present=False;okay=True
        for seed in spec['seeds']:
            try:
                if spec['method'] in ('baseline','combined_policy'):
                    check_cka_chain(trial,trial_args,seed)
                    for i,task in enumerate(spec['sequence']):
                        if metrics.checkpoint_complete(metrics.checkpoint_dir(trial_args.save_root,spec['suite'],spec['method'],seed,i,task)):
                            present=True
                else:
                    from paper_runs.baseline_runner import check_chain,path_for
                    check_chain(trial,trial_args,seed)
                    present=present or any((path_for(trial,seed,i)[0]/'baseline_manifest.json').is_file() for i in range(len(spec['sequence'])))
            except (OSError,ValueError,RuntimeError):okay=False;break
        if okay and present:
            trial['requested_run_root']=str(requested)
            atomic_json(spec_file,trial)
            print(f'[reuse-equivalent] Verified existing run: {directory} (requested {requested.name})',flush=True)
            return trial,trial_args
    return spec,args


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--check',action='store_true')
    p.add_argument('--baseline-task',type=int);opt=p.parse_args()
    spec=read_json(opt.spec)
    if not spec:raise ValueError('Invalid job spec')
    env=Path(spec['project_root'])/spec['environment'];sys.path.insert(0,str(env));os.chdir(env)
    args=benchmark_args(spec)
    if args.force_retrain:
        raise ValueError('The safe launcher refuses --force-retrain. Use a distinct --comment instead.')
    if not opt.check and os.environ.get('SLURM_JOB_ID') and not args.cpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in the GPU allocation; refusing an unintended CPU training job')
        print(f'[gpu] {torch.cuda.get_device_name(0)}; CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}',flush=True)
    if opt.baseline_task is None:
        spec,args=reuse_equivalent_folder(spec,args,opt.spec)
    if opt.baseline_task is not None:
        from paper_runs.baseline_runner import train_one
        train_one(spec,args,opt.baseline_task);return 0
    if spec['method']=='scratch':
        good=check_scratch(spec,args,train=not opt.check);return 10 if good and opt.check else 0
    if spec['method'] in ('baseline','combined_policy'):return execute_cka(spec,args,opt.check)
    from paper_runs import baseline_runner
    if spec['phase']=='train':
        good=baseline_runner.check_chain(spec,args,spec['seed'],verify_bytes=opt.check)
        if good:print('[complete] Baseline chain matches',flush=True);return 10 if opt.check else 0
        if opt.check:return 0
        with lock(Path(spec['run_root'])/'.paper_locks'/f'seed_{spec["seed"]}.lock',blocking=False):
            baseline_runner.train_chain(spec,args,opt.spec)
    else:
        try:connect_reference(spec,args)
        except FileNotFoundError as exc:
            if opt.check:
                print(f'[not ready] {exc}',flush=True);return 12
            raise
        if opt.check:
            ready=[]
            for seed in spec['seeds']:
                try:
                    if baseline_runner.check_chain(spec,args,seed,verify_bytes=True):ready.append(seed)
                except (OSError,ValueError,RuntimeError) as exc:
                    print(f'[baseline preflight] seed {seed}: {exc}',flush=True)
            print(f'[baseline preflight] complete seeds: {ready}',flush=True)
            return 0 if ready else 12
        else:
            from paper_runs.evaluation import evaluate_baseline
            with lock(Path(spec['run_root'])/'.paper_locks'/'evaluation.lock',blocking=False):evaluate_baseline(spec,args)
    return 0

if __name__=='__main__':
    try:code=main()
    except Exception as exc:
        import traceback;traceback.print_exc();code=11
    sys.exit(code)
