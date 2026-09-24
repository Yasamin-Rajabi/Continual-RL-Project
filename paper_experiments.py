#!/usr/bin/env python3
"""Additive Kuma launcher. No third-party imports needed on the login node."""
from __future__ import annotations
import argparse, os, re, shlex, subprocess, sys
from pathlib import Path
from paper_runs.config import ENVIRONMENTS,METHODS,read_job,cli_dict,replace_cli,suite_info,make_cases,parse_override
from paper_runs.io import atomic_json,digest,read_json,lock

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environments',nargs='+',choices=ENVIRONMENTS,default=list(ENVIRONMENTS))
    p.add_argument('--groups',nargs='+',choices=['main','kl','lineage','pool','warmup','no_merge','all'],default=['main','kl','lineage','pool','warmup'])
    p.add_argument('--methods',nargs='+',choices=METHODS,help='Restrict selected methods; ablations only apply to combined_policy')
    p.add_argument('--phase',choices=['train','scratch','eval','collect','status'],default='train')
    p.add_argument('--comment',default=os.environ.get('RUN_COMMENT',''))
    p.add_argument('--pool-sizes',nargs='+',type=int,default=[3,5,8])
    p.add_argument('--warmup-steps',nargs='+',type=int,default=[0,5000,10000])
    p.add_argument('--baseline-eval-protocol',choices=['reward_route','native','latest'],default='reward_route')
    p.add_argument('--eval-mode',choices=['deterministic','stochastic'],help='Override the existing EVAL_MODES explicitly')
    p.add_argument('--seeds',nargs='+',type=int,help='Explicit continual seed subset, defaults to existing MAIN_SEEDS')
    p.add_argument('--set',action='append',default=[],metavar='KEY=VALUE',help='Explicit common argument override (does not edit job.sh)')
    p.add_argument('--env-set',action='append',default=[],metavar='ENV:KEY=VALUE')
    p.add_argument('--no-reuse-equivalent',action='store_true',help='Only use the exact requested run folder; do not search matching older comments')
    p.add_argument('--no-scratch',action='store_true',help='Do not submit scratch during --phase train')
    p.add_argument('--submit',action='store_true',help='Actually submit; without this flag only print the plan')
    p.add_argument('--no-precheck',action='store_true',help='Skip login-node container preflight; workers still validate before use')
    p.add_argument('--dry-run',action='store_true',help='Explicit alias for the default no-submit preview')
    p.add_argument('--project-root',default=str(Path(__file__).resolve().parent))
    args=p.parse_args()
    if args.submit and args.dry_run:p.error('--submit and --dry-run conflict')
    if args.comment and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.comment):p.error('Unsafe comment')
    if any(x<1 for x in args.pool_sizes) or any(x<0 for x in args.warmup_steps):p.error('Invalid ablation values')
    root=Path(args.project_root).resolve()
    groups=(['main','kl','lineage','pool','warmup','no_merge'] if 'all' in args.groups else args.groups)
    count=0
    active_job_ids=None
    for env in args.environments:
        base=read_job(root,env)
        for override in args.set+[x.split(':',1)[1] for x in args.env_set if x.startswith(env+':')]:
            k,v=override.split('=',1)
            v=parse_override(v)
            base['COMMON_ARGS']=replace_cli(base['COMMON_ARGS'],k,v)
        cfg=cli_dict(base['COMMON_ARGS']);suite=cfg['task_suites']
        if not isinstance(suite,str):p.error('Select one suite per environment in the job preset')
        sequence,task_count=suite_info(root,env,suite)
        if 'task_sequence' in cfg:
            s=cfg['task_sequence'];sequence=[int(x) for x in (s if isinstance(s,list) else [s])]
        base['sequence']=sequence;base['suite']=suite;base['num_tasks']=task_count
        cases=make_cases(base,groups,args.pool_sizes,args.warmup_steps,sequence)
        if args.methods:cases=[c for c in cases if c[0] in args.methods]
        seeds=args.seeds or [int(x) for x in base['MAIN_SEEDS']]
        modes=[args.eval_mode] if args.eval_mode else base['EVAL_MODES']
        exp=Path(base['EXPERIMENT_ROOT']); specs=[]
        # Scratch uses the shared reference settings, not each algorithm ablation.
        for mode in modes:
            if args.phase=='scratch' or (args.phase=='train' and not args.no_scratch):
                for seed in base['SCRATCH_SEEDS']:
                    specs.append(dict(base,method='scratch',phase='scratch',seed=int(seed),seeds=seeds,
                        mode=mode,comment='',arguments=base['COMMON_ARGS'],baseline_eval_protocol=args.baseline_eval_protocol))
            if args.phase=='scratch':continue
            if args.phase=='collect':
                cmd=['apptainer','exec','--bind',f'{root}:{root}',base['IMAGE'],'/opt/conda/bin/python',str(root/'collect_paper_metrics.py'),
                     '--experiment-root',str(exp),'--suite',suite,'--eval-mode',mode]
                print(shlex.join(cmd))
                if args.submit:
                    e=os.environ.copy();e['APPTAINERENV_PYTHONNOUSERSITE']='1';subprocess.run(cmd,env=e,check=True)
                continue
            for method,label,argv in cases:
                comment='_'.join(s for s in (args.comment,label) if s)
                name=f'{method}_{mode}'+(f'_{comment}' if comment else '')
                actual_phase='eval' if args.phase in ('eval','status') else 'train'
                for seed in (seeds if actual_phase=='train' else [None]):
                    specs.append(dict(base,method=method,phase=actual_phase,seed=seed,seeds=seeds,mode=mode,
                        comment=comment,arguments=argv,run_root=str(exp/'main'/name),
                        baseline_eval_protocol=args.baseline_eval_protocol))
        for spec in specs:
            spec['project_root']=str(root)
            spec['reuse_equivalent']=not args.no_reuse_equivalent
            spec['spec_id']=digest(spec)[:20]
        plan_key=digest(specs)[:16];plan_dir=exp/'.paper_suite'/plan_key
        print(f'\n[{env}] suite={suite}, occurrences={len(sequence)}, modes={modes}, seeds={seeds}')
        print(f'  Source settings: {base["job_file"]}')
        if args.phase!='collect':print(f'  Planned workers: {len(specs)} (existing/active matches checked before submission)')
        for i,spec in enumerate(specs):
            stem=f'{spec["phase"]}_{spec["method"]}_{spec["mode"]}'
            if spec['seed'] is not None:stem+=f'_seed{spec["seed"]}'
            if spec['comment']:stem+='_'+spec['comment']
            path=plan_dir/(stem+'.json')
            print(f'  {stem} -> {spec.get("run_root",str(exp/"scratch"/spec["mode"]))}')
            if spec['method']=='combined_policy' and int(cli_dict(spec['arguments'])['pool_size'])>=len(sequence):
                print('    diagnostic: no overflow; larger memory budget, not a matched-capacity comparison')
            if not args.submit and args.phase!='status':continue
            saved_spec=read_json(path)
            if saved_spec is not None:
                if saved_spec.get('spec_id')!=spec['spec_id']:
                    raise RuntimeError(f'Existing immutable spec has a different identity: {path}')
                spec=saved_spec  # Keep any verified existing-folder resolution.
            else:
                atomic_json(path,spec)
            e=os.environ.copy();e['APPTAINERENV_PYTHONNOUSERSITE']='1'
            precheck=['apptainer','exec','--bind',f'{root}:{root}','--pwd',str(root/env),base['IMAGE'],
                      '/opt/conda/bin/python',str(root/'paper_runs/worker.py'),'--spec',str(path),'--check']
            if not args.no_precheck or args.phase=='status':
                status=subprocess.run(precheck,env=e)
                # Exit 10 means all requested output already complete, 11 incompatible data.
                if status.returncode==10:print('    SKIP: already complete');continue
                if status.returncode==12:print('    SKIP evaluation: no ready compatible seed/reference');continue
                spec=read_json(path,spec)
                if status.returncode not in (0,10):raise SystemExit(f'Preflight failed for {stem}; no overwrite performed')
            if args.phase=='status':continue
            receipt=exp/'.paper_suite'/'submitted'/(spec['spec_id']+'.json')
            with lock(receipt.with_suffix('.lock')):
                prior=read_json(receipt,{})
                if prior.get('job_id'):
                    if active_job_ids is None:
                        import getpass
                        q=subprocess.run(['squeue','-h','-u',os.environ.get('USER',getpass.getuser()),'-o','%A'],stdout=subprocess.PIPE,text=True)
                        if q.returncode!=0:raise RuntimeError('Cannot check existing Slurm jobs; refusing duplicate submission')
                        active_job_ids=set(q.stdout.split())
                    if str(prior['job_id']) in active_job_ids:
                        print(f'    SKIP: active job {prior["job_id"]}');continue
                (exp/'logs').mkdir(parents=True,exist_ok=True)
                resources=[y for x in base['resources'] for y in shlex.split(x)]
                cmd=['sbatch','--parsable','--job-name=causal',*resources,'--chdir='+str(root/env),
                     '--output='+str(exp/'logs'/f'paper_{stem}_%j.out'),
                     '--error='+str(exp/'logs'/f'paper_{stem}_%j.err'),
                     str(root/'paper_worker.sh'),str(path)]
                result=subprocess.run(cmd,stdout=subprocess.PIPE,text=True,check=True)
                jobid=result.stdout.strip().split(';')[0]
                if not jobid.isdigit():raise RuntimeError(f'Unexpected sbatch output: {result.stdout}')
                if active_job_ids is not None:active_job_ids.add(jobid)
                atomic_json(receipt,{'job_id':jobid,'spec':str(path)})
                print(f'    SUBMITTED {jobid}');count+=1
    if args.phase=='train':print('\nScratch and training can run concurrently. Wait for both before --phase eval.')
    print(f'\nSubmitted {count} jobs.' if args.submit else '\nPreview only. Add --submit after checking settings and run names.')

if __name__=='__main__':main()
