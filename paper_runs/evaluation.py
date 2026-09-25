"""Frozen retained-baseline evaluation, using the main metric definitions.

Default reward_route: same immediate-reward routing heuristic and interaction
budget as combined_policy. NO task-to-module oracle is used. Native/latest are
explicit alternative protocols written under a separate plots subdirectory.
Completed checkpoint/task cells are cached atomically for interrupted jobs.
"""
from __future__ import annotations
from pathlib import Path
import copy
import numpy as np
import torch
from torch import nn
from paper_runs.io import atomic_json,read_json,digest,write_csv,file_hash
from paper_runs.baseline_runner import path_for,valid
from paper_runs.baselines import freeze
from policy_utils import bound_log_std
from policy_composition import sample_action,representative_action
from tasks import get_task
import metrics


class HistoricalRouter(nn.Module):
    def __init__(self,policies,discrete=False):
        super().__init__();self.experts=nn.ModuleList(policies);self.discrete=bool(discrete)
        self.alpha=nn.Parameter(torch.zeros(len(policies)))
        for expert in self.experts:freeze(expert)
    def policy_components(self,obs):
        with torch.no_grad():parts=[p(obs) for p in self.experts]
        head_a=torch.stack([x[0] for x in parts],dim=1)
        head_b=torch.stack([x[1] if self.discrete else bound_log_std(x[1]) for x in parts],dim=1)
        return head_a,head_b,torch.softmax(self.alpha,dim=0)


def finite_mean(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    return float(a.mean()) if a.size else float('nan')


def eval_cell(agent,spec,args,task,seed,device):
    policies=agent.candidates();protocol=spec.get('baseline_eval_protocol','reward_route')
    if protocol=='latest':policies=policies[-1:]
    if protocol=='native':
        # Oracle is explicit; task identities select only a policy already stored.
        matches=[i for i,t in enumerate(agent.tasks) if t==task]
        if len(policies)>1:policies=[policies[matches[-1] if matches else -1]]
    env=get_task(task,task_suite=spec['suite'])
    discrete=hasattr(env.action_space,'n')
    policy=HistoricalRouter(policies,discrete=discrete).to(device).eval();policy.requires_grad_(False)
    adapt=int(args.test_adapt_steps) if protocol=='reward_route' else 0
    if adapt and len(policies)>1:policy.alpha.requires_grad_(True)
    opt=torch.optim.Adam([policy.alpha],lr=args.test_adapt_lr) if policy.alpha.requires_grad else None
    if discrete:
        scale=bias=None
    else:
        scale=torch.as_tensor((env.action_space.high-env.action_space.low)/2,dtype=torch.float32,device=device)
        bias=torch.as_tensor((env.action_space.high+env.action_space.low)/2,dtype=torch.float32,device=device)
    cuda=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    evaluation_steps=0
    try:
        with torch.random.fork_rng(devices=cuda):
            torch.manual_seed(seed+10000*int(task))
            if adapt:
                obs,_=env.reset(seed=seed)
                for _ in range(adapt):
                    x=torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)
                    if discrete:
                        action,logp,_=sample_action(policy,x,score_function=True)
                        env_action=int(action[0].detach().item())
                    else:
                        action,logp,_=sample_action(policy,x,scale,bias,score_function=True)
                        env_action=action[0].detach().cpu().numpy()
                    obs,reward,done,trunc,_=env.step(env_action)
                    if opt is not None:
                        loss=-logp.mean()*float(reward);opt.zero_grad();loss.backward();opt.step()
                    if done or trunc:obs,_=env.reset()
            policy.requires_grad_(False)
            returns=[];successes=[];errors=[]
            for ep in range(args.retention_eval_episodes):
                obs,_=env.reset(seed=seed+10000*int(task)+ep);ret=0.;succ=[];error=[]
                while True:
                    x=torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)
                    with torch.no_grad():
                        if discrete:
                            a=(sample_action(policy,x)[0] if spec['mode']=='stochastic'
                               else representative_action(policy,x))
                        else:
                            a=(sample_action(policy,x,scale,bias)[0] if spec['mode']=='stochastic'
                               else representative_action(policy,x,scale,bias))
                    env_action=int(a[0].item()) if discrete else a[0].cpu().numpy()
                    obs,r,done,trunc,info=env.step(env_action);ret+=float(r);evaluation_steps+=1
                    if 'success' in info:succ.append(float(info['success']))
                    if metrics.ERROR_KEY in info:error.append(float(info[metrics.ERROR_KEY]))
                    if done or trunc:break
                returns.append(ret)
                successes.append(float(max(succ)) if succ and metrics.EPISODIC_SUCCESS else finite_mean(succ))
                errors.append(finite_mean(error))
        return {'return':finite_mean(returns),'success':finite_mean(successes),metrics.ERROR_KEY:finite_mean(errors),
                'evaluation_interactions':evaluation_steps,'adaptation_interactions':adapt,
                'routing_weights':torch.softmax(policy.alpha,0).detach().cpu().tolist()}
    finally:env.close()


def forward_transfer(spec,args,seed,reference):
    import scratch_baselines as scratch
    result={};positions=metrics._first_unseen_positions(spec['sequence'])
    for name,tag,upper in [('success','charts/test_success',1.),('return','charts/test_episodic_return',metrics.RETURN_UPPER_BOUND)]:
        scores=[];deltas=[];used=[];missing=[]
        for index,task in positions:
            _,events,_=path_for(spec,seed,index)
            auc=metrics._auc(*metrics.load_scalar(events,tag))
            baselines=[]
            for s in map(int,spec['SCRATCH_SEEDS']):
                bd=scratch.scratch_event_dir(reference/'runs',spec['suite'],task,args.total_timesteps,s,'plain')
                a=metrics._auc(*metrics.load_scalar(bd,tag))
                if a is None or not np.isfinite(a):missing.append(f'task {task}, scratch seed {s}, {tag}')
                else:baselines.append(a)
            if auc is None or not np.isfinite(auc):missing.append(f'task {task}, continual seed {seed}, {tag}');continue
            if len(baselines)!=len(spec['SCRATCH_SEEDS']):continue
            ref=float(np.mean(baselines));deltas.append(auc-ref)
            if upper is not None and upper-ref>1e-12:
                scores.append((auc-ref)/(upper-ref));used.append({'seq_idx':index,'task_id':task})
        # Missing curves must not silently turn an incomplete run into a paper mean.
        result[f'FT_{name}']=finite_mean(scores) if not missing else float('nan')
        result[f'FT_{name}_per_position']=scores
        result[f'FT_{name}_positions']=used
        result[f'FT_{name}_auc_delta']=finite_mean(deltas) if not missing else float('nan')
        result[f'FT_{name}_missing']=missing
        if missing:print(f'[FT unavailable] {name}: {missing}',flush=True)
    result['FT_return_upper_bound']=metrics.RETURN_UPPER_BOUND
    result['FT_reference']='common plain SAC (same environment, training budget and core SAC settings)'
    return result


def evaluate_baseline(spec,args):
    from paper_runs.worker import reference_root
    import plots
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    protocol=spec.get('baseline_eval_protocol','reward_route')
    if protocol!='reward_route':args.plots_root=str(Path(args.plots_root)/f'baseline_{protocol}')
    target=Path(args.plots_root)/spec['suite'];target.mkdir(parents=True,exist_ok=True)
    eval_config={'sequence':spec['sequence'],'episodes':args.retention_eval_episodes,
                 'adapt_steps':args.test_adapt_steps,'adapt_lr':args.test_adapt_lr,'mode':spec['mode'],
                 'protocol':protocol,'evaluation_source':file_hash(__file__)}
    atomic_json(target/'evaluation_protocol.json',dict(eval_config,
        task_identity_used=(protocol=='native'),stored_weights_frozen=True,
        note='reward_route uses the main immediate-reward REINFORCE heuristic; native/latest have no adaptation'))
    all_payloads=[];surveys=[];good_seeds=[];invalid={}
    ref=reference_root(spec,args)
    for seed in spec['seeds']:
        try:
            parent=None;signatures=[]
            for i in range(len(spec['sequence'])):
                okay,reason,m=valid(spec,args,seed,i,parent,verify_bytes=True)
                if not okay:raise RuntimeError(f'seq {i}: {reason}')
                parent=m['signature'];signatures.append(parent)
            cachepath=target/'cell_cache'/f'{spec["method"]}_seed_{seed}.json'
            key=digest([eval_config,signatures])
            cache=read_json(cachepath,{})
            if cache.get('identity')!=key:cache={'identity':key,'cells':{}}
            tasks=sorted(set(spec['sequence']))
            values={k:[] for k in ['return','success',metrics.ERROR_KEY,'evaluation_interactions','adaptation_interactions']}
            for i,trained_task in enumerate(spec['sequence']):
                run,_,_=path_for(spec,seed,i);agent=None
                for task in tasks:
                    k=f'{i}:{task}'
                    if k in cache['cells']:continue
                    if agent is None:agent=torch.load(run/'agent_state.pt',map_location='cpu',weights_only=False)
                    cache['cells'][k]=eval_cell(agent,spec,args,task,seed,device)
                    atomic_json(cachepath,cache)
                    print(f'[baseline eval] {spec["method"]} seed={seed} seq={i} task={task} saved',flush=True)
                del agent
                for metric in values:values[metric].append([cache['cells'][f'{i}:{t}'][metric] for t in tasks])
            payload=dict(suite=spec['suite'],condition=spec['method'],seed=seed,sequence=spec['sequence'],
                         eval_task_ids=tasks,episodes=args.retention_eval_episodes,**values)
            atomic_json(target/'retention_data'/f'{spec["method"]}_seed_{seed}.json',payload)
            diagonal={str(i):cache['cells'][f'{i}:{t}']['success'] for i,t in enumerate(spec['sequence'])}
            final={str(t):cache['cells'][f'{len(spec["sequence"])-1}:{t}']['success'] for t in tasks}
            survey=dict(suite=spec['suite'],condition=spec['method'],seed=seed,sequence=spec['sequence'],
                        A_N=metrics.compute_A_N(final),**metrics.compute_fg_bwt(diagonal,final,spec['sequence']),
                        **forward_transfer(spec,args,seed,ref),p_diagonal=diagonal,p_final_row=final,
                        baseline_eval_protocol=protocol,native_task_id=(protocol=='native'))
            atomic_json(target/'survey_metrics'/f'{spec["method"]}_seed_{seed}.json',survey)
            all_payloads.append(payload);surveys.append(survey);good_seeds.append(seed)
        except (OSError,ValueError,RuntimeError,EOFError) as exc:
            invalid[str(seed)]=f'{type(exc).__name__}: {exc}'
            print(f'[baseline eval] SKIP seed {seed}: {invalid[str(seed)]}',flush=True)
    atomic_json(target/'seed_status.json',dict(requested=spec['seeds'],used=good_seeds,skipped=invalid))
    if not good_seeds:raise RuntimeError('All baseline seeds invalid; no summary was written')
    args.seeds=good_seeds
    # Reuse the existing aggregate schemas so collect_paper_metrics still works.
    plots.write_summary_csv(args,spec['suite'],[spec['method']],{spec['method']:all_payloads})
    plots.write_survey_metrics_csv(args,spec['suite'],[spec['method']],{spec['method']:surveys})
    # Include raw-return FT (essential for AntDir, which has no finite reward bound).
    write_csv(target/'baseline_metrics_detail.csv',[
        {k:v for k,v in s.items() if isinstance(v,(str,int,float,bool)) or v is None} for s in surveys])
    plots.plot_retention(args,spec['suite'],[spec['method']],{spec['method']:all_payloads})
