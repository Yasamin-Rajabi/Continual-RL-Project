"""Read settings from existing job.sh declarations without running that job.

Only selected simple variable/array assignments are passed to a clean Bash
process. Functions, sbatch calls and the job body are NEVER sourced.
"""
from __future__ import annotations
import importlib.util, json, os, re, subprocess, sys
from pathlib import Path

ENVIRONMENTS=('half-cheetah','Walker2D','AntDir','minigrid')
METHODS=('baseline','ft_n','prognet','packnet','masknet','crelus','componet','cbpnet','combined_policy')
ARRAYS=('MAIN_SEEDS','SCRATCH_SEEDS','EVAL_MODES','COMMON_ARGS','COMBINED_POLICY_ARGS')
VARIABLES=('PROJECT_ROOT','IMAGE','BASE_STORAGE','EXPERIMENT_ROOT')

def declaration(text,name,array=False):
    pattern=(r'^'+re.escape(name)+r'=\([^\n]*\)(?=\s*$)|^'+re.escape(name)+r'=\(\s*\n.*?^\)' if array else
             r'^'+re.escape(name)+r'=.*$')
    found=list(re.finditer(pattern,text,re.M|re.S if array else re.M))
    if not found:return None
    return found[-1].group()

def read_job(project,environment):
    project=Path(project).resolve();path=project/environment/'job.sh';text=path.read_text()
    decls=[]
    for name in VARIABLES+ARRAYS:
        decl=declaration(text,name,name in ARRAYS)
        if decl is not None:
            if '$(' in decl or '`' in decl or '\n#SBATCH' in decl:
                raise ValueError(f'Cannot safely extract computed declaration {name}; use simple assignments in {path}')
            decls.append(decl)
    program='set -eu\n'+'\n'.join(decls)+'\n'
    for name in VARIABLES:
        program+=f'printf "%s\\0" "${{{name}:-}}"\n'
    for name in ARRAYS:
        program+=f'printf "%s\\0" "__{name}__" "${{{name}[@]}}"\n'
    env=os.environ.copy();env['PROJECT_ROOT']=str(project)
    # A RUN_COMMENT affects only output paths later; it is not a hyperparameter.
    result=subprocess.run(['bash','--noprofile','--norc','-c',program],env=env,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    words=result.stdout.decode().split('\0');d=dict(zip(VARIABLES,words[:len(VARIABLES)]))
    rest=words[len(VARIABLES):];key=None
    for word in rest:
        if word.startswith('__') and word.endswith('__'):
            key=word[2:-2];d[key]=[]
        elif key and word:d[key].append(word)
    resources=[]
    for line in text.splitlines():
        if line.startswith('#SBATCH '):
            opt=line[len('#SBATCH '):].strip()
            if opt.startswith(('--output','--error','--job-name')):continue
            resources.append(opt)
    d['resources']=resources;d['environment']=environment;d['job_file']=str(path)
    if not d.get('MAIN_SEEDS'):d['MAIN_SEEDS']=['1','2','3']
    if not d.get('SCRATCH_SEEDS'):d['SCRATCH_SEEDS']=['101','102','103']
    return d

def cli_dict(argv):
    result={};i=0
    while i<len(argv):
        item=argv[i]
        if not item.startswith('--'):raise ValueError(f'Unexpected argument {item}')
        if '=' in item:
            key,value=item[2:].split('=',1);values=[value];i+=1
        else:
            key=item[2:];i+=1;values=[]
            while i<len(argv) and not argv[i].startswith('--'):
                values.append(argv[i]);i+=1
        if key.startswith('no-') and not values:result[key[3:].replace('-','_')]=False
        elif not values:result[key.replace('-','_')]=True
        else:result[key.replace('-','_')]=values[0] if len(values)==1 else values
    return result

def replace_cli(argv,key,value):
    key=key.lstrip('-').replace('_','-');out=[];i=0
    aliases={key,'no-'+key}
    if key=='distill-extra-steps':aliases.add('distill-buffer-steps')
    if key=='distill-buffer-steps':aliases.add('distill-extra-steps')
    while i<len(argv):
        item=argv[i];k=item[2:].split('=',1)[0];i+=1;values=[]
        while i<len(argv) and not argv[i].startswith('--'):values.append(argv[i]);i+=1
        if k not in aliases:out.extend([item]+values)
    if isinstance(value,bool):out.append('--'+('' if value else 'no-')+key)
    elif isinstance(value,(list,tuple)):out.extend(['--'+key]+[str(x) for x in value])
    else:out.extend(['--'+key,str(value)])
    return out

def suite_info(project,env,suite):
    path=Path(project)/env/'tasks.py';name='_paper_task_config_'+env.replace('-','_')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m)
    if suite not in m.TASK_SUITES:raise ValueError(f'{suite} not in {env}/tasks.py')
    seq=(m.default_sequence(suite) if hasattr(m,'default_sequence') else m.DEFAULT_CONTINUAL_SEQUENCE)
    return list(seq),len(m.TASK_SUITES[suite])

def make_cases(base,groups,pool_sizes,warmups,sequence):
    cases=[]
    def add(method,label='',changes=None):
        args=list(base['COMMON_ARGS'])
        if method=='combined_policy':args+=base.get('COMBINED_POLICY_ARGS',[])
        for k,v in (changes or {}).items():args=replace_cli(args,k,v)
        cases.append((method,label,args))
    if 'main' in groups:
        for method in METHODS:add(method)
    else:add('combined_policy') # shared reference for each ablation family
    if 'kl' in groups:
        add('combined_policy','random_merge',{'merge-ablation':'random_merge'})
        add('combined_policy','kl_discard',{'merge-ablation':'kl_discard'})
    if 'lineage' in groups:
        current=cli_dict(base['COMMON_ARGS']).get('balance_source_lineages',False)
        add('combined_policy','lineage_off' if current else 'lineage_on',{'balance-source-lineages':not current})
    if 'pool' in groups:
        for n in pool_sizes:add('combined_policy',f'pool{n}',{'pool-size':n})
    if 'warmup' in groups:
        for n in warmups:add('combined_policy',f'warmup{n}',{'alpha-warmup-steps':n})
    if 'no_merge' in groups:add('combined_policy','no_merge',{'pool-size':len(sequence)})
    # An ablation equal to the reference is the same run, not a second job.
    seen=set();unique=[]
    for method,label,args in cases:
        normalized=cli_dict(args)
        normalized.setdefault('merge_ablation','kl_merge')
        k=(method,json.dumps(normalized,sort_keys=True))
        if k in seen:continue
        seen.add(k);unique.append((method,label,args))
    return unique


def parse_override(value):
    """Accept booleans, JSON lists, or whitespace-separated CLI list values."""
    import shlex
    if value.lower() in ('true','false'):return value.lower()=='true'
    if value.startswith('['):
        v=json.loads(value)
        if not isinstance(v,list):raise ValueError('Expected an argument list')
        return v
    parts=shlex.split(value)
    return parts if len(parts)>1 else value
