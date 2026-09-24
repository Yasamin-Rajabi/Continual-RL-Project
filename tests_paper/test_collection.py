import csv,json,os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sample(root,raw=False):
    folder=root/'main'/'ft_n_deterministic'
    out=folder/'plots'/'tiny';out.mkdir(parents=True)
    fields=['suite','condition','seed','A_N','FG','BWT','FT_success','FT_return']
    if raw:fields+=['FT_return_auc_delta']
    with (out/'survey_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for seed in [1,2]:
            r=dict(suite='tiny',condition='ft_n',seed=seed,A_N=.7,FG=.2,BWT=-.2,FT_success=.3,FT_return=float('nan') if raw else .4)
            if raw:r['FT_return_auc_delta']=25.
            w.writerow(r)
    (out/'summary_metrics.csv').write_text('seed,final_avg_return_all_eval_tasks,final_avg_success_all_eval_tasks\n1,80,0.7\n2,80,0.7\n')
    for seed in [1,2]:
        for i,t in enumerate([0,1,0]):
            run=folder/'runs'/'tiny'/'ft_n'/f'seed_{seed}'/f'seq_{i}'/f'tiny__task_{t}__ft_n__run_sac__{seed}'
            run.mkdir(parents=True)
            (run/'scalars.csv').write_text(f'wall_time,step,tag,value\n0,100,charts/final_success,0.9\n0,100,charts/final_return,{100+seed+i}\n')
    return folder

def test_ant_raw_metric_is_separate_and_perf_is_acquisition(tmp_path):
    sample(tmp_path,raw=True)
    subprocess.run([sys.executable,str(ROOT/'collect_paper_metrics.py'),'--experiment-root',str(tmp_path),
                    '--suite','tiny','--eval-mode','deterministic'],check=True,capture_output=True,text=True)
    rows=list(csv.DictReader((tmp_path/'paper_metrics_deterministic.csv').open()))
    r=rows[0]
    assert float(r['PERF_success_mean'])==.9 and float(r['A_N_mean'])==.7
    assert r['FT_return_mean']=='nan' and float(r['FT_return_auc_delta_mean'])==25
    assert int(r['PERF_return_n'])==2 and float(r['PERF_return_mean'])==102.5

def test_existing_survey_schema_has_no_added_return_normalization(tmp_path):
    sample(tmp_path,raw=False)
    subprocess.run([sys.executable,str(ROOT/'collect_paper_metrics.py'),'--experiment-root',str(tmp_path),
                    '--suite','tiny','--eval-mode','deterministic'],check=True,capture_output=True,text=True)
    r=list(csv.DictReader((tmp_path/'paper_metrics_deterministic.csv').open()))[0]
    assert 'FT_return_auc_delta_mean' not in r and float(r['FT_return_mean'])==.4
