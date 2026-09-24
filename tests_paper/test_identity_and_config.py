import importlib.util,sys,json,copy,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'half-cheetah')]
import experiment_identity as identity
from paper_runs.config import read_job,cli_dict,make_cases,replace_cli,suite_info

def test_original_default_manifest_accepted_but_new_ablation_rejected(tmp_path):
    new=identity.source_fingerprint(ROOT/'half-cheetah')
    assert new in identity.DEFAULT_ABLATION_SOURCE_TRANSITIONS
    old=next(iter(identity.DEFAULT_ABLATION_SOURCE_TRANSITIONS[new]))
    cfg={'seed':1,'model_type':'cka-rl','alpha_lr':.005,'alpha_mass_lr':.005,
         'balance_source_lineages':True}
    data={'schema_version':2,'training_config':cfg,'source_fingerprint':old,
          'runtime_versions':identity.runtime_versions(),'pretrained_encoder_sha256':None,
          'parent_signatures':[],'run_signature':'old'}
    (tmp_path/'run_manifest.json').write_text(json.dumps(data))
    assert identity.checkpoint_matches(tmp_path,cfg,root=ROOT/'half-cheetah')==(True,'match')
    ok,reason=identity.checkpoint_matches(tmp_path,dict(cfg,merge_ablation='random_merge'),root=ROOT/'half-cheetah')
    assert not ok and 'merge_ablation' in reason

def test_future_training_edit_is_not_silently_whitelisted(tmp_path):
    import shutil
    source=tmp_path/'source';shutil.copytree(ROOT/'half-cheetah',source)
    current=identity.source_fingerprint(ROOT/'half-cheetah')
    old=next(iter(identity.DEFAULT_ABLATION_SOURCE_TRANSITIONS[current]))
    data={'schema_version':2,'training_config':{'seed':1},'source_fingerprint':old,
      'runtime_versions':identity.runtime_versions(),'pretrained_encoder_sha256':None,'parent_signatures':[]}
    checkpoint=tmp_path/'ckpt';checkpoint.mkdir();(checkpoint/'run_manifest.json').write_text(json.dumps(data))
    p=source/'shared_arch.py';p.write_text(p.read_text()+'\n# meaningful source edit must trigger review\n')
    ok,why=identity.checkpoint_matches(checkpoint,{'seed':1},root=source)
    assert not ok and 'source' in why

def test_shell_settings_and_case_dedup():
    base=read_job(ROOT,'half-cheetah');cfg=cli_dict(base['COMMON_ARGS'])
    assert cfg['total_timesteps']=='80000' and cfg['pool_size']=='5'
    assert cfg['balance_source_lineages'] is True
    assert base['EVAL_MODES']==['deterministic']
    assert '--exclude=kh023,kh032' in base['resources']
    seq,_=suite_info(ROOT,'half-cheetah','halfcheetah_wind_vel')
    cases=make_cases(base,['main','pool','lineage','kl'],[5,3],[],seq)
    assert sum(method=='combined_policy' and cli_dict(a)['pool_size']=='5' and label=='' for method,label,a in cases)==1
    assert not any(label=='pool5' for method,label,a in cases)

def test_antdir_does_not_overwrite_velocity_suite():
    assert (ROOT/'ant/tasks.py').exists()
    seq,n=suite_info(ROOT,'AntDir','ant_dir')
    assert n==8 and seq==list(range(8))*2
    assert 'RETURN_UPPER_BOUND = None' in (ROOT/'AntDir/metrics.py').read_text()
