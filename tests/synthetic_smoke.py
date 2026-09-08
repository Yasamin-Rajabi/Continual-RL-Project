"""Run the REAL SAC/benchmark code with a tiny synthetic Gym/replay adapter.

This catches Python integration, snapshot, budget and optimizer bugs. It is NOT
MuJoCo/MetaWorld validation and makes no claims about task performance. Missing
simulator libraries are deliberately substituted, not silently simulated.
Usage: python tests/synthetic_smoke.py --family half-cheetah
"""
from __future__ import annotations
import argparse
import dataclasses
import importlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--family', default='half-cheetah')
parser.add_argument('--output', default=None)
parser.add_argument('--suite', default=None)
options = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT/options.family
sys.path.insert(0, str(FOLDER))
torch.set_num_threads(1)


def module(name, **attributes):
    obj = types.ModuleType(name)
    obj.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    obj.__dict__.update(attributes)
    sys.modules[name] = obj
    return obj


class Box:
    def __init__(self, shape):
        self.shape = shape
        self.high = np.ones(shape, np.float32)
        self.low = -self.high
        self.dtype = np.float32
        self.rng = np.random.default_rng(0)
    def seed(self, seed): self.rng = np.random.default_rng(seed)
    def sample(self): return self.rng.uniform(self.low, self.high).astype(self.dtype)


ENVIRONMENTS = []
class SyntheticEnv:
    def __init__(self):
        self.observation_space, self.action_space = Box((6,)), Box((3,))
        self.steps = 0
        self.obs = np.zeros(6, np.float32)
        self.t = 0
        ENVIRONMENTS.append(self)
    def reset(self, seed=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.obs = self.rng.normal(0, 0.02, size=6).astype(np.float32)
        self.t = 0
        return self.obs.copy(), {}
    def step(self, action):
        self.t += 1; self.steps += 1
        self.obs[:3] = 0.8*self.obs[:3] + 0.1*np.asarray(action)
        error = float(np.linalg.norm(self.obs[:3] - 0.1))
        info = dict(success=float(error < .2), velocity_error=error, task_error=error, x_velocity=float(self.obs[0]))
        return self.obs.copy(), -error, self.t >= 5, False, info
    def close(self): pass


class Vector:
    def __init__(self, thunks):
        self.env = thunks[0]()
        self.num_envs = 1
        self.single_action_space = self.env.action_space
        self.single_observation_space = self.env.observation_space
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs[None], info
    def step(self, actions):
        obs, reward, terminated, truncated, info = self.env.step(actions[0])
        final = obs.copy()
        info = {k: np.array([v]) for k, v in info.items()}
        if terminated or truncated:
            obs, _ = self.env.reset()
            info['final_observation'] = np.array([final])
            info['_final_observation'] = np.array([True])
        return obs[None], np.array([reward]), np.array([terminated]), np.array([truncated]), info
    def close(self): self.env.close()


class Replay:
    def __init__(self, size, obs, act, device, **kwargs):
        self.device = device
        self.data = []
    def add(self, obs, nxt, actions, rewards, dones, infos):
        self.data.append([np.asarray(v).copy() for v in (obs, nxt, actions, rewards, dones)])
    def sample(self, batch_size):
        batch = [self.data[i] for i in np.random.randint(len(self.data), size=batch_size)]
        names = ('observations', 'next_observations', 'actions', 'rewards', 'dones')
        values = [torch.as_tensor(np.concatenate([row[i] for row in batch]), dtype=torch.float32,
                                  device=self.device) for i in range(5)]
        values[3], values[4] = values[3][:, None], values[4][:, None]
        return types.SimpleNamespace(**dict(zip(names, values)))


class Writer:
    def __init__(self, *args, **kwargs): pass
    def add_scalar(self, *args, **kwargs): pass
    def add_text(self, *args, **kwargs): pass
    def flush(self): pass
    def close(self): pass

module('gymnasium', spaces=types.SimpleNamespace(Box=Box),
       wrappers=types.SimpleNamespace(RecordEpisodeStatistics=lambda env: env),
       vector=types.SimpleNamespace(SyncVectorEnv=Vector))
module('stable_baselines3'); module('stable_baselines3.common')
module('stable_baselines3.common.buffers', ReplayBuffer=Replay)
module('torch.utils.tensorboard', SummaryWriter=Writer)
module('tensorboard'); module('tensorboard.backend'); module('tensorboard.backend.event_processing')
module('tensorboard.backend.event_processing.event_accumulator')
OVERRIDES = {}
module('tyro', cli=lambda cls: cls(**OVERRIDES))
# Fast progress logging, not algorithm logic.
module('tqdm', tqdm=lambda obj, **kwargs: obj)
import tasks
tasks.get_task = lambda task_id, task_suite=None, **kwargs: SyntheticEnv()
import run_continual_benchmark as benchmark
import metrics
from checkpoint_evaluation import evaluate

records = []
with tempfile.TemporaryDirectory(prefix='crl_synthetic_') as temp:
    saved_argv = sys.argv
    sys.argv = ['benchmark', '--total-timesteps', '40', '--learning-starts', '2',
                '--random-actions-end', '2', '--alpha-warmup-steps', '10',
                '--batch-size', '8', '--pool-size', '2', '--eval-every', '10',
                '--num-evals', '1', '--distill-buffer-steps', '8', '--projection-epochs', '2',
                '--distill-epochs', '2', '--similarity-samples', '8', '--distill-max-samples', '16',
                '--distill-batch-size', '8', '--max-distill-buffer', '16', '--cpu',
                '--save-root', temp+'/models', '--runs-root', temp+'/runs',
                '--analysis-root', temp+'/analysis', '--plots-root', temp+'/plots',
                '--skip-forward-transfer']
    args = benchmark.parse_args()
    sys.argv = saved_argv
    args.task_sequence = [0, 1, 0]
    args.analysis_log_every = 5
    args.retention_eval_episodes = 1
    suite = options.suite or next(iter(tasks.TASK_SUITES))
    # Numeric metric regression, including repeated task IDs.
    result = metrics.compute_fg_bwt({"0": 0.8, "1": 0.4, "2": 0.7}, {"0": 0.7, "1": 0.6}, [0, 1, 0])
    assert abs(result['FG'] - .05) < 1e-8 and abs(result['BWT'] - .05) < 1e-8, result
    def run_subprocess(cmd, check=True):
        global OVERRIDES
        # Parse the benchmark-generated argv into actual run_sac.Args fields.
        import run_sac
        defaults = run_sac.Args()
        fields = {f.name: f for f in dataclasses.fields(defaults)}
        data = {}; i = 2
        while i < len(cmd):
            arg = cmd[i]
            if arg == '--prev-units':
                data['prev_units'] = tuple(Path(p) for p in cmd[i+1:]); break
            assert arg.startswith('--'), arg
            body = arg[2:]
            if '=' in body:
                key, value = body.split('=', 1); key = key.replace('-', '_')
                default = getattr(defaults, key)
                if isinstance(default, int) and not isinstance(default, bool): value = int(value)
                elif isinstance(default, float): value = float(value)
                data[key] = value
            elif body.startswith('no-'):
                data[body[3:].replace('-', '_')] = False
            else: data[body.replace('-', '_')] = True
            i += 1
        assert set(data) <= fields.keys()
        OVERRIDES = data
        count = len(ENVIRONMENTS)
        runpy.run_path(str(FOLDER/'run_sac.py'), run_name='__main__')
        # First environment is training; the second is monitor evaluation.
        assert ENVIRONMENTS[count].steps == 40, ENVIRONMENTS[count].steps
        assert ENVIRONMENTS[count + 1].steps > 0
    benchmark.subprocess.run = run_subprocess
    for space in ('parameter', 'policy'):
        for condition in ('baseline', 'combined'):
            label = condition + ('_policy' if space == 'policy' else '')
            cfg = {**benchmark.CONDITIONS[condition], 'composition_space': space}
            paths = benchmark.train_chain(args, suite, label, cfg, seed=1)
            latest = paths[-1]
            budget = json.loads((latest/'interaction_budget.json').read_text())
            assert budget['Delta'] == 40 and budget['optimization_phase_steps'] == 32 and budget['frozen_tail_steps'] == 8
            for frozen in ('pool', 'snapshot'):
                for action_mode in ('deterministic', 'stochastic'):
                    result = evaluate(latest, suite, 0, 2, 1, torch.device('cpu'), frozen_policy=frozen, action_mode=action_mode)
                    assert result['adaptation_interactions'] == 0 and result['evaluation_interactions'] == 10
                    assert np.isfinite(result['return'])
            adapted = evaluate(latest, suite, 0, 1, 1, torch.device('cpu'), adapt_steps=3)
            assert adapted['adaptation_interactions'] == 3
            retention = metrics.build_retention_matrix(args, suite, label, 1, torch.device('cpu'))
            assert np.asarray(retention['evaluation_interactions']).shape == (3, 2)
            assert np.all(np.asarray(retention['adaptation_interactions']) == 0)
            summary = metrics.compute_survey_metrics(args, suite, label, 1, torch.device('cpu'), [101], 40)
            assert np.isfinite(summary['A_N']) and np.isnan(summary['FT_success'])
            # Exercise both FT formulas with known, normalized constant AUCs.
            original_loader = metrics.load_scalar
            def fake_curve(directory, tag):
                is_scratch = 'scratch' in str(directory)
                if 'success' in tag:
                    value = .2 if is_scratch else .4
                else:
                    value = metrics.RETURN_UPPER_BOUND - (100 if is_scratch else 50)
                return np.array([0., 40.]), np.array([value, value])
            metrics.load_scalar = fake_curve
            success_ft = metrics.compute_forward_transfer_success(args, suite, label, 1, [101], 40)
            return_ft = metrics.compute_forward_transfer_return(args, suite, label, 1, [101], 40)
            metrics.load_scalar = original_loader
            assert abs(success_ft['FT_success'] - .25) < 1e-8
            assert abs(return_ft['FT_return'] - .5) < 1e-8
            records.append(dict(family=options.family, condition=label, trained_tasks=len(paths),
                                exact_Delta=40, evaluations_checked=4, adaptation_checked=True))
report = {'type': 'SYNTHETIC adapters, NOT real simulators', 'passed': True, 'runs': records}
if options.output: Path(options.output).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
