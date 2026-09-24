"""Unit-test doubles only; NEVER used by a submitted experiment.

The development sandbox lacks Gymnasium/MuJoCo/SB3/TensorBoard. These stand-ins
exercise the real PyTorch optimizer, lifecycle, serialization, CSV and metrics
paths. paper_runs/smoke_real.py separately tests actual container dependencies.
"""
import sys,types
from pathlib import Path
import numpy as np
import torch

def install(monkeypatch):
    def mod(name):
        m=types.ModuleType(name);monkeypatch.setitem(sys.modules,name,m);return m
    gym=mod('gymnasium');gym.spaces=types.SimpleNamespace(Box=Box)
    gym.wrappers=types.SimpleNamespace(RecordEpisodeStatistics=lambda env:env)
    gym.vector=types.SimpleNamespace(SyncVectorEnv=VectorEnv,AutoresetMode=types.SimpleNamespace(SAME_STEP='same'))
    mod('stable_baselines3');mod('stable_baselines3.common');mod('stable_baselines3.common.buffers').ReplayBuffer=ReplayBuffer
    mod('torch.utils.tensorboard').SummaryWriter=Writer
    mod('tensorboard');mod('tensorboard.backend');mod('tensorboard.backend.event_processing')
    ea=mod('tensorboard.backend.event_processing.event_accumulator');ea.SCALARS='scalars';ea.EventAccumulator=Accumulator

class Box:
    def __init__(self,n):self.shape=(n,);self.low=-np.ones(n,dtype='float32');self.high=-self.low;self.dtype=np.float32;self.seed(0)
    def seed(self,seed):self.rng=np.random.default_rng(seed)
    def sample(self):return self.rng.uniform(-1,1,self.shape).astype('float32')
class TinyEnv:
    def __init__(self,task=0):self.task=task;self.observation_space=Box(6);self.action_space=Box(3)
    def reset(self,seed=None):
        if seed is not None or not hasattr(self, 'rng'):
            self.rng=np.random.default_rng(seed)
        self.x=self.rng.normal(0,.1,6).astype('float32');self.steps=0;self.ret=0
        return self.x.copy(),{}
    def step(self,a):
        a=np.asarray(a,dtype='float32');self.x[:3]=.85*self.x[:3]+.15*a;self.x[3:]=a
        r=-float(np.square(a-.2*self.task).mean());self.ret+=r;self.steps+=1
        info={'success':float(r>-.5),'velocity_error':float(np.sqrt(-r)),'x_velocity':float(self.x[0])}
        if self.steps>=8:info['episode']={'r':self.ret,'l':self.steps}
        return self.x.copy(),r,False,self.steps>=8,info
    def close(self):pass
class VectorEnv:
    def __init__(self,thunks,autoreset_mode=None):
        self.envs=[t() for t in thunks];self.num_envs=len(self.envs)
        self.single_observation_space=self.envs[0].observation_space;self.single_action_space=self.envs[0].action_space
    def reset(self,seed=None):return np.array([e.reset(seed=seed)[0] for e in self.envs]),{}
    def step(self,actions):
        obs=[];rewards=[];ds=[];ts=[];final=[];finalobs=[]
        for e,a in zip(self.envs,actions):
            o,r,d,t,info=e.step(a);rewards.append(r);ds.append(d);ts.append(t)
            final.append(info if d or t else None);finalobs.append(o.copy() if d or t else None)
            if d or t:o,_=e.reset()
            obs.append(o)
        return np.array(obs),np.array(rewards),np.array(ds),np.array(ts),dict(final_info=final,_final_info=np.logical_or(ds,ts),final_observation=finalobs)
    def close(self):
        for e in self.envs:e.close()
class ReplayBuffer:
    def __init__(self,size,obs,act,device,**kw):self.rows=[];self.device=device
    def add(self,o,n,a,r,d,infos):self.rows.append([np.array(x).copy() for x in (o,n,a,r,d)])
    def sample(self,n):
        rows=[self.rows[i] for i in np.random.randint(len(self.rows),size=n)]
        return types.SimpleNamespace(**{name:torch.as_tensor(np.stack([r[i][0] for r in rows]),dtype=torch.float32,device=self.device).reshape(n,-1)
            for i,name in enumerate(['observations','next_observations','actions','rewards','dones'])})
class Writer:
    def __init__(self,log_dir=None,**kw):self.log_dir=str(log_dir);Path(log_dir).mkdir(parents=True,exist_ok=True)
    def add_scalar(self,*a,**kw):pass
    def add_text(self,*a,**kw):pass
    def flush(self):pass
    def close(self):pass
class Accumulator:
    def __init__(self,*a,**kw):pass
    def Reload(self):return self
    def Tags(self):return {'scalars':[]}
