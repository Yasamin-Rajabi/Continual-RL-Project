"""Continuous-control ports of the supplied baseline mechanisms.

One capacity unit per task OCCURRENCE: task identities are not used to choose
an old model at training boundaries. FT-N preserves full actors (not FT-1).
Evaluation can reward-route over the *current* retained bank, use its latest
member, or explicitly request the task-aware native/oracle protocol.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from shared_arch import shared
from paper_runs.layers import CReLU, DoubleHead, CatHead, layer_init
from paper_runs.vendor.progressive_column import ProgressiveColumn
from paper_runs.vendor.componet_modules import CompoNet, FirstModuleWrapper
from paper_runs.vendor.mask_modules import (MultitaskMaskLinear, NEW_MASK_LINEAR_COMB,
    consolidate_mask, set_model_task, set_num_tasks_learned)
from paper_runs.vendor.cbp_modules import CbpGaussianActor, GnT

METHODS=('ft_n','prognet','packnet','masknet','crelus','componet','cbpnet')

@dataclass(frozen=True)
class TaskContext:
    task_id:int
    seq_idx:int
    suite:str
    seed:int
    first_encounter:bool=True
    @property
    def is_root(self): return self.seq_idx==0


def freeze(module):
    module.eval()
    for p in module.parameters(): p.requires_grad_(False)


class DensePolicy(nn.Module):
    def __init__(self, obs, act, hidden=128, linear_out=False, kind='ft_n'):
        super().__init__()
        self.fc=shared(obs,linear_out=linear_out)
        self.act_dim=act
        if kind=='crelus':
            half=hidden//2
            self.head=nn.Sequential(layer_init(nn.Linear(256,half)), CReLU(),
                                    layer_init(nn.Linear(2*half,2*act),std=.01))
        elif kind=='cbpnet': self.head=CbpGaussianActor(256,hidden,act)
        else: self.head=DoubleHead(256,hidden,act)
    def forward(self,obs):
        out=self.head(self.fc(obs))
        return out if isinstance(out,tuple) else out.split(self.act_dim,dim=-1)


class ProgressivePolicy(nn.Module):
    def __init__(self,obs,act,hidden=256):
        super().__init__();self.obs=obs;self.act=act;self.hidden=hidden
        self.columns=nn.ModuleList();self.active=-1
    def grow(self):
        for m in self.columns: freeze(m)
        self.columns.append(ProgressiveColumn(self.obs,self.hidden,2*self.act,len(self.columns)))
        self.active=len(self.columns)-1
    def forward(self,obs):
        h1s,h2s=[],[]
        for i,column in enumerate(self.columns[:self.active+1]):
            if i<self.active:
                with torch.no_grad():
                    raw,h1,h2=(column.forward_first(obs) if i==0 else column.forward_other(obs,h1s,h2s))
            else:
                raw,h1,h2=(column.forward_first(obs) if i==0 else column.forward_other(obs,h1s,h2s))
            h1s.append(h1);h2s.append(h2)
        return raw.split(self.act,dim=-1)


class PackedPolicy(nn.Module):
    """Full-actor PackNet; functional masks plus post-Adam restoration.

    The donor's equal_share allocation is retained. Unlike merely masking
    gradients, post-step restoration also protects against optimizer momentum.
    Biases are learned on the first task and then frozen, as in donor PackNet.
    """
    def __init__(self,obs,act,n,hidden=128,linear_out=False,capacity='equal_share',keep=1.0):
        super().__init__();self.fc=shared(obs,linear_out=linear_out)
        self.head=DoubleHead(256,hidden,act);self.n=n;self.capacity=capacity;self.keep=keep
        self.active=0;self.pruned=False;self._names=[];self._saved=[]
        for name,p in list(self.named_parameters()):
            if name.endswith('weight'):
                self.register_buffer(f'owner_{len(self._names)}',torch.zeros_like(p,dtype=torch.long))
                self._names.append(name)
    def weights(self):
        params=dict(self.named_parameters())
        return [(params[name],getattr(self,f'owner_{i}')) for i,name in enumerate(self._names)]
    def maskable_forward(self,module,x,prefix):
        if isinstance(module,nn.Linear):
            i=self._names.index(prefix+'.weight');owner=getattr(self,f'owner_{i}')
            mask=((owner>0)&(owner<=self.active+1))
            if not self.pruned:mask=mask|(owner==0)
            return F.linear(x,module.weight*mask,module.bias)
        for name,layer in module.named_children():
            if isinstance(layer,nn.Linear): x=self.maskable_forward(layer,x,prefix+'.'+name)
            else:x=layer(x)
        return x
    def forward(self,obs):
        z=self.maskable_forward(self.fc,obs,'fc')
        return (self.maskable_forward(self.head.mean,z,'head.mean'),
                self.maskable_forward(self.head.logstd,z,'head.logstd'))
    @torch.no_grad()
    def prune(self):
        for p,owner in self.weights():
            free=torch.nonzero(owner.flatten()==0,as_tuple=False).flatten()
            if not free.numel():continue
            keep=(int(p.numel()*self.keep/self.n) if self.capacity=='equal_share'
                  else int(free.numel()*self.keep))
            if self.active==self.n-1 and self.keep==1.0:keep=free.numel()
            keep=min(free.numel(),max(1,keep))
            ranked=free[torch.argsort(p.flatten()[free].abs(),descending=True)]
            owner.flatten()[ranked[:keep]]=self.active+1
            p.flatten()[ranked[keep:]]=0
        self.pruned=True
    @torch.no_grad()
    def before_update(self):
        self._saved=[]
        for p,owner in self.weights():
            writable=(owner==self.active+1) if self.pruned else (owner==0)
            if p.grad is not None:p.grad.mul_(writable)
            self._saved.append((p,~writable,p.detach().clone()))
    @torch.no_grad()
    def after_update(self):
        for p,mask,old in self._saved:p[mask]=old[mask]
        self._saved=[]


class SupermaskPolicy(nn.Module):
    """MaskNet donor supermask head, shared root encoder frozen thereafter."""
    def __init__(self,obs,act,n,hidden=128,linear_out=False):
        super().__init__();self.fc=shared(obs,linear_out=linear_out);self.act=act
        self.head=nn.Sequential(
            MultitaskMaskLinear(256,hidden,num_tasks=n,new_mask_type=NEW_MASK_LINEAR_COMB),nn.ReLU(),
            MultitaskMaskLinear(hidden,2*act,num_tasks=n,new_mask_type=NEW_MASK_LINEAR_COMB))
        self.active=0
    def forward(self,obs):return self.head(self.fc(obs)).split(self.act,dim=-1)
    def set_active(self,i,training=False):
        self.active=int(i);set_model_task(self.head,i,new_task=training)
        for m in self.head.modules():
            if isinstance(m,MultitaskMaskLinear):
                for j,p in enumerate(m.scores):p.requires_grad_(training and j==i)
                if m.betas is not None:m.betas.requires_grad_(training and i>0)


class ComposedPolicy(nn.Module):
    """Flat module list; no recursively duplicated previous-unit trees."""
    def __init__(self,obs,act,hidden=128,linear_out=False):
        super().__init__();self.fc=shared(obs,linear_out=linear_out)
        self.units=nn.ModuleList();self.act=act;self.hidden=hidden;self.active=-1
    def grow(self):
        for unit in self.units:freeze(unit)
        if not self.units:
            unit=FirstModuleWrapper(CatHead(256,self.hidden,self.act))
        else:
            internal=nn.Sequential(layer_init(nn.Linear(256+self.hidden,self.hidden)),nn.ReLU(),
                                   layer_init(nn.Linear(self.hidden,2*self.act),std=.01))
            unit=CompoNet(list(self.units),256,self.hidden,2*self.act,internal)
            del unit.previous_units
            freeze(self.fc)
        unit.is_prev=True
        self.units.append(unit);self.active=len(self.units)-1
    def forward(self,obs):
        z=self.fc(obs);x=z
        for i,unit in enumerate(self.units[:self.active+1]):
            if i<self.active:
                with torch.no_grad():x=unit(x)
            else:x=unit(x)
        phi,_=x
        return phi[:,-1,:].split(self.act,dim=-1)


class BaselineAgent(nn.Module):
    def __init__(self,method,obs_dim,act_dim,num_occurrences,*,hidden_dim=128,
                 encoder_linear_out=False,packnet_capacity='equal_share',packnet_keep=1.0,
                 packnet_retrain_fraction=.3,cbp_replacement_rate=1e-4,
                 cbp_maturity_threshold=100,cbp_decay_rate=.99):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method;self.obs_dim=obs_dim;self.act_dim=act_dim;self.hidden_dim=hidden_dim
        self.num_occurrences=num_occurrences;self.active_seq=-1;self.tasks=[]
        self.packnet_retrain_fraction=packnet_retrain_fraction
        self.cbp_cfg=(cbp_replacement_rate,cbp_maturity_threshold,cbp_decay_rate)
        self._gnt=None;self._gnt_state=None;self._optimizer=None;self._saved_masks=[]
        self.preserved=nn.ModuleList()
        if method in ('ft_n','crelus','cbpnet'):
            self.policy=DensePolicy(obs_dim,act_dim,hidden_dim,encoder_linear_out,method)
        elif method=='prognet':self.policy=ProgressivePolicy(obs_dim,act_dim,256)
        elif method=='packnet':
            self.policy=PackedPolicy(obs_dim,act_dim,num_occurrences,hidden_dim,encoder_linear_out,
                                     packnet_capacity,packnet_keep)
        elif method=='masknet':self.policy=SupermaskPolicy(obs_dim,act_dim,num_occurrences,hidden_dim,encoder_linear_out)
        else:self.policy=ComposedPolicy(obs_dim,act_dim,hidden_dim,encoder_linear_out)
    def on_task_start(self,ctx):
        if ctx.seq_idx!=len(self.tasks):raise ValueError('Non-contiguous baseline chain')
        self.active_seq=ctx.seq_idx;self.tasks.append(ctx.task_id)
        device=next(self.parameters(),torch.empty(0)).device
        if self.method in ('prognet','componet'):self.policy.grow();self.policy.to(device)
        elif self.method=='packnet':
            self.policy.active=ctx.seq_idx;self.policy.pruned=False
            for name,p in self.policy.named_parameters():p.requires_grad_(ctx.seq_idx==0 or not name.endswith('bias'))
        elif self.method=='masknet':
            if ctx.seq_idx>0:freeze(self.policy.fc)
            self.policy.set_active(ctx.seq_idx,training=True)
        else:
            for p in self.policy.parameters():p.requires_grad_(True)
        self.train()
    def on_task_end(self,ctx):
        if self.method=='ft_n':
            saved=copy.deepcopy(self.policy).cpu();freeze(saved);self.preserved.append(saved)
        elif self.method=='packnet':
            if not self.policy.pruned:self.policy.prune()
        elif self.method=='masknet':
            consolidate_mask(self.policy.head);set_num_tasks_learned(self.policy.head,ctx.seq_idx+1)
    def on_phase_boundary(self,step,budget):
        if self.method=='packnet' and not self.policy.pruned:
            if step>=int(budget.training*(1-self.packnet_retrain_fraction)):self.policy.prune()
    def trainable_actor_parameters(self):return [p for p in self.policy.parameters() if p.requires_grad]
    def auxiliary_loss(self):return None
    def attach_optimizer(self,opt,device):
        self._optimizer=opt
        if self.method=='cbpnet':
            rate,maturity,decay=self.cbp_cfg
            self._gnt=GnT(self.policy.head,opt,decay_rate=decay,replacement_rate=rate,
                          maturity_threshold=maturity,device=device)
            if self._gnt_state:
                for k,v in self._gnt_state.items():setattr(self._gnt,k,v.to(device) if torch.is_tensor(v) else v)
    def detach_optimizer(self):
        if self._gnt:
            # Only statistics, never live optimizer/module references, cross tasks.
            self._gnt_state={k:(v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k,v in vars(self._gnt).items() if torch.is_tensor(v) or isinstance(v,(float,int))}
        self._gnt=None;self._optimizer=None
    def before_optimizer_step(self):
        if self.method=='packnet':self.policy.before_update()
        if self.method=='masknet':
            self._saved_masks=[]
            for m in self.policy.head.modules():
                if isinstance(m,MultitaskMaskLinear) and m.betas is not None and m.betas.grad is not None:
                    mask=torch.ones_like(m.betas,dtype=torch.bool);mask[self.active_seq,:self.active_seq+1]=False
                    m.betas.grad[mask]=0;self._saved_masks.append((m.betas,mask,m.betas.detach().clone()))
    @torch.no_grad()
    def after_optimizer_step(self):
        if self.method=='packnet':self.policy.after_update()
        if self.method=='masknet':
            for p,mask,old in self._saved_masks:p[mask]=old[mask]
            self._saved_masks=[]
        if self._gnt:self._gnt.step()
    def scalars(self):
        out={'stored_occurrences':len(self.tasks),
             'actor_parameters':sum(p.numel() for p in self.parameters()),
             'trainable_actor_parameters':sum(p.numel() for p in self.trainable_actor_parameters())}
        if self.method=='packnet':
            pairs=self.policy.weights();n=sum(o.numel() for p,o in pairs)
            out['free_weight_fraction']=sum(int((o==0).sum()) for p,o in pairs)/n
        return out
    def candidates(self):
        """Frozen actions derivable from the LATEST retained state, not old checkpoints."""
        self.detach_optimizer()
        if self.method=='ft_n':return [copy.deepcopy(p) for p in self.preserved]
        if self.method in ('crelus','cbpnet'):return [copy.deepcopy(self.policy)]
        out=[]
        for i in range(len(self.tasks)):
            p=copy.deepcopy(self.policy)
            if self.method=='masknet':p.set_active(i,training=False)
            else:p.active=i
            if self.method=='packnet':p.pruned=True
            # An evaluated early column need not carry its future columns.
            if self.method=='prognet':p.columns=nn.ModuleList(list(p.columns[:i+1]))
            if self.method=='componet':p.units=nn.ModuleList(list(p.units[:i+1]))
            freeze(p);out.append(p)
        return out
