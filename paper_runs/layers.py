"""Donor initialization and Gaussian heads, independent of environment modules."""
import torch
from torch import nn

def layer_init(layer, std=2**0.5, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None: nn.init.constant_(layer.bias, bias_const)
    return layer

class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([torch.relu(x),torch.relu(-x)], dim=-1)

class DoubleHead(nn.Module):
    def __init__(self, dim, hidden, act):
        super().__init__()
        self.mean=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Linear(hidden,act))
        self.logstd=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Linear(hidden,act))
    def forward(self,z): return self.mean(z),self.logstd(z)

class CatHead(nn.Module):
    def __init__(self, dim, hidden, act):
        super().__init__()
        self.net=nn.Sequential(layer_init(nn.Linear(dim,hidden)),nn.ReLU(),
                               layer_init(nn.Linear(hidden,2*act),std=.01))
    def forward(self,z): return self.net(z)
