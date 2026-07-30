import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, ParameterList, init
from loguru import logger

class FuseLinear(nn.Module):
    r"""Applies a linear transformation with dual-mode fusion (Classic CKA or Weight-Delta)"""
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, 
                 out_features: int, 
                 bias: bool = True, 
                 num_weights: int = 0, 
                 alpha: nn.Parameter = None,
                 alpha_scale: nn.Parameter = torch.tensor(1.0, requires_grad=False),
                 fusion_mode: str = "classic_cka",
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha 
        self.alpha_scale = alpha_scale 
        self._bias = bias
        self.num_weights = num_weights 
        self.fusion_mode = fusion_mode # "classic_cka" or "weight_delta"
        
        if self.num_weights > 0:
            assert(self.alpha.shape[0] == self.num_weights)
        assert self.num_weights >= 0, "num_weights must be non-negative"
        
        if self.fusion_mode == "weight_delta":
            self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)
            self.b_weight = Parameter(torch.zeros((out_features, in_features), **factory_kwargs), requires_grad=True)
            if bias:
                self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
                self.b_bias = Parameter(torch.zeros(out_features, **factory_kwargs), requires_grad=True)
            else:
                self.register_parameter('bias', None)
                self.register_parameter('b_bias', None)
        else:
            self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=True)
            self.register_parameter('b_weight', None)
            if bias:
                self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=True)
            else:
                self.register_parameter('bias', None)
            self.register_parameter('b_bias', None)

        if self.num_weights > 0:
            self.weights = Parameter(torch.stack([torch.empty((out_features, in_features)) for _ in range(num_weights)], dim=0), requires_grad=False)
            if bias:
                self.biaes = Parameter(torch.stack([torch.empty(out_features) for _ in range(num_weights)], dim=0), requires_grad=False)
            else:
                self.register_parameter('biases', None)
        else:
            self.weights = None
            if bias:
                self.biaes = None
            else:
                self.register_parameter('biases', None)
                
        self.reset_parameters()

    @torch.no_grad()
    def merge_weight(self):
        if self.num_weights <= 0 and self.fusion_mode == "classic_cka":
            return
            
        hist_weight = 0
        hist_bias = 0
        if self.num_weights > 0:
            alphas_normalized = F.softmax(self.alpha * self.alpha_scale, dim=0)
            hist_weight = (alphas_normalized.view(-1, 1, 1) * self.weights.data).sum(dim = 0)
            if self._bias:
                hist_bias = (alphas_normalized.view(-1,1) * self.biaes.data).sum(dim=0)

        if self.fusion_mode == "weight_delta":
            self.weight.data += self.b_weight.data + hist_weight
            self.b_weight.data.zero_()
            if self._bias:
                self.bias.data += self.b_bias.data + hist_bias
                self.b_bias.data.zero_()
        else:
            self.weight.data += hist_weight
            if self._bias:
                self.bias.data += hist_bias

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        hist_weight = 0
        hist_bias = 0
        if self.alpha is not None and self.weights is not None:
            alphas_normalized = F.softmax(self.alpha * self.alpha_scale, dim=0)
            hist_weight = (alphas_normalized.view(-1, 1, 1) * self.weights).sum(dim = 0)
            if self._bias:
                hist_bias = (alphas_normalized.view(-1,1) * self.biaes).sum(dim=0)
        
        if self.fusion_mode == "weight_delta":
            eff_weight = self.weight + self.b_weight + hist_weight
            eff_bias = (self.bias + self.b_bias + hist_bias) if self._bias else None
        else:
            eff_weight = self.weight + hist_weight
            eff_bias = (self.bias + hist_bias) if self._bias else None
                
        return F.linear(input, eff_weight, eff_bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, num_weights={self.num_weights}, fusion={self.fusion_mode}'   
        
    def set_base_and_vectors(self, base, vectors):
        if base is not None:
            assert('weight' in base and 'bias' in base)
            self.weight.data.copy_(base['weight'])
            if self._bias:
                self.bias.data.copy_(base['bias'])
                
            if self.fusion_mode == "weight_delta":
                self.b_weight.data.zero_()
                if self._bias:
                    self.b_bias.data.zero_()
        else:
            return
            
        if vectors is not None: 
            assert('weight' in vectors and 'bias' in vectors)
            self.weights.data.copy_(vectors['weight'])
            if self._bias:
                self.biaes.data.copy_(vectors['bias'])

    def get_vectors(self, base = None):
        if self.fusion_mode == "weight_delta":
            hist_weight = 0
            hist_bias = 0
            if self.weights is not None:
                alphas_normalized = F.softmax(self.alpha * self.alpha_scale, dim=0)
                hist_weight = (alphas_normalized.view(-1, 1, 1) * self.weights.data).sum(dim=0)
                if self._bias:
                    hist_bias = (alphas_normalized.view(-1,1) * self.biaes.data).sum(dim=0)
            
            # V_t = b + sum(beta * V_i)
            new_weight = self.b_weight.data + hist_weight
            new_bias = self.b_bias.data + hist_bias if self._bias else None
        else:
            if base is None:
                base_weight = torch.zeros_like(self.weight)
                base_bias = torch.zeros_like(self.bias) if self._bias else None
            else:
                base_weight = base['weight']
                base_bias = base['bias']
                
            new_weight = self.weight.data - base_weight
            new_bias = self.bias.data - base_bias if self._bias else None
        
        if self.weights is not None:
            weights = torch.cat([new_weight.unsqueeze(0), self.weights.data], dim=0)
        else:
            weights = new_weight.unsqueeze(0)
            
        if self._bias:
            if self.biaes is not None:
                biaes = torch.cat([new_bias.unsqueeze(0), self.biaes.data], dim=0)
            else:
                biaes = new_bias.unsqueeze(0)
        else:
            biaes = None
            
        return {"weight":weights, "bias":biaes}, weights.shape[0]
    
    def get_base(self):
        return {"weight":self.weight, "bias":self.bias}

class FuseShared(nn.Module):
    def __init__(self, input_dim, 
                    layer_init=lambda x, **kwargs: x,
                    num_weights: int = 0, 
                    alpha: nn.Parameter = None,
                    alpha_scale: nn.Parameter = torch.tensor(1.0, requires_grad=False),
                    global_alpha: bool = True,
                    fusion_mode: str = "classic_cka"):
        super().__init__()
        self.fuse_layers = [0,2]
        self.fusion_mode = fusion_mode
        if global_alpha or num_weights == 0:
            self.network = nn.Sequential(
                FuseLinear(input_dim, 256, 
                            num_weights=num_weights, 
                            alpha=alpha, alpha_scale=alpha_scale,
                            fusion_mode=fusion_mode), 
                nn.ReLU(),
                FuseLinear(256, 256,             
                            num_weights=num_weights, 
                            alpha=alpha, alpha_scale=alpha_scale,
                            fusion_mode=fusion_mode),
                nn.ReLU(),
            )
        else:
            logger.debug("FuseShared using local alphas")
            self.alphas = ParameterList([Parameter(alpha.clone().detach().requires_grad_(alpha.requires_grad)) for _ in range(len(self.fuse_layers))])
            self.alpha_scales = ParameterList([Parameter(alpha_scale.clone().detach().requires_grad_(alpha_scale.requires_grad)) for _ in range(len(self.fuse_layers))])
            self.network = nn.Sequential(
                FuseLinear(input_dim, 256, num_weights=num_weights, 
                alpha=self.alphas[0], alpha_scale=self.alpha_scales[0],
                fusion_mode=fusion_mode),
                nn.ReLU(),
                FuseLinear(256, 256, num_weights=num_weights, 
                                    alpha=self.alphas[1], alpha_scale=self.alpha_scales[1],
                                    fusion_mode=fusion_mode),
                nn.ReLU(),
            )
        
    def load_base_and_vectors(self, base_dir, vector_dirs):
        base = []
        vectors = []
        num_weights = 0
        if base_dir:
            logger.info(f"Loading base from {base_dir}/model.pt")
            base_state_dict = torch.load(f"{base_dir}/model.pt").state_dict()
            prefix = list(base_state_dict.keys())[0].split('0')[0][:-1]
            for i in self.fuse_layers:
                base.append({"weight":base_state_dict[f"{prefix}.{i}.weight"],"bias":base_state_dict[f"{prefix}.{i}.bias"]})
        else:
            return [None,None],[None,None]

        for idx,i in enumerate(self.fuse_layers):
            vector_weight = []
            vector_bias = []
            for p in vector_dirs:
                vector_state_dict = torch.load(f"{p}/model.pt").state_dict()
                vector_weight.append(vector_state_dict[f"{prefix}.{i}.weight"] - base[idx]['weight'])
                vector_bias.append(vector_state_dict[f"{prefix}.{i}.bias"] - base[idx]['bias'])
            vectors.append({"weight":torch.stack(vector_weight),
                            "bias":torch.stack(vector_bias)})
        num_weights += vectors[0]["weight"].shape[0] if vectors else 0
        return base, vectors
        
    def set_base_and_vectors(self, base_dir, prevs_paths):
        base, vectors = self.load_base_and_vectors(base_dir, prevs_paths)
        if base[0] is None:
            logger.warning("Not base or vectors exist")
            return 
        for idx,i in enumerate(self.fuse_layers):
            self.network[i].set_base_and_vectors(base[idx],vectors[idx])
        
    def forward(self, x):
        return self.network(x)
    
    def merge_weight(self):
        for i in self.fuse_layers:
            self.network[i].merge_weight()
            
    def log_alphas(self):
        for i in self.fuse_layers:
            normalized_alpha = F.softmax(self.network[i].alpha * self.network[i].alpha_scale, dim=0)
            logger.info(f"Layer {i} alpha: {normalized_alpha}")