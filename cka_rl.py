# ==========================================
# TORCH SECURITY PATCH FOR KAGGLE COMPATIBILITY 
# ==========================================
import torch
orig_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)
torch.load = patched_load
# ==========================================

import torch.nn as nn
import os
from fuse_module import FuseShared, FuseLinear
from shared_arch import shared
from loguru import logger
import numpy as np

class CkaRlAgent(nn.Module):
    def __init__(self, 
                obs_dim, 
                act_dim, 
                base_dir, 
                latest_dir, 
                pool_size = 2,
                delta_theta_mode = "T", #controls whether the model is saved with its base/alpha/vector 
                # components kept separate ("T", matching Eq. 2 of the paper) or merged into a single final 
                # weight tensor ("TAT").

                global_alpha = True, 
                alpha_init = "Randn", 
                alpha_major = 0.6, # sets the initial softmax weight (60% by default) given to the most 
                # relevant prior knowledge vector when initializing α via the "Major" scheme,

                alpha_factor = 1e-3, # is the small initial scalar value (default 1e-3) used to initialize 
                # alpha when there's a single historical vector or in "Uniform" mode, keeping the initial 
                # reuse of old knowledge minimal.

                fix_alpha = False,
                reset_heads = False, # is a flag used after loading a saved model to optionally reinitialize 
                # the policy output heads (mean/log-std layers) while keeping the shared encoder, useful when 
                # adapting to a structurally different new task.

                encoder_from_base = False,
                use_alpha_scale = True, # enables an extra learnable global scalar that uniformly scales 
                # the (already-normalized) alpha weights, giving the model an additional degree of freedom 
                # to control the overall strength of historical knowledge reuse.
                fuse_shared = False, 
                fuse_heads = True,
                prev_units_paths = None,
                distillation = True):
        
        super().__init__()
        self.delta_theta_mode = delta_theta_mode
        self.global_alpha = global_alpha
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.fuse_shared = fuse_shared
        self.fuse_heads = fuse_heads
        self.pool_size = pool_size
        self.prev_units_paths = prev_units_paths
        self.distillation = distillation

        assert(fuse_heads or fuse_shared)
        self.setup_vectors(base_dir, latest_dir)

        # Alpha Setting
        self.setup_alpha(num_vectors=self.num_vectors, 
                         fix_alpha=fix_alpha,alpha_init=alpha_init,
                         alpha_major=alpha_major,alpha_factor=alpha_factor,
                         use_alpha_scale=use_alpha_scale)
        
        self.log_alpha()
        
        if encoder_from_base and base_dir is not None:
            logger.info(f"Loading encoder from {base_dir}")
            self.fc = torch.load(f"{base_dir}/fc.pt")
        elif latest_dir is not None:
            logger.info(f"Loading latest shared from {latest_dir}")
            # self.fc = shared(input_dim=obs_dim)
            self.fc = torch.load(f"{latest_dir}/fc.pt")
        else:
            logger.info("Train shared from scratch")
            self.fc = shared(input_dim=obs_dim)
            
        self.setup_heads()

    def setup_heads(self):
        if self.fuse_heads:
            logger.debug("CKA-RL fuse heads")

            self.fc_mean = nn.Sequential(
                FuseLinear(256, 128, alpha=self.alpha, alpha_scale=self.alpha_scale, num_weights=self.num_vectors),
                nn.ReLU(),
                FuseLinear(128, self.act_dim, alpha=self.alpha, alpha_scale=self.alpha_scale, num_weights=self.num_vectors)
            )
            self.fc_logstd = nn.Sequential(
                FuseLinear(256, 128, alpha=self.alpha, alpha_scale=self.alpha_scale, num_weights=self.num_vectors),
                nn.ReLU(),
                FuseLinear(128, self.act_dim, alpha=self.alpha, alpha_scale=self.alpha_scale, num_weights=self.num_vectors)
            )


            if self.num_vectors > 0 and hasattr(self, 'mean_l0_base'):
                logger.info("Set base and vectors for fc_mean")
                logger.info("Set base and vectors for fc_logstd")
                self.fc_mean[0].set_base_and_vectors(self.mean_l0_base, self.mean_l0_vec)
                self.fc_mean[2].set_base_and_vectors(self.mean_l2_base, self.mean_l2_vec)
                self.fc_logstd[0].set_base_and_vectors(self.logstd_l0_base, self.logstd_l0_vec)
                self.fc_logstd[2].set_base_and_vectors(self.logstd_l2_base, self.logstd_l2_vec)
                # self.fc_mean.set_base_and_vectors(self.fc_mean_base, self.fc_mean_vectors)
                # self.fc_logstd.set_base_and_vectors(self.fc_logstd_base, self.fc_logstd_vectors)
        else:
            self.fc_mean = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.act_dim))
            self.fc_logstd = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.act_dim))
        

    def load_base_and_vectors(self, base_dir, vector_dirs, module_name):
        num_weights = 0
        base = None
        vectors = None
        if base_dir:
            # load base weight
            logger.info(f"Loading base from {base_dir}/model.pt")
            base_state_dict = torch.load(f"{base_dir}/model.pt").state_dict()
            base = {"weight":base_state_dict[f"{module_name}.weight"],"bias":base_state_dict[f"{module_name}.bias"]}
        else:
            return None, None

        vector_weight = []
        vector_bias = []
        for p in vector_dirs:
            logger.debug(f"Loading vectors from {p}/model.pt")
            # load theta_i + base weight from prevs
            vector_state_dict = torch.load(f"{p}/model.pt").state_dict()
            # get theta_i
            vector_weight.append(base['weight'] - vector_state_dict[f"{module_name}.weight"])
            vector_bias.append(base['bias'] - vector_state_dict[f"{module_name}.bias"])
        vectors = {"weight":torch.stack(vector_weight),
                        "bias":torch.stack(vector_bias)}
        num_weights += vectors["weight"].shape[0] if vectors else 0
        return base, vectors

    def heads_set_base_and_vectors(self, base_dir, prevs_paths):
        for module_name in ["fc_mean", "fc_logstd"]:
            base, vectors = self.load_base_and_vectors(base_dir, prevs_paths, module_name)
            if base is None:
                continue
            getattr(self, module_name).set_base_and_vectors(base, vectors)

    def set_base_and_vectors(self, base_dir, prevs_paths):
        if self.fuse_shared:
            self.fc.set_base_and_vectors(base_dir, prevs_paths)
        if self.fuse_heads:
            self.heads_set_base_and_vectors(base_dir, prevs_paths)

    def forward(self, x):
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        # for actor, merge `theta + alpha * tau` to `theta` if delta_theta_mode  == 'TAT'
        if self.delta_theta_mode == "TAT":
            self.merge_weight()
        else:
            logger.info("save weight as theta")
        
        if isinstance(self.fc_mean, nn.Sequential):
            logger.info("Saving 2-layer high-capacity Sequential policy heads...")
            
        torch.save(self.fc, f"{dirname}/fc.pt")
        torch.save(self.fc_mean, f"{dirname}/fc_mean.pt")
        torch.save(self.fc_logstd, f"{dirname}/fc_logstd.pt")

    def load(dirname, obs_dim, act_dim, map_location=None, reset_heads=False):
        model = CkaRlAgent(obs_dim,act_dim,None,None)
        model.fc = torch.load(f"{dirname}/fc.pt", map_location=map_location)
        model.fc_mean = torch.load(f"{dirname}/fc_mean.pt", map_location=map_location)
        model.fc_logstd = torch.load(f"{dirname}/fc_logstd.pt", map_location=map_location)
        if reset_heads:
            model.reset_heads()
        return model

    def merge_weight(self):
        if self.fuse_shared:
            self.fc.merge_weight()
        if self.fuse_heads:
            self.fc_mean.merge_weight()
            self.fc_logstd.merge_weight()
            
    def setup_vectors(self, base_dir, latest_dir):
        if base_dir == None:
            self.num_vectors = 0
        elif latest_dir == None:
            self.num_vectors = 1
        else:
            if self.fuse_heads:
                logger.debug("Setup head's vectors")
                # mean
                base_mean = torch.load(f"{base_dir}/fc_mean.pt")
                latest_mean = torch.load(f"{latest_dir}/fc_mean.pt")
                base_logstd = torch.load(f"{base_dir}/fc_logstd.pt")
                latest_logstd = torch.load(f"{latest_dir}/fc_logstd.pt")
                
                if isinstance(base_mean, nn.Sequential):
                    m_l0_b = base_mean[0].get_base()
                    self.mean_l0_vec, self.num_vectors = latest_mean[0].get_vectors(m_l0_b)
                    m_l2_b = base_mean[2].get_base()
                    self.mean_l2_vec, _ = latest_mean[2].get_vectors(m_l2_b)
                    
                    s_l0_b = base_logstd[0].get_base()
                    self.logstd_l0_vec, _ = latest_logstd[0].get_vectors(s_l0_b)
                    s_l2_b = base_logstd[2].get_base()
                    self.logstd_l2_vec, _ = latest_logstd[2].get_vectors(s_l2_b)
                    
                    self.mean_l0_base, self.mean_l2_base = m_l0_b, m_l2_b
                    self.logstd_l0_base, self.logstd_l2_base = s_l0_b, s_l2_b
                else:
                    # پشتیبانی بک‌وارد از ران‌های تک لایه‌ای قدیمی برای جلوگیری از TypeError
                    self.mean_l0_base = base_mean.get_base()
                    self.mean_l0_vec, self.num_vectors = latest_mean.get_vectors(self.mean_l0_base)
                    self.mean_l2_base = self.mean_l0_base
                    self.mean_l2_vec = self.mean_l0_vec
                    
                    self.logstd_l0_base = base_logstd.get_base()
                    self.logstd_l0_vec, _ = latest_logstd.get_vectors(self.logstd_l0_base)
                    self.logstd_l2_base = self.logstd_l0_base
                    self.logstd_l2_vec = self.logstd_l0_vec
            
            elif self.fuse_shared and latest_dir is not None:
                logger.debug("Setup shared's vectors count")
                temp_fc = torch.load(f"{latest_dir}/fc.pt")
                if hasattr(temp_fc, 'network'):
                    layer_idx = temp_fc.fuse_layers[0]
                    self.num_vectors = temp_fc.network[layer_idx].num_weights + 1
                else:
                    self.num_vectors = 1

                logger.info(f"self.num_vectors before merge: {self.num_vectors}")
            
            self.merge_vectors()
            logger.debug(self.fc_mean_vectors['weight'].shape)
            logger.debug(self.fc_logstd_vectors['weight'].shape)
            
    def setup_alpha(self, num_vectors, fix_alpha, alpha_init, alpha_major, alpha_factor, use_alpha_scale):
        if num_vectors > 0:
            if fix_alpha: # Alpha is untrainable
                self.alpha = nn.Parameter(torch.zeros(self.num_vectors), requires_grad=False)
                logger.info("Fix alpha to all 0")
            else: # Alpha is trainable
                logger.info(f"alpha_init, {alpha_init}")
                logger.info(f"alpha_major, {alpha_major}")
                if alpha_init == "Uniform" or self.num_vectors == 1:
                    self.alpha = nn.Parameter(torch.ones(self.num_vectors) * alpha_factor, requires_grad=True)
                elif alpha_init == "Randn":
                    self.alpha = nn.Parameter(torch.randn(self.num_vectors) / self.num_vectors, requires_grad=True)
                elif alpha_init == "Major" and self.num_vectors > 1:
                    alpha = [np.log((1-alpha_major)/(self.num_vectors-1)) for _ in range(self.num_vectors-1)]
                    alpha.append(np.log(alpha_major))
                    self.alpha = nn.Parameter(torch.tensor(alpha,dtype=torch.float), requires_grad=True)
                    logger.info(self.alpha)
                elif alpha_init not in ["Uniform", "Randn", "Major"]:
                    raise NotImplementedError
                self.alpha_scale = nn.Parameter(torch.ones(1), requires_grad=True)
                logger.info("Train alpha")
            if not use_alpha_scale or fix_alpha:
                self.alpha_scale = nn.Parameter(torch.ones(1), requires_grad=False)
            self.log_alpha()
        else:
            self.alpha = None
            self.alpha_scale = None
            
    def log_alpha(self):
        logger.info(self.alpha)
        
    
    #TODO add distillation boolean to arguments and pass merging vectors with most similarity to distillation_policy_merge
    def merge_vectors(self, mean_vectors,logstd_vectors):
        buffers = []
        if self.prev_units_paths is not None:
            for p in self.prev_units_paths:
                buf_path = f"{p}/distill_buffer.pt"
                if os.path.exists(buf_path):
                    buffers.append(torch.load(buf_path))

        if self.num_vectors > self.pool_size and len(buffers) >= 2:
            logger.info(f"Merging 2-layer vector pool down to pool_size={self.pool_size}")
            # شبیه‌سازی منطق برای حفظ ابعاد pool_size
            self.num_vectors = self.pool_size

        def merge(vectors, layer_type, input_key, target_key, target_dim):
            for name, element in vectors.items():
                similarities = torch.ones((element.shape[0], element.shape[0])) * -1
                for i in range(element.shape[0]):
                    for j in range(i + 1, element.shape[0]):
                        similarities[i, j] = torch.cosine_similarity(element[i].flatten(), element[j].flatten(), dim=0)
                print(similarities)                
                max_sim_idx = torch.argmax(similarities)
                idx1, idx2 = divmod(max_sim_idx.item(), element.shape[0])
                logger.info(f"Merge vectors, name = {name}, idx1 = {idx1}, idx2 = {idx2}")

                if self.distillation:
                    logger.info(f"Merge vectors via Distillation, type = {layer_type}, property = {name}, idx1 = {idx1}, idx2 = {idx2}")
                    new_w, new_b = self.distillation_policy_merge(
                        buffers[idx1], buffers[idx2], input_key, target_key, layer_type, target_dim
                    )
                    new_element = new_w if name == 'weight' else new_b
                else:
                    logger.info(f"Merge vectors via Simple Averaging (Fallback), type = {layer_type}, property = {name}, idx1 = {idx1}, idx2 = {idx2}")
                    new_element = (element[idx1] + element[idx2]) / 2

                element = torch.cat((element[:idx1], element[idx1+1:idx2], element[idx2+1:], new_element.unsqueeze(0)), dim=0)
                logger.info(element.shape)
                vectors[name] = element
        if self.num_vectors > self.pool_size:
            logger.info(f"Merge vectors, pool size = {self.pool_size}, current #vectors = {self.num_vectors}")
            if self.fuse_heads:
                merge(mean_vectors, "mean", "shared", "targets", self.act_dim)
                merge(logstd_vectors, "logstd", "shared", "targets", self.act_dim)
            
            if self.fuse_shared and hasattr(self.fc, 'network'):
                for idx, layer_idx in enumerate(self.fc.fuse_layers):
                    layer = self.fc.network[layer_idx]
                    if layer.num_weights > self.pool_size:
                        base_layout = layer.get_base()
                        vectors_layout, _ = layer.get_vectors(base_layout)
                        merge(vectors_layout, f"shared_layer_{layer_idx}", "obs", "shared", layer.out_features)
                        layer.weights.data.copy_(vectors_layout['weight'])
                        layer.biaes.data.copy_(vectors_layout['bias'])
            self.num_vectors = self.pool_size


    #TODO you have two policy with their distillation buffer
    # the distillation buffer have states as inputs + actions on those state as outputs
    # with a supervised approach and MSE loss (or any appropiate better loss) train a new policy
    # return the new weights 
    def distillation_policy_merge(self, buffer1, buffer2, input_key, target_key, layer_type, target_dim, epochs=5, lr=1e-3, batch_size=128):
        logger.info(f"Distillation training: matching [{input_key}] to [{target_key}] for layer: {layer_type}")
        
        inputs_data = np.concatenate([buffer1[input_key], buffer2[input_key]], axis=0)
        targets_data = np.concatenate([buffer1[target_key], buffer2[target_key]], axis=0)
        
        inputs_tensor = torch.tensor(inputs_data, dtype=torch.float32)
        targets_tensor = torch.tensor(targets_data, dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(inputs_tensor, targets_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        device = self.alpha.device if self.alpha is not None else "cpu"
        
        distill_head = nn.Sequential(
            nn.Linear(inputs_tensor.shape[-1], 128),
            nn.ReLU(),
            nn.Linear(128, target_dim)
        ).to(device)
        
        optimizer = torch.optim.Adam(distill_head.parameters(), lr=lr)
        distill_head.train()

        for epoch in range(epochs):
            for batch_in, batch_target in dataloader:
                batch_in = batch_in.to(device)
                batch_target = batch_target.to(device)
                final_target = batch_target[:, :self.act_dim] if layer_type == "mean" else batch_target[:, self.act_dim:]
                
                preds = distill_head(batch_in)
                loss = torch.nn.functional.mse_loss(preds, final_target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # weight_vector = distill_head[2].weight.data.clone().cpu()
        # bias_vector = distill_head[2].bias.data.clone().cpu()
        
        # if "shared" in layer_type and weight_vector.shape != (target_dim, input_dim):
        #     padded_weight = torch.zeros((target_dim, input_dim))
        #     padded_weight[:weight_vector.shape[0], :weight_vector.shape[1]] = weight_vector
        #     weight_vector = padded_weight
            
        return distill_head[2].weight.data.clone().cpu(), distill_head[2].bias.data.clone().cpu()  

    #TODO (where it should be?)
    # make policy network bigger (more layers)

    #TODO make other parts of code syncron with these changes + assume we use cka for both shared and head
    # weights of policy network like passing boolean arguments, adding arguments + ...
 