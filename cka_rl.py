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
                distillation = True,
                max_distill_buffer = 50_000):
        
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
        self.max_distill_buffer = max_distill_buffer

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
        # Persist the (already-merged-if-applicable) buffer pool this task inherited,
        # so the *next* task can prepend its own new buffer to it the same way
        # get_vectors() prepends this task's own weight delta to the inherited `.weights`.
        torch.save(getattr(self, "buffer_pool", []), f"{dirname}/distill_buffer_pool.pt")

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
        self.buffer_pool = []
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
            
            if self.fuse_heads and latest_dir is not None:
                # Load the buffer pool exactly the way get_vectors() loads the weight
                # pool: `latest_dir`'s own freshly-collected buffer (its "new_weight"
                # equivalent) gets prepended to whatever pool it itself inherited (and
                # possibly already reduced via its own merge_vectors() call).
                inherited_pool = []
                pool_path = f"{latest_dir}/distill_buffer_pool.pt"
                if os.path.exists(pool_path):
                    inherited_pool = torch.load(pool_path)
                own_buffer_path = f"{latest_dir}/distill_buffer.pt"
                if os.path.exists(own_buffer_path):
                    self.buffer_pool = [torch.load(own_buffer_path)] + inherited_pool
                else:
                    self.buffer_pool = inherited_pool

            self.merge_vectors()
            if self.fuse_heads:
                logger.debug(self.mean_l2_vec['weight'].shape)
                logger.debug(self.logstd_l2_vec['weight'].shape)
            
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
        
    
    def merge_vectors(self):
        """Reduce the accumulated vector pool down to `self.pool_size`.

        Called with no arguments (see `setup_vectors`); reads the per-layer
        vector pools directly off `self` (`self.mean_l0_vec`, `self.mean_l2_vec`,
        `self.logstd_l0_vec`, `self.logstd_l2_vec`) and the buffer pool
        (`self.buffer_pool`), all of which `setup_vectors` populates before
        calling this.

        IMPORTANT: all four FuseLinear layers (mean_l0, mean_l2, logstd_l0,
        logstd_l2) share the exact same `self.alpha` parameter (see
        `setup_heads`). That means slot `i` in every one of these pools MUST
        refer to the same historical task -- alpha[i] can't mean "task A" in
        one layer and "task B" in another. So we pick exactly ONE
        (idx1, idx2) pair per round (from the combined weight-delta across
        all four layers), and apply it everywhere: all four vector pools AND
        the buffer pool collapse together, in one shot, not independently.
        """
        if self.fuse_heads and self.num_vectors > self.pool_size:
            logger.info(f"Merge head vectors, pool size = {self.pool_size}, current #vectors = {self.num_vectors}")

            n = self.num_vectors
            flat_per_slot = torch.cat([
                self.mean_l0_vec['weight'].reshape(n, -1), self.mean_l0_vec['bias'].reshape(n, -1),
                self.mean_l2_vec['weight'].reshape(n, -1), self.mean_l2_vec['bias'].reshape(n, -1),
                self.logstd_l0_vec['weight'].reshape(n, -1), self.logstd_l0_vec['bias'].reshape(n, -1),
                self.logstd_l2_vec['weight'].reshape(n, -1), self.logstd_l2_vec['bias'].reshape(n, -1),
            ], dim=1)

            similarities = torch.ones((n, n)) * -1
            for i in range(n):
                for j in range(i + 1, n):
                    similarities[i, j] = torch.cosine_similarity(flat_per_slot[i], flat_per_slot[j], dim=0)
            max_sim_idx = torch.argmax(similarities)
            idx1, idx2 = divmod(max_sim_idx.item(), n)
            logger.info(f"Merging pool slots idx1={idx1}, idx2={idx2} (one shared pair for all vectors + buffers)")

            buffers = getattr(self, "buffer_pool", [])
            use_distillation = self.distillation and len(buffers) > max(idx1, idx2)
            if self.distillation and not use_distillation:
                logger.warning(
                    f"Distillation requested but buffer pool is unavailable/insufficient "
                    f"(have {len(buffers)}, need index {max(idx1, idx2)}); "
                    f"falling back to simple averaging for this round."
                )

            # Output layer (128 -> act_dim) maps directly onto the mean/log_std values the
            # distillation buffer's `targets` were recorded from, so distillation-based
            # merging is well-defined here. Trained once per head, reused for weight+bias.
            distilled = {}
            if use_distillation:
                distilled["mean"] = self.distillation_policy_merge(
                    buffers[idx1], buffers[idx2], "shared", "targets", "mean_l2", self.act_dim
                )
                distilled["logstd"] = self.distillation_policy_merge(
                    buffers[idx1], buffers[idx2], "shared", "targets", "logstd_l2", self.act_dim
                )

            def collapse(vectors, distilled_wb=None):
                for name, element in vectors.items():
                    if distilled_wb is not None:
                        new_w, new_b = distilled_wb
                        new_element = new_w if name == 'weight' else new_b
                    else:
                        new_element = (element[idx1] + element[idx2]) / 2
                    # distillation_policy_merge always returns CPU tensors, but `element`
                    # (the loaded pool) can be on cuda:0 if it was torch.load'd without
                    # map_location from a checkpoint saved while the model was on GPU.
                    # Match devices before concatenating either way.
                    new_element = new_element.to(element.device)
                    element = torch.cat(
                        (element[:idx1], element[idx1+1:idx2], element[idx2+1:], new_element.unsqueeze(0)), dim=0
                    )
                    vectors[name] = element

            collapse(self.mean_l2_vec, distilled.get("mean"))
            collapse(self.logstd_l2_vec, distilled.get("logstd"))
            # Hidden layer (256 -> 128) has no supervised target recorded in the distillation
            # buffer (only the shared features and the final mean/log_std are stored), so it
            # always uses simple averaging, regardless of the distillation flag.
            collapse(self.mean_l0_vec, None)
            collapse(self.logstd_l0_vec, None)

            # Collapse the buffer pool with the SAME (idx1, idx2), capped to max_distill_buffer.
            if buffers:
                b1, b2 = buffers[idx1], buffers[idx2]
                merged_buffer = {k: np.concatenate([b1[k], b2[k]], axis=0) for k in ("obs", "shared", "targets")}
                n_rows = merged_buffer["obs"].shape[0]
                if n_rows > self.max_distill_buffer:
                    keep = np.random.choice(n_rows, size=self.max_distill_buffer, replace=False)
                    merged_buffer = {k: v[keep] for k, v in merged_buffer.items()}
                self.buffer_pool = buffers[:idx1] + buffers[idx1+1:idx2] + buffers[idx2+1:] + [merged_buffer]

            self.num_vectors = self.pool_size

        # fuse_shared is unused by run_sac.py (hardcoded False) -- left as its original,
        # independent per-tensor merge. If you ever enable fuse_shared alongside fuse_heads,
        # revisit this to share one (idx1, idx2) pair too, for the same alpha-consistency
        # reason described above.
        if self.fuse_shared and hasattr(self.fc, 'network'):
            def merge(vectors, layer_type, input_key, target_key, target_dim, allow_distillation=True):
                for name, element in vectors.items():
                    similarities = torch.ones((element.shape[0], element.shape[0])) * -1
                    for i in range(element.shape[0]):
                        for j in range(i + 1, element.shape[0]):
                            similarities[i, j] = torch.cosine_similarity(element[i].flatten(), element[j].flatten(), dim=0)
                    max_sim_idx = torch.argmax(similarities)
                    idx1, idx2 = divmod(max_sim_idx.item(), element.shape[0])
                    new_element = (element[idx1] + element[idx2]) / 2
                    element = torch.cat((element[:idx1], element[idx1+1:idx2], element[idx2+1:], new_element.unsqueeze(0)), dim=0)
                    vectors[name] = element

            for layer_idx in self.fc.fuse_layers:
                layer = self.fc.network[layer_idx]
                if layer.num_weights > self.pool_size:
                    base_layout = layer.get_base()
                    vectors_layout, _ = layer.get_vectors(base_layout)
                    merge(vectors_layout, f"shared_layer_{layer_idx}", "obs", "shared", layer.out_features)
                    layer.weights.data.copy_(vectors_layout['weight'])
                    layer.biaes.data.copy_(vectors_layout['bias'])


    # Trains a small supervised MLP (input_key -> target_key) on the pooled distillation
    # buffers of the two most-similar vectors, then returns its output layer's weight/bias
    # as the merged replacement vector.
    def distillation_policy_merge(self, buffer1, buffer2, input_key, target_key, layer_type, target_dim, epochs=5, lr=1e-3, batch_size=128):
        logger.info(f"Distillation training: matching [{input_key}] to [{target_key}] for layer: {layer_type}")
        
        inputs_data = np.concatenate([buffer1[input_key], buffer2[input_key]], axis=0)
        targets_data = np.concatenate([buffer1[target_key], buffer2[target_key]], axis=0)
        
        inputs_tensor = torch.tensor(inputs_data, dtype=torch.float32)
        targets_tensor = torch.tensor(targets_data, dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(inputs_tensor, targets_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # NOTE: this runs from inside setup_vectors() -> merge_vectors(), which is called
        # in __init__ BEFORE self.alpha is created (setup_alpha() runs after setup_vectors()
        # because it needs the num_vectors count that setup_vectors/merge_vectors finalizes).
        # So this can't read self.alpha -- it doesn't exist yet at this point. The returned
        # weight/bias are moved to .cpu() below anyway, so training this small MLP on
        # whatever device is available is fine regardless of where the model ends up.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
                final_target = batch_target[:, :self.act_dim] if layer_type.startswith("mean") else batch_target[:, self.act_dim:]
                
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
 