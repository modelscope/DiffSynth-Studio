import torch
from einops import rearrange, repeat
from .flux_dit import RoPEEmbedding, TimestepEmbeddings, FluxJointTransformerBlock, FluxSingleTransformerBlock, RMSNorm
from .utils import hash_state_dict_keys, init_weights_on_device



class FluxControlNet(torch.nn.Module):
    def __init__(self, disable_guidance_embedder=False, num_joint_blocks=5, num_single_blocks=10, num_mode=0, mode_dict={}, additional_input_dim=0):
        super().__init__()
        self.pos_embedder = RoPEEmbedding(3072, 10000, [16, 56, 56])
        self.time_embedder = TimestepEmbeddings(256, 3072)
        self.guidance_embedder = None if disable_guidance_embedder else TimestepEmbeddings(256, 3072)
        self.pooled_text_embedder = torch.nn.Sequential(torch.nn.Linear(768, 3072), torch.nn.SiLU(), torch.nn.Linear(3072, 3072))
        self.context_embedder = torch.nn.Linear(4096, 3072)
        self.x_embedder = torch.nn.Linear(64, 3072)

        self.blocks = torch.nn.ModuleList([FluxJointTransformerBlock(3072, 24) for _ in range(num_joint_blocks)])
        self.single_blocks = torch.nn.ModuleList([FluxSingleTransformerBlock(3072, 24) for _ in range(num_single_blocks)])

        self.controlnet_blocks = torch.nn.ModuleList([torch.nn.Linear(3072, 3072) for _ in range(num_joint_blocks)])
        self.controlnet_single_blocks = torch.nn.ModuleList([torch.nn.Linear(3072, 3072) for _ in range(num_single_blocks)])
        
        self.mode_dict = mode_dict
        self.controlnet_mode_embedder = torch.nn.Embedding(num_mode, 3072) if len(mode_dict) > 0 else None
        self.controlnet_x_embedder = torch.nn.Linear(64 + additional_input_dim, 3072)


    def prepare_image_ids(self, latents):
        batch_size, _, height, width = latents.shape
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        latent_image_ids = latent_image_ids.to(device=latents.device, dtype=latents.dtype)

        return latent_image_ids
    

    def patchify(self, hidden_states):
        hidden_states = rearrange(hidden_states, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        return hidden_states
    

    def align_res_stack_to_original_blocks(self, res_stack, num_blocks, hidden_states):
        if len(res_stack) == 0:
            return [torch.zeros_like(hidden_states)] * num_blocks
        interval = (num_blocks + len(res_stack) - 1) // len(res_stack)
        aligned_res_stack = [res_stack[block_id // interval] for block_id in range(num_blocks)]
        return aligned_res_stack


    def forward(
        self,
        hidden_states,
        controlnet_conditioning,
        timestep, prompt_emb, pooled_prompt_emb, guidance, text_ids, image_ids=None,
        processor_id=None,
        tiled=False, tile_size=128, tile_stride=64,
        **kwargs
    ):
        if image_ids is None:
            image_ids = self.prepare_image_ids(hidden_states)

        conditioning = self.time_embedder(timestep, hidden_states.dtype) + self.pooled_text_embedder(pooled_prompt_emb)
        if self.guidance_embedder is not None:
            guidance = guidance * 1000
            conditioning = conditioning + self.guidance_embedder(guidance, hidden_states.dtype)
        prompt_emb = self.context_embedder(prompt_emb)
        if self.controlnet_mode_embedder is not None: # Different from FluxDiT
            processor_id = torch.tensor([self.mode_dict[processor_id]], dtype=torch.int)
            processor_id = repeat(processor_id, "D -> B D", B=1).to(text_ids.device)
            prompt_emb = torch.concat([self.controlnet_mode_embedder(processor_id), prompt_emb], dim=1)
            text_ids = torch.cat([text_ids[:, :1], text_ids], dim=1)
        image_rotary_emb = self.pos_embedder(torch.cat((text_ids, image_ids), dim=1))

        hidden_states = self.patchify(hidden_states)
        hidden_states = self.x_embedder(hidden_states)
        controlnet_conditioning = self.patchify(controlnet_conditioning) # Different from FluxDiT
        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_conditioning) # Different from FluxDiT

        controlnet_res_stack = []
        for block, controlnet_block in zip(self.blocks, self.controlnet_blocks):
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)
            controlnet_res_stack.append(controlnet_block(hidden_states))

        controlnet_single_res_stack = []
        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        for block, controlnet_block in zip(self.single_blocks, self.controlnet_single_blocks):
            hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)
            controlnet_single_res_stack.append(controlnet_block(hidden_states[:, prompt_emb.shape[1]:]))

        controlnet_res_stack = self.align_res_stack_to_original_blocks(controlnet_res_stack, 19, hidden_states[:, prompt_emb.shape[1]:])
        controlnet_single_res_stack = self.align_res_stack_to_original_blocks(controlnet_single_res_stack, 38, hidden_states[:, prompt_emb.shape[1]:])

        return controlnet_res_stack, controlnet_single_res_stack


    @staticmethod
    def state_dict_converter():
        return FluxControlNetStateDictConverter()
    
    def quantize(self):
        def cast_to(weight, dtype=None, device=None, copy=False):
            if device is None or weight.device == device:
                if not copy:
                    if dtype is None or weight.dtype == dtype:
                        return weight
                return weight.to(dtype=dtype, copy=copy)

            r = torch.empty_like(weight, dtype=dtype, device=device)
            r.copy_(weight)
            return r

        def cast_weight(s, input=None, dtype=None, device=None):
            if input is not None:
                if dtype is None:
                    dtype = input.dtype
                if device is None:
                    device = input.device
            weight = cast_to(s.weight, dtype, device)
            return weight

        def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None):
            if input is not None:
                if dtype is None:
                    dtype = input.dtype
                if bias_dtype is None:
                    bias_dtype = dtype
                if device is None:
                    device = input.device
            bias = None
            weight = cast_to(s.weight, dtype, device)
            bias = cast_to(s.bias, bias_dtype, device)
            return weight, bias

        class quantized_layer:
            class QLinear(torch.nn.Linear):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    
                def forward(self,input,**kwargs):
                    weight,bias= cast_bias_weight(self,input)
                    return torch.nn.functional.linear(input,weight,bias)
            
            class QRMSNorm(torch.nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                    
                def forward(self,hidden_states,**kwargs):
                    weight= cast_weight(self.module,hidden_states)
                    input_dtype = hidden_states.dtype
                    variance = hidden_states.to(torch.float32).square().mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + self.module.eps)
                    hidden_states = hidden_states.to(input_dtype) * weight
                    return hidden_states
            
            class QEmbedding(torch.nn.Embedding):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    
                def forward(self,input,**kwargs):
                    weight= cast_weight(self,input)
                    return torch.nn.functional.embedding(
                        input, weight, self.padding_idx, self.max_norm,
                        self.norm_type, self.scale_grad_by_freq, self.sparse)
            
        def replace_layer(model):
            for name, module in model.named_children():
                if isinstance(module,quantized_layer.QRMSNorm):
                    continue
                if isinstance(module, torch.nn.Linear):
                    with init_weights_on_device():
                        new_layer = quantized_layer.QLinear(module.in_features,module.out_features)
                    new_layer.weight = module.weight
                    if module.bias is not None:
                        new_layer.bias = module.bias
                    setattr(model, name, new_layer)
                elif isinstance(module, RMSNorm):
                    if hasattr(module,"quantized"):
                        continue
                    module.quantized= True
                    new_layer = quantized_layer.QRMSNorm(module)
                    setattr(model, name, new_layer)
                elif isinstance(module,torch.nn.Embedding):
                    rows, cols = module.weight.shape
                    new_layer = quantized_layer.QEmbedding(
                        num_embeddings=rows,
                        embedding_dim=cols,
                        _weight=module.weight,
                        # _freeze=module.freeze,
                        padding_idx=module.padding_idx,
                        max_norm=module.max_norm,
                        norm_type=module.norm_type,
                        scale_grad_by_freq=module.scale_grad_by_freq,
                        sparse=module.sparse)
                    setattr(model, name, new_layer)
                else:
                    replace_layer(module)

        replace_layer(self)
    


class FluxControlNetStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        hash_value = hash_state_dict_keys(state_dict)
        global_rename_dict = {
            "context_embedder": "context_embedder",
            "x_embedder": "x_embedder",
            "time_text_embed.timestep_embedder.linear_1": "time_embedder.timestep_embedder.0",
            "time_text_embed.timestep_embedder.linear_2": "time_embedder.timestep_embedder.2",
            "time_text_embed.guidance_embedder.linear_1": "guidance_embedder.timestep_embedder.0",
            "time_text_embed.guidance_embedder.linear_2": "guidance_embedder.timestep_embedder.2",
            "time_text_embed.text_embedder.linear_1": "pooled_text_embedder.0",
            "time_text_embed.text_embedder.linear_2": "pooled_text_embedder.2",
            "norm_out.linear": "final_norm_out.linear",
            "proj_out": "final_proj_out",
        }
        rename_dict = {
            "proj_out": "proj_out",
            "norm1.linear": "norm1_a.linear",
            "norm1_context.linear": "norm1_b.linear",
            "attn.to_q": "attn.a_to_q",
            "attn.to_k": "attn.a_to_k",
            "attn.to_v": "attn.a_to_v",
            "attn.to_out.0": "attn.a_to_out",
            "attn.add_q_proj": "attn.b_to_q",
            "attn.add_k_proj": "attn.b_to_k",
            "attn.add_v_proj": "attn.b_to_v",
            "attn.to_add_out": "attn.b_to_out",
            "ff.net.0.proj": "ff_a.0",
            "ff.net.2": "ff_a.2",
            "ff_context.net.0.proj": "ff_b.0",
            "ff_context.net.2": "ff_b.2",
            "attn.norm_q": "attn.norm_q_a",
            "attn.norm_k": "attn.norm_k_a",
            "attn.norm_added_q": "attn.norm_q_b",
            "attn.norm_added_k": "attn.norm_k_b",
        }
        rename_dict_single = {
            "attn.to_q": "a_to_q",
            "attn.to_k": "a_to_k",
            "attn.to_v": "a_to_v",
            "attn.norm_q": "norm_q_a",
            "attn.norm_k": "norm_k_a",
            "norm.linear": "norm.linear",
            "proj_mlp": "proj_in_besides_attn",
            "proj_out": "proj_out",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name.endswith(".weight") or name.endswith(".bias"):
                suffix = ".weight" if name.endswith(".weight") else ".bias"
                prefix = name[:-len(suffix)]
                if prefix in global_rename_dict:
                    state_dict_[global_rename_dict[prefix] + suffix] = param
                elif prefix.startswith("transformer_blocks."):
                    names = prefix.split(".")
                    names[0] = "blocks"
                    middle = ".".join(names[2:])
                    if middle in rename_dict:
                        name_ = ".".join(names[:2] + [rename_dict[middle]] + [suffix[1:]])
                        state_dict_[name_] = param
                elif prefix.startswith("single_transformer_blocks."):
                    names = prefix.split(".")
                    names[0] = "single_blocks"
                    middle = ".".join(names[2:])
                    if middle in rename_dict_single:
                        name_ = ".".join(names[:2] + [rename_dict_single[middle]] + [suffix[1:]])
                        state_dict_[name_] = param
                    else:
                        state_dict_[name] = param
                else:
                    state_dict_[name] = param
        for name in list(state_dict_.keys()):
            if ".proj_in_besides_attn." in name:
                name_ = name.replace(".proj_in_besides_attn.", ".to_qkv_mlp.")
                param = torch.concat([
                    state_dict_[name.replace(".proj_in_besides_attn.", f".a_to_q.")],
                    state_dict_[name.replace(".proj_in_besides_attn.", f".a_to_k.")],
                    state_dict_[name.replace(".proj_in_besides_attn.", f".a_to_v.")],
                    state_dict_[name],
                ], dim=0)
                state_dict_[name_] = param
                state_dict_.pop(name.replace(".proj_in_besides_attn.", f".a_to_q."))
                state_dict_.pop(name.replace(".proj_in_besides_attn.", f".a_to_k."))
                state_dict_.pop(name.replace(".proj_in_besides_attn.", f".a_to_v."))
                state_dict_.pop(name)
        for name in list(state_dict_.keys()):
            for component in ["a", "b"]:
                if f".{component}_to_q." in name:
                    name_ = name.replace(f".{component}_to_q.", f".{component}_to_qkv.")
                    param = torch.concat([
                        state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_q.")],
                        state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_k.")],
                        state_dict_[name.replace(f".{component}_to_q.", f".{component}_to_v.")],
                    ], dim=0)
                    state_dict_[name_] = param
                    state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_q."))
                    state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_k."))
                    state_dict_.pop(name.replace(f".{component}_to_q.", f".{component}_to_v."))
        if hash_value == "78d18b9101345ff695f312e7e62538c0":
            extra_kwargs = {"num_mode": 10, "mode_dict": {"canny": 0, "tile": 1, "depth": 2, "blur": 3, "pose": 4, "gray": 5, "lq": 6}}
        elif hash_value == "b001c89139b5f053c715fe772362dd2a":
            extra_kwargs = {"num_single_blocks": 0}
        elif hash_value == "52357cb26250681367488a8954c271e8":
            extra_kwargs = {"num_joint_blocks": 6, "num_single_blocks": 0, "additional_input_dim": 4}
        elif hash_value == "0cfd1740758423a2a854d67c136d1e8c":
            extra_kwargs = {"num_joint_blocks": 4, "num_single_blocks": 1}
        elif hash_value == "7f9583eb8ba86642abb9a21a4b2c9e16":
            extra_kwargs = {"num_joint_blocks": 4, "num_single_blocks": 10}
        elif hash_value == "43ad5aaa27dd4ee01b832ed16773fa52":
            extra_kwargs = {"num_joint_blocks": 6, "num_single_blocks": 0}
        else:
            extra_kwargs = {}
        return state_dict_, extra_kwargs
    

    def from_civitai(self, state_dict):
        return self.from_diffusers(state_dict)
