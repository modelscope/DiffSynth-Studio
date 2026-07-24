import json, math
from functools import wraps
from pathlib import Path
import torch
import torch.nn.functional as F
from ...core import load_state_dict


class FluxTimestepLoRAComponent(torch.nn.Module):

    def __init__(
        self,
        lora_a,
        lora_b,
        gamma_weight,
        gamma_bias,
        beta_weight,
        beta_bias,
        scaling,
        output_slice=None,
    ):
        super().__init__()
        self.register_buffer("lora_a", lora_a)
        self.register_buffer("lora_b", lora_b)
        self.register_buffer("gamma_weight", gamma_weight)
        self.register_buffer("gamma_bias", gamma_bias)
        self.register_buffer("beta_weight", beta_weight)
        self.register_buffer("beta_bias", beta_bias)
        self.scaling = scaling
        self.output_slice = output_slice

    def forward(self, hidden_states, timestep_embedding):
        hidden_states_lora = F.linear(hidden_states.to(self.lora_a.dtype), self.lora_a)
        timestep_embedding = timestep_embedding.to(device=hidden_states_lora.device, dtype=hidden_states_lora.dtype)
        gamma = F.linear(timestep_embedding, self.gamma_weight, self.gamma_bias)
        beta = F.linear(timestep_embedding, self.beta_weight, self.beta_bias)
        hidden_states_lora = hidden_states_lora * gamma
        hidden_states_lora = hidden_states_lora + beta
        return F.linear(hidden_states_lora, self.lora_b) * self.scaling


class FluxTimestepLoRALinear(torch.nn.Module):

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.components = torch.nn.ModuleList()
        self.timestep_embedding = None

    def forward(self, hidden_states):
        output = self.base_layer(hidden_states)
        if self.timestep_embedding is None:
            return output
        for component in self.components:
            delta = component(hidden_states, self.timestep_embedding).to(device=output.device, dtype=output.dtype)
            if component.output_slice is None:
                output = output + delta
            else:
                output[..., component.output_slice] = output[..., component.output_slice] + delta
        return output


class FluxTimestepLoRALoader:

    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype

    def load_adapter_scaling(self, lora_file, state_dict):
        config_path = lora_file.parent / "adapter_config.json"
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        rank = int(config["r"])
        denominator = math.sqrt(rank) if config.get("use_rslora", False) else rank
        return float(config["lora_alpha"]) / denominator

    @staticmethod
    def fetch_tensor(state_dict, layer_name, suffix):
        prefix = f"base_model.model.{layer_name}"
        key = f"{prefix}.{suffix}"
        return state_dict[key]

    @staticmethod
    def get_or_wrap_linear(dit, module_path):
        module = dit.get_submodule(module_path)
        if isinstance(module, FluxTimestepLoRALinear):
            return module
        wrapper = FluxTimestepLoRALinear(module)
        parent_path, child_name = module_path.rsplit(".", 1)
        parent = dit.get_submodule(parent_path)
        if child_name.isdigit():
            parent[int(child_name)] = wrapper
        else:
            setattr(parent, child_name, wrapper)
        return wrapper

    def add_component(self, dit, state_dict, layer_name, module_path, scaling, output_slice=None):
        wrapper = self.get_or_wrap_linear(dit, module_path)
        wrapper.components.append(
            FluxTimestepLoRAComponent(
                lora_a=self.fetch_tensor(state_dict, layer_name, "lora_A.weight"),
                lora_b=self.fetch_tensor(state_dict, layer_name, "lora_B.weight"),
                gamma_weight=self.fetch_tensor(state_dict, layer_name, "lora_time_gamma.weight"),
                gamma_bias=self.fetch_tensor(state_dict, layer_name, "lora_time_gamma.bias"),
                beta_weight=self.fetch_tensor(state_dict, layer_name, "lora_time_beta.weight"),
                beta_bias=self.fetch_tensor(state_dict, layer_name, "lora_time_beta.bias"),
                scaling=scaling,
                output_slice=output_slice,
            )
        )

    def add_qkv_components(self, dit, state_dict, layer_prefix, module_path, scaling, source_names):
        output_dim = self.fetch_tensor(state_dict, f"{layer_prefix}.{source_names[0]}", "lora_B.weight").shape[0]
        for component_id, source_name in enumerate(source_names):
            output_slice = slice(component_id * output_dim, (component_id + 1) * output_dim)
            self.add_component(dit, state_dict, f"{layer_prefix}.{source_name}", module_path, scaling, output_slice)

    @staticmethod
    def set_timestep_embedding(dit, timestep_embedding):
        for module in dit.modules():
            if isinstance(module, FluxTimestepLoRALinear):
                module.timestep_embedding = timestep_embedding

    def wrap_model_fn(self, model_fn):
        @wraps(model_fn)
        def model_fn_with_timestep_lora(*args, **kwargs):
            dit = kwargs.get("dit")
            timestep = kwargs.get("timestep")
            latents = kwargs.get("latents")
            if dit is None or timestep is None or latents is None:
                return model_fn(*args, **kwargs)

            timestep_embedding = dit.time_embedder.time_proj(timestep).to(device=latents.device, dtype=latents.dtype)
            self.set_timestep_embedding(dit, timestep_embedding)
            try:
                return model_fn(*args, **kwargs)
            finally:
                self.set_timestep_embedding(dit, None)

        return model_fn_with_timestep_lora

    def load(self, dit, lora_config, alpha=1.0):
        lora_config.download_if_necessary()
        lora_file = lora_config.path[0] if isinstance(lora_config.path, list) else lora_config.path
        state_dict = load_state_dict(lora_file, torch_dtype=self.torch_dtype, device=self.device)
        scaling = self.load_adapter_scaling(Path(lora_file), state_dict) * alpha

        for block_id in range(len(dit.blocks)):
            source_prefix = f"transformer_blocks.{block_id}"
            target_prefix = f"blocks.{block_id}"
            self.add_qkv_components(dit, state_dict, f"{source_prefix}.attn", f"{target_prefix}.attn.a_to_qkv", scaling, ("to_q", "to_k", "to_v"))
            self.add_qkv_components(dit, state_dict, f"{source_prefix}.attn", f"{target_prefix}.attn.b_to_qkv", scaling, ("add_q_proj", "add_k_proj", "add_v_proj"))
            self.add_component(dit, state_dict, f"{source_prefix}.attn.to_out.0", f"{target_prefix}.attn.a_to_out", scaling)
            self.add_component(dit, state_dict, f"{source_prefix}.attn.to_add_out", f"{target_prefix}.attn.b_to_out", scaling)
            self.add_component(dit, state_dict, f"{source_prefix}.ff.net.0.proj", f"{target_prefix}.ff_a.0", scaling)
            self.add_component(dit, state_dict, f"{source_prefix}.ff.net.2", f"{target_prefix}.ff_a.2", scaling)
            self.add_component(dit, state_dict, f"{source_prefix}.ff_context.net.0.proj", f"{target_prefix}.ff_b.0", scaling)
            self.add_component(dit, state_dict, f"{source_prefix}.ff_context.net.2", f"{target_prefix}.ff_b.2", scaling)

        for block_id in range(len(dit.single_blocks)):
            self.add_qkv_components(dit, state_dict, f"single_transformer_blocks.{block_id}.attn", f"single_blocks.{block_id}.to_qkv_mlp", scaling, ("to_q", "to_k", "to_v"))
