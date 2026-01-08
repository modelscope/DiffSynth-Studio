from PIL import Image
import torch
import numpy as np
from einops import repeat, reduce
from typing import Union, Optional, Dict, List
from ..core import AutoTorchModule, AutoWrappedLinear, load_state_dict, ModelConfig, parse_device_type
from ..utils.lora import GeneralLoRALoader
from ..models.model_loader import ModelPool
from ..utils.controlnet import ControlNetInput


def get_available_gpus() -> List[int]:
    """Get list of available GPU device IDs."""
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def get_gpu_memory_map() -> Dict[int, float]:
    """Get available memory (in GB) for each GPU."""
    memory_map = {}
    for gpu_id in get_available_gpus():
        free, total = torch.cuda.mem_get_info(gpu_id)
        memory_map[gpu_id] = free / (1024 ** 3)
    return memory_map


class PipelineUnit:
    def __init__(
        self,
        seperate_cfg: bool = False,
        take_over: bool = False,
        input_params: tuple[str] = None,
        output_params: tuple[str] = None,
        input_params_posi: dict[str, str] = None,
        input_params_nega: dict[str, str] = None,
        onload_model_names: tuple[str] = None
    ):
        self.seperate_cfg = seperate_cfg
        self.take_over = take_over
        self.input_params = input_params
        self.output_params = output_params
        self.input_params_posi = input_params_posi
        self.input_params_nega = input_params_nega
        self.onload_model_names = onload_model_names

    def fetch_input_params(self):
        params = []
        if self.input_params is not None:
            for param in self.input_params:
                params.append(param)
        if self.input_params_posi is not None:
            for _, param in self.input_params_posi.items():
                params.append(param)
        if self.input_params_nega is not None:
            for _, param in self.input_params_nega.items():
                params.append(param)
        params = sorted(list(set(params)))
        return params
    
    def fetch_output_params(self):
        params = []
        if self.output_params is not None:
            for param in self.output_params:
                params.append(param)
        return params

    def process(self, pipe, **kwargs) -> dict:
        return {}
    
    def post_process(self, pipe, **kwargs) -> dict:
        return {}


class BasePipeline(torch.nn.Module):

    def __init__(
        self,
        device="cuda", torch_dtype=torch.float16,
        height_division_factor=64, width_division_factor=64,
        time_division_factor=None, time_division_remainder=None,
    ):
        super().__init__()
        # The device and torch_dtype is used for the storage of intermediate variables, not models.
        self.device = device
        self.torch_dtype = torch_dtype
        self.device_type = parse_device_type(device)
        # The following parameters are used for shape check.
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # VRAM management
        self.vram_management_enabled = False
        # Pipeline Unit Runner
        self.unit_runner = PipelineUnitRunner()
        # LoRA Loader
        self.lora_loader = GeneralLoRALoader
        
        
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self


    def check_resize_height_width(self, height, width, num_frames=None):
        # Shape check
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        else:
            if num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
                print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
            return height, width, num_frames


    def preprocess_image(self, image, torch_dtype=None, device=None, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a PIL.Image to torch.Tensor
        image = torch.Tensor(np.array(image, dtype=np.float32))
        image = image.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        image = image * ((max_value - min_value) / 255) + min_value
        image = repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))
        return image


    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a list of PIL.Image to torch.Tensor
        video = [self.preprocess_image(image, torch_dtype=torch_dtype, device=device, min_value=min_value, max_value=max_value) for image in video]
        video = torch.stack(video, dim=pattern.index("T") // 2)
        return video


    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to PIL.Image
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        image = image.to(device="cpu", dtype=torch.uint8)
        image = Image.fromarray(image.numpy())
        return image


    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        # Transform a torch.Tensor to list of PIL.Image
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        video = [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]
        return video


    def load_models_to_device(self, model_names):
        if self.vram_management_enabled:
            # offload models
            for name, model in self.named_children():
                if name not in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        if hasattr(model, "offload"):
                            model.offload()
                        else:
                            for module in model.modules():
                                if hasattr(module, "offload"):
                                    module.offload()
            # Clear cache if available (only CUDA has empty_cache)
            device_module = getattr(torch, self.device_type, None)
            if device_module is not None and hasattr(device_module, "empty_cache"):
                device_module.empty_cache()
            # onload models
            for name, model in self.named_children():
                if name in model_names:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        if hasattr(model, "onload"):
                            model.onload()
                        else:
                            for module in model.modules():
                                if hasattr(module, "onload"):
                                    module.onload()


    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        # Initialize Gaussian noise
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        noise = noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)
        return noise

        
    def get_vram(self):
        device = self.device if self.device != "npu" else "npu:0"
        return getattr(torch, self.device_type).mem_get_info(device)[1] / (1024 ** 3)
    
    def get_module(self, model, name):
        if "." in name:
            name, suffix = name[:name.index(".")], name[name.index(".") + 1:]
            if name.isdigit():
                return self.get_module(model[int(name)], suffix)
            else:
                return self.get_module(getattr(model, name), suffix)
        else:
            return getattr(model, name)
    
    def freeze_except(self, model_names):
        self.eval()
        self.requires_grad_(False)
        for name in model_names:
            module = self.get_module(self, name)
            if module is None:
                print(f"No {name} models in the pipeline. We cannot enable training on the model. If this occurs during the data processing stage, it is normal.")
                continue
            module.train()
            module.requires_grad_(True)
                
    
    def blend_with_mask(self, base, addition, mask):
        return base * (1 - mask) + addition * mask
    
    
    def step(self, scheduler, latents, progress_id, noise_pred, input_latents=None, inpaint_mask=None, **kwargs):
        timestep = scheduler.timesteps[progress_id]
        if inpaint_mask is not None:
            noise_pred_expected = scheduler.return_to_timestep(scheduler.timesteps[progress_id], latents, input_latents)
            noise_pred = self.blend_with_mask(noise_pred_expected, noise_pred, inpaint_mask)
        latents_next = scheduler.step(noise_pred, timestep, latents)
        return latents_next
    
    
    def split_pipeline_units(self, model_names: list[str]):
        return PipelineUnitGraph().split_pipeline_units(self.units, model_names)
    
    
    def flush_vram_management_device(self, device):
        for module in self.modules():
            if isinstance(module, AutoTorchModule):
                module.offload_device = device
                module.onload_device = device
                module.preparing_device = device
                module.computation_device = device
                
    
    def load_lora(
        self,
        module: torch.nn.Module,
        lora_config: Union[ModelConfig, str] = None,
        alpha=1,
        hotload=None,
        state_dict=None,
        verbose=1,
    ):
        if state_dict is None:
            if isinstance(lora_config, str):
                lora = load_state_dict(lora_config, torch_dtype=self.torch_dtype, device=self.device)
            else:
                lora_config.download_if_necessary()
                lora = load_state_dict(lora_config.path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        lora_loader = self.lora_loader(torch_dtype=self.torch_dtype, device=self.device)
        lora = lora_loader.convert_state_dict(lora)
        if hotload is None:
            hotload = hasattr(module, "vram_management_enabled") and getattr(module, "vram_management_enabled")
        if hotload:
            if not (hasattr(module, "vram_management_enabled") and getattr(module, "vram_management_enabled")):
                raise ValueError("VRAM Management is not enabled. LoRA hotloading is not supported.")
            updated_num = 0
            for _, module in module.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    name = module.name
                    lora_a_name = f'{name}.lora_A.weight'
                    lora_b_name = f'{name}.lora_B.weight'
                    if lora_a_name in lora and lora_b_name in lora:
                        updated_num += 1
                        module.lora_A_weights.append(lora[lora_a_name] * alpha)
                        module.lora_B_weights.append(lora[lora_b_name])
            if verbose >= 1:
                print(f"{updated_num} tensors are patched by LoRA. You can use `pipe.clear_lora()` to clear all LoRA layers.")
        else:
            lora_loader.fuse_lora_to_base_model(module, lora, alpha=alpha)
            
            
    def clear_lora(self, verbose=1):
        cleared_num = 0
        for name, module in self.named_modules():
            if isinstance(module, AutoWrappedLinear):
                if hasattr(module, "lora_A_weights"):
                    if len(module.lora_A_weights) > 0:
                        cleared_num += 1
                    module.lora_A_weights.clear()
                if hasattr(module, "lora_B_weights"):
                    module.lora_B_weights.clear()
        if verbose >= 1:
            print(f"{cleared_num} LoRA layers are cleared.")
        
    
    def download_and_load_models(self, model_configs: list[ModelConfig] = [], vram_limit: float = None):
        model_pool = ModelPool()
        for model_config in model_configs:
            model_config.download_if_necessary()
            vram_config = model_config.vram_config()
            vram_config["computation_dtype"] = vram_config["computation_dtype"] or self.torch_dtype
            vram_config["computation_device"] = vram_config["computation_device"] or self.device
            model_pool.auto_load_model(
                model_config.path,
                vram_config=vram_config,
                vram_limit=vram_limit,
                clear_parameters=model_config.clear_parameters,
            )
        return model_pool
    
    
    def check_vram_management_state(self):
        vram_management_enabled = False
        for module in self.children():
            if hasattr(module, "vram_management_enabled") and getattr(module, "vram_management_enabled"):
                vram_management_enabled = True
        return vram_management_enabled


    def enable_multi_gpu(
        self,
        mode: str = "auto",
        device_map: Optional[Dict[str, str]] = None,
        tensor_parallel_layers: Optional[List[str]] = None,
    ):
        """
        Enable multi-GPU support for this pipeline.

        Args:
            mode: Parallelism mode:
                - "auto": Automatically select best strategy
                - "model": Distribute different models to different GPUs
                - "tensor": Split large layers across GPUs (requires torchrun)
                - "data": Same model on all GPUs, different data (for batch processing)
            device_map: Manual device mapping, e.g., {"dit": "cuda:0", "text_encoder": "cuda:1"}
            tensor_parallel_layers: Layer names to apply tensor parallelism to

        Returns:
            self for method chaining
        """
        try:
            from ..distributed import MultiGPUPipeline, enable_multi_gpu
        except ImportError:
            print("Warning: Distributed module not available. Multi-GPU support disabled.")
            return self

        num_gpus = len(get_available_gpus())
        if num_gpus <= 1:
            print("Only one GPU available. Multi-GPU support not needed.")
            return self

        print(f"Enabling multi-GPU support with {num_gpus} GPUs, mode={mode}")

        if mode == "model" and device_map is None:
            # Auto-create device map for model parallel
            device_map = self._auto_create_device_map()

        if device_map is not None:
            self._apply_device_map(device_map)
            print(f"Applied device map: {device_map}")

        return self


    def _auto_create_device_map(self) -> Dict[str, str]:
        """
        Automatically create a device map for model parallelism.

        Distributes model components across available GPUs based on their size.
        """
        num_gpus = len(get_available_gpus())
        if num_gpus <= 1:
            return {}

        # Get model components and their sizes
        components = {}
        for name, module in self.named_children():
            if module is not None and isinstance(module, torch.nn.Module):
                num_params = sum(p.numel() for p in module.parameters())
                if num_params > 0:
                    components[name] = num_params

        if not components:
            return {}

        # Sort by size (largest first) and assign to GPUs
        sorted_components = sorted(components.items(), key=lambda x: -x[1])
        device_map = {}
        gpu_loads = [0] * num_gpus

        for name, size in sorted_components:
            # Assign to GPU with least load
            min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
            device_map[name] = f"cuda:{min_gpu}"
            gpu_loads[min_gpu] += size

        return device_map


    def _apply_device_map(self, device_map: Dict[str, str]):
        """
        Apply a device map to distribute models across GPUs.

        Args:
            device_map: Mapping of model names to devices
        """
        for name, device in device_map.items():
            if hasattr(self, name):
                module = getattr(self, name)
                if module is not None:
                    module.to(device)
                    print(f"  Moved {name} to {device}")


    def get_model_distribution(self) -> Dict[str, str]:
        """
        Get the current distribution of models across devices.

        Returns:
            Dictionary mapping model names to their current devices
        """
        distribution = {}
        for name, module in self.named_children():
            if module is not None and isinstance(module, torch.nn.Module):
                try:
                    device = next(module.parameters()).device
                    distribution[name] = str(device)
                except StopIteration:
                    distribution[name] = "no parameters"
        return distribution


    def print_model_distribution(self):
        """Print the current distribution of models across devices."""
        distribution = self.get_model_distribution()
        print("Model distribution:")
        for name, device in distribution.items():
            print(f"  {name}: {device}")
    
    
    def cfg_guided_model_fn(self, model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega, **inputs_others):
        if inputs_shared.get("positive_only_lora", None) is not None:
            self.clear_lora(verbose=0)
            self.load_lora(self.dit, state_dict=inputs_shared["positive_only_lora"], verbose=0)
        noise_pred_posi = model_fn(**inputs_posi, **inputs_shared, **inputs_others)
        if cfg_scale != 1.0:
            if inputs_shared.get("positive_only_lora", None) is not None:
                self.clear_lora(verbose=0)
            noise_pred_nega = model_fn(**inputs_nega, **inputs_shared, **inputs_others)
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi
        return noise_pred


class PipelineUnitGraph:
    def __init__(self):
        pass
    
    def build_edges(self, units: list[PipelineUnit]):
        # Establish dependencies between units
        # to search for subsequent related computation units.
        last_compute_unit_id = {}
        edges = []
        for unit_id, unit in enumerate(units):
            for input_param in unit.fetch_input_params():
                if input_param in last_compute_unit_id:
                    edges.append((last_compute_unit_id[input_param], unit_id))
            for output_param in unit.fetch_output_params():
                last_compute_unit_id[output_param] = unit_id
        return edges
    
    def build_chains(self, units: list[PipelineUnit]):
        # Establish updating chains for each variable
        # to track their computation process.
        params = sum([unit.fetch_input_params() + unit.fetch_output_params() for unit in units], [])
        params = sorted(list(set(params)))
        chains = {param: [] for param in params}
        for unit_id, unit in enumerate(units):
            for param in unit.fetch_output_params():
                chains[param].append(unit_id)
        return chains
    
    def search_direct_unit_ids(self, units: list[PipelineUnit], model_names: list[str]):
        # Search for units that directly participate in the model's computation.
        related_unit_ids = []
        for unit_id, unit in enumerate(units):
            for model_name in model_names:
                if unit.onload_model_names is not None and model_name in unit.onload_model_names:
                    related_unit_ids.append(unit_id)
                    break
        return related_unit_ids
    
    def search_related_unit_ids(self, edges, start_unit_ids, direction="target"):
        # Search for subsequent related computation units.
        related_unit_ids = [unit_id for unit_id in start_unit_ids]
        while True:
            neighbors = []
            for source, target in edges:
                if direction == "target" and source in related_unit_ids and target not in related_unit_ids:
                    neighbors.append(target)
                elif direction == "source" and source not in related_unit_ids and target in related_unit_ids:
                    neighbors.append(source)
            neighbors = sorted(list(set(neighbors)))
            if len(neighbors) == 0:
                break
            else:
                related_unit_ids.extend(neighbors)
        related_unit_ids = sorted(list(set(related_unit_ids)))
        return related_unit_ids
    
    def search_updating_unit_ids(self, units: list[PipelineUnit], chains, related_unit_ids):
        # If the input parameters of this subgraph are updated outside the subgraph,
        # search for the units where these updates occur.
        first_compute_unit_id = {}
        for unit_id in related_unit_ids:
            for param in units[unit_id].fetch_input_params():
                if param not in first_compute_unit_id:
                    first_compute_unit_id[param] = unit_id
        updating_unit_ids = []
        for param in first_compute_unit_id:
            unit_id = first_compute_unit_id[param]
            chain = chains[param]
            if unit_id in chain and chain.index(unit_id) != len(chain) - 1:
                for unit_id_ in chain[chain.index(unit_id) + 1:]:
                    if unit_id_ not in related_unit_ids:
                        updating_unit_ids.append(unit_id_)
        related_unit_ids.extend(updating_unit_ids)
        related_unit_ids = sorted(list(set(related_unit_ids)))
        return related_unit_ids
    
    def split_pipeline_units(self, units: list[PipelineUnit], model_names: list[str]):
        # Split the computation graph,
        # separating all model-related computations.
        related_unit_ids = self.search_direct_unit_ids(units, model_names)
        edges = self.build_edges(units)
        chains = self.build_chains(units)
        while True:
            num_related_unit_ids = len(related_unit_ids)
            related_unit_ids = self.search_related_unit_ids(edges, related_unit_ids, "target")
            related_unit_ids = self.search_updating_unit_ids(units, chains, related_unit_ids)
            if len(related_unit_ids) == num_related_unit_ids:
                break
            else:
                num_related_unit_ids = len(related_unit_ids)
        related_units = [units[i] for i in related_unit_ids]
        unrelated_units = [units[i] for i in range(len(units)) if i not in related_unit_ids]
        return related_units, unrelated_units


class PipelineUnitRunner:
    def __init__(self):
        pass

    def __call__(self, unit: PipelineUnit, pipe: BasePipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict) -> tuple[dict, dict]:
        if unit.take_over:
            # Let the pipeline unit take over this function.
            inputs_shared, inputs_posi, inputs_nega = unit.process(pipe, inputs_shared=inputs_shared, inputs_posi=inputs_posi, inputs_nega=inputs_nega)
        elif unit.seperate_cfg:
            # Positive side
            processor_inputs = {name: inputs_posi.get(name_) for name, name_ in unit.input_params_posi.items()}
            if unit.input_params is not None:
                for name in unit.input_params:
                    processor_inputs[name] = inputs_shared.get(name)
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_posi.update(processor_outputs)
            # Negative side
            if inputs_shared["cfg_scale"] != 1:
                processor_inputs = {name: inputs_nega.get(name_) for name, name_ in unit.input_params_nega.items()}
                if unit.input_params is not None:
                    for name in unit.input_params:
                        processor_inputs[name] = inputs_shared.get(name)
                processor_outputs = unit.process(pipe, **processor_inputs)
                inputs_nega.update(processor_outputs)
            else:
                inputs_nega.update(processor_outputs)
        else:
            processor_inputs = {name: inputs_shared.get(name) for name in unit.input_params}
            processor_outputs = unit.process(pipe, **processor_inputs)
            inputs_shared.update(processor_outputs)
        return inputs_shared, inputs_posi, inputs_nega
