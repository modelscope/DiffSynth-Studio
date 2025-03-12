import torch
import time
from tqdm import tqdm
import psutil
import gc
import os
import platform
import multiprocessing
from .sd_unet import SDUNet
from .sdxl_unet import SDXLUNet
from .sd_text_encoder import SDTextEncoder
from .sdxl_text_encoder import SDXLTextEncoder, SDXLTextEncoder2
from .sd3_dit import SD3DiT
from .flux_dit import FluxDiT
from .hunyuan_dit import HunyuanDiT
from .cog_dit import CogDiT
from .hunyuan_video_dit import HunyuanVideoDiT
from .wan_video_dit import WanModel

# Global debug variable: when set to False, only minimal info is printed.
DEBUG = False

def debug_print(*args, **kwargs):
    """Print debug messages only if DEBUG is True."""
    if DEBUG:
        print(*args, **kwargs)

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if DEBUG:
            print(f"⏱️ {func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

def memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return f"{memory_info.rss / (1024 * 1024):.1f} MB"

def optimize_cpu_threading():
    """Set optimal thread configuration for the current CPU"""
    cpu_count = multiprocessing.cpu_count()
    
    # Get processor information
    processor = platform.processor().lower()
    
    if "amd" in processor:
        optimal_threads = max(1, cpu_count)
    else:  # Intel or other
        optimal_threads = max(1, cpu_count // 2)
    
    os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
    os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
    
    blas_info = "unknown"
    try:
        import torch.__config__
        config_info = torch.__config__.show()
        if "mkl" in config_info.lower():
            blas_info = "MKL"
        elif "openblas" in config_info.lower():
            blas_info = "OpenBLAS"
    except:
        pass
    
    if DEBUG:
        print(f"CPU Optimization: {platform.processor()}")
        print(f"Physical cores: {cpu_count // 2}, Total threads: {cpu_count}")
        print(f"Using {optimal_threads} threads for computation")
        print(f"BLAS backend: {blas_info}")
    else:
        print(f"CPU threading optimized: using {optimal_threads} threads")
    
    torch.set_num_threads(optimal_threads)
    return optimal_threads

class LoRAFromCivitai:
    def __init__(self):
        self.supported_model_classes = []
        self.lora_prefix = []
        self.renamed_lora_prefix = {}
        self.special_keys = {}
        self.stats = {
            "tensor_movements_to_gpu": 0,
            "tensor_movements_to_cpu": 0,
            "lora_weights_processed": 0,
            "format_conversions": 0,
        }
        # Set optimal thread count for CPU operations
        self.optimal_threads = optimize_cpu_threading()
        self.use_gpu = torch.cuda.is_available()
        
        # Enable tensor cores for matrix operations if available
        if self.use_gpu and hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True

    @timing_decorator
    def convert_state_dict(self, state_dict, lora_prefix="lora_unet_", alpha=1.0):
        if DEBUG:
            print(f"Converting state dict with prefix {lora_prefix}, memory usage: {memory_usage()}")
        # Detect format
        for key in state_dict:
            if ".lora_up" in key:
                if DEBUG:
                    print(f"Detected up/down format, keys: {len(state_dict)}")
                return self.convert_state_dict_up_down(state_dict, lora_prefix, alpha)
        if DEBUG:
            print(f"Detected A/B format, keys: {len(state_dict)}")
        return self.convert_state_dict_AB(state_dict, lora_prefix, alpha)

    @timing_decorator
    def convert_state_dict_up_down(self, state_dict, lora_prefix="lora_unet_", alpha=1.0):
        renamed_lora_prefix = self.renamed_lora_prefix.get(lora_prefix, "")
        state_dict_ = {}
        if DEBUG:
            print(f"Processing up/down conversion for {len(state_dict)} tensors...")
            
        # Determine optimal processing device
        device = "cuda" if self.use_gpu else "cpu"
        torch_dtype = torch.float16 if self.use_gpu else torch.float32
        
        # Count applicable keys first
        applicable_keys = []
        for key in state_dict:
            if ".lora_up" in key and key.startswith(lora_prefix):
                applicable_keys.append(key)
                
        # Prepare batches for processing
        BATCH_SIZE = 16  # Adjust based on memory constraints
        if DEBUG:
            print(f"Processing {len(applicable_keys)} tensors in batches of {BATCH_SIZE}...")
            
        with tqdm(total=len(applicable_keys), desc="Converting up/down weights") as pbar:
            for i in range(0, len(applicable_keys), BATCH_SIZE):
                batch_keys = applicable_keys[i:i+BATCH_SIZE]
                for key in batch_keys:
                    # Track GPU tensor movements
                    weight_up = state_dict[key].to(device=device, dtype=torch_dtype)
                    weight_down = state_dict[key.replace(".lora_up", ".lora_down")].to(device=device, dtype=torch_dtype)
                    self.stats["tensor_movements_to_gpu"] += 2
                    
                    # Matrix multiplication - faster on GPU, or optimized CPU
                    if len(weight_up.shape) == 4:
                        weight_up = weight_up.squeeze(3).squeeze(2).to(torch.float32)
                        weight_down = weight_down.squeeze(3).squeeze(2).to(torch.float32)
                        lora_weight = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                    else:
                        lora_weight = alpha * torch.mm(weight_up, weight_down)
                    
                    target_key = key.split(".")[0].replace(lora_prefix, renamed_lora_prefix).replace("_", ".") + ".weight"
                    state_dict_[target_key] = lora_weight.cpu()
                    self.stats["tensor_movements_to_cpu"] += 1
                    self.stats["lora_weights_processed"] += 1
                    
                    # Apply special key replacements
                    for special_key in self.special_keys:
                        if special_key in target_key:
                            state_dict_[target_key] = state_dict_[target_key].replace(special_key, self.special_keys[special_key])
                    
                    pbar.update(1)
                
                # Clear memory after each batch
                del weight_up, weight_down, lora_weight
                if self.use_gpu:
                    torch.cuda.empty_cache()
        
        if DEBUG:
            print(f"Up/down conversion complete, resulting in {len(state_dict_)} tensors, memory: {memory_usage()}")
        else:
            print(f"LoRA conversion complete: {len(state_dict_)} tensors processed")
        return state_dict_
    
    @timing_decorator
    def convert_state_dict_AB(self, state_dict, lora_prefix="", alpha=1.0):
        state_dict_ = {}
        # Determine optimal processing device
        device = "cuda" if self.use_gpu else "cpu"
        torch_dtype = torch.float16 if self.use_gpu else torch.float32
        
        # Collect applicable keys first
        applicable_keys = []
        for key in state_dict:
            if ".lora_B." in key and key.startswith(lora_prefix):
                applicable_keys.append(key)
        
        # Prepare batches for processing
        BATCH_SIZE = 16  # Adjust based on memory constraints
        if DEBUG:
            print(f"Processing {len(applicable_keys)} tensors in batches of {BATCH_SIZE}...")
            
        with tqdm(total=len(applicable_keys), desc="Converting A/B weights") as pbar:
            for i in range(0, len(applicable_keys), BATCH_SIZE):
                batch_keys = applicable_keys[i:i+BATCH_SIZE]
                for key in batch_keys:
                    # Load and process tensors
                    weight_up = state_dict[key].to(device=device, dtype=torch_dtype)
                    weight_down = state_dict[key.replace(".lora_B.", ".lora_A.")].to(device=device, dtype=torch_dtype)
                    self.stats["tensor_movements_to_gpu"] += 2
                    
                    if len(weight_up.shape) == 4:
                        weight_up = weight_up.squeeze(3).squeeze(2)
                        weight_down = weight_down.squeeze(3).squeeze(2)
                        lora_weight = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                    else:
                        lora_weight = alpha * torch.mm(weight_up, weight_down)
                    
                    # Extract target name
                    keys = key.split(".")
                    keys.pop(keys.index("lora_B"))
                    target_name = ".".join(keys)
                    target_name = target_name[len(lora_prefix):]
                    
                    # Store result
                    state_dict_[target_name] = lora_weight.cpu()
                    self.stats["tensor_movements_to_cpu"] += 1
                    self.stats["lora_weights_processed"] += 1
                    pbar.update(1)
                
                # Clear memory after each batch
                del weight_up, weight_down, lora_weight
                if self.use_gpu:
                    torch.cuda.empty_cache()
        
        if DEBUG:
            print(f"A/B conversion complete, resulting in {len(state_dict_)} tensors, memory: {memory_usage()}")
        else:
            print(f"LoRA conversion complete: {len(state_dict_)} tensors processed")
        return state_dict_
    
    @timing_decorator
    def load(self, model, state_dict_lora, lora_prefix, alpha=1.0, model_resource=None):
        print(f"Starting LoRA loading process for {model.__class__.__name__}...")
        
        # Measure state dict loading time - use direct parameter access
        start_state_dict = time.time()
        state_dict_model = {}
        for name, param in model.named_parameters():
            state_dict_model[name] = param
        end_state_dict = time.time()
        if DEBUG:
            print(f"⏱️ Loading model parameters took {end_state_dict - start_state_dict:.4f} seconds, size: {len(state_dict_model)} tensors")
        else:
            print(f"Model parameters mapped: {len(state_dict_model)} parameters")
        
        # Measure LoRA conversion time
        start_convert = time.time()
        state_dict_lora = self.convert_state_dict(state_dict_lora, lora_prefix=lora_prefix, alpha=alpha)
        self.stats["format_conversions"] += 1
        end_convert = time.time()
        if DEBUG:
            print(f"⏱️ LoRA conversion took {end_convert - start_convert:.4f} seconds")
        
        # Measure format conversion time if applicable
        if model_resource:
            if DEBUG:
                print(f"Converting format from {model_resource}...")
            start_format = time.time()
            if model_resource == "diffusers":
                state_dict_lora = model.__class__.state_dict_converter().from_diffusers(state_dict_lora)
            elif model_resource == "civitai":
                state_dict_lora = model.__class__.state_dict_converter().from_civitai(state_dict_lora)
            self.stats["format_conversions"] += 1
            end_format = time.time()
            if DEBUG:
                print(f"⏱️ Format conversion took {end_format - start_format:.4f} seconds")
        
        if isinstance(state_dict_lora, tuple):
            state_dict_lora = state_dict_lora[0]
        
        if len(state_dict_lora) > 0:
            if DEBUG:
                print(f"Applying {len(state_dict_lora)} LoRA tensors to model weights...")
            else:
                print("Applying LoRA weights...")
            
            # Process in batches
            BATCH_SIZE = 32
            lora_keys = list(state_dict_lora.keys())
            
            start_update = time.time()
            with tqdm(total=len(lora_keys), desc="Applying LoRA weights") as pbar:
                for i in range(0, len(lora_keys), BATCH_SIZE):
                    batch_keys = lora_keys[i:i+BATCH_SIZE]
                    for name in batch_keys:
                        if name not in state_dict_model:
                            pbar.update(1)
                            continue
                            
                        param = state_dict_model[name]
                        
                        # Handle FP8 tensors
                        fp8 = False
                        if param.dtype == torch.float8_e4m3fn:
                            param_data = param.to(state_dict_lora[name].dtype)
                            fp8 = True
                        else:
                            param_data = param.data
                        
                        # Apply direct update (avoids load_state_dict overhead)
                        param.data = param_data + state_dict_lora[name].to(
                            dtype=param_data.dtype, device=param_data.device)
                        
                        if fp8:
                            param.data = param.data.to(torch.float8_e4m3fn)
                        
                        pbar.update(1)
                    
                    # Clear memory after each batch
                    if self.use_gpu:
                        torch.cuda.empty_cache()
                        
            end_update = time.time()
            if DEBUG:
                print(f"⏱️ Weight update took {end_update - start_update:.4f} seconds")
            else:
                print("Weight update complete.")
        else:
            print("No LoRA tensors to apply!")
            
        if DEBUG:
            print("\n==== LoRA LOADING STATISTICS ====")
            print(f"Total tensor movements to GPU: {self.stats['tensor_movements_to_gpu']}")
            print(f"Total tensor movements to CPU: {self.stats['tensor_movements_to_cpu']}")
            print(f"Total LoRA weights processed: {self.stats['lora_weights_processed']}")
            print(f"Total format conversions: {self.stats['format_conversions']}")
            print(f"Final memory usage: {memory_usage()}")
            print("================================")
        else:
            print(f"LoRA load complete: {self.stats['lora_weights_processed']} weights processed, GPU moves: {self.stats['tensor_movements_to_gpu']}, CPU moves: {self.stats['tensor_movements_to_cpu']}.")
        
        # Clear temporary data and run garbage collection
        del state_dict_lora
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @timing_decorator
    def match(self, model, state_dict_lora):
        if DEBUG:
            print(f"Trying to match LoRA format for {model.__class__.__name__}, memory usage: {memory_usage()}")
        match_results = []
        
        for i, (lora_prefix, model_class) in enumerate(zip(self.lora_prefix, self.supported_model_classes)):
            if not isinstance(model, model_class):
                continue
                
            if DEBUG:
                print(f"Checking prefix '{lora_prefix}' for model class {model_class.__name__}")
            
            # Get parameter names
            param_names = set(name for name, _ in model.named_parameters())
            
            for model_resource in ["diffusers", "civitai"]:
                try:
                    if DEBUG:
                        print(f"  Attempting {model_resource} format...")
                    start_time = time.time()
                    
                    # Try conversion
                    state_dict_lora_ = self.convert_state_dict(state_dict_lora, lora_prefix=lora_prefix, alpha=1.0)
                    converter_fn = model.__class__.state_dict_converter().from_diffusers if model_resource == "diffusers" \
                        else model.__class__.state_dict_converter().from_civitai
                    state_dict_lora_ = converter_fn(state_dict_lora_)
                    
                    if isinstance(state_dict_lora_, tuple):
                        state_dict_lora_ = state_dict_lora_[0]
                        
                    if len(state_dict_lora_) == 0:
                        if DEBUG:
                            print(f"  ❌ No matching tensors found for {model_resource} format")
                        continue
                        
                    # Verify the keys actually match the model (sample check)
                    valid_keys = 0
                    for name in list(state_dict_lora_.keys())[:10]:
                        if name in param_names:
                            valid_keys += 1
                        else:
                            if DEBUG:
                                print(f"  ⚠️ Key not found in model: {name}")
                            break
                    
                    end_time = time.time()
                    
                    if valid_keys > 0:
                        if DEBUG:
                            print(f"  ✅ Match found! Prefix: {lora_prefix}, Format: {model_resource}, Valid keys: {valid_keys}")
                            print(f"  ⏱️ Match verification took {end_time - start_time:.4f} seconds")
                        else:
                            print("Matching format found.")
                        return lora_prefix, model_resource
                    else:
                        if DEBUG:
                            print(f"  ❌ No valid keys found for this format")
                        
                except Exception as e:
                    if DEBUG:
                        print(f"  ❌ Error during matching: {str(e)}")
        if DEBUG:
            print("❌ No match found for any format or prefix")
        return None

# Specialized classes derived from LoRAFromCivitai
class SDLoRAFromCivitai(LoRAFromCivitai):
    def __init__(self):
        super().__init__()
        self.supported_model_classes = [SDUNet, SDTextEncoder]
        self.lora_prefix = ["lora_unet_", "lora_te_"]
        self.special_keys = {
            "down.blocks": "down_blocks",
            "up.blocks": "up_blocks",
            "mid.block": "mid_block",
            "proj.in": "proj_in",
            "proj.out": "proj_out",
            "transformer.blocks": "transformer_blocks",
            "to.q": "to_q",
            "to.k": "to_k",
            "to.v": "to_v",
            "to.out": "to_out",
            "text.model": "text_model",
            "self.attn.q.proj": "self_attn.q_proj",
            "self.attn.k.proj": "self_attn.k_proj",
            "self.attn.v.proj": "self_attn.v_proj",
            "self.attn.out.proj": "self_attn.out_proj",
            "input.blocks": "model.diffusion_model.input_blocks",
            "middle.block": "model.diffusion_model.middle_block",
            "output.blocks": "model.diffusion_model.output_blocks",
        }

class SDXLLoRAFromCivitai(LoRAFromCivitai):
    def __init__(self):
        super().__init__()
        self.supported_model_classes = [SDXLUNet, SDXLTextEncoder, SDXLTextEncoder2]
        self.lora_prefix = ["lora_unet_", "lora_te1_", "lora_te2_"]
        self.renamed_lora_prefix = {"lora_te2_": "2"}
        self.special_keys = {
            "down.blocks": "down_blocks",
            "up.blocks": "up_blocks",
            "mid.block": "mid_block",
            "proj.in": "proj_in",
            "proj.out": "proj_out",
            "transformer.blocks": "transformer_blocks",
            "to.q": "to_q",
            "to.k": "to_k",
            "to.v": "to_v",
            "to.out": "to_out",
            "text.model": "conditioner.embedders.0.transformer.text_model",
            "self.attn.q.proj": "self_attn.q_proj",
            "self.attn.k.proj": "self_attn.k_proj",
            "self.attn.v.proj": "self_attn.v_proj",
            "self.attn.out.proj": "self_attn.out_proj",
            "input.blocks": "model.diffusion_model.input_blocks",
            "middle.block": "model.diffusion_model.middle_block",
            "output.blocks": "model.diffusion_model.output_blocks",
            "2conditioner.embedders.0.transformer.text_model.encoder.layers": "text_model.encoder.layers"
        }

class FluxLoRAFromCivitai(LoRAFromCivitai):
    def __init__(self):
        super().__init__()
        self.supported_model_classes = [FluxDiT, FluxDiT]
        self.lora_prefix = ["lora_unet_", "transformer."]
        self.renamed_lora_prefix = {}
        self.special_keys = {
            "single.blocks": "single_blocks",
            "double.blocks": "double_blocks",
            "img.attn": "img_attn",
            "img.mlp": "img_mlp",
            "img.mod": "img_mod",
            "txt.attn": "txt_attn",
            "txt.mlp": "txt_mlp",
            "txt.mod": "txt_mod",
        }

class HunyuanVideoLoRAFromCivitai(LoRAFromCivitai):
    def __init__(self):
        super().__init__()
        self.supported_model_classes = [HunyuanVideoDiT, HunyuanVideoDiT]
        self.lora_prefix = ["diffusion_model.", "transformer."]
        self.special_keys = {}

class GeneralLoRAFromPeft:
    def __init__(self):
        self.supported_model_classes = [SDUNet, SDXLUNet, SD3DiT, HunyuanDiT, FluxDiT, CogDiT, WanModel]
        self.stats = {
            "tensor_movements_to_gpu": 0,
            "tensor_movements_to_cpu": 0,
            "lora_weights_processed": 0,
            "parameter_updates": 0,
        }
        # Set optimal thread count for CPU operations
        self.optimal_threads = optimize_cpu_threading()
        self.use_gpu = torch.cuda.is_available()
        
        # Enable tensor cores for matrix operations if available
        if self.use_gpu and hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True

    def _get_target_name(self, key):
        """Extract target parameter name from LoRA key"""
        keys = key.split(".")
        if len(keys) > keys.index("lora_B") + 2:
            keys.pop(keys.index("lora_B") + 1)
        keys.pop(keys.index("lora_B"))
        target_name = ".".join(keys)
        if target_name.startswith("diffusion_model."):
            target_name = target_name[len("diffusion_model."):]
        return target_name

    @timing_decorator
    def convert_state_dict(self, state_dict, alpha=1.0, target_state_dict={}):
        if DEBUG:
            print(f"Converting state dict with GeneralLoRAFromPeft, memory: {memory_usage()}")
        device = "cuda" if self.use_gpu else "cpu"
        torch_dtype = torch.float16 if self.use_gpu else torch.float32
            
        state_dict_ = {}
        
        # Count applicable keys
        applicable_keys = [key for key in state_dict if ".lora_B." in key]
        
        # Process in batches
        BATCH_SIZE = 16
        if DEBUG:
            print(f"Processing {len(applicable_keys)} tensors in batches of {BATCH_SIZE}...")
        
        with tqdm(total=len(applicable_keys), desc="Converting LoRA weights") as pbar:
            for i in range(0, len(applicable_keys), BATCH_SIZE):
                batch_keys = applicable_keys[i:i+BATCH_SIZE]
                for key in batch_keys:
                    weight_up = state_dict[key].to(device=device, dtype=torch_dtype)
                    weight_down = state_dict[key.replace(".lora_B.", ".lora_A.")].to(device=device, dtype=torch_dtype)
                    self.stats["tensor_movements_to_gpu"] += 2
                    
                    if len(weight_up.shape) == 4:
                        weight_up = weight_up.squeeze(3).squeeze(2)
                        weight_down = weight_down.squeeze(3).squeeze(2)
                        lora_weight = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                    else:
                        lora_weight = alpha * torch.mm(weight_up, weight_down)
                    
                    target_name = self._get_target_name(key)
                    
                    if target_state_dict and target_name not in target_state_dict:
                        pbar.update(1)
                        continue
                        
                    state_dict_[target_name] = lora_weight.cpu()
                    self.stats["tensor_movements_to_cpu"] += 1
                    self.stats["lora_weights_processed"] += 1
                    pbar.update(1)
                
                # Clear memory after each batch
                del weight_up, weight_down, lora_weight
                if self.use_gpu:
                    torch.cuda.empty_cache()
                
        if DEBUG:
            print(f"Conversion complete, resulting in {len(state_dict_)} tensors, memory: {memory_usage()}")
        else:
            print(f"General LoRA conversion complete: {len(state_dict_)} weights processed")
        return state_dict_

    @timing_decorator
    def load(self, model, state_dict_lora, lora_prefix="", alpha=1.0, model_resource=""):
        """Apply LoRA weights directly to model parameters with batched processing"""
        print(f"Starting optimized LoRA loading for {model.__class__.__name__}...")
        
        # Create parameter lookup dict 
        start_map = time.time()
        param_dict = {name: param for name, param in model.named_parameters()}
        end_map = time.time()
        if DEBUG:
            print(f"⏱️ Parameter mapping took {end_map - start_map:.4f} seconds, found {len(param_dict)} parameters")
        else:
            print(f"Mapped {len(param_dict)} model parameters")
        
        # Count applicable LoRA parameters
        lora_b_keys = [key for key in state_dict_lora if ".lora_B." in key]
        print(f"Found {len(lora_b_keys)} LoRA parameter pairs to process")
        
        # Group parameters by shape for better memory access patterns
        shape_groups = {}
        for key in lora_b_keys:
            target_name = self._get_target_name(key)
            if target_name not in param_dict:
                continue
            shape = state_dict_lora[key].shape
            if shape not in shape_groups:
                shape_groups[shape] = []
            shape_groups[shape].append((key, target_name))
        
        if DEBUG:
            print(f"Organized into {len(shape_groups)} shape groups for efficient processing")
        
        # Process each shape group in batches
        BATCH_SIZE = 32
        modified_count = 0
        
        for shape, key_pairs in shape_groups.items():
            if DEBUG:
                print(f"Processing {len(key_pairs)} parameters with shape {shape}")
            for i in range(0, len(key_pairs), BATCH_SIZE):
                batch = key_pairs[i:i+BATCH_SIZE]
                for lora_key, target_name in batch:
                    param = param_dict[target_name]
                    dtype_for_calc = torch.float32 if param.dtype == torch.float8_e4m3fn else param.dtype
                    
                    # Load weights and compute LoRA update
                    weight_b = state_dict_lora[lora_key].to(device="cuda" if self.use_gpu else "cpu", dtype=dtype_for_calc)
                    weight_a = state_dict_lora[lora_key.replace(".lora_B.", ".lora_A.")].to(device="cuda" if self.use_gpu else "cpu", dtype=dtype_for_calc)
                    self.stats["tensor_movements_to_gpu"] += 2
                    
                    if len(weight_b.shape) == 4:
                        weight_b = weight_b.squeeze(3).squeeze(2)
                        weight_a = weight_a.squeeze(3).squeeze(2)
                        lora_weight = alpha * torch.mm(weight_b, weight_a).unsqueeze(2).unsqueeze(3)
                    else:
                        lora_weight = alpha * torch.mm(weight_b, weight_a)
                    
                    # Apply update directly to parameter
                    if param.dtype == torch.float8_e4m3fn:
                        param_float = param.to(torch.float32)
                        param.data = (param_float + lora_weight).to(param.dtype)
                        del param_float
                    else:
                        param.data += lora_weight.to(dtype=param.dtype, device=param.device)
                    
                    del weight_a, weight_b, lora_weight
                    self.stats["parameter_updates"] += 1
                    modified_count += 1
                
                if self.use_gpu:
                    torch.cuda.empty_cache()
            
        if DEBUG:
            print("\n==== OPTIMIZED LORA LOADING STATISTICS ====")
            print(f"Total tensor movements to GPU: {self.stats['tensor_movements_to_gpu']}")
            print(f"Total LoRA weights processed: {self.stats['lora_weights_processed']}")
            print(f"Total parameters updated: {self.stats['parameter_updates']}")
            print(f"Final memory usage: {memory_usage()}")
            print("==========================================")
        else:
            print(f"Optimized LoRA load complete: updated {modified_count} tensors")
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"⏱️ {modified_count} tensors were updated successfully")

    @timing_decorator
    def match(self, model, state_dict_lora):
        """Check if LoRA parameters match model parameters"""
        if DEBUG:
            print(f"Checking General LoRA compatibility for {model.__class__.__name__}...")
        
        for model_class in self.supported_model_classes:
            if not isinstance(model, model_class):
                continue
            
            # Create set of parameter names
            start_param = time.time()
            param_names = set(name for name, _ in model.named_parameters())
            end_param = time.time()
            if DEBUG:
                print(f"⏱️ Parameter name collection took {end_param - start_param:.4f} seconds, found {len(param_names)} names")
            
            # Check if a sample of LoRA keys map to model parameters
            matched_count = 0
            checked_count = 0
            
            start_check = time.time()
            for key in state_dict_lora:
                if ".lora_B." not in key:
                    continue
                
                target_name = self._get_target_name(key)
                if target_name in param_names:
                    matched_count += 1
                
                checked_count += 1
                if matched_count >= 5:  # Found enough matches
                    break
                if checked_count >= 50 and matched_count == 0:  # Checked enough without matches
                    break
            end_check = time.time()
            
            if DEBUG:
                print(f"⏱️ Match check took {end_check - start_check:.4f} seconds")
                print(f"Matched {matched_count}/{checked_count} checked parameters")
            
            if matched_count > 0:
                if DEBUG:
                    print(f"✅ Compatible with GeneralLoRAFromPeft")
                else:
                    print("LoRA compatibility check: PASS")
                return "", ""
        if DEBUG:        
            print("❌ Not compatible with GeneralLoRAFromPeft")
        return None

class FluxLoRAConverter:
    def __init__(self):
        pass

    @staticmethod
    @timing_decorator
    def align_to_opensource_format(state_dict, alpha=1.0):
        if DEBUG:
            print(f"Converting Flux LoRA to opensource format, input keys: {len(state_dict)}")
        prefix_rename_dict = {
            "single_blocks": "lora_unet_single_blocks",
            "blocks": "lora_unet_double_blocks",
        }
        middle_rename_dict = {
            "norm.linear": "modulation_lin",
            "to_qkv_mlp": "linear1",
            "proj_out": "linear2",
            "norm1_a.linear": "img_mod_lin",
            "norm1_b.linear": "txt_mod_lin",
            "attn.a_to_qkv": "img_attn_qkv",
            "attn.b_to_qkv": "txt_attn_qkv",
            "attn.a_to_out": "img_attn_proj",
            "attn.b_to_out": "txt_attn_proj",
            "ff_a.0": "img_mlp_0",
            "ff_a.2": "img_mlp_2",
            "ff_b.0": "txt_mlp_0",
            "ff_b.2": "txt_mlp_2",
        }
        suffix_rename_dict = {
            "lora_B.weight": "lora_up.weight",
            "lora_A.weight": "lora_down.weight",
        }
        state_dict_ = {}
        for name, param in tqdm(state_dict.items(), desc="Aligning to opensource format"):
            names = name.split(".")
            if names[-2] != "lora_A" and names[-2] != "lora_B":
                names.pop(-2)
            prefix = names[0]
            middle = ".".join(names[2:-2])
            suffix = ".".join(names[-2:])
            block_id = names[1]
            if middle not in middle_rename_dict:
                continue
            rename = prefix_rename_dict[prefix] + "_" + block_id + "_" + middle_rename_dict[middle] + "." + suffix_rename_dict[suffix]
            state_dict_[rename] = param
            if rename.endswith("lora_up.weight"):
                state_dict_[rename.replace("lora_up.weight", "alpha")] = torch.tensor((alpha,))[0]
        if DEBUG:
            print(f"Conversion complete, output keys: {len(state_dict_)}")
        else:
            print(f"Flux LoRA conversion complete: {len(state_dict_)} keys")
        return state_dict_
    
    @staticmethod
    @timing_decorator
    def align_to_diffsynth_format(state_dict):
        if DEBUG:
            print(f"Converting to diffsynth format, input keys: {len(state_dict)}")
        rename_dict = {
            "lora_unet_double_blocks_blockid_img_mod_lin.lora_down.weight": "blocks.blockid.norm1_a.linear.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_img_mod_lin.lora_up.weight": "blocks.blockid.norm1_a.linear.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_txt_mod_lin.lora_down.weight": "blocks.blockid.norm1_b.linear.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_txt_mod_lin.lora_up.weight": "blocks.blockid.norm1_b.linear.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_img_attn_qkv.lora_down.weight": "blocks.blockid.attn.a_to_qkv.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_img_attn_qkv.lora_up.weight": "blocks.blockid.attn.a_to_qkv.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_txt_attn_qkv.lora_down.weight": "blocks.blockid.attn.b_to_qkv.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_txt_attn_qkv.lora_up.weight": "blocks.blockid.attn.b_to_qkv.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_img_attn_proj.lora_down.weight": "blocks.blockid.attn.a_to_out.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_img_attn_proj.lora_up.weight": "blocks.blockid.attn.a_to_out.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_txt_attn_proj.lora_down.weight": "blocks.blockid.attn.b_to_out.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_txt_attn_proj.lora_up.weight": "blocks.blockid.attn.b_to_out.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_img_mlp_0.lora_down.weight": "blocks.blockid.ff_a.0.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_img_mlp_0.lora_up.weight": "blocks.blockid.ff_a.0.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_img_mlp_2.lora_down.weight": "blocks.blockid.ff_a.2.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_img_mlp_2.lora_up.weight": "blocks.blockid.ff_a.2.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_txt_mlp_0.lora_down.weight": "blocks.blockid.ff_b.0.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_txt_mlp_0.lora_up.weight": "blocks.blockid.ff_b.0.lora_B.default.weight",
            "lora_unet_double_blocks_blockid_txt_mlp_2.lora_down.weight": "blocks.blockid.ff_b.2.lora_A.default.weight",
            "lora_unet_double_blocks_blockid_txt_mlp_2.lora_up.weight": "blocks.blockid.ff_b.2.lora_B.default.weight",
            "lora_unet_single_blocks_blockid_modulation_lin.lora_down.weight": "single_blocks.blockid.norm.linear.lora_A.default.weight",
            "lora_unet_single_blocks_blockid_modulation_lin.lora_up.weight": "single_blocks.blockid.norm.linear.lora_B.default.weight",
            "lora_unet_single_blocks_blockid_linear1.lora_down.weight": "single_blocks.blockid.to_qkv_mlp.lora_A.default.weight",
            "lora_unet_single_blocks_blockid_linear1.lora_up.weight": "single_blocks.blockid.to_qkv_mlp.lora_B.default.weight",
            "lora_unet_single_blocks_blockid_linear2.lora_down.weight": "single_blocks.blockid.proj_out.lora_A.default.weight",
            "lora_unet_single_blocks_blockid_linear2.lora_up.weight": "single_blocks.blockid.proj_out.lora_B.default.weight",
        }
        def guess_block_id(name):
            names = name.split("_")
            for i in names:
                if i.isdigit():
                    return i, name.replace(f"_{i}_", "_blockid_")
            return None, None
        state_dict_ = {}
        for name, param in tqdm(state_dict.items(), desc="Aligning to diffsynth format"):
            block_id, source_name = guess_block_id(name)
            if source_name in rename_dict:
                target_name = rename_dict[source_name]
                target_name = target_name.replace(".blockid.", f".{block_id}.")
                state_dict_[target_name] = param
            else:
                state_dict_[name] = param
        if DEBUG:
            print(f"Conversion complete, output keys: {len(state_dict_)}")
        else:
            print(f"Diffsynth conversion complete: {len(state_dict_)} keys")
        return state_dict_

def get_lora_loaders():
    return [SDLoRAFromCivitai(), SDXLLoRAFromCivitai(), FluxLoRAFromCivitai(), HunyuanVideoLoRAFromCivitai(), GeneralLoRAFromPeft()]
