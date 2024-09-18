import torch
from torch import nn
from diffsynth import download_models, ModelManager, OmostPromter, FluxImagePipeline

from diffsynth.models.flux_dit import RMSNorm


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
    class Linear(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self,input,**kwargs):
            weight,bias= cast_bias_weight(self.module,input)
            return torch.nn.functional.linear(input,weight,bias)
    
    class RMSNorm(torch.nn.Module):
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
    
def replace_layer(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            new_layer = quantized_layer.Linear(module)
            setattr(model, name, new_layer)
        elif isinstance(module, RMSNorm):
            new_layer = quantized_layer.RMSNorm(module)
            setattr(model, name, new_layer)
        else:
            replace_layer(module)
            
def print_layers(model):
    for name, module in model.named_modules():
        print(type(module))
    

def fetch_models(self, model_manager: ModelManager, model_manager2: ModelManager, prompt_refiner_classes=[], prompt_extender_classes=[]):
    self.text_encoder_1 = model_manager.fetch_model("flux_text_encoder_1")
    self.text_encoder_2 = model_manager.fetch_model("flux_text_encoder_2")
    self.dit = model_manager2.fetch_model("flux_dit")
    self.vae_decoder = model_manager.fetch_model("flux_vae_decoder")
    self.vae_encoder = model_manager.fetch_model("flux_vae_encoder")
    self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2)
    self.prompter.load_prompt_refiners(model_manager, prompt_refiner_classes)
    self.prompter.load_prompt_extenders(model_manager, prompt_extender_classes)
    
download_models(["FLUX.1-dev"])

model_manager = ModelManager(torch_dtype=torch.bfloat16, device='cuda')
model_manager.load_models([
    "models/FLUX/FLUX.1-dev/text_encoder/model.safetensors",
    "models/FLUX/FLUX.1-dev/text_encoder_2",
    "models/FLUX/FLUX.1-dev/ae.safetensors",
])

model_manager2 = ModelManager(torch_dtype=torch.float8_e4m3fn, device="cuda")
model_manager2.load_models(["models/FLUX/FLUX.1-dev/flux1-dev.safetensors"])

pipe =FluxImagePipeline(device="cuda",torch_dtype=torch.bfloat16)
fetch_models(pipe,model_manager,model_manager2)
# pipe.enable_cpu_offload()

trans = pipe.dit
replace_layer(trans)

prompt = "CG. Full body. A captivating fantasy magic woman portrait in the deep sea. The woman, with blue spaghetti strap silk dress, swims in the sea. Her flowing silver hair shimmers with every color of the rainbow and cascades down, merging with the floating flora around her. Smooth, delicate and fair skin."

torch.manual_seed(6)
image = pipe(
    prompt=prompt,
    num_inference_steps=30, embedded_guidance=3.5
)
image.save("image_1024_float8.jpg")