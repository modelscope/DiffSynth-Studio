"""Validate HiDream-O1-Image LoRA training checkpoint."""
import torch, os
from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline, ModelConfig
from peft import PeftModel

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

lora_path = './models/train/HiDream-O1-Image_lora_test/epoch-1.safetensors'

if not os.path.exists(lora_path):
    print(f'LoRA checkpoint not found: {lora_path}')
    exit(1)

print(f'Loading LoRA from: {lora_path}', flush=True)

# Load base pipeline
model_configs = [ModelConfig(model_id='HiDream-ai/HiDream-O1-Image', origin_file_pattern='model-*.safetensors')]
pipe = HiDreamO1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16, device='cuda', model_configs=model_configs,
)
print(f'Pipeline loaded. Setting to eval mode.', flush=True)

# Load LoRA into the DiT model
from peft import LoraConfig, inject_adapter_in_model
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
)
pipe.dit = inject_adapter_in_model(lora_config, pipe.dit)

# Load weights
state_dict = torch.load(lora_path, map_location='cuda', weights_only=True)
print(f'Loaded {len(state_dict)} keys from LoRA checkpoint', flush=True)

# Filter keys that belong to our model
filtered_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('pipe.dit.'):
        new_k = k.replace('pipe.dit.', '')
        filtered_state_dict[new_k] = v

missing, unexpected = pipe.dit.load_state_dict(filtered_state_dict, strict=False)
print(f'Missing keys: {len(missing)}', flush=True)
print(f'Unexpected keys: {len(unexpected)}', flush=True)

pipe.dit.eval()

# Run inference
print('Running inference...', flush=True)
output = pipe(
    prompt='a dog sitting in the grass',
    height=512, width=512,
    num_inference_steps=20,
    seed=42,
)
output.save('/tmp/lora_test_output.jpg')
print(f'Output image shape: {output.size}', flush=True)
print('LORA VALIDATION SUCCESS!', flush=True)
