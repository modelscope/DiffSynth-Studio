from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
from PIL import Image
from modelscope import dataset_snapshot_download


pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.dit, "models/train/Qwen-Image-EliGen_lora/epoch-4.safetensors")

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_image_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="eligen/*.png"
)


entity_prompts = ["A beautiful girl", "sign 'Entity Control'", "shorts", "shirt"]
global_prompt = "A beautiful girl wearing shirt and shorts in the street,  holding a sign 'Entity Control'"
masks = [Image.open(f"data/example_image_dataset/eligen/{i}.png").convert('RGB') for i in range(len(entity_prompts))]

image = pipe(global_prompt,
             seed=0,
             height=1024,
             width=1024,
             eligen_entity_prompts=entity_prompts,
             eligen_entity_masks=masks)
image.save("Qwen-Image_EliGen.jpg")
