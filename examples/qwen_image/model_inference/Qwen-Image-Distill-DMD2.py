from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.core import load_state_dict
from modelscope import snapshot_download
import torch, math


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

snapshot_download("MusePublic/Qwen-Image-Distill", allow_file_pattern="qwen_image_distill_3step.safetensors", cache_dir="models")
lora_state_dict = load_state_dict("models/MusePublic/Qwen-Image-Distill/qwen_image_distill_3step.safetensors")
lora_state_dict = {i.replace("base_model.model.", ""): j for i, j in lora_state_dict.items()}
pipe.load_lora(pipe.dit, state_dict=lora_state_dict)

prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
image = pipe(prompt, seed=0, num_inference_steps=3, cfg_scale=1, exponential_shift_mu=math.log(2.5))
image.save("image.jpg")
