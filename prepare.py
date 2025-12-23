from diffsynth import load_state_dict, skip_model_initialization
from diffsynth.models.z_image_image2lora import ZImageImage2LoRAModel
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig, ZImageUnit_Image2LoRAEncode, ZImageUnit_Image2LoRADecode
import torch, os
from PIL import Image
from safetensors.torch import save_file


model = ZImageImage2LoRAModel(compress_dim=256).to("cuda").to(torch.bfloat16)
model.initialize_weights()
os.makedirs("models/train/Z-Image-i2L_v12", exist_ok=True)
save_file(model.state_dict(), "models/train/Z-Image-i2L_v12/model.safetensors")

# check loading
pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig("models/train/Z-Image-i2L_v12/model.safetensors"),
    ],
)