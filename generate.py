import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from diffsynth.models.utils import load_state_dict
from diffsynth import ModelManager
import os
from diffsynth.models.value_control import SingleValueEncoder

device = "cuda:5"
# value_embedder_color = SingleValueEncoder(256, 3072, 32).to(dtype=torch.bfloat16, device=device)
value_embedder_photo = SingleValueEncoder(256, 3072, 32).to(dtype=torch.bfloat16, device=device)
value_embedder_detail = SingleValueEncoder(256, 3072, 32).to(dtype=torch.bfloat16, device=device)
value_embedder_logic = SingleValueEncoder(256, 3072, 32).to(dtype=torch.bfloat16, device=device)


value_embedder = {
    # 'color': value_embedder_color,
    'photo': value_embedder_photo,
    'detail': value_embedder_detail,
    'logic': value_embedder_logic,
}
save_dir = "/shark/dchen/Work-Duan/prefer/DiffSynth-Studio/dchen"
# for name in ['color', 'photo', 'detail', 'logic']:
for name in ['photo', 'detail', 'logic']:
    state_dict = load_state_dict(f"/shark/dchen/Work-Duan/prefer/DiffSynth-Studio/dchen/{name}.ckpt")
    missing1, unexpected1 = value_embedder[name].load_state_dict(state_dict, strict=False)
    print(f"{name} Missing: {missing1}, Unexpected: {unexpected1}")


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device=device,
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="color.ckpt"),
    ],
)
pipe.load_lora(pipe.dit, "/shark/dchen/Work-Duan/prefer/DiffSynth-Studio/dchen/dit.ckpt", alpha=1.0/8)
# pipe.value_embedder_color = value_embedder_color
print("pipe.value_embedder_color",pipe.value_embedder_color)
pipe.value_embedder_photo = value_embedder_photo
pipe.value_embedder_detail = value_embedder_detail
pipe.value_embedder_logic = value_embedder_logic


value = {'color': 0.5,'photo': 0.5,'detail': 0.5,'logic': 0.5,}
value_len = 32
image = pipe(prompt="a cat", value=value, seed=0, value_len=value_len)
image.save("prefer_image4.jpg")