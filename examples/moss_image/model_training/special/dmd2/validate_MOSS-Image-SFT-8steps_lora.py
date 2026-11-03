import torch
from diffsynth.pipelines.moss_image import MOSSImagePipeline, ModelConfig

lora_checkpoint = "workspace/your_training_run/step-1000.safetensors"

pipe = MOSSImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", computation_dtype=torch.float32),
    ],
    tokenizer_config=ModelConfig(model_id="openmoss/MOSS-Image-SFT", origin_file_pattern="tokenizer/"),
)

pipe.load_lora(pipe.dit, lora_checkpoint)

prompt = "a dog"
image = pipe(prompt, height=1024, width=1024, seed=0, num_inference_steps=8, cfg_scale=1.0)
image.save("image_moss_dmd2_lora_8steps.jpg")
