from diffsynth.pipelines.anima_image import AnimaPipeline, ModelConfig
import torch


pipe = AnimaPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="circlestone-labs/Anima", origin_file_pattern="split_files/diffusion_models/anima-preview.safetensors"),
        ModelConfig(model_id="circlestone-labs/Anima", origin_file_pattern="split_files/text_encoders/qwen_3_06b_base.safetensors"),
        ModelConfig(model_id="circlestone-labs/Anima", origin_file_pattern="split_files/vae/qwen_image_vae.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen3-0.6B", origin_file_pattern="./"),
)
prompt = """masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top and denim shorts is standing outdoors. She's holding a rectangular sign out in front of her that reads "ANIMA". She's looking at the viewer with a smile. The background features some trees and blue sky with clouds."""
negative_prompt = """worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"""
image = pipe(prompt, seed=0, num_inference_steps=50, rand_device="cuda")
image.save("image.jpg")
