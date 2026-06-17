from diffsynth.pipelines.ideogram4 import Ideogram4Pipeline
from diffsynth.core import ModelConfig
import torch

pipe = Ideogram4Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/ideogram-4-bf16-repackage", origin_file_pattern="transformer/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/ideogram-4-bf16-repackage", origin_file_pattern="unconditional_transformer/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/ideogram-4-bf16-repackage", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/ideogram-4-bf16-repackage", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="ideogram-ai/ideogram-4-fp8", origin_file_pattern="tokenizer/"),
)
# pipe.load_lora(pipe.dit, "models/train/Ideogram-4-bf16-repackage_lora/epoch-1.safetensors", alpha=1)
# pipe.load_lora(pipe.dit_uncond, "models/train/Ideogram-4-bf16-repackage_lora/epoch-1.safetensors", alpha=1)

prompt = "{\"high_level_description\":\"A close-up photograph of a happy Pembroke Welsh Corgi sitting on a concrete wall, panting with its tongue out, set against a backdrop of blurred pink cherry blossoms and blue sky.\",\"style_description\":{\"aesthetics\":\"joyful, vibrant, spring-like, cute, energetic\",\"lighting\":\"bright natural daylight, soft diffuse sunlight, shallow depth of field\",\"photo\":\"85mm lens, f/2.0, bokeh background, sharp focus on dog face\",\"medium\":\"photograph\",\"color_palette\":[\"#E2725B\",\"#FFFFFF\",\"#FFB7C5\",\"#87CEEB\",\"#A9A9A9\"]},\"compositional_deconstruction\":{\"background\":\"Softly blurred background of pink cherry blossom branches against a pale blue sky. The bokeh effect creates a dreamy spring atmosphere. The background is out of focus to highlight the sharp details of the dog in the foreground.\",\"elements\":[{\"type\":\"obj\",\"bbox\":[150,200,900,850],\"desc\":\"A Pembroke Welsh Corgi with fluffy orange and white fur. Its mouth is open, panting with a pink tongue hanging out, expression is happy and excited. Ears are perked up. Sharp focus on the face and eyes.\"},{\"type\":\"obj\",\"bbox\":[850,0,1000,1000],\"desc\":\"A grey concrete wall or ledge at the bottom of the frame. The dog's front paws are resting near the edge. Rough texture.\"}]}}"
image = pipe(prompt=prompt, height=1024, width=1024, num_inference_steps=48, cfg_scale=7.0, seed=0)
image.save("image_lora.jpg")
