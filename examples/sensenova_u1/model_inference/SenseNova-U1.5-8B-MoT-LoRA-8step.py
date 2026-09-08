from diffsynth.pipelines.sensenova_u1_image import SenseNovaU1ImagePipeline, ModelConfig
import torch

pipe = SenseNovaU1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="model*.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT", origin_file_pattern="./"),
)
pipe.load_lora(pipe.dit, ModelConfig(model_id="SenseNova/SenseNova-U1.5-8B-MoT-LoRAs", origin_file_pattern="SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors"))

prompt = "A neon bar sign that clearly reads \"OPEN LATE\", dark interior, moody reflections, easy text rendering. Any text in the image must be rendered exactly as written in quotation marks, with correct spelling, clean typography, and strong readability."
image = pipe(prompt=prompt, seed=42, height=2048, width=2048, num_inference_steps=8, cfg_scale=1.0, shift=3.0)
image.save("image_SenseNova-U1.5-8B-MoT-LoRA-8step.jpg")

# think_mode
image = pipe(prompt=prompt, seed=42, height=2048, width=2048, num_inference_steps=8, cfg_scale=1.0, shift=3.0, think_mode=True)
image.save("image_SenseNova-U1.5-8B-MoT-LoRA-8step-Think.jpg")
