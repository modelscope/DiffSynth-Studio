from diffsynth.pipelines.hiuni import HiUniPipeline, ModelConfig
import torch


pipe = HiUniPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="model-*.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="tokenizer*.json"),
    processor_config=ModelConfig(model_id="ernie-research/HiUni", origin_file_pattern="preprocessor_config.json"),
)
pipe.load_lora(pipe.dit, "models/train/HiUni_lora/epoch-4.safetensors")
prompt = "dog, white and brown dog, sitting on wall, under pink flowers"
image = pipe(prompt, height=1024, width=1024, seed=0)
image.save("image.jpg")
