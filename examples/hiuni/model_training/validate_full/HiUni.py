from diffsynth.pipelines.hiuni import HiUniPipeline, ModelConfig
from diffsynth import load_state_dict
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
state_dict = load_state_dict("models/train/HiUni_full/epoch-1.safetensors")
pipe.dit.load_state_dict(state_dict)
prompt = "dog, white and brown dog, sitting on wall, under pink flowers"
image = pipe(prompt, height=1024, width=1024, seed=0)
image.save("image.jpg")
