from diffsynth.pipelines.ace_step_audio import AceStepAudioPipeline, ModelConfig
import torch


pipe = AceStepAudioPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/model.safetensors"),
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="acestep-v15-turbo/model.safetensors"),
        ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B"),
)
