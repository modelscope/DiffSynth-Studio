from diffsynth.core.loader import ModelConfig
from diffsynth.core.quant import QuantizeConfig, MixedQuantizeConfig
from diffsynth.models.model_loader import ModelPool
from diffsynth.core.loader import hash_model_file
from safetensors.torch import save_file
import os


def save_quantized_model(model_config: ModelConfig, output_path: str, device="cuda") -> str:
    quantize = model_config.quantize
    if quantize is None:
        raise ValueError(
            "`model_config.quantize` is required, e.g. ModelConfig(..., quantize=QuantizeConfig(method=\"bitsandbytes_nf4\"))."
        )
    model_config.download_if_necessary()
    pool = ModelPool()
    vram_config = pool.default_vram_config()
    vram_config["computation_device"] = device
    pool.auto_load_model(model_config.path, vram_config=vram_config, quantize=quantize)
    if len(pool.model) != 1:
        raise ValueError(f"Expected the file to map to exactly one model, but {len(pool.model)} were loaded: {pool.model_name}.")
    model = pool.model[0]
    state_dict = {key: value.cpu() for key, value in model.state_dict().items()}
    tensors, metadata = quantize.flatten_state_dict(state_dict)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(tensors, output_path, metadata=metadata)
    print(
        f"saved: {output_path} ({os.path.getsize(output_path) / 1024**3:.2f} GB, {len(tensors)} tensors)"
    )
    return hash_model_file(output_path)


def test_save_quantized_model():
    dit_quantize = QuantizeConfig(
        method="bitsandbytes_nf4",
        mode="dynamic",
        exclude_modules=[
            "time_embedder.proj_in", "time_embedder.proj_out",
            "video_patch_proj", "audio_patch_proj", "condition_proj",
            "final_layer.video_out", "final_layer.audio_out",
        ],
    )
    dit_config = ModelConfig(model_id="MiniMaxAI/MiniMax-H3", origin_file_pattern="Ref2VA/transformer/model*.safetensors", quantize=dit_quantize)
    print(save_quantized_model(dit_config, "models/DiffSynth-Studio/MiniMax-H3-NF4/minimax-h3-ref2va-nf4.safetensors"))


def test_save_mixed_quantized_model():
    mod_layers = [
        "img_mod.1", "txt_mod.1", "norm_out.linear", "img_in", "txt_in", "proj_out",
        "timestep_embedder.linear_1", "timestep_embedder.linear_2",
    ]
    dit_quantize = MixedQuantizeConfig(configs=[
        QuantizeConfig(method="bitsandbytes_nf4", exclude_modules=mod_layers),
        QuantizeConfig(method="torchao_int8_w8a16", target_modules=mod_layers),
    ])
    dit_config = ModelConfig(model_id="Qwen/Qwen-Image-2512", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors", quantize=dit_quantize)
    print(save_quantized_model(dit_config, "models/DiffSynth-Studio/Qwen-Image-2512-mixed/qwen-image-2512-nf4-int8mod.safetensors"))
