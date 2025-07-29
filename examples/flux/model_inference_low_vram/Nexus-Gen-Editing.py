import importlib
import torch
from PIL import Image
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig
from modelscope import dataset_snapshot_download


if importlib.util.find_spec("transformers") is None:
    raise ImportError("You are using Nexus-GenV2. It depends on transformers, which is not installed. Please install it with `pip install transformers==4.49.0`.")
else:
    import transformers
    assert transformers.__version__ == "4.49.0", "Nexus-GenV2 requires transformers==4.49.0, please install it with `pip install transformers==4.49.0`."


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="edit_decoder.bin", offload_device="cpu"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors", offload_device="cpu"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/", offload_device="cpu"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors", offload_device="cpu"),
    ],
    nexus_gen_processor_config=ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="processor/"),
)
pipe.enable_vram_management()

dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern=f"data/examples/nexusgen/cat.jpg")
ref_image = Image.open("data/examples/nexusgen/cat.jpg").convert("RGB")
prompt = "Add a crown."
image = pipe(
    prompt=prompt, negative_prompt="",
    seed=42, cfg_scale=2.0, num_inference_steps=50,
    nexus_gen_reference_image=ref_image,
    height=512, width=512,
)
image.save("cat_crown.jpg")
