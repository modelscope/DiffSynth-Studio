import importlib
import torch
from diffsynth import ModelManager, FluxImagePipeline, download_models, ControlNetConfigUnit
from modelscope import dataset_snapshot_download
from PIL import Image

if importlib.util.find_spec("facexlib") is None:
    raise ImportError("You are using InifiniteYou. It depends on facexlib, which is not installed. Please install it with `pip install facexlib`.")
if importlib.util.find_spec("insightface") is None:
    raise ImportError("You are using InifiniteYou. It depends on insightface, which is not installed. Please install it with `pip install insightface`.")

download_models(["InfiniteYou"])
model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda", model_id_list=["FLUX.1-dev"])
model_manager.load_models([
    [
        "models/InfiniteYou/InfuseNetModel/diffusion_pytorch_model-00001-of-00002.safetensors",
        "models/InfiniteYou/InfuseNetModel/diffusion_pytorch_model-00002-of-00002.safetensors"
    ],
    "models/InfiniteYou/image_proj_model.bin",
])


pipe = FluxImagePipeline.from_model_manager(
    model_manager,
    controlnet_config_units=[
        ControlNetConfigUnit(processor_id="none",
                             model_path=[
                                 'models/InfiniteYou/InfuseNetModel/diffusion_pytorch_model-00001-of-00002.safetensors',
                                 'models/InfiniteYou/InfuseNetModel/diffusion_pytorch_model-00002-of-00002.safetensors'
                             ],
                             scale=1.0)])
dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern=f"data/examples/infiniteyou/*")

prompt = "A man, portrait, cinematic"
id_image = "data/examples/infiniteyou/man.jpg"
id_image = Image.open(id_image).convert('RGB')
image = pipe(
    prompt=prompt, seed=1,
    id_image=id_image, controlnet_guidance=1.0,
    num_inference_steps=50, embedded_guidance=3.5,
    height=1024, width=1024,
)
image.save("man.jpg")

prompt = "A woman, portrait, cinematic"
id_image = "data/examples/infiniteyou/woman.jpg"
id_image = Image.open(id_image).convert('RGB')
image = pipe(
    prompt=prompt, seed=1,
    id_image=id_image, controlnet_guidance=1.0,
    num_inference_steps=50, embedded_guidance=3.5,
    height=1024, width=1024,
)
image.save("woman.jpg")
