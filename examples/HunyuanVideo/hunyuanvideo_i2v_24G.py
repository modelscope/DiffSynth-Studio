import torch
from diffsynth import ModelManager, HunyuanVideoPipeline, download_models, save_video
from modelscope import dataset_snapshot_download
from PIL import Image


download_models(["HunyuanVideoI2V"])
model_manager = ModelManager()

# The DiT model is loaded in bfloat16.
model_manager.load_models(
    [
        "models/HunyuanVideoI2V/transformers/mp_rank_00_model_states.pt"
    ],
    torch_dtype=torch.bfloat16,
    device="cpu"
)

# The other modules are loaded in float16.
model_manager.load_models(
    [
        "models/HunyuanVideoI2V/text_encoder/model.safetensors",
        "models/HunyuanVideoI2V/text_encoder_2",
        'models/HunyuanVideoI2V/vae/pytorch_model.pt'
    ],
    torch_dtype=torch.float16,
    device="cpu"
)
# The computation device is "cuda".
pipe = HunyuanVideoPipeline.from_model_manager(model_manager,
                                               torch_dtype=torch.bfloat16,
                                               device="cuda",
                                               enable_vram_management=True)

dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth",
                          local_dir="./",
                          allow_file_pattern=f"data/examples/hunyuanvideo/*")

i2v_resolution = "720p"
prompt = "An Asian man with short hair in black tactical uniform and white clothes waves a firework stick."
images = [Image.open("data/examples/hunyuanvideo/0.jpg").convert('RGB')]
video = pipe(prompt, input_images=images, num_inference_steps=50, seed=0, i2v_resolution=i2v_resolution)
save_video(video, f"video_{i2v_resolution}_low_vram.mp4", fps=30, quality=6)
