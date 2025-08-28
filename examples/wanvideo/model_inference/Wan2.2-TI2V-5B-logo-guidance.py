import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

# This is a placeholder path. Please replace it with the actual path to your classifier model.
LOGO_CLASSIFIER_PATH = "path/to/your/logo_classifier.pt"
# This is a placeholder path. Please replace it with the actual path to your logo detector model.
LOGO_DETECTOR_PATH = "path/to/your/logo_detector.pt"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)
pipe.enable_vram_management()

# Load the classifier
pipe.load_classifier(LOGO_CLASSIFIER_PATH)

# Image-to-video with logo guidance
dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern=["data/examples/wan/cat_fightning.jpg"]
)
input_image = Image.open("data/examples/wan/cat_fightning.jpg").resize((1248, 704))

video = pipe(
    prompt="A video of a cat fighting, with a brand logo in the corner.",
    negative_prompt="worst quality, low quality",
    seed=0, tiled=True,
    height=704, width=1248,
    input_image=input_image,
    num_frames=121,
    classifier_guidance_scale=5.0,
    classifier_class_id=1, # Replace with the class id of your logo
    logo_detector_path=LOGO_DETECTOR_PATH,
)
save_video(video, "video_with_logo_guidance.mp4", fps=15, quality=5)
