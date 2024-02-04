from diffsynth import ModelManager, SDVideoPipeline, ControlNetConfigUnit, VideoData, save_video
from diffsynth.processors.FastBlend import FastBlendSmoother
from diffsynth.processors.PILEditor import ContrastEditor, SharpnessEditor
from diffsynth.processors.sequencial_processor import SequencialProcessor
import torch


# Download models
# `models/stable_diffusion/dreamshaper_8.safetensors`: [link](https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16)
# `models/ControlNet/control_v11f1p_sd15_depth.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth)
# `models/ControlNet/control_v11p_sd15_softedge.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth)
# `models/Annotators/dpt_hybrid-midas-501f0c75.pt`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/dpt_hybrid-midas-501f0c75.pt)
# `models/Annotators/ControlNetHED.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth)

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "models/stable_diffusion/dreamshaper_8.safetensors",
    "models/ControlNet/control_v11f1p_sd15_depth.pth",
    "models/ControlNet/control_v11p_sd15_softedge.pth"
])
pipe = SDVideoPipeline.from_model_manager(
    model_manager,
    [
        ControlNetConfigUnit(
            processor_id="depth",
            model_path=rf"models/ControlNet/control_v11f1p_sd15_depth.pth",
            scale=0.5
        ),
        ControlNetConfigUnit(
            processor_id="softedge",
            model_path=rf"models/ControlNet/control_v11p_sd15_softedge.pth",
            scale=0.5
        )
    ]
)
smoother = SequencialProcessor([FastBlendSmoother(), ContrastEditor(rate=1.1), SharpnessEditor(rate=1.1)])

# Load video
# Original video: https://pixabay.com/videos/flow-rocks-water-fluent-stones-159627/
video = VideoData(video_file="data/pixabay100/159627 (1080p).mp4", height=512, width=768)
input_video = [video[i] for i in range(128)]

# Rerender
torch.manual_seed(0)
output_video = pipe(
    prompt="winter, ice, snow, water, river",
    negative_prompt="", cfg_scale=7,
    input_frames=input_video, controlnet_frames=input_video, num_frames=len(input_video),
    num_inference_steps=20, height=512, width=768,
    animatediff_batch_size=8, animatediff_stride=4, unet_batch_size=8,
    cross_frame_attention=True,
    smoother=smoother, smoother_progress_ids=[4, 9, 14, 19]
)

# Save images and video
save_video(output_video, "output_video.mp4", fps=30)
