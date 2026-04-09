import torch
from PIL import Image
from diffsynth.core import load_state_dict
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)

lora_path = "models/train/Wan2.1-VACE-14B_lora/epoch-4.safetensors"
lora_state_dict = load_state_dict(lora_path, torch_dtype=torch.bfloat16, device="cuda")
pipe.load_lora(pipe.vace, state_dict=lora_state_dict, alpha=1)

# If you also finetuned the VACE context embedder (vace_patch_embedding), its weights must be loaded too.
for prefix in ("", "vace."):
    w_key = f"{prefix}vace_patch_embedding.weight"
    b_key = f"{prefix}vace_patch_embedding.bias"
    if w_key in lora_state_dict:
        patch_sd = {"weight": lora_state_dict[w_key]}
        if b_key in lora_state_dict:
            patch_sd["bias"] = lora_state_dict[b_key]
            print(f"Loading Finetuned bias for VACE context embedder from {lora_path}")
        print(f"Loading Finetuned VACE context embedder weights from {lora_path} ({prefix or 'no prefix'})")
        pipe.vace.vace_patch_embedding.load_state_dict(patch_sd, strict=False)
        break

video = VideoData("data/example_video_dataset/video1_softedge.mp4", height=480, width=832)
video = [video[i] for i in range(17)]
reference_image = VideoData("data/example_video_dataset/video1.mp4", height=480, width=832)[0]

video = pipe(
    prompt="from sunset to night, a small town, light, house, river",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=video, vace_reference_image=reference_image, num_frames=17,
    seed=1, tiled=True
)
save_video(video, "video_Wan2.1-VACE-14B.mp4", fps=15, quality=5)
