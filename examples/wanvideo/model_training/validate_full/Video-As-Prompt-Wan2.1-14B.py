import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="ByteDance/Video-As-Prompt-Wan2.1-14B", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
)
state_dict = load_state_dict("models/train/Video-As-Prompt-Wan2.1-14B_full/epoch-1.safetensors")
pipe.vap.load_state_dict(state_dict)

ref_video_path = 'data/example_video_dataset/wanvap/vap_ref.mp4'
target_image_path = 'data/example_video_dataset/wanvap/input_image.jpg'

image = Image.open(target_image_path).convert("RGB")
ref_video = VideoData(ref_video_path, height=480, width=832)
ref_frames = [ref_video[i] for i in range(49)]

vap_prompt = "A man stands with his back to the camera on a dirt path overlooking sun-drenched, rolling green tea plantations. He wears a blue and green plaid shirt, dark pants, and white shoes. As he turns to face the camera and spreads his arms, a brief, magical burst of sparkling golden light particles envelops him. Through this shimmer, he seamlessly transforms into a Labubu toy character. His head morphs into the iconic large, furry-eared head of the toy, featuring a wide grin with pointed teeth and red cheek markings. The character retains the man's original plaid shirt and clothing, which now fit its stylized, cartoonish body. The camera remains static throughout the transformation, positioned low among the tea bushes, maintaining a consistent view of the subject and the expansive scenery."
prompt = "A young woman with curly hair, wearing a green hijab and a floral dress, plays a violin in front of a vintage green car on a tree-lined street. She executes a swift counter-clockwise turn to face the camera. During the turn, a brilliant shower of golden, sparkling particles erupts and momentarily obscures her figure. As the particles fade, she is revealed to have seamlessly transformed into a Labubu toy character. This new figure, now with the toy's signature large ears, big eyes, and toothy grin, maintains the original pose and continues playing the violin. The character's clothing—the green hijab, floral dress, and black overcoat—remains identical to the woman's. Throughout this transition, the camera stays static, and the street-side environment remains completely consistent."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    input_image=image,
    seed=42, tiled=True,
    height=480, width=832,
    num_frames=49,
    vap_video=ref_frames,
    vap_prompt=vap_prompt,
    negative_vap_prompt=negative_prompt,
)
save_video(video, "video_Video-As-Prompt-Wan2.1-14B.mp4", fps=15, quality=5)
