import torch
from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline
from diffsynth.core.loader.config import ModelConfig
from diffsynth.diffusion import HiDreamO1FlashScheduler
from PIL import Image
import os
from modelscope import dataset_snapshot_download


pipe = HiDreamO1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="HiDream-ai/HiDream-O1-Image-Dev", origin_file_pattern="model-*.safetensors"),
    ],
    processor_config=ModelConfig(model_id="HiDream-ai/HiDream-O1-Image-Dev", origin_file_pattern="./"),
)
pipe.scheduler = HiDreamO1FlashScheduler(noise_scale_start=7.5, noise_scale_end=7.5, noise_clip_std=2.5)

# Text-to-Image
image = pipe(
    prompt="medium shot, eye-level, front view. A woman is seated in an ornate bedroom, illuminated by candlelight, with a calm and composed expression. The subject is a young woman with fair skin, light brown hair styled in an updo with loose tendrils framing her face, and blue eyes. She wears a cream-colored satin robe with delicate floral embroidery and lace trim along the neckline. Her ears are adorned with pearl drop earrings. She is seated on a bed with a dark, intricately carved wooden headboard. To her left, a wooden nightstand holds three lit white candles and a candelabra with multiple lit candles in the background. The bed is covered with patterned pillows and a dark, textured blanket. The walls are paneled with dark wood and feature a large, ornate tapestry with muted earth tones. The lighting creates soft highlights on her face and robe, with warm shadows cast across the room.",
    cfg_scale=1.0,
    height=2048,
    width=2048,
    seed=42,
    num_inference_steps=28,
    model_type="dev",
    noise_scale=7.5,
)
image.save("image.jpg")

# Image-to-Image
image = pipe(
    prompt="change her clothes to blue",
    cfg_scale=1.0,
    height=2048,
    width=2048,
    seed=43,
    num_inference_steps=28,
    model_type="dev",
    noise_scale=7.5,
    edit_image=[image],
)
image.save("image_edit.jpg")

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/diffsynth_example_dataset",
    local_dir="data/diffsynth_example_dataset",
    allow_file_pattern="hidream_o1_image/HiDream-O1-Image-Dev/*",
)
# Multi-Reference Subject-Driven Personalization
ref_image_dir = "./data/diffsynth_example_dataset/hidream_o1_image/HiDream-O1-Image-Dev/IP/"
ref_images = [Image.open(os.path.join(ref_image_dir, f)).convert("RGB") for f in os.listdir(ref_image_dir) if f.endswith(".jpg")]
prompt = "A young boy with blonde hair stands on steps wearing light blue jeans, a white t-shirt with logo, and blue and white sneakers. He wears a brown cord necklace with beads, a black wristwatch with digital display, and carries a yellow fanny pack with white zipper. In his hand is a red boxing glove with white top, a teal plastic toy car, and a plastic toy figure of Captain America. He wears a straw hat with cream band. Natural light illuminates the scene."
image = pipe(
    prompt=prompt,
    cfg_scale=1.0,
    height=2048,
    width=2048,
    seed=43,
    num_inference_steps=28,
    model_type="dev",
    noise_scale=7.5,
    edit_image=ref_images,
)
image.save("ip.jpg")

# Multi-Reference Subject-Driven Personalization with Skeleton
ref_image_dir = "./data/diffsynth_example_dataset/hidream_o1_image/HiDream-O1-Image-Dev/IP_skeleton/"
ref_images = [Image.open(os.path.join(ref_image_dir, f)).convert("RGB") for f in os.listdir(ref_image_dir) if f.endswith(".jpg")]
prompt = "Create a realistic try-on image of the person wearing the provided clothing."
image = pipe(
    prompt=prompt,
    cfg_scale=1.0,
    height=2048,
    width=2048,
    seed=43,
    num_inference_steps=28,
    model_type="dev",
    noise_scale=7.5,
    edit_image=ref_images,
)
image.save("ip_skeleton.jpg")

# Multi-Reference Subject-Driven Personalization with Layout：relative coordinates [x1, x2, y1, y2]
layout_bboxes = [[0.21, 0.44, 0.49, 0.74], [0.58, 0.80, 0.09, 0.34]]
ref_image_dir = "./data/diffsynth_example_dataset/hidream_o1_image/HiDream-O1-Image-Dev/IP_layout/"
ref_images = [Image.open(os.path.join(ref_image_dir, f)).convert("RGB") for f in os.listdir(ref_image_dir) if f.endswith(".jpg")]
prompt = "City council members pose with relaxed smiles on a sunlit terrace, warm approachable mood, golden hour, cinematic soft glow."
image = pipe(
    prompt=prompt,
    cfg_scale=1.0,
    height=2048,
    width=2048,
    seed=43,
    num_inference_steps=28,
    model_type="dev",
    noise_scale=7.5,
    edit_image=ref_images,
    layout_bboxes=layout_bboxes,
)
image.save("ip_layout.jpg")
