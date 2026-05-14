import torch
from PIL import Image
from diffsynth.pipelines.hidream_o1_image import HiDreamO1ImagePipeline
from diffsynth.core.loader.config import ModelConfig


pipe = HiDreamO1ImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="HiDream-ai/HiDream-O1-Image",
            origin_file_pattern="model-*.safetensors",
        ),
    ],
    processor_config=ModelConfig(
        model_id="HiDream-ai/HiDream-O1-Image",
        origin_file_pattern="./",
    ),
)

ref_image_1 = Image.open("/mnt/nas1/zhanghong/project26/main_project/opencode/packages/hidream-o1-image/HiDream-O1-Image/assets/edit/test.jpg").convert("RGB")

image = pipe(
    prompt="remove the earphones",
    negative_prompt=" ",
    cfg_scale=4.0,
    height=1792,
    width=2304,
    seed=42,
    num_inference_steps=50,
    edit_image=[ref_image_1],
)
image.save("image_edit_demo_base1.jpg")

# workdirs = "workdirs/edit_base/"
# import os
# os.makedirs(workdirs, exist_ok=True)
# for seed in range(20):
#     # Load two reference images
#     ref_image_1 = Image.open("image.jpg").convert("RGB")

#     image = pipe(
#         prompt="change her clothes to blue",
#         negative_prompt=" ",
#         cfg_scale=4.0,
#         height=2048,
#         width=2048,
#         seed=seed,
#         num_inference_steps=50,
#         edit_image=[ref_image_1],
#     )
#     image.save(os.path.join(workdirs, f"image_edit_{seed}.jpg"))
