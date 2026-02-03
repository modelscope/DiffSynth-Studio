from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig, FlowMatchScheduler
from modelscope import dataset_snapshot_download
from PIL import Image
import torch

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)

lora = ModelConfig(
    model_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
    origin_file_pattern="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
)
pipe.load_lora(pipe.dit, lora, alpha=8/64)
pipe.scheduler = FlowMatchScheduler("Qwen-Image-Lightning")


dataset_snapshot_download(
    "DiffSynth-Studio/example_image_dataset",
    allow_file_pattern="qwen_image_edit/*",
    local_dir="data/example_image_dataset",
)

prompt = "生成这两个人的合影"
edit_image = [
    Image.open("data/example_image_dataset/qwen_image_edit/image1.jpg"),
    Image.open("data/example_image_dataset/qwen_image_edit/image2.jpg"),
]
image = pipe(
    prompt,
    edit_image=edit_image,
    seed=1,
    num_inference_steps=4,
    height=1152,
    width=896,
    edit_image_auto_resize=True,
    zero_cond_t=True, # This is a special parameter introduced by Qwen-Image-Edit-2511
    cfg_scale=1.0,
)
image.save("image.jpg")

# Qwen-Image-Edit-2511 is a multi-image editing model.
# Please use a list to input `edit_image`, even if the input contains only one image.
# edit_image = [Image.open("image.jpg")]
# Please do not input the image directly.
# edit_image = Image.open("image.jpg")
