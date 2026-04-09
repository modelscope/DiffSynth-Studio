from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig, ControlNetInput
from modelscope import dataset_snapshot_download
from PIL import Image
import torch


pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="PAI/Z-Image-Turbo-Fun-Controlnet-Union-2.1", origin_file_pattern="Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="transformer/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="text_encoder/*.safetensors"),
        ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
)

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/examples_in_diffsynth",
    local_dir="./",
    allow_file_pattern="data/examples/upscale/low_res.png"
)
controlnet_image = Image.open("data/examples/upscale/low_res.png").resize((1024, 1024))
prompt = "这是一张充满都市气息的户外人物肖像照片。画面中是一位年轻男性，他展现出时尚而自信的形象。人物拥有精心打理的短发发型，两侧修剪得较短，顶部保留一定长度，呈现出流行的Undercut造型。他佩戴着一副时尚的浅色墨镜或透明镜框眼镜，为整体造型增添了潮流感。脸上洋溢着温和友善的笑容，神情放松自然，给人以阳光开朗的印象。他身穿一件经典的牛仔外套，这件单品永不过时，展现出休闲又有型的穿衣风格。牛仔外套的蓝色调与整体氛围十分协调，领口处隐约可见内搭的衣物。照片的背景是典型的城市街景，可以看到模糊的建筑物、街道和行人，营造出繁华都市的氛围。背景经过了恰当的虚化处理，使人物主体更加突出。光线明亮而柔和，可能是白天的自然光，为照片带来清新通透的视觉效果。整张照片构图专业，景深控制得当，完美捕捉了一个现代都市年轻人充满活力和自信的瞬间，展现出积极向上的生活态度。"
image = pipe(prompt=prompt, seed=0, height=1024, width=1024, controlnet_inputs=[ControlNetInput(image=controlnet_image, scale=0.7)])
image.save("image_tile.jpg")
