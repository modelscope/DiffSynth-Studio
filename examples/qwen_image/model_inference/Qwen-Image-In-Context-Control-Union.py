from PIL import Image
import torch
from modelscope import dataset_snapshot_download, snapshot_download
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from diffsynth.controlnets.processors import Annotator

allow_file_pattern = ["sk_model.pth", "sk_model2.pth", "dpt_hybrid-midas-501f0c75.pt", "ControlNetHED.pth", "body_pose_model.pth", "hand_pose_model.pth", "facenet.pth", "scannet.pt"]
snapshot_download("lllyasviel/Annotators", local_dir="models/Annotators", allow_file_pattern=allow_file_pattern)

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
snapshot_download("DiffSynth-Studio/Qwen-Image-In-Context-Control-Union", local_dir="models/DiffSynth-Studio/Qwen-Image-In-Context-Control-Union", allow_file_pattern="model.safetensors")
pipe.load_lora(pipe.dit, "models/DiffSynth-Studio/Qwen-Image-In-Context-Control-Union/model.safetensors")

dataset_snapshot_download(dataset_id="DiffSynth-Studio/examples_in_diffsynth", local_dir="./", allow_file_pattern=f"data/examples/qwen-image-context-control/image.jpg")
origin_image = Image.open("data/examples/qwen-image-context-control/image.jpg").resize((1024, 1024))
annotator_ids = ['openpose', 'canny', 'depth', 'lineart', 'softedge', 'normal']
for annotator_id in annotator_ids:
    annotator = Annotator(processor_id=annotator_id, device="cuda")
    control_image = annotator(origin_image)
    control_image.save(f"{annotator.processor_id}.png")

    control_prompt = "Context_Control. "
    prompt = f"{control_prompt}一个穿着淡蓝色的漂亮女孩正在翩翩起舞，背景是梦幻的星空，光影交错，细节精致。"
    negative_prompt = "网格化，规则的网格，模糊, 低分辨率, 低质量, 变形, 畸形, 错误的解剖学, 变形的手, 变形的身体, 变形的脸, 变形的头发, 变形的眼睛, 变形的嘴巴"
    image = pipe(prompt, seed=1, negative_prompt=negative_prompt, context_image=control_image, height=1024, width=1024)
    image.save(f"image_{annotator.processor_id}.png")
