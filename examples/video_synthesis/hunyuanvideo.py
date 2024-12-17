from diffsynth import ModelManager, HunyuanVideoPipeline, download_models
import torch


# Download models (automatically)
download_models(["HunyuanVideo"])

# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_models([
    "t2i_models/HunyuanVideo/text_encoder/model.safetensors",
    "t2i_models/HunyuanVideo/text_encoder_2",
])
pipe = HunyuanVideoPipeline.from_model_manager(model_manager)
prompt = 'A cat walks on the grass, realistic style.'
pipe(prompt)
