import os

import numpy as np

from safetensors import safe_open
from safetensors.torch import save_file

from diffsynth import hash_state_dict_keys
from diffsynth.core import load_state_dict
from diffsynth.models.model_loader import ModelPool

model_pool = ModelPool()
os.makedirs("models/DiffSynth-Studio/LTX-2.5-Repackage", exist_ok=True)

SOURCES = (
    "models/Lightricks/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "models/Lightricks/LTX-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
)

def target_name(name):
    if name.startswith("text_embedding_projection."):
        return "feature_extractor." + name.removeprefix("text_embedding_projection.")
    if name.startswith("model.diffusion_model.video_embeddings_connector."):
        return "connectors.video_connector." + name.removeprefix("model.diffusion_model.video_embeddings_connector.")
    if name.startswith("model.diffusion_model.audio_embeddings_connector."):
        return "connectors.audio_connector." + name.removeprefix("model.diffusion_model.audio_embeddings_connector.")
    return None


text_encoder_post_modules_state_dict = {}
for path in SOURCES:
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            new_name = target_name(name)
            if new_name is not None:
                text_encoder_post_modules_state_dict[new_name] = handle.get_tensor(name)

save_file(text_encoder_post_modules_state_dict, "models/DiffSynth-Studio/LTX-2.5-Repackage/text_encoder_post_modules.safetensors")
print(f"text_encoder_post_modules keys hash: {hash_state_dict_keys(text_encoder_post_modules_state_dict)}")
model_pool.auto_load_model("models/DiffSynth-Studio/LTX-2.5-Repackage/text_encoder_post_modules.safetensors")

tokenizer_dir = "models/DiffSynth-Studio/LTX-2.5-Repackage/tokenizer"
os.makedirs(tokenizer_dir, exist_ok=True)
with safe_open(SOURCES[0], framework="pt", device="cpu") as handle:
    tokenizer_bytes = handle.get_tensor("tokenizer_json").detach().cpu().numpy().astype(np.uint8).tobytes()
    config_bytes = handle.get_tensor("hf_asset__tokenizer_config.json").detach().cpu().numpy().astype(np.uint8).tobytes()
with open(os.path.join(tokenizer_dir, "tokenizer.json"), "wb") as f:
    f.write(tokenizer_bytes)
with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "wb") as f:
    f.write(config_bytes)
print(f"tokenizer assets written to {tokenizer_dir}")
