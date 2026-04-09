from safetensors.torch import save_file
from diffsynth import hash_state_dict_keys
from diffsynth.core import load_state_dict
from diffsynth.models.model_loader import ModelPool

model_pool = ModelPool()
state_dict = load_state_dict("models/Lightricks/LTX-2/ltx-2-19b-dev.safetensors")

dit_state_dict = {}
for name in state_dict:
    if name.startswith("model.diffusion_model."):
        new_name = name.replace("model.diffusion_model.", "")
        if new_name.startswith("audio_embeddings_connector.") or new_name.startswith("video_embeddings_connector."):
            continue
        dit_state_dict[name] = state_dict[name]

print(f"dit_state_dict keys hash: {hash_state_dict_keys(dit_state_dict)}")
save_file(dit_state_dict, "models/DiffSynth-Studio/LTX-2-Repackage/transformer.safetensors")
model_pool.auto_load_model(
    "models/DiffSynth-Studio/LTX-2-Repackage/transformer.safetensors",
)


video_vae_encoder_state_dict = {}
for name in state_dict:
    if name.startswith("vae.encoder."):
        video_vae_encoder_state_dict[name] = state_dict[name]
    elif name.startswith("vae.per_channel_statistics."):
        video_vae_encoder_state_dict[name] = state_dict[name]

save_file(video_vae_encoder_state_dict, "models/DiffSynth-Studio/LTX-2-Repackage/video_vae_encoder.safetensors")
print(f"video_vae_encoder keys hash: {hash_state_dict_keys(video_vae_encoder_state_dict)}")
model_pool.auto_load_model("models/DiffSynth-Studio/LTX-2-Repackage/video_vae_encoder.safetensors")


video_vae_decoder_state_dict = {}
for name in state_dict:
    if name.startswith("vae.decoder."):
        video_vae_decoder_state_dict[name] = state_dict[name]
    elif name.startswith("vae.per_channel_statistics."):
        video_vae_decoder_state_dict[name] = state_dict[name]
save_file(video_vae_decoder_state_dict, "models/DiffSynth-Studio/LTX-2-Repackage/video_vae_decoder.safetensors")
print(f"video_vae_decoder keys hash: {hash_state_dict_keys(video_vae_decoder_state_dict)}")
model_pool.auto_load_model("models/DiffSynth-Studio/LTX-2-Repackage/video_vae_decoder.safetensors")


audio_vae_decoder_state_dict = {}
for name in state_dict:
    if name.startswith("audio_vae.decoder."):
        audio_vae_decoder_state_dict[name] = state_dict[name]
    elif name.startswith("audio_vae.per_channel_statistics."):
        audio_vae_decoder_state_dict[name] = state_dict[name]
save_file(audio_vae_decoder_state_dict, "models/DiffSynth-Studio/LTX-2-Repackage/audio_vae_decoder.safetensors")
print(f"audio_vae_decoder keys hash: {hash_state_dict_keys(audio_vae_decoder_state_dict)}")
model_pool.auto_load_model("models/DiffSynth-Studio/LTX-2-Repackage/audio_vae_decoder.safetensors")


audio_vae_encoder_state_dict = {}
for name in state_dict:
    if name.startswith("audio_vae.encoder."):
        audio_vae_encoder_state_dict[name] = state_dict[name]
    elif name.startswith("audio_vae.per_channel_statistics."):
        audio_vae_encoder_state_dict[name] = state_dict[name]
save_file(audio_vae_encoder_state_dict, "models/DiffSynth-Studio/LTX-2-Repackage/audio_vae_encoder.safetensors")
print(f"audio_vae_encoder keys hash: {hash_state_dict_keys(audio_vae_encoder_state_dict)}")
model_pool.auto_load_model("models/DiffSynth-Studio/LTX-2-Repackage/audio_vae_encoder.safetensors")


audio_vocoder_state_dict = {}
for name in state_dict:
    if name.startswith("vocoder."):
        audio_vocoder_state_dict[name] = state_dict[name]
save_file(audio_vocoder_state_dict, "models/DiffSynth-Studio/LTX-2-Repackage/audio_vocoder.safetensors")
print(f"audio_vocoder keys hash: {hash_state_dict_keys(audio_vocoder_state_dict)}")
model_pool.auto_load_model("models/DiffSynth-Studio/LTX-2-Repackage/audio_vocoder.safetensors")


text_encoder_post_modules_state_dict = {}
for name in state_dict:
    if name.startswith("text_embedding_projection."):
        text_encoder_post_modules_state_dict[name] = state_dict[name]
    elif name.startswith("model.diffusion_model.video_embeddings_connector."):
        text_encoder_post_modules_state_dict[name] = state_dict[name]
    elif name.startswith("model.diffusion_model.audio_embeddings_connector."):
        text_encoder_post_modules_state_dict[name] = state_dict[name]
save_file(text_encoder_post_modules_state_dict, "models/DiffSynth-Studio/LTX-2-Repackage/text_encoder_post_modules.safetensors")
print(f"text_encoder_post_modules keys hash: {hash_state_dict_keys(text_encoder_post_modules_state_dict)}")
model_pool.auto_load_model("models/DiffSynth-Studio/LTX-2-Repackage/text_encoder_post_modules.safetensors")


state_dict = load_state_dict("models/Lightricks/LTX-2/ltx-2-19b-distilled.safetensors")
dit_state_dict = {}
for name in state_dict:
    if name.startswith("model.diffusion_model."):
        new_name = name.replace("model.diffusion_model.", "")
        if new_name.startswith("audio_embeddings_connector.") or new_name.startswith("video_embeddings_connector."):
            continue
        dit_state_dict[name] = state_dict[name]

print(f"dit_state_dict keys hash: {hash_state_dict_keys(dit_state_dict)}")
save_file(dit_state_dict, "models/DiffSynth-Studio/LTX-2-Repackage/transformer_distilled.safetensors")
model_pool.auto_load_model(
    "models/DiffSynth-Studio/LTX-2-Repackage/transformer_distilled.safetensors",
)