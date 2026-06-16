import torch, torchaudio
from diffsynth import load_model, ModelConfig
from diffsynth.models.demucs import HTDemucs

class AudioTrackSeparator(torch.nn.Module):
    def __init__(self, torch_dtype=torch.float32, device="cuda", model_config=ModelConfig(model_id="DiffSynth-Studio/Demucs-Repackage", origin_file_pattern="model.safetensors")):
        super().__init__()
        model_config.download_if_necessary()
        self.model = load_model(HTDemucs, model_config.path, torch_dtype=torch_dtype, device=device)

    @torch.no_grad()
    def __call__(self, audio, target_sample_rate=48000, **kwargs):
        if isinstance(audio, str):
            audio, sample_rate = torchaudio.load(audio)
        else:
            audio, sample_rate = audio
        audio = audio.to(dtype=next(iter(self.model.parameters())).dtype, device=next(iter(self.model.parameters())).device)
        vocals = self.model.extract_track(audio, sample_rate)
        if target_sample_rate != 44100:
            vocals = torchaudio.functional.resample(vocals, 44100, target_sample_rate)
        return vocals
