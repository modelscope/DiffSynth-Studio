import re, torch, warnings, torchaudio
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.diffsynth_music_dit import DiffSynthMusicDiTModel
from ..models.ace_step_conditioner import AceStepConditionEncoder
from ..models.ace_step_text_encoder import AceStepTextEncoder
from ..models.ace_step_vae import AceStepVAE
from ..models.demucs import HTDemucs
from transformers import AutoTokenizer


class DiffSynthMusicPipeline(BasePipeline):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler("FLUX.2")
        self.text_encoder: AceStepTextEncoder = None
        self.conditioner: AceStepConditionEncoder = None
        self.dit: DiffSynthMusicDiTModel = None
        self.vae: AceStepVAE = None
        self.track_separator: HTDemucs = None
        self.tokenizer: AutoTokenizer = None

        self.in_iteration_models = ("dit",)
        self.units = [
            DiffSynthMusic_PromptEmbedder(),
            DiffSynthMusic_ConditionEmbedder(),
            DiffSynthMusic_NoiseInitializer(),
            DiffSynthMusic_InputAudioEmbedder(),
        ]
        self.model_fn = model_fn_ace_step
        self.compilable_models = ["dit"]
        self.default_negative_prompt = (
            "A barren, low-fidelity audio mess with a strong sense of technical failure. "
            "The track kicks off with an awkward silence where the backing instrumentation is completely absent, "
            "leaving only a hollow void. A layer of harsh, static-like noise and environmental hiss dominates the soundscape, "
            "drowning out any potential clarity. A distorted male vocal enters with severe clipping and digital artifacts, "
            "cracking painfully on every syllable. The singing is consistently off-key and wandering, lacking any rhythmic anchor "
            "or melodic coherence. The overall mood is grating, amateurish, and unintentionally comical."
        )

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/DiffSynth-Music-Tools", origin_file_pattern="text_encoder/"),
        vram_limit: float = None,
    ):
        pipe = DiffSynthMusicPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("ace_step_text_encoder")
        pipe.conditioner = model_pool.fetch_model("ace_step_conditioner")
        pipe.dit = model_pool.fetch_model("diffsynth_music_dit")
        pipe.vae = model_pool.fetch_model("ace_step_vae")
        if pipe.vae is not None: pipe.vae.remove_weight_norm()
        pipe.track_separator = model_pool.fetch_model("demucs")

        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def extract_track(self, audio, sample_rate=48000, track="vocals"):
        self.load_models_to_device(["track_separator"])
        audio = audio.to(dtype=next(iter(self.track_separator.parameters())).dtype, device=self.device)
        output = self.track_separator.extract_track(audio, sample_rate, track)
        output = torchaudio.functional.resample(output, 44100, sample_rate)
        return output

    @torch.no_grad()
    def fuse_track(self, audio, target_audio, target_track):
        if not isinstance(target_track, list):
            target_track = [target_track]
        kept_track = [i for i in self.track_separator.sources if i not in target_track]
        target_audio = self.extract_track(target_audio, track=target_track)
        audio = self.extract_track(audio, track=kept_track)
        audio, target_audio = self._balance_volumes(audio, target_audio)
        audio = audio + target_audio
        return audio

    @staticmethod
    def _balance_volumes(audio_a, audio_b, silence_threshold=1e-4):
        rms_a = audio_a.pow(2).mean().sqrt().item()
        rms_b = audio_b.pow(2).mean().sqrt().item()
        a_silent = rms_a < silence_threshold
        b_silent = rms_b < silence_threshold
        if a_silent or b_silent:
            return audio_a, audio_b
        target_rms = min(rms_a, rms_b)
        scale_a = target_rms / rms_a
        scale_b = target_rms / rms_b
        return audio_a * scale_a, audio_b * scale_b

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str = "",
        negative_prompt: str = "",
        lyrics: str = "",
        cfg_scale: float = 1.0,
        # Metadata
        bpm: float = None,
        timesignature: str = None,
        keyscale: str = None,
        # Input Audio
        input_audio: Optional[torch.Tensor] = None,
        denoising_strength: float = 1.0,
        # Duration
        duration: int = 160,
        # Rand
        seed: int = None,
        rand_device: str = "cpu",
        # Tiled VAE
        tiled: bool = False,
        tile_size: int = 512,
        tile_stride: int = 256,
        # KV Cache
        kv_cache = None,
        negative_kv_cache = None,
        # Force Control
        target_audio = None,
        target_track = "vocals",
        # Steps
        num_inference_steps: int = 50,
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=4096)

        # Parameters
        inputs_posi = {"prompt": prompt, "lyrics": lyrics, "kv_cache": kv_cache}
        inputs_nega = {"prompt": negative_prompt, "lyrics": "", "kv_cache": negative_kv_cache}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "bpm": bpm, "timesignature": timesignature, "keyscale": keyscale,
            "input_audio": input_audio,
            "duration": duration,
            "seed": seed,
            "rand_device": rand_device,
        }

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id,
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

        # Decode
        self.load_models_to_device(['vae'])
        audio = self.vae_output_to_audio(inputs_shared["latents"], tiled, tile_size, tile_stride)
        self.load_models_to_device([])
        if target_audio is not None:
            audio = self.fuse_track(audio, target_audio, target_track)
        return audio
    
    def vae_output_to_audio(self, vae_output, tiled=False, tile_size=512, tile_stride=256):
        audio = self.vae.decode(vae_output.transpose(1, 2), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        peak = torch.max(torch.abs(audio))
        if peak < 1e-6: return audio
        audio = audio * (10 ** (-1 / 20.0) / peak)
        audio = self.output_audio_format_check(audio)
        return audio


class DiffSynthMusic_PromptEmbedder(PipelineUnit):
    LYRIC_PROMPT = "# Languages\n{}\n\n# Lyric\n{}<|endoftext|>"

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "lyrics": "lyrics"},
            input_params_nega={"prompt": "prompt", "lyrics": "lyrics"},
            input_params=("bpm", "timesignature", "keyscale", "duration"),
            output_params=("text_hidden_states", "text_attention_mask", "lyric_hidden_states", "lyric_attention_mask"),
            onload_model_names=("text_encoder",)
        )

    def _encode_text(self, pipe: DiffSynthMusicPipeline, text, max_length=256):
        text_inputs = pipe.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(pipe.device)
        attention_mask = text_inputs.attention_mask.bool().to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids, attention_mask)
        return hidden_states, attention_mask

    def _encode_lyrics(self, pipe: DiffSynthMusicPipeline, lyric_text, max_length=2048):
        text_inputs = pipe.tokenizer(
            lyric_text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(pipe.device)
        attention_mask = text_inputs.attention_mask.bool().to(pipe.device)
        hidden_states = pipe.text_encoder.model.embed_tokens(input_ids)
        return hidden_states, attention_mask

    def process(self, pipe: DiffSynthMusicPipeline, prompt, lyrics, bpm, timesignature, keyscale, duration):
        pipe.load_models_to_device(['text_encoder'])
        if bpm is None: bpm = 100
        if timesignature is None: timesignature = "4"
        if keyscale is None: keyscale = "B minor"
        if duration is None: duration = "300"
        prompt = f"""
# Instruction
Fill the audio semantic mask based on the given conditions:

# Caption
{prompt}

# Metas
- bpm: {bpm}
- timesignature: {timesignature}
- keyscale: {keyscale}
- duration: {duration} seconds
<|endoftext|>
""".strip()
        text_hidden_states, text_attention_mask = self._encode_text(pipe, prompt, max_length=256)
        lyric_text = self.LYRIC_PROMPT.format("N/A", lyrics)
        lyric_hidden_states, lyric_attention_mask = self._encode_lyrics(pipe, lyric_text, max_length=2048)
        return {
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "lyric_hidden_states": lyric_hidden_states,
            "lyric_attention_mask": lyric_attention_mask,
        }


class DiffSynthMusic_ConditionEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={
                "text_hidden_states": "text_hidden_states",
                "text_attention_mask": "text_attention_mask",
                "lyric_hidden_states": "lyric_hidden_states",
                "lyric_attention_mask": "lyric_attention_mask"
            },
            input_params_nega={
                "text_hidden_states": "text_hidden_states",
                "text_attention_mask": "text_attention_mask",
                "lyric_hidden_states": "lyric_hidden_states",
                "lyric_attention_mask": "lyric_attention_mask"
            },
            output_params=("encoder_hidden_states",),
            onload_model_names=("conditioner",),
        )

    def process(self, pipe: DiffSynthMusicPipeline, text_hidden_states, text_attention_mask, lyric_hidden_states, lyric_attention_mask):
        pipe.load_models_to_device(["conditioner"])
        encoder_hidden_states, _ = pipe.conditioner(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            reference_latents=None,
            refer_audio_order_mask=None,
        )
        return {"encoder_hidden_states": encoder_hidden_states}


class DiffSynthMusic_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("duration", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: DiffSynthMusicPipeline, duration, seed, rand_device):
        noise = pipe.generate_noise((1, int(round(duration * 48000 / 1920 / 2)) * 2, 64), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}


class DiffSynthMusic_InputAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("noise", "input_audio"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def encode_audio(self, pipe: DiffSynthMusicPipeline, audio):
        if audio.shape[-1] == 64:
            return audio
        else:
            audio = torch.clamp(audio, -1.0, 1.0).unsqueeze(0)
            audio = pipe.vae.encode(audio.to(dtype=pipe.torch_dtype, device=pipe.device)).transpose(1, 2)
            return audio

    def process(self, pipe: DiffSynthMusicPipeline, noise, input_audio):
        if input_audio is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        input_latents = self.encode_audio(pipe, input_audio)
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {"latents": latents, "input_latents": input_latents}


def model_fn_ace_step(
    dit: DiffSynthMusicDiTModel,
    latents=None,
    timestep=None,
    encoder_hidden_states=None,
    kv_cache=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    decoder_outputs = dit(
        x=latents,
        timestep=timestep,
        y=encoder_hidden_states,
        kv_cache=kv_cache,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return decoder_outputs
