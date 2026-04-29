"""
ACE-Step Pipeline for DiffSynth-Studio.

Text-to-Music generation pipeline using ACE-Step 1.5 model.
"""
import re, torch
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
import random, math
import torch.nn.functional as F
from einops import rearrange

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.ace_step_dit import AceStepDiTModel
from ..models.ace_step_conditioner import AceStepConditionEncoder
from ..models.ace_step_text_encoder import AceStepTextEncoder
from ..models.ace_step_vae import AceStepVAE
from ..models.ace_step_tokenizer import AceStepTokenizer


class AceStepPipeline(BasePipeline):
    """Pipeline for ACE-Step text-to-music generation."""

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=1,
            width_division_factor=1,
        )
        self.scheduler = FlowMatchScheduler("ACE-Step")
        self.text_encoder: AceStepTextEncoder = None
        self.conditioner: AceStepConditionEncoder = None
        self.dit: AceStepDiTModel = None
        self.vae: AceStepVAE = None
        self.tokenizer_model: AceStepTokenizer = None

        self.in_iteration_models = ("dit",)
        self.units = [
            AceStepUnit_TaskTypeChecker(),
            AceStepUnit_PromptEmbedder(),
            AceStepUnit_ReferenceAudioEmbedder(),
            AceStepUnit_ContextLatentBuilder(),
            AceStepUnit_ConditionEmbedder(),
            AceStepUnit_NoiseInitializer(),
            AceStepUnit_InputAudioEmbedder(),
        ]
        self.model_fn = model_fn_ace_step
        self.compilable_models = ["dit"]

        self.sample_rate = 48000

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        text_tokenizer_config: ModelConfig = ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B/"),
        silence_latent_config: ModelConfig = ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="acestep-v15-turbo/silence_latent.pt"),
        vram_limit: float = None,
    ):
        pipe = AceStepPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("ace_step_text_encoder")
        pipe.conditioner = model_pool.fetch_model("ace_step_conditioner")
        pipe.dit = model_pool.fetch_model("ace_step_dit")
        pipe.vae = model_pool.fetch_model("ace_step_vae")
        pipe.vae.remove_weight_norm()
        pipe.tokenizer_model = model_pool.fetch_model("ace_step_tokenizer")

        if text_tokenizer_config is not None:
            text_tokenizer_config.download_if_necessary()
            from transformers import AutoTokenizer
            pipe.tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_config.path)
        if silence_latent_config is not None:
            silence_latent_config.download_if_necessary()
            pipe.silence_latent = torch.load(silence_latent_config.path, weights_only=True).transpose(1, 2).to(dtype=pipe.torch_dtype, device=pipe.device)

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        cfg_scale: float = 1.0,
        # Lyrics
        lyrics: str = "",
        # Task type
        task_type: Optional[str] = "text2music",
        # Reference audio
        reference_audios: List[torch.Tensor] = None,
        # Source audio
        src_audio: torch.Tensor = None,
        denoising_strength: float = 1.0,  # denoising_strength = 1 - cover_noise_strength
        audio_cover_strength: float = 1.0,
        # Audio codes
        audio_code_string: Optional[str] = None,
        # Inpainting
        repainting_ranges: Optional[List[Tuple[float, float]]] = None,
        repainting_strength: float = 1.0,
        # Shape
        duration: int = 60,
        # Audio Meta
        bpm: Optional[int] = 100,
        keyscale: Optional[str] = "B minor",
        timesignature: Optional[str] = "4",
        vocal_language: Optional[str] = "unknown",
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 8,
        # Scheduler-specific parameters
        shift: float = 3.0,
        # Progress
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, denoising_strength=denoising_strength, shift=shift)

        # Parameters
        inputs_posi = {"prompt": prompt, "positive": True}
        inputs_nega = {"positive": False}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "lyrics": lyrics,
            "task_type": task_type,
            "reference_audios": reference_audios,
            "src_audio": src_audio, "audio_cover_strength": audio_cover_strength, "audio_code_string": audio_code_string,
            "repainting_ranges": repainting_ranges, "repainting_strength": repainting_strength,
            "duration": duration,
            "bpm": bpm, "keyscale": keyscale, "timesignature": timesignature, "vocal_language": vocal_language,
            "seed": seed,
            "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "shift": shift,
        }

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            self.switch_noncover_condition(inputs_shared, inputs_posi, inputs_nega, progress_id)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id,
            )
            inputs_shared["latents"] = self.step(
                self.scheduler, inpaint_mask=inputs_shared.get("denoise_mask", None), input_latents=inputs_shared.get("src_latents", None),
                progress_id=progress_id, noise_pred=noise_pred, **inputs_shared,
            )

        # Decode
        self.load_models_to_device(['vae'])
        # DiT output is [B, T, 64] (channels-last), VAE expects [B, 64, T] (channels-first)
        latents = inputs_shared["latents"].transpose(1, 2)
        vae_output = self.vae.decode(latents)
        audio_output = self.normalize_audio(vae_output, target_db=-1.0)
        audio = self.output_audio_format_check(audio_output)
        self.load_models_to_device([])
        return audio

    def normalize_audio(self, audio: torch.Tensor, target_db: float = -1.0) -> torch.Tensor:
        peak = torch.max(torch.abs(audio))
        if peak < 1e-6:
            return audio
        target_amp = 10 ** (target_db / 20.0)
        gain = target_amp / peak
        return audio * gain

    def switch_noncover_condition(self, inputs_shared, inputs_posi, inputs_nega, progress_id):
        if inputs_shared["task_type"] != "cover" or inputs_shared["audio_cover_strength"] >= 1.0:
            return
        if inputs_shared.get("shared_noncover", None) is None:
            return
        cover_steps = int(len(self.scheduler.timesteps) * inputs_shared["audio_cover_strength"])
        if progress_id >= cover_steps:
            inputs_shared.update(inputs_shared.pop("shared_noncover", {}))
            inputs_posi.update(inputs_shared.pop("posi_noncover", {}))
            if inputs_shared["cfg_scale"] != 1.0:
                inputs_nega.update(inputs_shared.pop("nega_noncover", {}))


class AceStepUnit_TaskTypeChecker(PipelineUnit):
    """Check and compute sequence length from duration."""
    def __init__(self):
        super().__init__(
            input_params=("task_type", "src_audio", "repainting_ranges", "audio_code_string"),
            output_params=("task_type",),
        )

    def process(self, pipe, task_type, src_audio, repainting_ranges, audio_code_string):
        assert task_type in ["text2music", "cover", "repaint"], f"Unsupported task_type: {task_type}"
        if task_type == "cover":
            assert (src_audio is not None) or (audio_code_string is not None), "For cover task, either src_audio or audio_code_string must be provided."
        elif task_type == "repaint":
            assert src_audio is not None, "For repaint task, src_audio must be provided."
            assert repainting_ranges is not None and len(repainting_ranges) > 0, "For repaint task, inpainting_ranges must be provided and non-empty."
        return {}


class AceStepUnit_PromptEmbedder(PipelineUnit):
    SFT_GEN_PROMPT = "# Instruction\n{}\n\n# Caption\n{}\n\n# Metas\n{}<|endoftext|>\n"
    INSTRUCTION_MAP = {
        "text2music": "Fill the audio semantic mask based on the given conditions:",
        "cover": "Generate audio semantic tokens based on the given conditions:",
        "repaint": "Repaint the mask area based on the given conditions:",
        "extract": "Extract the {TRACK_NAME} track from the audio:",
        "extract_default": "Extract the track from the audio:",
        "lego": "Generate the {TRACK_NAME} track based on the audio context:",
        "lego_default": "Generate the track based on the audio context:",
        "complete": "Complete the input track with {TRACK_CLASSES}:",
        "complete_default": "Complete the input track:",
    }
    LYRIC_PROMPT = "# Languages\n{}\n\n# Lyric\n{}<|endoftext|>"

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "prompt", "positive": "positive"},
            input_params=("lyrics", "duration", "bpm", "keyscale", "timesignature", "vocal_language", "task_type"),
            output_params=("text_hidden_states", "text_attention_mask", "lyric_hidden_states", "lyric_attention_mask"),
            onload_model_names=("text_encoder",)
        )

    def _encode_text(self, pipe, text, max_length=256):
        """Encode text using Qwen3-Embedding → [B, T, 1024]."""
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

    def _encode_lyrics(self, pipe, lyric_text, max_length=2048):
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

    def _dict_to_meta_string(self, meta_dict: Dict[str, Any]) -> str:
        bpm = meta_dict.get("bpm", "N/A")
        timesignature = meta_dict.get("timesignature", "N/A")
        keyscale = meta_dict.get("keyscale", "N/A")
        duration = meta_dict.get("duration", 30)
        duration = f"{int(duration)} seconds"
        return (
            f"- bpm: {bpm}\n"
            f"- timesignature: {timesignature}\n"
            f"- keyscale: {keyscale}\n"
            f"- duration: {duration}\n"
        )

    def process(self, pipe, prompt, positive, lyrics, duration, bpm, keyscale, timesignature, vocal_language, task_type):
        if not positive:
            return {}
        pipe.load_models_to_device(['text_encoder'])
        meta_dict = {"bpm": bpm, "keyscale": keyscale, "timesignature": timesignature, "duration": duration}
        INSTRUCTION = self.INSTRUCTION_MAP.get(task_type, self.INSTRUCTION_MAP["text2music"])
        prompt = self.SFT_GEN_PROMPT.format(INSTRUCTION, prompt, self._dict_to_meta_string(meta_dict))
        text_hidden_states, text_attention_mask = self._encode_text(pipe, prompt, max_length=256)

        lyric_text = self.LYRIC_PROMPT.format(vocal_language, lyrics)
        lyric_hidden_states, lyric_attention_mask = self._encode_lyrics(pipe, lyric_text, max_length=2048)

        return {
            "text_hidden_states": text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "lyric_hidden_states": lyric_hidden_states,
            "lyric_attention_mask": lyric_attention_mask,
        }


class AceStepUnit_ReferenceAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_audios",),
            output_params=("reference_latents", "refer_audio_order_mask"),
            onload_model_names=("vae",)
        )

    def process(self, pipe, reference_audios):
        if reference_audios is not None:
            pipe.load_models_to_device(['vae'])
            reference_audios = [
                self.process_reference_audio(reference_audio).to(dtype=pipe.torch_dtype, device=pipe.device)
                for reference_audio in reference_audios
            ]
            reference_latents, refer_audio_order_mask = self.infer_refer_latent(pipe, [reference_audios])
        else:
            reference_audios = [[torch.zeros(2, 30 * pipe.vae.sampling_rate).to(dtype=pipe.torch_dtype, device=pipe.device)]]
            reference_latents, refer_audio_order_mask = self.infer_refer_latent(pipe, reference_audios)
        return {"reference_latents": reference_latents, "refer_audio_order_mask": refer_audio_order_mask}

    def process_reference_audio(self, audio) -> Optional[torch.Tensor]:
        if audio.ndim == 3 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        target_frames = 30 * 48000
        segment_frames = 10 * 48000
        if audio.shape[-1] < target_frames:
            repeat_times = math.ceil(target_frames / audio.shape[-1])
            audio = audio.repeat(1, repeat_times)
        total_frames = audio.shape[-1]
        segment_size = total_frames // 3
        front_start = random.randint(0, max(0, segment_size - segment_frames))
        front_audio = audio[:, front_start:front_start + segment_frames]
        middle_start = segment_size + random.randint(0, max(0, segment_size - segment_frames))
        middle_audio = audio[:, middle_start:middle_start + segment_frames]
        back_start = 2 * segment_size + random.randint(0, max(0, (total_frames - 2 * segment_size) - segment_frames))
        back_audio = audio[:, back_start:back_start + segment_frames]
        return torch.cat([front_audio, middle_audio, back_audio], dim=-1).unsqueeze(0)

    def infer_refer_latent(self, pipe, refer_audioss: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer packed reference-audio latents and order mask."""
        refer_audio_order_mask = []
        refer_audio_latents = []
        for batch_idx, refer_audios in enumerate(refer_audioss):
            if len(refer_audios) == 1 and torch.all(refer_audios[0] == 0.0):
                refer_audio_latent = pipe.silence_latent[:, :750, :]
                refer_audio_latents.append(refer_audio_latent)
                refer_audio_order_mask.append(batch_idx)
            else:
                for refer_audio in refer_audios:
                    refer_audio_latent = pipe.vae.encode(refer_audio).transpose(1, 2).to(dtype=pipe.torch_dtype, device=pipe.device)
                    refer_audio_latents.append(refer_audio_latent)
                    refer_audio_order_mask.append(batch_idx)
        refer_audio_latents = torch.cat(refer_audio_latents, dim=0)
        refer_audio_order_mask = torch.tensor(refer_audio_order_mask, device=pipe.device, dtype=torch.long)
        return refer_audio_latents, refer_audio_order_mask


class AceStepUnit_ConditionEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            output_params=("encoder_hidden_states", "encoder_attention_mask"),
            onload_model_names=("conditioner",),
        )

    def process(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        pipe.load_models_to_device(['conditioner'])
        encoder_hidden_states, encoder_attention_mask = pipe.conditioner(
            text_hidden_states=inputs_posi.get("text_hidden_states", None),
            text_attention_mask=inputs_posi.get("text_attention_mask", None),
            lyric_hidden_states=inputs_posi.get("lyric_hidden_states", None),
            lyric_attention_mask=inputs_posi.get("lyric_attention_mask", None),
            reference_latents=inputs_shared.get("reference_latents", None),
            refer_audio_order_mask=inputs_shared.get("refer_audio_order_mask", None),
        )
        inputs_posi["encoder_hidden_states"] = encoder_hidden_states
        inputs_posi["encoder_attention_mask"] = encoder_attention_mask
        if inputs_shared["cfg_scale"] != 1.0:
            inputs_nega["encoder_hidden_states"] = pipe.conditioner.null_condition_emb.expand_as(encoder_hidden_states).to(
                dtype=encoder_hidden_states.dtype, device=encoder_hidden_states.device,
            )
            inputs_nega["encoder_attention_mask"] = encoder_attention_mask
        if inputs_shared["task_type"] == "cover" and inputs_shared["audio_cover_strength"] < 1.0:
            hidden_states_noncover = AceStepUnit_PromptEmbedder().process(
                pipe, inputs_posi["prompt"], True, inputs_shared["lyrics"], inputs_shared["duration"],
                inputs_shared["bpm"], inputs_shared["keyscale"], inputs_shared["timesignature"],
                inputs_shared["vocal_language"], "text2music")
            encoder_hidden_states_noncover, encoder_attention_mask_noncover = pipe.conditioner(
                **hidden_states_noncover,
                reference_latents=inputs_shared.get("reference_latents", None),
                refer_audio_order_mask=inputs_shared.get("refer_audio_order_mask", None),
            )
            duration = inputs_shared["context_latents"].shape[1] * 1920 / pipe.vae.sampling_rate
            context_latents_noncover = AceStepUnit_ContextLatentBuilder().process(pipe, duration, None, None)["context_latents"]
            inputs_shared["shared_noncover"] = {"context_latents": context_latents_noncover}
            inputs_shared["posi_noncover"] = {"encoder_hidden_states": encoder_hidden_states_noncover, "encoder_attention_mask": encoder_attention_mask_noncover}
            if inputs_shared["cfg_scale"] != 1.0:
                inputs_shared["nega_noncover"] = {
                    "encoder_hidden_states": pipe.conditioner.null_condition_emb.expand_as(encoder_hidden_states_noncover).to(
                        dtype=encoder_hidden_states_noncover.dtype, device=encoder_hidden_states_noncover.device,
                    ),
                    "encoder_attention_mask": encoder_attention_mask_noncover,
                }
        return inputs_shared, inputs_posi, inputs_nega


class AceStepUnit_ContextLatentBuilder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("duration", "src_audio", "audio_code_string", "task_type", "repainting_ranges", "repainting_strength"),
            output_params=("context_latents", "src_latents", "chunk_masks", "attention_mask"),
            onload_model_names=("vae", "tokenizer_model",),
        )

    def _get_silence_latent_slice(self, pipe, length: int) -> torch.Tensor:
        available = pipe.silence_latent.shape[1]
        if length <= available:
            return pipe.silence_latent[0, :length, :]
        repeats = (length + available - 1) // available
        tiled = pipe.silence_latent[0].repeat(repeats, 1)
        return tiled[:length, :]

    def tokenize(self, tokenizer, x, silence_latent, pool_window_size):
        if x.shape[1] % pool_window_size != 0:
            pad_len = pool_window_size - (x.shape[1] % pool_window_size)
            x = torch.cat([x,  silence_latent[:1,:pad_len].repeat(x.shape[0],1,1)], dim=1)
        x = rearrange(x, 'n (t_patch p) d -> n t_patch p d', p=pool_window_size)
        quantized, indices = tokenizer(x)
        return quantized

    @staticmethod
    def _parse_audio_code_string(code_str: str) -> list:
        """Extract integer audio codes from tokens like <|audio_code_123|>."""
        if not code_str:
            return []
        try:
            codes = []
            max_audio_code = 63999
            for x in re.findall(r"<\|audio_code_(\d+)\|>", code_str):
                code_value = int(x)
                codes.append(max(0, min(code_value, max_audio_code)))
        except Exception as e:
            raise ValueError(f"Invalid audio_code_string format: {e}")
        return codes

    def pad_src_audio(self, pipe, src_audio, task_type, repainting_ranges):
        if task_type != "repaint" or repainting_ranges is None:
            return src_audio, repainting_ranges, None, None
        min_left = min([start for start, end in repainting_ranges])
        max_right = max([end for start, end in repainting_ranges])
        pad_left = max(0, -min_left)
        padding_frames_left = int(pad_left * pipe.vae.sampling_rate)
        padding_frames_right = max(int(max_right * pipe.vae.sampling_rate) - src_audio.shape[-1], 0)
        if padding_frames_left > 0 or padding_frames_right > 0:
            src_audio = F.pad(src_audio, (padding_frames_left, padding_frames_right), value=0.0)
        repainting_ranges = [(start + pad_left, end + pad_left) for start, end in repainting_ranges]
        return src_audio, repainting_ranges, padding_frames_left, padding_frames_right

    def parse_repaint_masks(self, pipe, src_latents, task_type, repainting_ranges, repainting_strength, padding_frames_left, padding_frames_right):
        if task_type != "repaint" or repainting_ranges is None:
            return None, src_latents
        # let repainting area be repainting_strength, non-repainting area be 0.0, and blend at the boundary with cf_frames.
        max_latent_length = src_latents.shape[1]
        denoise_mask = torch.zeros((1, max_latent_length, 1), dtype=pipe.torch_dtype, device=pipe.device)
        for start, end in repainting_ranges:
            start_frame = int(start * pipe.vae.sampling_rate / 1920)
            end_frame = int(end * pipe.vae.sampling_rate / 1920)
            denoise_mask[:, start_frame:end_frame, :] = repainting_strength
        # set padding areas to 1.0 (full repaint) to avoid artifacts at the boundaries caused by padding
        pad_left_frames =  int(padding_frames_left / 1920)
        pad_right_frames = int(padding_frames_right / 1920)
        denoise_mask[:, :pad_left_frames, :] = 1
        denoise_mask[:, max_latent_length - pad_right_frames:, :] = 1

        silent_latents = self._get_silence_latent_slice(pipe, max_latent_length).unsqueeze(0)
        src_latents = src_latents * (1 - denoise_mask) + silent_latents * denoise_mask
        return denoise_mask, src_latents

    def process(self, pipe, duration, src_audio, audio_code_string, task_type=None, repainting_ranges=None, repainting_strength=None):
        # get src_latents from audio_code_string > src_audio > silence
        source_latents = None
        denoise_mask = None
        if audio_code_string is not None:
            # use audio_cede_string to get src_latents.
            pipe.load_models_to_device(self.onload_model_names)
            code_ids = self._parse_audio_code_string(audio_code_string)
            quantizer = pipe.tokenizer_model.tokenizer.quantizer.to(device=pipe.device)
            indices = torch.tensor(code_ids, device=quantizer.codebooks.device, dtype=torch.long).unsqueeze(0).unsqueeze(-1)
            codes = quantizer.get_codes_from_indices(indices)
            quantized = codes.sum(dim=0).to(pipe.torch_dtype).to(pipe.device)
            quantized = quantizer.project_out(quantized)
            src_latents = pipe.tokenizer_model.detokenizer(quantized).to(pipe.device)
            max_latent_length = src_latents.shape[1]
        elif src_audio is not None:
            # use src_audio to get src_latents.
            pipe.load_models_to_device(self.onload_model_names)
            src_audio = src_audio.unsqueeze(0) if src_audio.dim() == 2 else src_audio
            src_audio = torch.clamp(src_audio, -1.0, 1.0)

            src_audio, repainting_ranges, pad_left, pad_right = self.pad_src_audio(pipe, src_audio, task_type, repainting_ranges)

            src_latents = pipe.vae.encode(src_audio.to(dtype=pipe.torch_dtype, device=pipe.device)).transpose(1, 2)
            source_latents = src_latents # cache for potential use in audio inpainting tasks
            denoise_mask, src_latents = self.parse_repaint_masks(pipe, src_latents, task_type, repainting_ranges, repainting_strength, pad_left, pad_right)
            if task_type == "cover":
                lm_hints_5Hz = self.tokenize(pipe.tokenizer_model.tokenizer, src_latents, pipe.silence_latent, pipe.tokenizer_model.tokenizer.pool_window_size)
                src_latents = pipe.tokenizer_model.detokenizer(lm_hints_5Hz)
                if src_latents.shape[1] > source_latents.shape[1]:
                    source_latents = torch.cat([source_latents, src_latents[:, source_latents.shape[1]:]], dim=1)
            max_latent_length = src_latents.shape[1]
        else:
            # use silence latents.
            max_latent_length = round(duration * pipe.sample_rate  / 1920)
            src_latents = self._get_silence_latent_slice(pipe, max_latent_length).unsqueeze(0)
        chunk_masks = torch.ones((1, max_latent_length, src_latents.shape[-1]), dtype=torch.bool, device=pipe.device)
        attention_mask = torch.ones((1, max_latent_length), device=src_latents.device, dtype=pipe.torch_dtype)
        context_latents = torch.cat([src_latents, chunk_masks], dim=-1)
        return {"context_latents": context_latents, "attention_mask": attention_mask, "src_latents": source_latents, "denoise_mask": denoise_mask}


class AceStepUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("context_latents", "seed", "rand_device", "src_latents"),
            output_params=("noise",),
        )

    def process(self, pipe, context_latents, seed, rand_device, src_latents):
        src_latents_shape = (context_latents.shape[0], context_latents.shape[1], context_latents.shape[-1] // 2)
        noise = pipe.generate_noise(src_latents_shape, seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        if src_latents is not None:
            noise = pipe.scheduler.add_noise(src_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {"noise": noise}


class AceStepUnit_InputAudioEmbedder(PipelineUnit):
    """Only for training."""
    def __init__(self):
        super().__init__(
            input_params=("noise", "input_audio"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, noise, input_audio):
        if input_audio is None:
            return {"latents": noise}
        if pipe.scheduler.training:
            pipe.load_models_to_device(self.onload_model_names)
            input_audio, sample_rate = input_audio
            input_audio = torch.clamp(input_audio, -1.0, 1.0)
            if input_audio.dim() == 2:
                input_audio = input_audio.unsqueeze(0)
            input_latents = pipe.vae.encode(input_audio.to(dtype=pipe.torch_dtype, device=pipe.device)).transpose(1, 2)
            # prevent potential size mismatch between context_latents and input_latents by cropping input_latents to the same temporal length as noise
            input_latents = input_latents[:, :noise.shape[1]]
            return {"input_latents": input_latents}


def model_fn_ace_step(
    dit: AceStepDiTModel,
    latents=None,
    timestep=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    context_latents=None,
    attention_mask=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    decoder_outputs = dit(
        hidden_states=latents,
        timestep=timestep,
        timestep_r=timestep,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        context_latents=context_latents,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )[0]
    return decoder_outputs
