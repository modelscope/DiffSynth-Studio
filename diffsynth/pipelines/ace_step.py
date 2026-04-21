"""
ACE-Step Pipeline for DiffSynth-Studio.

Text-to-Music generation pipeline using ACE-Step 1.5 model.
"""
import re
import torch
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm

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
        self.tokenizer_model: AceStepTokenizer = None  # AceStepTokenizer (tokenizer + detokenizer)

        self.in_iteration_models = ("dit",)
        self.units = [
            AceStepUnit_PromptEmbedder(),
            AceStepUnit_ReferenceAudioEmbedder(),
            AceStepUnit_ConditionEmbedder(),
            AceStepUnit_AudioCodeDecoder(), 
            AceStepUnit_ContextLatentBuilder(),
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
        """Load pipeline from pretrained checkpoints."""
        pipe = AceStepPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("ace_step_text_encoder")
        pipe.conditioner = model_pool.fetch_model("ace_step_conditioner")
        pipe.dit = model_pool.fetch_model("ace_step_dit")
        pipe.vae = model_pool.fetch_model("ace_step_vae")
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
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        # Lyrics
        lyrics: str = "",
        # Reference audio
        reference_audios: List[torch.Tensor] = None,
        # Src audio
        src_audio: torch.Tensor = None,
        denoising_strength: float = 1.0,
        # Audio codes
        audio_code_string: Optional[str] = None,
        # Shape
        duration: int = 60,
        # Audio Meta
        bpm: Optional[int] = 100,
        keyscale: Optional[str] = "B minor",
        timesignature: Optional[str] = "4",
        vocal_language: Optional[str] = 'unknown',
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 8,
        # Scheduler-specific parameters
        shift: float = 1.0,
        # Progress
        progress_bar_cmd=tqdm,
    ):
        # 1. Scheduler
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, denoising_strength=1.0, shift=shift)

        # 2. 三字典输入
        inputs_posi = {"prompt": prompt, "positive": True}
        inputs_nega = {"positive": False}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "lyrics": lyrics,
            "reference_audios": reference_audios,
            "src_audio": src_audio,
            "audio_code_string": audio_code_string,
            "duration": duration,
            "bpm": bpm, "keyscale": keyscale, "timesignature": timesignature, "vocal_language": vocal_language,
            "seed": seed,
            "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "shift": shift,
        }

        # 3. Unit 链执行
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # 4. Denoise loop
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id,
            )
            inputs_shared["latents"] = self.step(
                self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared
            )

        # 5. VAE 解码
        self.load_models_to_device(['vae'])
        # DiT output is [B, T, 64] (channels-last), VAE expects [B, 64, T] (channels-first)
        latents = inputs_shared["latents"].transpose(1, 2)
        vae_output = self.vae.decode(latents)
        # VAE returns OobleckDecoderOutput with .sample attribute
        audio_output = vae_output.sample if hasattr(vae_output, 'sample') else vae_output
        audio_output = self.normalize_audio(audio_output, target_db=-1.0)
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


class AceStepUnit_TaskTypeChecker(PipelineUnit):
    """Check and compute sequence length from duration."""
    def __init__(self):
        super().__init__(
            input_params=("src_audio", "audio_code_string"),
            output_params=("task_type",),
        )

    def process(self, pipe, src_audio, audio_code_string):
        if audio_code_string is not None:
            print("audio_code_string detected, setting task_type to 'cover'")
            task_type = "cover"
        else:
            task_type = "text2music"
        return {"task_type": task_type}


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

        # TODO: remove this
        newtext = prompt + "\n\n" + lyric_text
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
        pipe.load_models_to_device(['vae'])
        if reference_audios is not None and len(reference_audios) > 0:
            # TODO: implement reference audio embedding using VAE encode, and generate refer_audio_order_mask
            pass
        else:
            reference_audios = [[torch.zeros(2, 30 * pipe.vae.sampling_rate).to(dtype=pipe.torch_dtype, device=pipe.device)]]
            reference_latents, refer_audio_order_mask = self.infer_refer_latent(pipe, reference_audios)
        return {"reference_latents": reference_latents, "refer_audio_order_mask": refer_audio_order_mask}

    def infer_refer_latent(self, pipe, refer_audioss: List[List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer packed reference-audio latents and order mask."""
        refer_audio_order_mask = []
        refer_audio_latents = []

        def _normalize_audio_2d(a: torch.Tensor) -> torch.Tensor:
            if not isinstance(a, torch.Tensor):
                raise TypeError(f"refer_audio must be a torch.Tensor, got {type(a)!r}")
            if a.dim() == 3 and a.shape[0] == 1:
                a = a.squeeze(0)
            if a.dim() == 1:
                a = a.unsqueeze(0)
            if a.dim() != 2:
                raise ValueError(f"refer_audio must be 1D/2D/3D(1,2,T); got shape={tuple(a.shape)}")
            if a.shape[0] == 1:
                a = torch.cat([a, a], dim=0)
            return a[:2]

        def _ensure_latent_3d(z: torch.Tensor) -> torch.Tensor:
            if z.dim() == 4 and z.shape[0] == 1:
                z = z.squeeze(0)
            if z.dim() == 2:
                z = z.unsqueeze(0)
            return z

        refer_encode_cache: Dict[int, torch.Tensor] = {}
        for batch_idx, refer_audios in enumerate(refer_audioss):
            if len(refer_audios) == 1 and torch.all(refer_audios[0] == 0.0):
                refer_audio_latent = _ensure_latent_3d(pipe.silence_latent[:, :750, :])
                refer_audio_latents.append(refer_audio_latent)
                refer_audio_order_mask.append(batch_idx)
            else:
                # TODO: check
                for refer_audio in refer_audios:
                    cache_key = refer_audio.data_ptr()
                    if cache_key in refer_encode_cache:
                        refer_audio_latent = refer_encode_cache[cache_key].clone()
                    else:
                        refer_audio = _normalize_audio_2d(refer_audio)
                        refer_audio_latent = pipe.vae.encode(refer_audio)
                        refer_audio_latent = refer_audio_latent.to(dtype=pipe.torch_dtype, device=pipe.device)
                        if refer_audio_latent.dim() == 2:
                            refer_audio_latent = refer_audio_latent.unsqueeze(0)
                        refer_audio_latent = _ensure_latent_3d(refer_audio_latent.transpose(1, 2))
                        refer_encode_cache[cache_key] = refer_audio_latent
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
        inputs_nega["encoder_hidden_states"] = pipe.conditioner.null_condition_emb.expand_as(encoder_hidden_states)
        inputs_nega["encoder_attention_mask"] = encoder_attention_mask
        return inputs_shared, inputs_posi, inputs_nega


class AceStepUnit_ContextLatentBuilder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("duration", "src_audio", "lm_hints"),
            output_params=("context_latents", "src_latents", "chunk_masks", "attention_mask"),
        )

    def _get_silence_latent_slice(self, pipe, length: int) -> torch.Tensor:
        available = pipe.silence_latent.shape[1]
        if length <= available:
            return pipe.silence_latent[0, :length, :]
        repeats = (length + available - 1) // available
        tiled = pipe.silence_latent[0].repeat(repeats, 1)
        return tiled[:length, :]

    def process(self, pipe, duration, src_audio, lm_hints):
        if lm_hints is not None:
            max_latent_length = lm_hints.shape[1]
            src_latents = lm_hints.clone()
            chunk_masks = torch.ones((1, max_latent_length, src_latents.shape[-1]), dtype=torch.bool, device=pipe.device)
            attention_mask = torch.ones((1, max_latent_length), device=src_latents.device, dtype=pipe.torch_dtype)
            context_latents = torch.cat([src_latents, chunk_masks], dim=-1)
        elif src_audio is not None:
            raise NotImplementedError("src_audio conditioning is not implemented yet. Please set lm_hints to None.")
        else:
            max_latent_length = duration * pipe.sample_rate  // 1920
            src_latents = self._get_silence_latent_slice(pipe, max_latent_length).unsqueeze(0)
            chunk_masks = torch.ones((1, max_latent_length, src_latents.shape[-1]), dtype=torch.bool, device=pipe.device)
            attention_mask = torch.ones((1, max_latent_length), device=src_latents.device, dtype=pipe.torch_dtype)
            context_latents = torch.cat([src_latents, chunk_masks], dim=-1)
        return {"context_latents": context_latents, "attention_mask": attention_mask}


class AceStepUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("context_latents", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe, context_latents, seed, rand_device):
        src_latents_shape = (context_latents.shape[0], context_latents.shape[1], context_latents.shape[-1] // 2)
        noise = pipe.generate_noise(src_latents_shape, seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}


class AceStepUnit_InputAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("noise", "input_audio"),
            output_params=("latents", "input_latents"),
        )

    def process(self, pipe, noise, input_audio):
        if input_audio is None:
            return {"latents": noise}
        # TODO: support for train
        return {"latents": noise, "input_latents": None}


class AceStepUnit_AudioCodeDecoder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("audio_code_string",),
            output_params=("lm_hints",),
            onload_model_names=("tokenizer_model",),
        )

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

    def process(self, pipe, audio_code_string):
        if audio_code_string is None or not audio_code_string.strip():
            return {"lm_hints": None}
        code_ids = self._parse_audio_code_string(audio_code_string)
        if len(code_ids) == 0:
            return {"lm_hints": None}

        pipe.load_models_to_device(["tokenizer_model"])
        indices = torch.tensor(code_ids, device=pipe.device, dtype=torch.long)
        indices = indices.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        quantized = pipe.tokenizer_model.tokenizer.quantizer.get_output_from_indices(indices).to(pipe.torch_dtype)  # [1, N, 2048]
        lm_hints = pipe.tokenizer_model.detokenizer(quantized)  # [1, N*5, 64]
        return {"lm_hints": lm_hints}


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
    timestep = timestep.unsqueeze(0)
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
