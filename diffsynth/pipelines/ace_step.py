"""
ACE-Step Pipeline for DiffSynth-Studio.

Text-to-Music generation pipeline using ACE-Step 1.5 model.
"""
import torch
from typing import Optional
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.ace_step_dit import AceStepDiTModel
from ..models.ace_step_conditioner import AceStepConditionEncoder
from ..models.ace_step_text_encoder import AceStepTextEncoder
from ..models.ace_step_vae import AceStepVAE


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
        self.vae = None  # AutoencoderOobleck (diffusers) or AceStepVAE

        # Unit chain order — 7 units total
        #
        # 1. ShapeChecker: duration → seq_len
        # 2. PromptEmbedder: prompt/lyrics → text/lyric embeddings (shared for CFG)
        # 3. SilenceLatentInitializer: seq_len → src_latents + chunk_masks
        # 4. ContextLatentBuilder: src_latents + chunk_masks → context_latents (shared, same for CFG+)
        # 5. ConditionEmbedder: text/lyric → encoder_hidden_states (separate for CFG+/-)
        # 6. NoiseInitializer: context_latents → noise
        # 7. InputAudioEmbedder: noise → latents
        #
        # ContextLatentBuilder runs before ConditionEmbedder so that
        # context_latents is available for noise shape computation.
        self.in_iteration_models = ("dit",)
        self.units = [
            AceStepUnit_ShapeChecker(),
            AceStepUnit_PromptEmbedder(),
            AceStepUnit_SilenceLatentInitializer(),
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
        text_tokenizer_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        """Load pipeline from pretrained checkpoints."""
        pipe = AceStepPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("ace_step_text_encoder")
        pipe.conditioner = model_pool.fetch_model("ace_step_conditioner")
        pipe.dit = model_pool.fetch_model("ace_step_dit")
        pipe.vae = model_pool.fetch_model("ace_step_vae")

        if text_tokenizer_config is not None:
            text_tokenizer_config.download_if_necessary()
            from transformers import AutoTokenizer
            pipe.tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_config.path)

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
        # Reference audio (optional, for timbre conditioning)
        reference_audio = None,
        # Shape
        duration: float = 60.0,
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
        # 1. Scheduler
        self.scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            denoising_strength=1.0,
            shift=shift,
        )

        # 2. 三字典输入
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "lyrics": lyrics,
            "reference_audio": reference_audio,
            "duration": duration,
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
                **models, timestep=timestep, progress_id=progress_id
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
        audio = self.output_audio_format_check(audio_output)
        self.load_models_to_device([])
        return audio

    def output_audio_format_check(self, audio_output):
        """Convert VAE output to standard audio format [C, T], float32.

        VAE decode outputs [B, C, T] (audio waveform).
        We squeeze batch dim and return [C, T].
        """
        if audio_output.ndim == 3:
            audio_output = audio_output.squeeze(0)
        return audio_output.float()


class AceStepUnit_ShapeChecker(PipelineUnit):
    """Check and compute sequence length from duration."""
    def __init__(self):
        super().__init__(
            input_params=("duration",),
            output_params=("duration", "seq_len"),
        )

    def process(self, pipe, duration):
        # ACE-Step: 25 Hz latent rate
        seq_len = int(duration * 25)
        return {"duration": duration, "seq_len": seq_len}


class AceStepUnit_PromptEmbedder(PipelineUnit):
    """Encode prompt and lyrics using Qwen3-Embedding.

    Uses seperate_cfg=True to read prompt from inputs_posi (not inputs_shared).
    The negative condition uses null_condition_emb (handled by ConditionEmbedder),
    so negative text encoding is not needed here.
    """
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={},
            input_params=("lyrics",),
            output_params=("text_hidden_states", "text_attention_mask", "lyric_hidden_states", "lyric_attention_mask"),
            onload_model_names=("text_encoder",)
        )

    def _encode_text(self, pipe, text):
        """Encode text using Qwen3-Embedding → [B, T, 1024]."""
        if pipe.tokenizer is None:
            return None, None
        text_inputs = pipe.tokenizer(
            text,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(pipe.device)
        attention_mask = text_inputs.attention_mask.to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids, attention_mask)
        return hidden_states, attention_mask

    def process(self, pipe, prompt, lyrics, negative_prompt=None):
        pipe.load_models_to_device(['text_encoder'])

        text_hidden_states, text_attention_mask = self._encode_text(pipe, prompt)

        # Lyrics encoding — use empty string if not provided
        lyric_text = lyrics if lyrics else ""
        lyric_hidden_states, lyric_attention_mask = self._encode_text(pipe, lyric_text)

        if text_hidden_states is not None and lyric_hidden_states is not None:
            return {
                "text_hidden_states": text_hidden_states,
                "text_attention_mask": text_attention_mask,
                "lyric_hidden_states": lyric_hidden_states,
                "lyric_attention_mask": lyric_attention_mask,
            }
        return {}


class AceStepUnit_SilenceLatentInitializer(PipelineUnit):
    """Generate silence latent (all zeros) and chunk_masks for text2music.

    Target library reference: `prepare_condition()` line 1698-1699:
        context_latents = torch.cat([src_latents, chunk_masks.to(dtype)], dim=-1)

    For text2music mode:
    - src_latents = zeros [B, T, 64] (VAE latent dimension)
    - chunk_masks = ones [B, T, 64] (full visibility mask for text2music)
    - context_latents = [B, T, 128] (concat of src_latents + chunk_masks)
    """
    def __init__(self):
        super().__init__(
            input_params=("seq_len",),
            output_params=("silence_latent", "src_latents", "chunk_masks"),
        )

    def process(self, pipe, seq_len):
        # silence_latent shape: [B, T, 64] — 64 is the VAE latent dimension
        silence_latent = torch.zeros(1, seq_len, 64, device=pipe.device, dtype=pipe.torch_dtype)
        # For text2music: src_latents = silence_latent
        src_latents = silence_latent.clone()

        # chunk_masks: [B, T, 64] of ones (same shape as src_latents)
        # In text2music mode (is_covers=0), chunk_masks are all 1.0
        # This matches the target library's behavior at line 1699
        chunk_masks = torch.ones(1, seq_len, 64, device=pipe.device, dtype=pipe.torch_dtype)

        return {"silence_latent": silence_latent, "src_latents": src_latents, "chunk_masks": chunk_masks}


class AceStepUnit_ContextLatentBuilder(PipelineUnit):
    """Build context_latents from src_latents and chunk_masks.

    Target library reference: `prepare_condition()` line 1699:
        context_latents = torch.cat([src_latents, chunk_masks.to(dtype)], dim=-1)

    context_latents is the SAME for positive and negative CFG paths
    (it comes from src_latents + chunk_masks, not from text encoding).
    So this is a普通模式 Unit — outputs go to inputs_shared.
    """
    def __init__(self):
        super().__init__(
            input_params=("src_latents", "chunk_masks"),
            output_params=("context_latents", "attention_mask"),
        )

    def process(self, pipe, src_latents, chunk_masks):
        # context_latents: cat([src_latents, chunk_masks], dim=-1) → [B, T, 128]
        context_latents = torch.cat([src_latents, chunk_masks], dim=-1)

        # attention_mask for the DiT: ones [B, T]
        # The target library uses this for cross-attention with context_latents
        attention_mask = torch.ones(src_latents.shape[0], src_latents.shape[1],
                                     device=pipe.device, dtype=pipe.torch_dtype)

        return {"context_latents": context_latents, "attention_mask": attention_mask}


class AceStepUnit_ConditionEmbedder(PipelineUnit):
    """Generate encoder_hidden_states via ACEStepConditioner.

    Target library reference: `prepare_condition()` line 1674-1681:
        encoder_hidden_states, encoder_attention_mask = self.encoder(...)

    Uses seperate_cfg mode:
    - Positive: encode with full condition (text + lyrics + reference audio)
    - Negative: replace text with null_condition_emb, keep lyrics/timbre same

    context_latents is handled by ContextLatentBuilder (普通模式), not here.
    """
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={
                "text_hidden_states": "text_hidden_states",
                "text_attention_mask": "text_attention_mask",
                "lyric_hidden_states": "lyric_hidden_states",
                "lyric_attention_mask": "lyric_attention_mask",
                "reference_audio": "reference_audio",
                "refer_audio_order_mask": "refer_audio_order_mask",
            },
            input_params_nega={},
            input_params=("cfg_scale",),
            output_params=(
                "encoder_hidden_states", "encoder_attention_mask",
                "negative_encoder_hidden_states", "negative_encoder_attention_mask",
            ),
            onload_model_names=("conditioner",)
        )

    def _prepare_condition(self, pipe, text_hidden_states, text_attention_mask,
                           lyric_hidden_states, lyric_attention_mask,
                           refer_audio_acoustic_hidden_states_packed=None,
                           refer_audio_order_mask=None):
        """Call ACEStepConditioner forward to produce encoder_hidden_states."""
        pipe.load_models_to_device(['conditioner'])

        # Handle reference audio
        if refer_audio_acoustic_hidden_states_packed is None:
            # No reference audio: create 2D packed zeros [N=1, d=64]
            # TimbreEncoder.unpack expects [N, d], not [B, T, d]
            refer_audio_acoustic_hidden_states_packed = torch.zeros(
                1, 64, device=pipe.device, dtype=pipe.torch_dtype
            )
            refer_audio_order_mask = torch.LongTensor([0]).to(pipe.device)

        encoder_hidden_states, encoder_attention_mask = pipe.conditioner(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        return encoder_hidden_states, encoder_attention_mask

    def _prepare_negative_condition(self, pipe, lyric_hidden_states, lyric_attention_mask,
                                    refer_audio_acoustic_hidden_states_packed=None,
                                    refer_audio_order_mask=None):
        """Generate negative condition using null_condition_emb."""
        if pipe.conditioner is None or not hasattr(pipe.conditioner, 'null_condition_emb'):
            return None, None

        null_emb = pipe.conditioner.null_condition_emb  # [1, 1, hidden_size]
        bsz = 1
        if lyric_hidden_states is not None:
            bsz = lyric_hidden_states.shape[0]
        null_hidden_states = null_emb.expand(bsz, -1, -1)
        null_attn_mask = torch.ones(bsz, 1, device=pipe.device, dtype=pipe.torch_dtype)

        # For negative: use null_condition_emb as text, keep lyrics and timbre
        neg_encoder_hidden_states, neg_encoder_attention_mask = pipe.conditioner(
            text_hidden_states=null_hidden_states,
            text_attention_mask=null_attn_mask,
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyric_attention_mask,
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
        )

        return neg_encoder_hidden_states, neg_encoder_attention_mask

    def process(self, pipe, text_hidden_states, text_attention_mask,
                lyric_hidden_states, lyric_attention_mask,
                reference_audio=None, refer_audio_order_mask=None,
                negative_prompt=None, cfg_scale=1.0):

        # Positive condition
        pos_enc_hs, pos_enc_mask = self._prepare_condition(
            pipe, text_hidden_states, text_attention_mask,
            lyric_hidden_states, lyric_attention_mask,
            None, refer_audio_order_mask,
        )

        # Negative condition: only needed when CFG is active (cfg_scale > 1.0)
        # For cfg_scale=1.0 (turbo), skip to avoid null_condition_emb dimension mismatch
        result = {
            "encoder_hidden_states": pos_enc_hs,
            "encoder_attention_mask": pos_enc_mask,
        }

        if cfg_scale > 1.0:
            neg_enc_hs, neg_enc_mask = self._prepare_negative_condition(
                pipe, lyric_hidden_states, lyric_attention_mask,
                None, refer_audio_order_mask,
            )
            if neg_enc_hs is not None:
                result["negative_encoder_hidden_states"] = neg_enc_hs
                result["negative_encoder_attention_mask"] = neg_enc_mask

        return result


class AceStepUnit_NoiseInitializer(PipelineUnit):
    """Generate initial noise tensor.

    Target library reference: `prepare_noise()` line 1781-1818:
        src_latents_shape = (bsz, context_latents.shape[1], context_latents.shape[-1] // 2)

    Noise shape = [B, T, context_latents.shape[-1] // 2] = [B, T, 128 // 2] = [B, T, 64]
    """
    def __init__(self):
        super().__init__(
            input_params=("seed", "seq_len", "rand_device", "context_latents"),
            output_params=("noise",),
        )

    def process(self, pipe, seed, seq_len, rand_device, context_latents):
        # Noise shape: [B, T, context_latents.shape[-1] // 2]
        # context_latents = [B, T, 128] → noise = [B, T, 64]
        # This matches the target library's prepare_noise() at line 1796
        noise_shape = (context_latents.shape[0], context_latents.shape[1],
                       context_latents.shape[-1] // 2)
        noise = pipe.generate_noise(
            noise_shape,
            seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype
        )
        return {"noise": noise}


class AceStepUnit_InputAudioEmbedder(PipelineUnit):
    """Set up latents for denoise loop.

    For text2music (no input audio): latents = noise, input_latents = None.

    Target library reference: `generate_audio()` line 1972:
        xt = noise  (when cover_noise_strength == 0)
    """
    def __init__(self):
        super().__init__(
            input_params=("noise",),
            output_params=("latents", "input_latents"),
        )

    def process(self, pipe, noise):
        # For text2music: start from pure noise
        return {"latents": noise, "input_latents": None}


def model_fn_ace_step(
    dit: AceStepDiTModel,
    latents=None,
    timestep=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    context_latents=None,
    attention_mask=None,
    past_key_values=None,
    negative_encoder_hidden_states=None,
    negative_encoder_attention_mask=None,
    negative_context_latents=None,
    **kwargs,
):
    """Model function for ACE-Step DiT forward.

    Timestep is already in [0, 1] range — no scaling needed.

    Target library reference: `generate_audio()` line 2009-2020:
        decoder_outputs = self.decoder(
            hidden_states=x, timestep=t_curr_tensor, timestep_r=t_curr_tensor,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
            use_cache=True, past_key_values=past_key_values,
        )

    Args:
        dit: AceStepDiTModel
        latents: [B, T, 64] noise/latent tensor (same shape as src_latents)
        timestep: scalar tensor in [0, 1]
        encoder_hidden_states: [B, T_text, 2048] condition from Conditioner
            (positive or negative depending on CFG pass — the cfg_guided_model_fn
            passes inputs_posi for positive, inputs_nega for negative)
        encoder_attention_mask: [B, T_text]
        context_latents: [B, T, 128] = cat([src_latents, chunk_masks], dim=-1)
            (same for both CFG+/- paths in text2music mode)
        attention_mask: [B, T] ones mask for DiT
        past_key_values: EncoderDecoderCache for KV caching

    The DiT internally concatenates: cat([context_latents, latents], dim=-1) = [B, T, 192]
    as the actual input (128 + 64 = 192 channels).
    """
    # ACE-Step uses timestep directly in [0, 1] range — no /1000 scaling
    timestep = timestep.squeeze()

    # Expand timestep to match batch size
    bsz = latents.shape[0]
    timestep = timestep.expand(bsz)

    decoder_outputs = dit(
        hidden_states=latents,
        timestep=timestep,
        timestep_r=timestep,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        context_latents=context_latents,
        use_cache=True,
        past_key_values=past_key_values,
    )

    # Return velocity prediction (first element of decoder_outputs)
    return decoder_outputs[0]
