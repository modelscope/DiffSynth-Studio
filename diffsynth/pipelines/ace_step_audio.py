import torch, math
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import numpy as np
from math import prod
from transformers import AutoTokenizer

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput
from ..utils.lora.merge import merge_lora

from ..core.device.npu_compatible_device import get_device_type
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline
from ..models.ace_step_text_encoder import AceStepTextEncoder
from ..models.ace_step_vae import AceStepVAE
from ..models.ace_step_dit import AceStepConditionGenerationModelWrapper


class AceStepAudioPipeline(BasePipeline):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.text_encoder: AceStepTextEncoder = None
        self.dit: AceStepConditionGenerationModelWrapper = None
        self.vae: AceStepVAE = None

        self.scheduler = FlowMatchScheduler()
        self.tokenizer: AutoTokenizer = None
        self.in_iteration_models = ("dit",)
        self.units = []

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="ACE-Step/Ace-Step1.5", origin_file_pattern="Qwen3-Embedding-0.6B"),
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = AceStepAudioPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("ace_step_text_encoder")
        pipe.dit = model_pool.fetch_model("ace_step_dit")
        pipe.vae = model_pool.fetch_model("ace_step_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        caption: str,
        lyrics: str = "",
        duration: float = 160,
        bpm: int = None,
        keyscale: str = "",
        timesignature: str = "",
        vocal_language: str = "zh",
        instrumental: bool = False,
        inference_steps: int = 8,
        guidance_scale: float = 3.0,
        seed: int = None,
    ):
        # Format text prompt with metadata
        text_prompt = self._format_text_prompt(caption, bpm, keyscale, timesignature, duration)
        lyrics_text = self._format_lyrics(lyrics, vocal_language, instrumental)

        # Tokenize
        text_inputs = self.tokenizer(
            text_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        lyrics_inputs = self.tokenizer(
            lyrics_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Encode text and lyrics
        text_outputs = self.text_encoder(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )

        lyrics_outputs = self.text_encoder(
            input_ids=lyrics_inputs["input_ids"],
            attention_mask=lyrics_inputs["attention_mask"],
        )

        # Get hidden states
        text_hidden_states = text_outputs.last_hidden_state
        lyric_hidden_states = lyrics_outputs.last_hidden_state

        # Prepare generation parameters
        latent_frames = int(duration * 46.875)  # 48000 / 1024 ≈ 46.875 Hz

        # For text2music task, use silence_latent as src_latents
        # silence_latent will be tokenized/detokenized to get lm_hints_25Hz (127 dims)
        # which will be used as context for generation
        if self.silence_latent is not None:
            # Slice or pad silence_latent to match latent_frames
            if self.silence_latent.shape[1] >= latent_frames:
                src_latents = self.silence_latent[:, :latent_frames, :].to(device=self.device, dtype=self.torch_dtype)
            else:
                # Pad with zeros if silence_latent is shorter
                pad_len = latent_frames - self.silence_latent.shape[1]
                src_latents = torch.cat([
                    self.silence_latent.to(device=self.device, dtype=self.torch_dtype),
                    torch.zeros(1, pad_len, self.src_latent_channels, device=self.device, dtype=self.torch_dtype)
                ], dim=1)
        else:
            # Fallback: create random latents if silence_latent is not loaded
            src_latents = torch.randn(1, latent_frames, self.src_latent_channels,
                                     device=self.device, dtype=self.torch_dtype)

        # Create attention mask
        attention_mask = torch.ones(1, latent_frames, device=self.device, dtype=self.torch_dtype)

        # Use silence_latent for the silence_latent parameter as well
        silence_latent = src_latents

        # Chunk masks and is_covers (for text2music, these are all zeros)
        # chunk_masks shape: [batch, latent_frames, 1]
        chunk_masks = torch.zeros(1, latent_frames, 1, device=self.device, dtype=self.torch_dtype)
        is_covers = torch.zeros(1, device=self.device, dtype=self.torch_dtype)

        # Reference audio (empty for text2music)
        # For text2music mode, we need empty reference audio
        # refer_audio_acoustic_hidden_states_packed: [batch, num_segments, hidden_dim]
        # refer_audio_order_mask: [num_segments] - indicates which batch each segment belongs to
        refer_audio_acoustic_hidden_states_packed = torch.zeros(1, 1, 64, device=self.device, dtype=self.torch_dtype)
        refer_audio_order_mask = torch.zeros(1, device=self.device, dtype=torch.long)  # 1-d tensor

        # Generate audio latents using DiT model
        generation_result = self.dit.model.generate_audio(
            text_hidden_states=text_hidden_states,
            text_attention_mask=text_inputs["attention_mask"],
            lyric_hidden_states=lyric_hidden_states,
            lyric_attention_mask=lyrics_inputs["attention_mask"],
            refer_audio_acoustic_hidden_states_packed=refer_audio_acoustic_hidden_states_packed,
            refer_audio_order_mask=refer_audio_order_mask,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            is_covers=is_covers,
            silence_latent=silence_latent,
            attention_mask=attention_mask,
            seed=seed if seed is not None else 42,
            fix_nfe=inference_steps,
            shift=guidance_scale,
        )

        # Extract target latents from result dictionary
        generated_latents = generation_result["target_latents"]

        # Decode latents to audio
        # generated_latents shape: [batch, latent_frames, 64]
        # VAE expects: [batch, latent_frames, 64]
        audio_output = self.vae.decode(generated_latents, return_dict=True)
        audio = audio_output.sample

        # Post-process audio
        audio = self._postprocess_audio(audio)

        self.load_models_to_device([])
        return audio

    def _format_text_prompt(self, caption, bpm, keyscale, timesignature, duration):
        """Format text prompt with metadata"""
        prompt = "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n"
        prompt += f"# Caption\n{caption}\n\n"
        prompt += "# Metas\n"
        if bpm:
            prompt += f"- bpm: {bpm}\n"
        if timesignature:
            prompt += f"- timesignature: {timesignature}\n"
        if keyscale:
            prompt += f"- keyscale: {keyscale}\n"
        prompt += f"- duration: {int(duration)} seconds\n"
        prompt += "<|endoftext|>"
        return prompt

    def _format_lyrics(self, lyrics, vocal_language, instrumental):
        """Format lyrics with language"""
        if instrumental or not lyrics:
            lyrics = "[Instrumental]"

        lyrics_text = f"# Languages\n{vocal_language}\n\n# Lyric\n{lyrics}<|endoftext|>"
        return lyrics_text

    def _postprocess_audio(self, audio):
        """Post-process audio tensor"""
        # Ensure audio is on CPU and in float32
        audio = audio.to(device="cpu", dtype=torch.float32)

        # Normalize to [-1, 1]
        max_val = torch.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        return audio
