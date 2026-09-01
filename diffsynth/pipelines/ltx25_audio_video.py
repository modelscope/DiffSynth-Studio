from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from tqdm import tqdm

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..models.ltx25_tokenizer import LTX25GemmaTokenizer
from .ltx2_audio_video import (
    LTX2AudioVideoPipeline,
    LTX2AudioVideoUnit_PromptEmbedder,
)


def _seconds_to_num_frames(seconds: float, frame_rate: float, min_frames: int = 1, max_frames: int = 1024) -> int:
    raw_frames = max(min_frames, min(round(seconds * frame_rate), max_frames))
    frames = ((raw_frames - 1) // 8) * 8 + 1
    if frames < min_frames:
        frames = min(-(-(min_frames - 1) // 8) * 8 + 1, max_frames)
    return frames


class LTX25AudioVideoUnit_PromptEmbedder(LTX2AudioVideoUnit_PromptEmbedder):
    def _preprocess_text(self, pipe, text: str):
        token_pairs = pipe.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[token_id for token_id, _ in token_pairs]], device=pipe.device)
        attention_mask = torch.tensor([[weight for _, weight in token_pairs]], device=pipe.device)

        outputs = pipe.text_encoder.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs.hidden_states, attention_mask


class LTX25AudioVideoPipeline(LTX2AudioVideoPipeline):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.duration_head = None
        self.units[2] = LTX25AudioVideoUnit_PromptEmbedder()

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        gemma_path: str | Path | None = None,
        vram_limit: float | None = None,
        load_duration_head: bool = False,
        stage2_lora_config: ModelConfig | None = None,
        stage2_lora_strength: float = 1.0,
    ) -> "LTX25AudioVideoPipeline":
        if gemma_path is None:
            raise ValueError("gemma_path is required for the packed LTX-2.5 Gemma4 tokenizer assets.")
        pipe = LTX25AudioVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("ltx25_text_encoder")
        pipe.text_encoder_post_modules = model_pool.fetch_model("ltx25_text_encoder_post_modules")
        pipe.dit = model_pool.fetch_model("ltx25_dit")
        pipe.video_vae_encoder = model_pool.fetch_model("ltx25_video_vae_encoder")
        pipe.video_vae_decoder = model_pool.fetch_model("ltx25_diffusion_video_vae_decoder")
        pipe.audio_vae_decoder = model_pool.fetch_model("ltx25_audio_vae_decoder")
        pipe.audio_vocoder = model_pool.fetch_model("ltx25_audio_vocoder")
        pipe.audio_vae_encoder = model_pool.fetch_model("ltx25_audio_vae_encoder")

        pipe.upsampler = model_pool.fetch_model("ltx2_latent_upsampler")
        if load_duration_head:
            pipe.duration_head = model_pool.fetch_model("ltx25_duration_head")
        if stage2_lora_config is not None:
            stage2_lora_config.download_if_necessary()
            pipe.stage2_lora_config = stage2_lora_config
            pipe.stage2_lora_strength = stage2_lora_strength
        pipe.tokenizer = LTX25GemmaTokenizer(gemma_path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def predict_num_frames(self, prompt: str, frame_rate: float = 24.0) -> int:
        if self.duration_head is None:
            raise ValueError("Automatic duration requires from_pretrained(..., load_duration_head=True) and its ModelConfig.")
        self.load_models_to_device(("text_encoder", "text_encoder_post_modules", "duration_head"))
        embedder = self.units[2]
        hidden_states, attention_mask = embedder._preprocess_text(self, prompt)
        video_context, audio_context, _ = self.text_encoder_post_modules.process_hidden_states(hidden_states, attention_mask)
        seconds = float(self.duration_head(video_context, audio_context).item())
        return _seconds_to_num_frames(seconds, frame_rate)

    @torch.no_grad()
    def __call__(
        self,
        *args,
        use_two_stage_pipeline: bool = True,
        use_distilled_pipeline: bool = True,
        cfg_scale: float = 1.0,
        num_inference_steps: int = 8,
        progress_bar_cmd=tqdm,
        **kwargs,
    ):
        if use_distilled_pipeline and not use_two_stage_pipeline:
            raise ValueError("LTX-2.5 distilled inference requires the two-stage refinement flow.")
        if use_distilled_pipeline and cfg_scale != 1.0:
            raise ValueError("LTX-2.5 distilled inference requires cfg_scale=1.0.")
        if use_two_stage_pipeline and not use_distilled_pipeline and not hasattr(self, "stage2_lora_config"):
            raise ValueError("LTX-2.5 Dev two-stage inference requires stage2_lora_config.")
        if kwargs.get("num_frames") is None:
            prompt = kwargs.get("prompt", args[0] if args else "")
            kwargs["num_frames"] = self.predict_num_frames(prompt, kwargs.get("frame_rate", 24.0))
        return super().__call__(
            *args,
            use_two_stage_pipeline=use_two_stage_pipeline,
            use_distilled_pipeline=use_distilled_pipeline,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            progress_bar_cmd=progress_bar_cmd,
            **kwargs,
        )
