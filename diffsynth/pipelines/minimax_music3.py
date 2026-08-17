import re
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..core import ModelConfig
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.minimax_music3_dit import MiniMaxMusic3DiT
from ..models.minimax_music3_condition_encoder import MiniMaxMusic3ConditionEncoder
from ..models.minimax_music3_vocoder import MiniMaxMusic3Vocoder
from ..models.minimax_music3_rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder
from ..models.minimax_music3_text_encoder import MiniMaxMusic3TextEncoder


class MiniMaxMusic3Pipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.text_encoder: MiniMaxMusic3TextEncoder = None
        self.rvq_depth_decoder: MiniMaxMusic3RVQDepthDecoder = None
        self.condition_encoder: MiniMaxMusic3ConditionEncoder = None
        self.dit: MiniMaxMusic3DiT = None
        self.vocoder: MiniMaxMusic3Vocoder = None
        self.tokenizer = None
        self.scheduler = FlowMatchScheduler("MiniMax-Music3")

        self.in_iteration_models = ("condition_encoder", "dit")
        self.units = [
            MiniMaxMusic3Unit_PromptEmbedder(),
            MiniMaxMusic3Unit_SemanticGenerator(),
            MiniMaxMusic3Unit_ChunkDenoiser(),
            MiniMaxMusic3Unit_Vocoder(),
        ]
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = MiniMaxMusic3Pipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("minimax_music3_text_encoder")
        pipe.rvq_depth_decoder = model_pool.fetch_model("minimax_music3_rvq_depth_decoder")
        pipe.condition_encoder = model_pool.fetch_model("minimax_music3_condition_encoder")
        pipe.dit = model_pool.fetch_model("minimax_music3_dit")
        pipe.vocoder = model_pool.fetch_model("minimax_music3_vocoder")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import AutoTokenizer
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        lyrics: str = " ",
        max_audio_duration: float = 60.0,
        num_inference_steps: int = 30,
        cfg_scale: float = 1.7,
        seed: int = None,
        rand_device: str = get_device_type(),
        progress_bar_cmd=tqdm,
    ):
        inputs_posi = {}
        inputs_nega = {}
        inputs_shared = {
            "prompt": prompt,
            "lyrics": lyrics,
            "max_audio_duration": max_audio_duration,
            "num_inference_steps": num_inference_steps,
            "cfg_scale": cfg_scale,
            "generator": None if seed is None else torch.Generator(rand_device).manual_seed(seed),
            "rand_device": rand_device,
            "progress_bar_cmd": progress_bar_cmd,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        self.load_models_to_device([])
        return inputs_shared["audio"]


class MiniMaxMusic3Unit_PromptEmbedder(PipelineUnit):

    im_start, im_end = "<|im_start|>", "<|im_end|>"
    caption_start, caption_end = "<|caption_start|>", "<|caption_end|>"
    lyrics_start, lyrics_end = "<|lyrics_start|>", "<|lyrics_end|>"
    audio_start = "<|audio_start|>"
    audio_cfg_token_id = 151654
    max_prompt_tokens = 5_000
    special_tag_re = re.compile(r"<\|([^|]*)\|>")
    leading_tags_re = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")

    def __init__(self):
        super().__init__(
            input_params=("prompt", "lyrics"),
            output_params=("text_ids",),
        )

    def clean_caption(self, caption: str) -> str:
        def rewrite_special_tag(match: re.Match) -> str:
            inner = match.group(1).strip()
            parts = inner.split(None, 1)
            return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

        text = self.special_tag_re.sub(rewrite_special_tag, caption)
        lines_out = []
        for line in text.splitlines():
            line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
            line = re.sub(r"^\s*[*+-]\s+", "", line)
            line = re.sub(r"^\s*\*\s+", "", line)
            while "**" in line:
                updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
                if updated == line:
                    break
                line = updated
            line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
            lines_out.append(line.rstrip())
        text = "\n".join(lines_out)
        text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
        text = text.replace("• ", "").replace("    ", "")
        return re.sub(r"\n{2,}", "\n", text)

    def normalize_lyrics(self, lyrics: str) -> str:
        output = []
        for line in lyrics.split("\n"):
            match = self.leading_tags_re.match(line)
            output.append(match.group(1).strip() if match else line)
        text = "\n".join(output)
        text = text.replace("] ", "]\n")
        text = text.replace(" [", "\n[")
        text = text.replace(" ^ ", "\n")
        text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
        return f"[start]\n{text}"

    def process(self, pipe: MiniMaxMusic3Pipeline, prompt, lyrics):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"`prompt` (the music description) must be a non-empty string, got {prompt!r}")
        text = (
            f"{self.im_start}{self.caption_start}{self.clean_caption(prompt)}{self.caption_end}"
            f"{self.lyrics_start}{self.normalize_lyrics(lyrics)}{self.lyrics_end}{self.im_end}{self.audio_start}"
        )
        input_ids = pipe.tokenizer(text, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] > self.max_prompt_tokens:
            raise ValueError(f"The assembled prompt has {input_ids.shape[1]} tokens; the maximum is {self.max_prompt_tokens}")
        unconditional_ids = input_ids.clone()
        unconditional_ids[:, 1:-2] = self.audio_cfg_token_id
        return {"text_ids": torch.cat((input_ids, unconditional_ids), dim=0).to(pipe.device)}


class MiniMaxMusic3Unit_SemanticGenerator(PipelineUnit):

    audio_end_token_id = 151670
    audio_code_offset = 151675
    semantic_vocab_size = 16384
    max_audio_frames = 9_000
    frame_rate = 25.0
    num_codebooks = 8
    audio_vocab_size = 1024
    ar_cfg_scale = 1.5
    ar_cfg_top_k = 50
    ar_sampling_top_k = 50

    def __init__(self):
        super().__init__(
            input_params=("text_ids", "max_audio_duration", "generator", "progress_bar_cmd"),
            output_params=("frame_hiddens",),
            onload_model_names=("text_encoder", "rvq_depth_decoder"),
        )

    def sample_top_k(self, logits: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
        top_k = min(self.ar_sampling_top_k, values.shape[-1])
        threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
        values = values.masked_fill(values < threshold, -float("inf"))
        probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sample_device = generator.device if generator is not None else probs.device
        return torch.multinomial(probs.to(sample_device), 1, generator=generator).squeeze(-1).to(probs.device)

    def embed_audio_frame(self, pipe: MiniMaxMusic3Pipeline, frame_codes):
        embed_tokens = pipe.text_encoder.model.model.embed_tokens
        embeds = embed_tokens(frame_codes[:, :1] + self.audio_code_offset)
        offsets = (torch.arange(self.num_codebooks - 1, device=frame_codes.device) * self.audio_vocab_size).unsqueeze(0)
        extra = pipe.rvq_depth_decoder.audio_embeddings(frame_codes[:, 1:] + offsets).sum(dim=1, keepdim=True)
        embeds = embeds + extra.to(embeds.dtype)
        return embeds * self.num_codebooks**-0.5

    def generate_depth_codes(self, pipe: MiniMaxMusic3Pipeline, last_hidden, semantic_code, generator):
        rvq = pipe.rvq_depth_decoder
        embed_tokens = pipe.text_encoder.model.model.embed_tokens
        sequence = [rvq.projection(last_hidden).unsqueeze(1)]
        code_embed = embed_tokens(semantic_code + self.audio_code_offset)
        sequence.append(rvq.projection(code_embed).unsqueeze(1))
        codes = [semantic_code]
        hidden_parts = []
        for index in range(1, self.num_codebooks):
            hidden = rvq(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden[:1])
            logits = rvq.audio_heads[index - 1](hidden)
            conditional, unconditional = logits[:1].float(), logits[1:2].float()
            logits = unconditional + (conditional - unconditional) * self.ar_cfg_scale
            code = self.sample_top_k(logits, generator).repeat(2)
            codes.append(code)
            if index < self.num_codebooks - 1:
                embed = rvq.audio_embeddings(code + (index - 1) * self.audio_vocab_size)
                sequence.append(rvq.projection(embed).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    def process(self, pipe: MiniMaxMusic3Pipeline, text_ids, max_audio_duration, generator, progress_bar_cmd):
        pipe.load_models_to_device(self.onload_model_names)
        if max_audio_duration <= 0:
            raise ValueError(f"`max_audio_duration` must be positive, got {max_audio_duration}")
        max_frames = min(int(max_audio_duration * self.frame_rate), self.max_audio_frames)
        backbone = pipe.text_encoder.model.model
        lm_head = pipe.text_encoder.model.lm_head
        vocab_size = pipe.text_encoder.config.vocab_size

        output = backbone(inputs_embeds=backbone.embed_tokens(text_ids), use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]

        vocab_mask = torch.ones(vocab_size, dtype=torch.bool, device=text_ids.device)
        vocab_mask[self.audio_code_offset : self.audio_code_offset + self.semantic_vocab_size] = False
        vocab_mask[self.audio_end_token_id] = False

        frame_hiddens = []
        for frame_index in progress_bar_cmd(
            range(max_frames + 1),
            desc="Generating semantic tokens",
            unit="tokens",
            bar_format="{desc}: {n_fmt} tokens ({rate_fmt})",
        ):
            logits = lm_head(last_hidden).float()
            logits = logits.masked_fill(vocab_mask, -float("inf"))
            conditional, unconditional = logits[0:1], logits[1:2]
            guided = unconditional + (conditional - unconditional) * self.ar_cfg_scale
            threshold = torch.topk(conditional, self.ar_cfg_top_k, dim=-1).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
            sampled = self.sample_top_k(guided, generator)
            if int(sampled.item()) == self.audio_end_token_id:
                break
            semantic_code = sampled - self.audio_code_offset
            frame_codes, depth_hidden = self.generate_depth_codes(pipe, last_hidden, semantic_code.repeat(2), generator)
            if frame_index > 0:
                frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                if len(frame_hiddens) >= max_frames:
                    break
            feedback = self.embed_audio_frame(pipe, frame_codes)
            output = backbone(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]

        if not frame_hiddens:
            raise ValueError("MiniMax Music 3 generated zero audio frames; the prompt ended generation immediately")
        return {"frame_hiddens": torch.stack(frame_hiddens, dim=1)}


class MiniMaxMusic3Unit_ChunkDenoiser(PipelineUnit):

    chunk_frames = 200
    chunk_hop = 100
    overlap_latent_length = 172
    crop_left_latent = overlap_latent_length // 2
    crop_right_latent = 2 * overlap_latent_length - crop_left_latent
    num_channels_latents = 128

    def __init__(self):
        super().__init__(
            input_params=("frame_hiddens", "num_inference_steps", "cfg_scale", "generator", "rand_device", "progress_bar_cmd"),
            output_params=("latent_chunks",),
            onload_model_names=("condition_encoder", "dit"),
        )

    def process(self, pipe: MiniMaxMusic3Pipeline, frame_hiddens, num_inference_steps, cfg_scale, generator, rand_device, progress_bar_cmd):
        pipe.load_models_to_device(self.onload_model_names)
        pipe.scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        timesteps = pipe.scheduler.timesteps.to(pipe.device)
        num_frames = frame_hiddens.shape[1]
        chunk_starts = [0] if num_frames <= self.chunk_frames else list(range(0, num_frames - self.chunk_hop, self.chunk_hop))

        latent_chunks = []
        previous_latent, previous_condition = None, None
        for chunk_start in progress_bar_cmd(chunk_starts):
            frames = frame_hiddens[:, chunk_start : chunk_start + self.chunk_frames].to(pipe.device)
            condition = pipe.condition_encoder(frames).to(pipe.torch_dtype)

            overlap = 0 if previous_latent is None else min(previous_latent.shape[-1], condition.shape[1])
            if overlap > 0:
                condition[:, :overlap] = previous_condition[:, :overlap]

            shape = (1, self.num_channels_latents, condition.shape[1])
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=condition.dtype).to(pipe.device)
            noise_prompt = latents[..., :overlap].clone()

            zeros = torch.zeros_like(condition)
            for i in range(num_inference_steps):
                t = (1.0 - timesteps[i] / pipe.scheduler.num_train_timesteps).to(latents.dtype)
                if overlap > 0:
                    latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * t) * noise_prompt + t * previous_latent[..., :overlap]
                timestep = t.expand(latents.shape[0])
                cond_pred = pipe.dit(latents, timestep, condition)
                uncond_pred = pipe.dit(latents, timestep, zeros)
                velocity = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
                latents = pipe.scheduler.step(-velocity, timesteps[i], latents)

            if overlap > 0:
                latents[..., :overlap] = previous_latent[..., :overlap]
            overlap_end = max(latents.shape[-1] - self.overlap_latent_length, 0)
            overlap_start = max(overlap_end - self.overlap_latent_length, 0)
            previous_latent = latents[..., overlap_start:overlap_end]
            previous_condition = condition[:, overlap_start:overlap_end]
            latent_chunks.append(latents)
        return {"latent_chunks": latent_chunks}


class MiniMaxMusic3Unit_Vocoder(PipelineUnit):

    latent_hop_length = 512

    def __init__(self):
        super().__init__(
            input_params=("latent_chunks", ),
            output_params=("audio", ),
            onload_model_names=("vocoder", ),
        )

    def process(self, pipe: MiniMaxMusic3Pipeline, latent_chunks):
        pipe.load_models_to_device(self.onload_model_names)
        crop_left = MiniMaxMusic3Unit_ChunkDenoiser.crop_left_latent * self.latent_hop_length
        crop_right = MiniMaxMusic3Unit_ChunkDenoiser.crop_right_latent * self.latent_hop_length
        waveform_chunks = []
        for chunk_index, latents in enumerate(latent_chunks):
            waveform = pipe.vocoder(latents.to(pipe.torch_dtype))
            left = 0 if chunk_index == 0 else crop_left
            right = 0 if chunk_index == len(latent_chunks) - 1 else crop_right
            waveform_chunks.append(waveform[...,
                                            left:waveform.shape[-1] - right])
        song = torch.cat(waveform_chunks, dim=-1)[0]
        return {"audio": song.float().clamp(-1.0, 1.0)}
