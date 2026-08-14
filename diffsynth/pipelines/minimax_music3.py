"""MiniMax Music 3 pipeline for DiffSynth-Studio: caption + lyrics -> song.

Two stages: an autoregressive global LLM (with an RVQ depth decoder) produces per-frame hidden states,
which condition a chunked flow-matching transformer whose Flow-VAE latents are vocoded to a stereo waveform.
The inference logic mirrors the target library's modular pipeline (autoregressive CFG 1.5 + top-50 sampling,
200-frame windows with 100-frame hop, overlap blending, inverted-sigma flow matching, waveform stitching).
"""
import re
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..core.device.npu_compatible_device import get_device_type
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.minimax_music3_dit import MiniMaxMusic3DiT
from ..models.minimax_music3_condition_encoder import MiniMaxMusic3ConditionEncoder
from ..models.minimax_music3_vocoder import MiniMaxMusic3Vocoder
from ..models.minimax_music3_rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder
from ..models.minimax_music3_text_encoder import MiniMaxMusic3TextEncoder


class MiniMaxMusic3Pipeline(BasePipeline):
    """Caption + lyrics -> music pipeline for MiniMax Music 3."""

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=1, width_division_factor=1)
        self.text_encoder: MiniMaxMusic3TextEncoder = None
        self.rvq_depth_decoder: MiniMaxMusic3RVQDepthDecoder = None
        self.condition_encoder: MiniMaxMusic3ConditionEncoder = None
        self.dit: MiniMaxMusic3DiT = None
        self.vocoder: MiniMaxMusic3Vocoder = None
        self.tokenizer = None

        # Checkpoint constants (see the official reference implementation's constants).
        # The DAV decoder synthesizes at 44.1 kHz; the delivered output contract is 32 kHz.
        self.dav_sample_rate = 44100
        self.sample_rate = 32000
        self.frame_rate = 25.0
        self.latent_hop_length = 512
        self.num_codebooks = 8
        self.audio_vocab_size = 1024
        self.num_channels_latents = 128

        self.in_iteration_models = ("dit",)
        self.units = [
            MiniMaxMusic3Unit_PromptEmbedder(),
            MiniMaxMusic3Unit_SemanticGenerator(),
            MiniMaxMusic3Unit_ChunkPlanner(),
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
    def denoise_chunks(self, frame_hiddens, chunk_starts, num_inference_steps, cfg_scale, generator, progress_bar_cmd):
        # Neighboring windows share 172 latent frames; the carry spans [L - 344, L - 172).
        overlap_latent_length = 172
        chunk_frames = MiniMaxMusic3Unit_ChunkPlanner.chunk_frames
        # Inverted-sigma flow matching, matching the target library's
        # FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0, invert_sigmas=True):
        # the schedule ascends from 0 (noise) to 1 (data) with a terminal sigma of 1.0.
        sig_in = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        sigmas = np.concatenate([1.0 - sig_in, [1.0]])
        timesteps = torch.tensor(sigmas[:-1], dtype=torch.float32, device=self.device)

        latent_chunks = []
        previous_latent = None
        previous_condition = None
        for k in progress_bar_cmd(range(len(chunk_starts))):
            chunk_start = chunk_starts[k]
            chunk_end = min(chunk_start + chunk_frames, frame_hiddens.shape[1])
            condition = self.condition_encoder(frame_hiddens[:, chunk_start:chunk_end].to(self.device))
            condition = condition.to(next(self.dit.parameters()).dtype)

            overlap = 0
            if previous_latent is not None:
                overlap = min(previous_latent.shape[-1], condition.shape[1])
                condition[:, :overlap] = previous_condition[:, :overlap]

            shape = (1, self.num_channels_latents, condition.shape[1])
            if generator is None:
                latents = self.generate_noise(shape, rand_device=self.device, rand_torch_dtype=condition.dtype, device=self.device)
            else:
                latents = torch.randn(shape, generator=generator, device=self.device, dtype=condition.dtype)
            noise_prompt = latents[..., :overlap].clone() if overlap > 0 else None

            # The unconditional branch conditions on zeros, not on a re-encoded empty prompt.
            zeros = torch.zeros_like(condition)
            for i in range(num_inference_steps):
                t = float(timesteps[i])
                if overlap > 0:
                    # Blend the overlap toward the previous window's trailing latents at every step.
                    latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * t) * noise_prompt + t * previous_latent[..., :overlap]
                timestep = timesteps[i].expand(latents.shape[0]).to(latents.dtype)
                cond_pred = self.dit(latents, timestep, condition)
                uncond_pred = self.dit(latents, timestep, zeros)
                velocity = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
                latents = latents + (sigmas[i + 1] - sigmas[i]) * velocity

            if overlap > 0:
                latents[..., :overlap] = previous_latent[..., :overlap]
            overlap_start = max(0, latents.shape[-1] - 2 * overlap_latent_length)
            overlap_end = max(overlap_start, latents.shape[-1] - overlap_latent_length)
            previous_latent = latents[..., overlap_start:overlap_end]
            previous_condition = condition[:, overlap_start:overlap_end]
            latent_chunks.append(latents)
        return latent_chunks

    @torch.no_grad()
    def decode(self, latent_chunks, output_type):
        # Neighboring windows share 172 latent frames: every window after the first drops its leading
        # 86 latent frames and every window before the last drops its trailing 344 - 86 latent frames,
        # so the kept spans tile the full song.
        crop_left_latent = 86
        crop_right_latent = 344 - 86
        hop_length = self.latent_hop_length
        num_chunks = len(latent_chunks)
        waveform_chunks = []
        for chunk_index, latents in enumerate(latent_chunks):
            waveform = self.vocoder(latents.to(next(self.vocoder.parameters()).dtype))
            left = 0 if chunk_index == 0 else crop_left_latent * hop_length
            right = 0 if chunk_index == num_chunks - 1 else crop_right_latent * hop_length
            waveform_chunks.append(waveform[..., left : waveform.shape[-1] - right])
        audios = torch.cat(waveform_chunks, dim=-1).float().clamp(-1.0, 1.0)
        if self.sample_rate != self.dav_sample_rate:
            import torchaudio.functional as AF
            audios = AF.resample(audios, self.dav_sample_rate, self.sample_rate)
        if output_type == "np":
            audios = audios.cpu().numpy()
        return audios

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        lyrics: str,
        audio_duration: float = 60.0,
        num_inference_steps: int = 30,
        cfg_scale: float = 1.7,
        seed: int = None,
        output_type: str = "np",
        progress_bar_cmd=tqdm,
    ):
        generator = None if seed is None else torch.Generator(self.device).manual_seed(seed)

        inputs_posi = {}
        inputs_nega = {}
        inputs_shared = {
            "prompt": prompt,
            "lyrics": lyrics,
            "audio_duration": audio_duration,
            "generator": generator,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        self.load_models_to_device(["condition_encoder", "dit"])
        latent_chunks = self.denoise_chunks(
            inputs_shared["frame_hiddens"], inputs_shared["chunk_starts"],
            num_inference_steps, cfg_scale, generator, progress_bar_cmd,
        )

        self.load_models_to_device(["vocoder"])
        audios = self.decode(latent_chunks, output_type)
        self.load_models_to_device([])
        return audios[0]


class MiniMaxMusic3Unit_PromptEmbedder(PipelineUnit):
    """Assembles the checkpoint's special-token prompt from the music description and the lyrics, then
    tokenizes it into the conditional/unconditional token id pair.

    The template, its token ids and the text normalization are part of the checkpoint contract: even
    whitespace-level changes to the assembled prompt change the generated audio.
    """

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
        # Strip the markdown forms accepted by the model's input contract.
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
        # Keep only consecutive structural tags at the start of a line; text on a tag line is dropped.
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
        if not isinstance(lyrics, str) or not lyrics.strip():
            raise ValueError(f"`lyrics` must be a non-empty string, got {lyrics!r}")
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
    """Autoregressive stage: frame by frame the global language model samples a semantic code with
    classifier-free guidance and the depth decoder samples the residual RVQ codes; the concatenated
    per-frame hidden states condition the flow-matching stage.
    """

    audio_end_token_id = 151670
    audio_code_offset = 151675
    semantic_vocab_size = 16384
    max_audio_frames = 9_000
    # The autoregressive stage's sampling parameters are fixed by the reference inference recipe.
    ar_cfg_scale = 1.5
    ar_cfg_top_k = 50
    ar_sampling_top_k = 50

    def __init__(self):
        super().__init__(
            input_params=("text_ids", "audio_duration", "generator"),
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
        # Sample on the generator's device so a CPU generator gives device-independent results.
        sample_device = generator.device if generator is not None else probs.device
        return torch.multinomial(probs.to(sample_device), 1, generator=generator).squeeze(-1).to(probs.device)

    def embed_audio_frame(self, pipe: MiniMaxMusic3Pipeline, frame_codes):
        # frame_codes: [2, num_codebooks]. Sum the semantic-code embedding with the residual-code embeddings.
        embed_tokens = pipe.text_encoder.model.model.embed_tokens
        embeds = embed_tokens(frame_codes[:, :1] + self.audio_code_offset)
        offsets = (torch.arange(pipe.num_codebooks - 1, device=frame_codes.device) * pipe.audio_vocab_size).unsqueeze(0)
        extra = pipe.rvq_depth_decoder.audio_extra_embedding(frame_codes[:, 1:] + offsets).sum(dim=1, keepdim=True)
        embeds = embeds + extra.to(embeds.dtype)
        return embeds * pipe.num_codebooks**-0.5

    def generate_depth_codes(self, pipe: MiniMaxMusic3Pipeline, last_hidden, semantic_code, generator):
        # Autoregressively sample the residual codes c1..c7 for one frame and collect their hidden states.
        rvq = pipe.rvq_depth_decoder
        embed_tokens = pipe.text_encoder.model.model.embed_tokens
        sequence = [rvq.audio_decoder.projection(last_hidden).unsqueeze(1)]
        code_embed = embed_tokens(semantic_code + self.audio_code_offset)
        sequence.append(rvq.audio_decoder.projection(code_embed).unsqueeze(1))
        codes = [semantic_code]
        hidden_parts = []
        for index in range(1, pipe.num_codebooks):
            hidden = rvq(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden[:1])
            logits = rvq.audio_decoder.audio_heads[index - 1](hidden)
            conditional, unconditional = logits[:1].float(), logits[1:2].float()
            logits = unconditional + (conditional - unconditional) * self.ar_cfg_scale
            # The sampled code is repeated so the language-model feedback keeps the [cond, uncond] rows.
            code = self.sample_top_k(logits, generator).repeat(2)
            codes.append(code)
            if index < pipe.num_codebooks - 1:
                embed = rvq.audio_extra_embedding(code + (index - 1) * pipe.audio_vocab_size)
                sequence.append(rvq.audio_decoder.projection(embed).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    def process(self, pipe: MiniMaxMusic3Pipeline, text_ids, audio_duration, generator):
        pipe.load_models_to_device(self.onload_model_names)
        if audio_duration <= 0:
            raise ValueError(f"`audio_duration` must be positive, got {audio_duration}")
        max_frames = min(int(audio_duration * pipe.frame_rate), self.max_audio_frames)
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
        # The first decode step only advances the state past `<|audio_start|>` and is not an emitted frame.
        for frame_index in range(max_frames + 1):
            logits = lm_head(last_hidden).float()
            logits = logits.masked_fill(vocab_mask, -float("inf"))
            conditional, unconditional = logits[0:1], logits[1:2]
            guided = unconditional + (conditional - unconditional) * self.ar_cfg_scale
            # Restrict the guided distribution to the conditional branch's top candidates, then re-mask:
            # guidance on two `-inf` logits produces NaN on masked positions.
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


class MiniMaxMusic3Unit_ChunkPlanner(PipelineUnit):
    """Splits the autoregressive frames into 200-frame denoising windows with a 100-frame hop; each window
    is flow-matched with the previous window's trailing latents as an overlap prompt.
    """

    chunk_frames = 200
    chunk_hop = 100

    def __init__(self):
        super().__init__(
            input_params=("frame_hiddens",),
            output_params=("chunk_starts",),
        )

    def process(self, pipe: MiniMaxMusic3Pipeline, frame_hiddens):
        num_frames = frame_hiddens.shape[1]
        chunk_starts = [0] if num_frames <= self.chunk_frames else list(range(0, num_frames - self.chunk_hop, self.chunk_hop))
        return {"chunk_starts": chunk_starts}
