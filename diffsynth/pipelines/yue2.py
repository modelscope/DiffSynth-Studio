"""YuE2 music generation pipeline: symbolic planning, AR semantic tokens, NAR flow matching and VAE decode."""
import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..diffusion.flow_match import FlowMatchScheduler
from ..models.yue2_mot import StaticKVCache, YuE2MoT
from ..models.yue2_tokenizer import YuE2TextTokenizer
from ..models.yue2_vae import YuE2VAEModel


class YuE2Pipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.mot: YuE2MoT = None
        self.vae: YuE2VAEModel = None
        self.tokenizer: YuE2TextTokenizer = None
        self.generation_config = None
        self.scheduler = FlowMatchScheduler("YuE2")
        self.scheduler.set_timesteps(1000, training=False)

        self.in_iteration_models = ("mot",)
        self.units = [
            YuE2Unit_ScorePlanner(),
            YuE2Unit_SemanticGenerator(),
            YuE2Unit_NARConditioner(),
            YuE2Unit_InputAudioEmbedder(),
        ]
        self.model_fn = model_fn_yue2

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        # Tokenizer and generation config
        tokenizer_config: ModelConfig = ModelConfig(model_id="m-a-p/YuE2-3B", origin_file_pattern="qwen.tiktoken"),
        generation_config: ModelConfig = ModelConfig(model_id="m-a-p/YuE2-3B", origin_file_pattern="yue2_generation_config.json"),
        # VRAM management
        vram_limit: float = None,
    ):
        pipe = YuE2Pipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.mot = model_pool.fetch_model("yue2_mot")
        pipe.vae = model_pool.fetch_model("yue2_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = YuE2TextTokenizer(tokenizer_config.path)
        if generation_config is not None:
            generation_config.download_if_necessary()
            pipe.generation_config = json.loads(Path(generation_config.path).read_text(encoding="utf-8"))
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def plan(
        self,
        # Prompt
        prompt: str,
        lyrics: str,
        # Symbolic plan
        cot: str = "full",
        seed: int = 831001,
        # Progress
        progress_bar_cmd=tqdm,
    ):
        """Return the editable ABC score planned from the prompt and lyrics."""
        if cot == "off":
            raise ValueError("cot=off has no symbolic plan")
        inputs_shared = {
            "prompt": prompt, "lyrics": lyrics, "cot": cot, "abc": None, "seed": seed,
            "progress_bar_cmd": progress_bar_cmd, "on_token": None,
        }
        inputs_shared, _, _ = self.unit_runner(YuE2Unit_ScorePlanner(), self, inputs_shared, {}, {})
        return inputs_shared["abc_text"]

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        lyrics: str,
        # Symbolic plan
        cot: str = "full",
        abc: str = None,
        cfg_scale: float = None,
        # Generation
        seed: int = 831001,
        num_inference_steps: int = 30,
        # Decode
        vae_core_frames: int = None,
        # Progress
        progress_bar_cmd=tqdm,
        on_token=None,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Parameters
        inputs_posi = {}
        inputs_nega = {}
        inputs_shared = {
            "prompt": prompt, "lyrics": lyrics,
            "cot": cot, "abc": abc, "cfg_scale": cfg_scale,
            "seed": seed, "num_inference_steps": num_inference_steps,
            "vae_core_frames": vae_core_frames,
            "progress_bar_cmd": progress_bar_cmd, "on_token": on_token,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        step_size = 1.0 / num_inference_steps
        chunks = inputs_shared["nar_chunks"]
        latents = []
        for chunk_index, chunk in enumerate(chunks):
            engine = _CachedNAR(models["mot"], chunk, self.device, self.torch_dtype)
            state = chunk.noise.to(device=engine.device, dtype=engine.dtype)
            progress = progress_bar_cmd(
                self.scheduler.timesteps,
                desc=f"Synthesizing audio ({chunk_index + 1}/{len(chunks)})",
                unit="step",
            )
            for progress_id, timestep in enumerate(progress):
                velocity = self.model_fn(**models, latents=state, timestep=timestep, engine=engine)
                midpoint = state - velocity * (step_size / 2)
                midpoint_velocity = self.model_fn(**models, latents=midpoint, timestep=timestep - 500 * step_size, engine=engine)
                state = self.step(self.scheduler, latents=state, progress_id=progress_id, noise_pred=midpoint_velocity)
            progress.close()
            latents.append(state.float().cpu())
            engine.close()
        inputs_shared["latents"] = torch.cat(latents, dim=0)

        # Decode
        self.load_models_to_device(["vae"])
        z = inputs_shared["latents"].T.unsqueeze(0)
        core_frames = vae_core_frames or self.vae.config.decode_core_frames
        tiles = (z.shape[-1] + core_frames - 1) // core_frames
        progress = progress_bar_cmd(total=tiles, desc="Decoding audio", unit="chunk")
        audio = self.vae.decode_tiled(z, core_frames=core_frames, halo_frames=16, output_device="cpu",
                                      on_progress=lambda completed, total: progress.update(1))
        progress.close()
        self.load_models_to_device([])
        return self.output_audio_format_check(audio).clamp(-1, 1)


class YuE2Unit_ARGenerator(PipelineUnit):

    eod = 151643
    abc_start, abc_end = 151847, 151848
    music_start, music_end = 151851, 151852
    codec_offset, codec_size = 151853, 32768
    latent_start, latent_end, latent_pad = 184621, 184622, 184623
    context = 24576
    instructions = {
        "off": "Generate music with codec tokens from the given conditions.",
        "melody": "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.",
        "full": "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.",
    }
    abc_sampling = {
        "temperature": 0.7, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.005,
        "penalty_window": 100, "min_tokens": 32, "max_tokens": 4096,
    }
    semantic_sampling = {
        "temperature": 1.0, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.2,
        "penalty_window": 50, "min_tokens": 200, "max_tokens": 9000,
    }

    @classmethod
    def guidance(cls, cot, cfg_scale):
        return (1.01 if cot == "off" else 1.0) if cfg_scale is None else cfg_scale

    @classmethod
    def text(cls, prompt, lyrics, cot):
        if cot not in cls.instructions:
            raise ValueError("cot must be off, melody or full")
        return f"{cls.instructions[cot]}\n[Tags]\n{prompt}\n[Lyrics]\n{lyrics}\n"

    @classmethod
    def prefix(cls, prompt, lyrics, cot, abc, tokenizer, abc_ids=None):
        base = [cls.eod] + tokenizer.encode(cls.text(prompt, lyrics, cot))
        if cot == "off":
            return base + [cls.abc_start, cls.abc_end, cls.music_start]
        if abc_ids is None:
            if abc is None:
                return base + [cls.abc_start]
            abc_ids = tokenizer.encode(abc)
        return base + [cls.abc_start] + list(abc_ids) + [cls.abc_end, cls.music_start]

    @classmethod
    def negative_prefix(cls, prompt, lyrics, cot, tokenizer, abc_ids):
        base = [cls.eod] + tokenizer.encode(cls.instructions[cot])
        if cot == "off":
            return base + [cls.music_start]
        return base + [cls.abc_start] + list(abc_ids) + [cls.abc_end, cls.music_start]

    def sampling(self, pipe, phase):
        sampling = dict(self.abc_sampling if phase == "abc" else self.semantic_sampling)
        if pipe.generation_config is not None:
            sampling.update(pipe.generation_config.get(phase, {}))
        return sampling

    @staticmethod
    def window_penalty(logits, recent_ids, penalty):
        if penalty == 1.0 or len(recent_ids) == 0:
            return logits
        recent = torch.as_tensor(recent_ids, dtype=torch.long, device=logits.device).reshape(1, -1)
        freq = torch.zeros_like(logits)
        freq.scatter_add_(-1, recent, torch.ones_like(recent, dtype=logits.dtype))
        alpha = penalty ** freq
        return torch.where(logits < 0, logits * alpha, logits / alpha)

    def distribution(self, logits, sampling, history, step, phase, legacy_off=False):
        scores = logits.clone() if legacy_off else logits.float().clone()
        end = self.abc_end if phase == "abc" else self.music_end
        allowed = torch.full_like(scores, float("-inf"))
        if phase == "abc":
            allowed[..., :self.eod] = 0
        else:
            allowed[..., self.codec_offset:self.codec_offset + self.codec_size] = 0
        allowed[..., end] = 0
        scores = scores + allowed
        if step < sampling["min_tokens"]:
            scores[..., end] = -torch.inf
        scores = self.window_penalty(scores, history[-sampling["penalty_window"]:], sampling["repetition_penalty"])
        if sampling["temperature"] == 0:
            return scores
        if sampling["temperature"] != 1:
            scores = scores / sampling["temperature"]
        threshold = scores.topk(min(sampling["top_k"], scores.shape[-1])).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
        if sampling["top_p"] < 1:
            values, indices = scores.sort(descending=True)
            probabilities = values.softmax(-1)
            removed = probabilities.cumsum(-1) - probabilities > sampling["top_p"]
            removed[..., :3 if legacy_off else 1] = False
            values = values.masked_fill(removed, -torch.inf)
            scores = values.scatter(-1, indices, values)
        return scores

    @torch.inference_mode()
    def generate(self, pipe, model, prefix, sampling, seed, phase, negative=None, cfg_scale=1.0,
                 legacy_off=False, progress_bar_cmd=tqdm, on_token=None):
        device = torch.device(pipe.device)
        dtype = pipe.torch_dtype
        if len(prefix) + sampling["max_tokens"] > self.context:
            raise ValueError("Prefix + requested generation budget exceeds the model context")
        if cfg_scale != 1 and negative is None:
            raise ValueError("CFG requires a negative prefix")
        if negative is not None and len(negative) + sampling["max_tokens"] > self.context:
            raise ValueError("Negative prefix + generation budget exceeds the context")
        progress_bar_cmd = progress_bar_cmd or tqdm
        rng_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
        generator = torch.Generator(device=rng_device).manual_seed(seed)
        config = model.config

        def prefill(ids):
            cache = StaticKVCache(num_layers=config.num_hidden_layers, batch_size=1,
                                  num_kv_heads=config.num_key_value_heads,
                                  max_seq_len=len(ids) + sampling["max_tokens"],
                                  head_dim=config.head_dim, dtype=dtype, device=device)
            output = model(torch.tensor([ids], device=device), past_key_values=cache,
                           use_cache=True, logits_to_keep=1)
            return output.logits[:, -1, :], output.past_key_values

        conditional, positive_cache = prefill(prefix)
        unconditional, negative_cache = None, None
        if cfg_scale != 1.0:
            unconditional, negative_cache = prefill(negative)
        history, eos = [], False
        end = self.abc_end if phase == "abc" else self.music_end
        progress = progress_bar_cmd(
            range(sampling["max_tokens"]),
            desc=f"Generating {phase} tokens",
            unit=" token",
            bar_format="{desc}: {n_fmt} tokens | {rate_fmt} | {elapsed}",
        )
        for step in progress:
            # Preserve historical BF16 CFG subtraction/multiply/add before upcast.
            logits = conditional if cfg_scale == 1.0 else unconditional + cfg_scale * (conditional - unconditional)
            scores = self.distribution(logits, sampling, history, step, phase, legacy_off)
            if sampling["temperature"] == 0:
                next_id = scores.argmax(-1, keepdim=True)
            else:
                probabilities = scores.softmax(-1)
                if device.type == "mps":
                    next_id = torch.multinomial(probabilities.cpu(), 1, generator=generator).to(device)
                else:
                    next_id = torch.multinomial(probabilities, 1, generator=generator)
            token = int(next_id.item())
            if on_token is not None:
                on_token(phase, token)
            if token == end:
                eos = True
                break
            history.append(token)
            if step + 1 < sampling["max_tokens"]:
                conditional = model(next_id, past_key_values=positive_cache, use_cache=True,
                                    logits_to_keep=1).logits[:, -1, :]
                if negative_cache is not None:
                    unconditional = model(next_id, past_key_values=negative_cache, use_cache=True,
                                          logits_to_keep=1).logits[:, -1, :]
        progress.close()
        return history, not eos


class YuE2Unit_ScorePlanner(YuE2Unit_ARGenerator):

    def __init__(self):
        # Declared without onload names so that `sft:data_process` keeps the AR stage in the preprocessing set.
        super().__init__(
            input_params=("prompt", "lyrics", "cot", "abc", "seed", "progress_bar_cmd", "on_token"),
            output_params=("prefix", "abc_text", "abc_ids"),
        )

    def process(self, pipe, prompt, lyrics, cot, abc, seed, progress_bar_cmd, on_token):
        if cot == "off":
            return {"prefix": self.prefix(prompt, lyrics, cot, abc, pipe.tokenizer), "abc_text": None, "abc_ids": []}
        if abc is not None:
            abc_ids = pipe.tokenizer.encode(abc)
            return {"prefix": self.prefix(prompt, lyrics, cot, abc, pipe.tokenizer, abc_ids), "abc_text": abc, "abc_ids": abc_ids}
        pipe.load_models_to_device(["mot"])
        prefix = self.prefix(prompt, lyrics, cot, abc, pipe.tokenizer)
        abc_ids, _ = self.generate(pipe, pipe.mot, prefix, self.sampling(pipe, "abc"), seed, "abc",
                                   progress_bar_cmd=progress_bar_cmd, on_token=on_token)
        return {"prefix": self.prefix(prompt, lyrics, cot, abc, pipe.tokenizer, abc_ids),
                "abc_text": pipe.tokenizer.decode(abc_ids), "abc_ids": abc_ids}


class YuE2Unit_SemanticGenerator(YuE2Unit_ARGenerator):

    def __init__(self):
        super().__init__(
            input_params=("prompt", "lyrics", "cot", "abc", "abc_ids", "prefix", "seed", "cfg_scale", "progress_bar_cmd", "on_token"),
            output_params=("semantic_tokens",),
        )

    def process(self, pipe, prompt, lyrics, cot, abc, abc_ids, prefix, seed, cfg_scale, progress_bar_cmd, on_token):
        if self.prefix(prompt, lyrics, cot, abc, pipe.tokenizer, abc_ids) != prefix:
            raise ValueError("Plan prefix disagrees with the request and ABC IDs")
        guidance = self.guidance(cot, cfg_scale)
        negative = self.negative_prefix(prompt, lyrics, cot, pipe.tokenizer, abc_ids) if guidance != 1 else None
        pipe.load_models_to_device(["mot"])
        ids, _ = self.generate(pipe, pipe.mot, prefix, self.sampling(pipe, "semantic"), seed, "semantic",
                               negative=negative, cfg_scale=guidance, legacy_off=cot == "off",
                               progress_bar_cmd=progress_bar_cmd, on_token=on_token)
        return {"semantic_tokens": [int(token) - self.codec_offset for token in ids]}


@dataclass
class _Chunk:
    ar_tokens: list[int]
    noise: torch.Tensor
    nar_cond_end: int = 0


def _integers(values, name):
    result = list(values)
    if not result or any(isinstance(v, bool) or not isinstance(v, Integral) for v in result):
        raise ValueError(f"{name} must be a nonempty sequence of integer token IDs")
    return [int(v) for v in result]


def _chunk_ranges(frames, prefix_tokens, context=YuE2Unit_ARGenerator.context):
    size = min((context - prefix_tokens - 3) // 2, context)
    if frames < 1 or size < 1:
        raise ValueError("Empty codec or prefix leaves no acoustic context")
    return [(a, min(a + size, frames)) for a in range(0, frames, size)]


def _song_chunks(prefix, codec, seed, context):
    prefix = _integers(prefix, "prefix")
    codec = _integers(codec, "codec")
    if min(prefix) < 0 or min(codec) < 0 or max(codec) >= YuE2Unit_ARGenerator.codec_size:
        raise ValueError("Token IDs are outside their allowed vocabulary")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    if isinstance(context, bool) or not isinstance(context, Integral) or not 1 <= context <= YuE2Unit_ARGenerator.context:
        raise ValueError("context must be a valid integer")
    ranges = _chunk_ranges(len(codec), len(prefix), int(context))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=generator)
    return [_Chunk(prefix + [value + YuE2Unit_ARGenerator.codec_offset for value in codec[a:b]]
                   + [YuE2Unit_ARGenerator.music_end], noise[a:b]) for a, b in ranges]


def _attention(q, k, v, *, causal=False, query_chunk_size=None):
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("Expected Q/K/V [tokens, heads, dim] with matching K/V")
    if min(q.shape) < 1 or min(k.shape) < 1 or q.shape[1] % k.shape[1]:
        raise ValueError("Invalid attention lengths or grouped-query head count")
    if causal and len(q) != len(k):
        raise ValueError("Causal prefill requires matching Q/K sequence lengths")
    if query_chunk_size is not None and (isinstance(query_chunk_size, bool) or
                                        not isinstance(query_chunk_size, Integral) or query_chunk_size < 1):
        raise ValueError("query_chunk_size must be a positive integer")
    block = query_chunk_size or (len(q) if q.device.type == "cuda" else 256)
    query = q.transpose(0, 1).unsqueeze(0)
    key = k.transpose(0, 1).unsqueeze(0)
    value = v.transpose(0, 1).unsqueeze(0)
    grouped = query.shape[1] != key.shape[1]
    if grouped and q.device.type == "mps":
        groups = query.shape[1] // key.shape[1]
        key, value = key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1)
        grouped = False
    outputs = []
    for start in range(0, len(q), block):
        end = min(start + block, len(q))
        used_key = key[..., :end, :] if causal else key
        used_value = value[..., :end, :] if causal else value
        mask = None
        if causal and start:
            mask = (torch.arange(end, device=q.device)[None, :] <=
                    torch.arange(start, end, device=q.device)[:, None])
        outputs.append(F.scaled_dot_product_attention(
            query[..., start:end, :], used_key, used_value,
            attn_mask=mask, is_causal=causal and start == 0, enable_gqa=grouped,
        ))
    return torch.cat(outputs, dim=-2)[0].transpose(0, 1)


class _CachedNAR:

    def __init__(self, model, chunk: _Chunk, device, dtype, query_chunk_size=None):
        self.model, self.chunk = model, chunk
        self.device, self.dtype = torch.device(device), dtype
        self.query_chunk_size = query_chunk_size
        if chunk.noise.ndim != 2 or chunk.noise.shape[1] != 64 or len(chunk.noise) < 1:
            raise ValueError("Expected nonempty acoustic noise [frames,64]")
        if not torch.isfinite(chunk.noise).all():
            raise ValueError("Acoustic noise contains non-finite values")
        self.ar_length, self.nar_length = len(chunk.ar_tokens), len(chunk.noise) + 2
        if self.ar_length < 1 or min(chunk.ar_tokens) < 0 or max(chunk.ar_tokens) >= model.config.vocab_size:
            raise ValueError("AR prefix is empty or outside the model vocabulary")
        if self.ar_length + self.nar_length > model.config.max_position_embeddings:
            raise ValueError("Original acoustic chunk exceeds the model context")
        if chunk.nar_cond_end < 0:
            raise ValueError("nar_cond_end must be nonnegative")
        self.visible_length = min(chunk.nar_cond_end, self.ar_length) if chunk.nar_cond_end else self.ar_length
        positions = torch.arange(self.ar_length, self.ar_length + self.nar_length, device=self.device)[None]
        self.cos, self.sin = model.model.rotary_emb(positions)
        local = torch.arange(self.nar_length, device=self.device).clamp(max=model.config.max_latent_frames - 1)
        self.pos_emb = model.latent_pos_embed(local)[None]
        self.cache = []
        self._prefill()

    def _attention(self, q, k, v, causal=False):
        return _attention(q, k, v, causal=causal, query_chunk_size=self.query_chunk_size)

    @torch.inference_mode()
    def _prefill(self):
        backbone = self.model.model
        ids = torch.tensor([self.chunk.ar_tokens], dtype=torch.long, device=self.device)
        positions = torch.arange(self.ar_length, device=self.device)[None]
        cos, sin = backbone.rotary_emb(positions)
        x = backbone.embed_tokens(ids)
        for layer in backbone.layers:
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
            # Clone only for restricted visibility; a slice would retain the storage of invisible codec tokens.
            cached = (k[0, :self.visible_length], v[0, :self.visible_length])
            if self.visible_length != self.ar_length:
                cached = tuple(t.clone() for t in cached)
            self.cache.append(cached)
            h = self._attention(q[0], k[0], v[0], causal=True)
            x = x + layer.self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.mlp(layer.post_attention_layernorm(x))

    @torch.inference_mode()
    def velocity(self, state, raw_t):
        model = self.model
        if tuple(state.shape) != tuple(self.chunk.noise.shape):
            raise ValueError("ODE state shape changed")
        x_nar = F.pad(state, (0, 0, 1, 1))
        shifted = model._shift_t_value(raw_t, self.device, self.dtype)
        x = model.vae2llm(x_nar[None])
        x = x + model.time_embedder(shifted.expand(self.nar_length))[None]
        x = x + self.pos_emb
        for layer, (ar_k, ar_v) in zip(model.model.layers, self.cache):
            q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), self.cos, self.sin)
            k, v = torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))
            h = self._attention(q[0], k, v)
            x = x + layer.nar_self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        return model.llm2vae(model.model.norm(x))[0, 1:-1]

    def close(self):
        self.cache.clear()
        self.cos = self.sin = self.pos_emb = None


class YuE2Unit_NARConditioner(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=("prefix", "semantic_tokens", "seed"),
            output_params=("nar_chunks",),
        )

    def process(self, pipe, prefix, semantic_tokens, seed):
        context = (pipe.generation_config or {}).get("context", YuE2Unit_ARGenerator.context)
        return {"nar_chunks": _song_chunks(prefix, semantic_tokens, seed, context)}


class YuE2Unit_InputAudioEmbedder(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=("input_audio",),
            output_params=("input_latents", "latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, input_audio):
        if not pipe.scheduler.training or input_audio is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        audio = input_audio[0] if isinstance(input_audio, tuple) else input_audio
        audio = torch.clamp(audio.float(), -1.0, 1.0)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        with torch.no_grad():
            input_latents = pipe.vae.encode(audio.to(device=pipe.device, dtype=torch.float32)).transpose(1, 2)
        return {"input_latents": input_latents, "latents": torch.zeros_like(input_latents)}


def model_fn_yue2(
    mot: YuE2MoT,
    latents=None,
    timestep=None,
    engine=None,
    input_latents=None,
    prefix=None,
    semantic_tokens=None,
    nar_cond_end=0,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    sigma = (timestep.to(device=latents.device).double().reshape(-1)[0] / 1000.0).clamp(1e-3, 1 - 1e-3)
    t_value = float(torch.log(sigma / (1 - sigma)).clamp(-20, 20))
    if engine is not None:
        return engine.velocity(latents, t_value)
    x_t = (latents if latents is not None else input_latents)[0].float()
    frames = x_t.shape[0]
    device = x_t.device
    prefix = torch.as_tensor(prefix, dtype=torch.long, device=device).reshape(1, -1)
    codec = torch.as_tensor(semantic_tokens, dtype=torch.long, device=device).reshape(1, -1) + YuE2Unit_ARGenerator.codec_offset
    nar_slots = torch.tensor([YuE2Unit_ARGenerator.latent_start] + [YuE2Unit_ARGenerator.latent_pad] * frames
                             + [YuE2Unit_ARGenerator.latent_end], dtype=torch.long, device=device).reshape(1, -1)
    tokens = torch.cat([prefix, codec,
                        torch.tensor([[YuE2Unit_ARGenerator.music_end]], dtype=torch.long, device=device), nar_slots], dim=1)
    sequence_length = tokens.shape[1]
    ar_length = prefix.shape[1] + codec.shape[1] + 1
    ar_mask = torch.zeros(1, sequence_length, dtype=torch.bool, device=device)
    ar_mask[:, :ar_length] = True
    nar_mask = ~ar_mask
    nar_content_mask = torch.zeros(1, sequence_length, dtype=torch.bool, device=device)
    nar_content_mask[:, ar_length + 1:ar_length + 1 + frames] = True
    velocity = mot.nar_velocity(
        tokens, ar_mask, nar_mask, nar_content_mask, x_t, t_value, nar_cond_end,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return velocity.unsqueeze(0)
