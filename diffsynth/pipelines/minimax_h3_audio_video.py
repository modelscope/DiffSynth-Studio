import numpy as np
import torch
from tqdm import tqdm

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.minimax_h3_dit import (
    MiniMaxH3DiT,
    patchify_video,
    unpatchify_video,
    pack_audio,
    unpack_audio,
)
from ..models.minimax_h3_text_encoder import MiniMaxH3TextEncoder
from ..models.minimax_h3_video_vae import MiniMaxH3VideoVAE
from ..models.minimax_h3_audio_vae import MiniMaxH3AudioVAE


class MiniMaxH3Pipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=32, width_division_factor=32,
        )
        # Two independent Rectified-Flow schedulers: video (shift=12) and audio
        # (shift=3). Both use the "MiniMax-H3" template; velocity is negated in
        # model_fn so the standard euler step matches the target's eta0 update.
        self.scheduler = FlowMatchScheduler("MiniMax-H3")
        self.scheduler_audio = FlowMatchScheduler("MiniMax-H3")
        self.text_encoder: MiniMaxH3TextEncoder = None
        self.dit: MiniMaxH3DiT = None
        self.video_vae: MiniMaxH3VideoVAE = None
        self.audio_vae: MiniMaxH3AudioVAE = None
        self.tokenizer = None
        self.in_iteration_models = ("dit",)
        self.units = [
            MiniMaxH3Unit_ShapeChecker(),
            MiniMaxH3Unit_PromptEmbedder(),
            MiniMaxH3Unit_NoiseInitializer(),
            MiniMaxH3Unit_PackedSequenceBuilder(),
        ]
        self.model_fn = model_fn_minimax_h3
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        pipe = MiniMaxH3Pipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("minimax_h3_text_encoder")
        pipe.dit = model_pool.fetch_model("minimax_h3_dit")
        pipe.video_vae = model_pool.fetch_model("minimax_h3_video_vae")
        pipe.audio_vae = model_pool.fetch_model("minimax_h3_audio_vae")
        if pipe.audio_vae is not None and hasattr(pipe.audio_vae, "remove_weight_norm"):
            pipe.audio_vae.remove_weight_norm()
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
        height: int = 768,
        width: int = 1344,
        num_frames: int = 24,
        num_inference_steps: int = 50,
        seed: int = 42,
        rand_device: str = "cpu",
        cfg_scale: float = 1.0,
        flow_shift: float = 12.0,
        audio_flow_shift: float = 3.0,
        tiled: bool = True,
        tile_size: int = None,
        tile_overlap: int = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        progress_bar_cmd=tqdm,
    ):
        # 1. Schedulers (video / audio独立 shift)
        self.scheduler.set_timesteps(num_inference_steps, shift=flow_shift)
        self.scheduler_audio.set_timesteps(num_inference_steps, shift=audio_flow_shift)

        # 2. three-dict inputs. MiniMax-H3 is CFG-distilled: cfg_scale defaults to
        # 1.0 (single forward, no negative prompt); prompt is a shared param.
        inputs_posi = {}
        inputs_nega = {}
        inputs_shared = {
            "prompt": prompt,
            "height": height, "width": width, "num_frames": num_frames,
            "seed": seed, "rand_device": rand_device,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": use_gradient_checkpointing_offload,
        }

        # 3. Unit chain
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # 4. Denoise loop. Video/audio are stepped by their own scheduler; the DiT
        # forward runs through cfg_guided_model_fn (cfg_scale=1.0 -> single forward).
        # model_fn negates velocity, so the standard euler step reproduces the
        # target's eta0 update. t_video/t_audio are the per-modality flow times.
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, _ in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            t_video = float(1.0 - self.scheduler.sigmas[progress_id])
            t_audio = float(1.0 - self.scheduler_audio.sigmas[progress_id])
            noise_pred_video, noise_pred_audio = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega,
                **models, t_video=t_video, t_audio=t_audio,
                device=self.device, torch_dtype=self.torch_dtype,
            )
            inputs_shared["video_latents"] = self.step(
                self.scheduler, inputs_shared["video_latents"], progress_id, noise_pred=noise_pred_video,
            )
            inputs_shared["audio_latents"] = self.step(
                self.scheduler_audio, inputs_shared["audio_latents"], progress_id, noise_pred=noise_pred_audio,
            )

        # 5. Decode (de-normalization + VAE decode live inside each VAE).
        self.load_models_to_device(["video_vae"])
        frames = self.video_vae.decode_video(
            inputs_shared["video_latents"], tiled=tiled, tile_size=tile_size, tile_overlap=tile_overlap,
        )  # [1,3,T,H,W] in [0,1]
        video = self.vae_output_to_video(frames, min_value=0, max_value=1)

        self.load_models_to_device(["audio_vae"])
        waveform = self.audio_vae.decode_audio(inputs_shared["audio_latents"])  # [1, C, L]
        audio = self.output_audio_format_check(waveform)  # (channels, samples)
        return video, audio


class MiniMaxH3Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "video_latent_t", "audio_latent_t", "latent_h", "latent_w"),
        )

    @staticmethod
    def _align_frame_count(frame_count: int) -> int:
        current = max(int(frame_count), 1)
        while current % 17 != 5:
            current += 1
        return current

    @staticmethod
    def _video_latent_t(frame_count: int) -> int:
        if frame_count <= 5:
            return 2
        return ((int(frame_count) - 5) // 17) * 5 + 2

    def process(self, pipe: MiniMaxH3Pipeline, height, width, num_frames):
        height, width = pipe.check_resize_height_width(height, width)
        frame_count = self._align_frame_count(num_frames)
        video_latent_t = self._video_latent_t(frame_count)
        audio_latent_t = int(round(float(num_frames) / 24.0 * 40.0))
        return {
            "height": height, "width": width,
            "video_latent_t": video_latent_t, "audio_latent_t": audio_latent_t,
            "latent_h": height // 16, "latent_w": width // 16,
        }


class MiniMaxH3Unit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("prompt",),
            output_params=("prompt_embeds",),
            onload_model_names=("text_encoder",),
        )

    def process(self, pipe: MiniMaxH3Pipeline, prompt):
        pipe.load_models_to_device(("text_encoder",))
        # t2va presentation: verbatim prompt ids, no special tokens (target
        # minimax_h3_text_only_ids + encode_ids -> Qwen3-VL layer-50 hidden).
        input_ids = torch.tensor(
            pipe.tokenizer(text=prompt, add_special_tokens=False)["input_ids"], dtype=torch.long
        )[None].to(pipe.device)
        attn = torch.ones_like(input_ids)
        hidden = pipe.text_encoder(input_ids=input_ids, attention_mask=attn)
        prompt_embeds = hidden[0].to(pipe.device, torch.bfloat16)
        return {"prompt_embeds": prompt_embeds}


class MiniMaxH3Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("seed", "video_latent_t", "latent_h", "latent_w", "audio_latent_t", "rand_device"),
            output_params=("video_latents", "audio_latents"),
        )

    def process(self, pipe: MiniMaxH3Pipeline, seed, video_latent_t, latent_h, latent_w, audio_latent_t, rand_device):
        if seed is None:
            seed = 42
        # Noise contract (target latent_preparation): CPU fp32, video drawn on the
        # raw 5D latent [1,24,T,H,W]; audio drawn as channel-major rows
        # [audio_t*2,32] with an INDEPENDENT generator re-seeded with the same seed.
        # generate_noise builds a fresh generator per call, so both draws are
        # independent yet share the seed.
        video_latents = pipe.generate_noise(
            (1, 24, video_latent_t, latent_h, latent_w),
            seed=seed, rand_device=rand_device, rand_torch_dtype=torch.float32, torch_dtype=torch.float32,
        )
        audio_rows = pipe.generate_noise(
            (audio_latent_t * 2, 32),
            seed=seed, rand_device=rand_device, rand_torch_dtype=torch.float32, torch_dtype=torch.float32,
        )
        audio_latents = unpack_audio(audio_rows, audio_channel=2, steps=audio_latent_t)
        return {"video_latents": video_latents, "audio_latents": audio_latents}


class MiniMaxH3Unit_PackedSequenceBuilder(PipelineUnit):
    # Packed-sequence constants (ported from target .../minimax_h3/packed_sequence.py).
    # t2va layout: [text | audio | video | pad].
    _INTERP = 32
    _T_GROUP = 5
    _FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    _FRAME_RESCALE = 5.0 / 3.0
    _SEQ_ALIGN = 64
    _PATCH_H, _PATCH_W = 2, 2
    _TEXT_ID, _AUDIO_FIRST_ID, _AUDIO_ID = -5, -15, -14
    _VIDEO_FIRST_ID, _VIDEO_ID, _VIDEO_LAST_ID, _PAD_ID = -3, -2, -4, -1

    def __init__(self):
        super().__init__(
            input_params=("prompt_embeds", "video_latent_t", "latent_h", "latent_w", "audio_latent_t"),
            output_params=("packed",),
        )

    @classmethod
    def _axis_from_sqrt_area(cls, dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
        ratio = dim / sqrt_area
        left = (1.0 - ratio) * 0.5
        right = left + ratio
        grid = np.linspace(left, right, dim // patch, endpoint=False) * cls._INTERP
        return torch.from_numpy(grid).to(torch.float64)

    @classmethod
    def _video_t_grid(cls, n: int, origin: float) -> torch.Tensor:
        spans = torch.tensor(
            [cls._FRAME_RESCALE * cls._FRAME_PER_TOKEN[k % cls._T_GROUP] for k in range(n)],
            dtype=torch.float64,
        )
        return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])

    @classmethod
    def _build_packed_t2va(cls, text_len, latent_t, latent_h, latent_w, audio_t, audio_channel=2):
        """t2va packed-sequence structural fields (target minimax_h3_packed_sequence,
        include_keyframe_cond=False)."""
        ph, pw = latent_h // cls._PATCH_H, latent_w // cls._PATCH_W
        frame_rows = ph * pw
        video_rows = latent_t * frame_rows
        audio_rows = audio_t * audio_channel
        used = text_len + audio_rows + video_rows
        seq_len = ((used + cls._SEQ_ALIGN - 1) // cls._SEQ_ALIGN) * cls._SEQ_ALIGN

        text_sl = slice(0, text_len)
        audio_sl = slice(text_len, text_len + audio_rows)
        video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)

        input_ids = torch.full((seq_len,), cls._PAD_ID, dtype=torch.int64)
        input_ids[text_sl] = cls._TEXT_ID
        input_ids[audio_sl] = cls._AUDIO_ID
        input_ids[audio_sl.start] = cls._AUDIO_FIRST_ID
        input_ids[video_sl] = cls._VIDEO_ID
        input_ids[video_sl.start] = cls._VIDEO_FIRST_ID
        input_ids[video_sl.stop - 1] = cls._VIDEO_LAST_ID

        img_pos = torch.arange(video_sl.start, video_sl.stop)
        update_mask = torch.ones(img_pos.shape[0], dtype=torch.bool)
        audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
        text_pos = torch.arange(0, text_len)

        g = torch.zeros(seq_len, 3, dtype=torch.float64)
        g[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)
        t_grid = cls._video_t_grid(latent_t, float(text_len))
        sqrt_area = np.sqrt(latent_h * latent_w)
        h_grid = cls._axis_from_sqrt_area(latent_h, cls._PATCH_H, sqrt_area)
        w_grid = cls._axis_from_sqrt_area(latent_w, cls._PATCH_W, sqrt_area)
        hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
        frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
        video_g = torch.empty(latent_t, frame_rows, 3, dtype=torch.float64)
        video_g[:, :, 0] = t_grid[:, None]
        video_g[:, :, 1:] = frame[None]
        g[video_sl] = video_g.reshape(-1, 3)
        audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
        g[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
        g[audio_sl, 2] = torch.cat([
            torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64),
            torch.full((audio_rows - audio_t,), float(w_grid[-1]), dtype=torch.float64),
        ])

        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[text_sl] = 1
        token_tags[audio_sl] = 2
        token_tags[img_pos] = 0

        cu = torch.tensor([0, used, seq_len], dtype=torch.int32)
        return {
            "seq_len": int(seq_len),
            "img_pos": img_pos,
            "audio_pos": audio_pos,
            "text_pos": text_pos,
            "update_mask": update_mask,
            "img_position_ids": g,
            "token_tags": token_tags,
            "cu_seqlens": cu,
            "text_len": int(text_len),
            "audio_channel": audio_channel,
            "audio_t": audio_t,
            "latent_t": latent_t,
            "latent_h_patched": ph,
            "latent_w_patched": pw,
        }

    def process(self, pipe: MiniMaxH3Pipeline, prompt_embeds, video_latent_t, latent_h, latent_w, audio_latent_t):
        text_len = int(prompt_embeds.shape[0])
        packed = self._build_packed_t2va(text_len, video_latent_t, latent_h, latent_w, audio_latent_t)
        dev = pipe.device
        packed["img_pos"] = packed["img_pos"].to(dev, torch.long)
        packed["audio_pos"] = packed["audio_pos"].to(dev, torch.long)
        packed["text_pos"] = packed["text_pos"].to(dev, torch.long)
        packed["update_mask"] = packed["update_mask"].to(dev)
        packed["img_position_ids"] = packed["img_position_ids"][None].to(dev)  # [1,S,3] fp64
        packed["token_tags"] = packed["token_tags"].to(dev, torch.long)
        packed["cu_seqlens"] = packed["cu_seqlens"].to(dev, torch.int32)
        return {"packed": packed}


def model_fn_minimax_h3(
    dit, video_latents, audio_latents, packed, prompt_embeds,
    t_video, t_audio, device, torch_dtype,
    use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
    **kwargs,
):
    # ---- patchify (in): natural latents -> packed rows ----
    video_rows = patchify_video(video_latents.to(device, torch.float32))
    audio_rows = pack_audio(audio_latents.to(device, torch.float32))

    seq_len = packed["seq_len"]
    img_pos = packed["img_pos"]
    audio_pos = packed["audio_pos"]
    text_pos = packed["text_pos"]
    cu = packed["cu_seqlens"]
    text_len = packed["text_len"]

    x = torch.zeros(1, seq_len, 96, dtype=torch.float32, device=device)
    x[0].index_copy_(0, img_pos, video_rows)
    audio_x = torch.zeros(1, seq_len, 32, dtype=torch.float32, device=device)
    audio_x[0].index_copy_(0, audio_pos, audio_rows)

    timesteps = torch.full((seq_len,), float(t_video), dtype=torch.float32, device=device)
    timesteps[audio_pos] = float(t_audio)
    unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)

    refiner_cu = torch.tensor([0, text_len, text_len], dtype=torch.int32, device=device)
    v_video_rows, v_audio_rows = dit(
        x=x,
        audio_x=audio_x,
        img_position_ids=packed["img_position_ids"],
        unique_timesteps=unique_timesteps,
        inverse_indices=inverse_indices,
        update_mask=packed["update_mask"],
        token_tags=packed["token_tags"],
        prompt_embeds=prompt_embeds.to(device, torch.bfloat16),
        img_pos_info={"position_ids": img_pos},
        audio_pos_info={"position_ids": audio_pos},
        text_pos_info={"position_ids": text_pos},
        img_pos_for_infer_output_info={"position_ids": img_pos},
        packed_seq_params={"cu_seqlens_q": cu, "max_seqlen_q": int(cu[1])},
        refiner_packed_seq_params={"cu_seqlens_q": refiner_cu, "max_seqlen_q": text_len},
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        skip_mask_out_condition=False,
    )

    # ---- unpatchify (out) + velocity negation ----
    v_video = unpatchify_video(v_video_rows.float(), packed["latent_t"], packed["latent_h_patched"], packed["latent_w_patched"])
    v_audio = unpack_audio(v_audio_rows.float(), packed["audio_channel"], packed["audio_t"])
    return -v_video, -v_audio
