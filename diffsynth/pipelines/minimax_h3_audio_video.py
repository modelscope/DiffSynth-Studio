import math

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.minimax_h3_dit import (
    MiniMaxH3DiT,
    patchify_video_latent,
    unpatchify_video_tokens,
    pack_audio_latent,
    unpack_audio_tokens,
)
from ..models.minimax_h3_text_encoder import MiniMaxH3TextEncoder
from ..models.minimax_h3_video_vae import MiniMaxH3VideoVAE
from ..models.minimax_h3_audio_vae import MiniMaxH3AudioVAE


# Per-channel latent normalization (from checkpoint video_vae/config.json and
# audio_vae/config.json). decode does: latent * std + mean, then VAE.decode.
_VIDEO_LATENTS_MEAN = [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279, -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264]
_VIDEO_LATENTS_STD = [1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264, 3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523]
_AUDIO_LATENTS_MEAN = [-0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373, 0.08179686214561671, -0.35782641352446604, 0.040623809960919084, -0.01552534501956604, -0.223362481667332, 0.1821006842509091, 0.2941778783780663, -0.07901167601970885, -0.056815072777201, -0.3699028221860095, -0.31616315591624855, 0.5905951377425391, -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163, -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344, 0.02868032184767538, -0.21217779266454084, -0.1678263169941987, 0.3221287889040614, -0.1223055851554907, 0.4356604928128464, -0.0502599202236253, 0.3979258376211797]
_AUDIO_LATENTS_STD = [1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828, 1.6390226546605453, 2.7788298348882177, 1.7659090095747236, 1.6199757612137327, 2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178, 1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373, 1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093, 1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912, 1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855, 1.718045989838149, 1.6307219190837705, 1.8661226051202384, 1.5613768203168363]


def _frames_to_pil(frames: torch.Tensor) -> list:
    # frames: [1, 3, T, H, W] in [0,1] -> list of T PIL RGB images.
    frames = frames[0].clamp(0, 1).mul(255).round().to(torch.uint8).cpu()  # [3,T,H,W]
    frames = frames.permute(1, 2, 3, 0).numpy()  # [T,H,W,3]
    return [Image.fromarray(frames[t]) for t in range(frames.shape[0])]


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
        flow_shift: float = 12.0,
        audio_flow_shift: float = 3.0,
        progress_bar_cmd=tqdm,
    ):
        # 1. Schedulers (video / audio独立 shift)
        self.scheduler.set_timesteps(num_inference_steps, shift=flow_shift)
        self.scheduler_audio.set_timesteps(num_inference_steps, shift=audio_flow_shift)

        # 2. three-dict inputs (no CFG/negative -> prompt is a shared param)
        inputs_posi = {}
        inputs_nega = {}
        inputs_shared = {
            "prompt": prompt,
            "height": height, "width": width, "num_frames": num_frames,
            "seed": seed, "rand_device": rand_device,
        }

        # 3. Unit chain
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # 4. Denoise loop (single-stream packed; video/audio stepped by their own
        # scheduler; no CFG). model_fn negates velocity and returns natural-form
        # noise_pred, so self.step reproduces the target's euler eta0 update.
        self.load_models_to_device(self.in_iteration_models)
        video_latents = inputs_shared["video_latents"]
        audio_latents = inputs_shared["audio_latents"]
        packed = inputs_shared["packed"]
        prompt_embeds = inputs_shared["prompt_embeds"]
        for progress_id, _ in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            t_video = float(1.0 - self.scheduler.sigmas[progress_id])
            t_audio = float(1.0 - self.scheduler_audio.sigmas[progress_id])
            v_video, v_audio = self.model_fn(
                self.dit, video_latents, audio_latents, packed, prompt_embeds,
                t_video=t_video, t_audio=t_audio, device=self.device, torch_dtype=self.torch_dtype,
            )
            video_latents = self.step(self.scheduler, video_latents, progress_id, noise_pred=v_video)
            audio_latents = self.step(self.scheduler_audio, audio_latents, progress_id, noise_pred=v_audio)

        # 5. Decode (de-normalize latents -> VAE); cast to each VAE's param dtype.
        self.load_models_to_device(["video_vae"])
        v_mean = torch.tensor(_VIDEO_LATENTS_MEAN, device=self.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        v_std = torch.tensor(_VIDEO_LATENTS_STD, device=self.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        v_dtype = next(self.video_vae.parameters()).dtype
        video_dec = (video_latents.to(self.device, torch.float32) * v_std + v_mean).to(v_dtype)
        frames = self.video_vae.decode_base(video_dec)
        frames = frames[0] if isinstance(frames, (tuple, list)) else frames
        # pixel de-normalization -> [0,1], then to a list of PIL frames
        frames = self.video_vae.processor.revert_tensor(frames.float())  # [1,3,T,H,W] in [0,1]
        video = _frames_to_pil(frames)

        self.load_models_to_device(["audio_vae"])
        a_mean = torch.tensor(_AUDIO_LATENTS_MEAN, device=self.device, dtype=torch.float32).view(1, -1, 1)
        a_std = torch.tensor(_AUDIO_LATENTS_STD, device=self.device, dtype=torch.float32).view(1, -1, 1)
        a_dtype = next(self.audio_vae.parameters()).dtype
        audio_dec = (audio_latents.to(self.device, torch.float32) * a_std + a_mean).to(a_dtype)
        waveform = self.audio_vae.decode(audio_dec)  # [C, 1, samples]
        audio = waveform.squeeze(1).float().cpu()    # -> (channels, samples) for write_video_audio
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
            input_params=("seed", "video_latent_t", "latent_h", "latent_w", "audio_latent_t"),
            output_params=("video_latents", "audio_latents"),
        )

    def process(self, pipe: MiniMaxH3Pipeline, seed, video_latent_t, latent_h, latent_w, audio_latent_t):
        if seed is None:
            seed = 42
        # Noise contract (target latent_preparation): CPU fp32, video drawn on the
        # raw 5D latent [1,24,T,H,W]; audio drawn as channel-major rows
        # [audio_t*2,32] with an INDEPENDENT generator re-seeded with the same seed.
        gen_v = torch.Generator().manual_seed(int(seed))
        video_latents = torch.randn(1, 24, video_latent_t, latent_h, latent_w, generator=gen_v, dtype=torch.float32)
        gen_a = torch.Generator().manual_seed(int(seed))
        audio_rows = torch.randn(audio_latent_t * 2, 32, generator=gen_a, dtype=torch.float32)
        audio_latents = unpack_audio_tokens(audio_rows, audio_channel=2, steps=audio_latent_t)
        return {
            "video_latents": video_latents.to(pipe.device),
            "audio_latents": audio_latents.to(pipe.device),
        }


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
    t_video, t_audio, device, torch_dtype, **kwargs,
):
    # ---- patchify (in): natural latents -> packed rows ----
    video_rows = patchify_video_latent(video_latents.to(device, torch.float32))
    audio_rows = pack_audio_latent(audio_latents.to(device, torch.float32))

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
        skip_mask_out_condition=False,
    )

    # ---- unpatchify (out) + velocity negation ----
    v_video = unpatchify_video_tokens(v_video_rows.float(), packed["latent_t"], packed["latent_h_patched"], packed["latent_w_patched"])
    v_audio = unpack_audio_tokens(v_audio_rows.float(), packed["audio_channel"], packed["audio_t"])
    return -v_video, -v_audio
