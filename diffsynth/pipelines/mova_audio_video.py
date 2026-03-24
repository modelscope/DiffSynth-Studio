import sys
import torch, types
from PIL import Image
from typing import Optional, Union
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d, set_to_torch_norm
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.mova_audio_dit import MovaAudioDit
from ..models.mova_audio_vae import DacVAE
from ..models.mova_dual_tower_bridge import DualTowerConditionalBridge
from ..utils.data.audio import convert_to_mono, resample_waveform


class MovaAudioVideoPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.video_dit: WanModel = None # high noise model
        self.video_dit2: WanModel = None # low noise model
        self.audio_dit: MovaAudioDit = None
        self.dual_tower_bridge: DualTowerConditionalBridge = None
        self.video_vae: WanVideoVAE = None
        self.audio_vae: DacVAE = None

        self.in_iteration_models = ("video_dit", "audio_dit", "dual_tower_bridge")
        self.in_iteration_models_2 = ("video_dit2", "audio_dit", "dual_tower_bridge")

        self.units = [
            MovaAudioVideoUnit_ShapeChecker(),
            MovaAudioVideoUnit_NoiseInitializer(),
            MovaAudioVideoUnit_InputVideoEmbedder(),
            MovaAudioVideoUnit_InputAudioEmbedder(),
            MovaAudioVideoUnit_PromptEmbedder(),
            MovaAudioVideoUnit_ImageEmbedderVAE(),
            MovaAudioVideoUnit_UnifiedSequenceParallel(),
        ]
        self.model_fn = model_fn_mova_audio_video
        self.compilable_models = ["video_dit", "video_dit2", "audio_dit"]

    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward
        for block in self.video_dit.blocks + self.audio_dit.blocks + self.video_dit2.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="openmoss/MOVA-720p", origin_file_pattern="tokenizer/"),
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp(device)
            import torch.distributed as dist
            from ..core.device.npu_compatible_device import get_device_name
            if dist.is_available() and dist.is_initialized():
                device = get_device_name()
        # Initialize pipeline
        pipe = MovaAudioVideoPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.video_dit, pipe.video_dit2 = dit
        else:
            pipe.video_dit = dit
        pipe.audio_dit = model_pool.fetch_model("mova_audio_dit")
        pipe.dual_tower_bridge = model_pool.fetch_model("mova_dual_tower_bridge")
        pipe.video_vae = model_pool.fetch_model("wan_video_vae")
        pipe.audio_vae = model_pool.fetch_model("mova_audio_vae")
        set_to_torch_norm([pipe.video_dit, pipe.audio_dit, pipe.dual_tower_bridge] + ([pipe.video_dit2] if pipe.video_dit2 is not None else []))

        # Size division factor
        if pipe.video_vae is not None:
            pipe.height_division_factor = pipe.video_vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.video_vae.upsampling_factor * 2

        # Initialize tokenizer and processor
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')

        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        denoising_strength: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 352,
        width: Optional[int] = 640,
        num_frames: Optional[int] = 81,
        frame_rate: Optional[int] = 24,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.9,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Inputs
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "denoising_strength": denoising_strength,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames, "frame_rate": frame_rate,
            "cfg_scale": cfg_scale,
            "sigma_shift": sigma_shift,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * 1000 and self.video_dit2 is not None and not models["video_dit"] is self.video_dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["video_dit"] = self.video_dit2
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred_video, noise_pred_audio = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            # Scheduler
            inputs_shared["video_latents"] = self.step(self.scheduler, inputs_shared["video_latents"], progress_id=progress_id, noise_pred=noise_pred_video, **inputs_shared)
            inputs_shared["audio_latents"] = self.step(self.scheduler, inputs_shared["audio_latents"], progress_id=progress_id, noise_pred=noise_pred_audio, **inputs_shared)

        # Decode
        self.load_models_to_device(['video_vae'])
        video = self.video_vae.decode(inputs_shared["video_latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device(["audio_vae"])
        audio = self.audio_vae.decode(inputs_shared["audio_latents"])
        audio = self.output_audio_format_check(audio)
        self.load_models_to_device([])
        return video, audio


class MovaAudioVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: MovaAudioVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class MovaAudioVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "frame_rate"),
            output_params=("video_noise", "audio_noise")
        )

    def process(self, pipe: MovaAudioVideoPipeline, height, width, num_frames, seed, rand_device, frame_rate):
        length = (num_frames - 1) // 4 + 1
        video_shape = (1, pipe.video_vae.model.z_dim, length, height // pipe.video_vae.upsampling_factor, width // pipe.video_vae.upsampling_factor)
        video_noise = pipe.generate_noise(video_shape, seed=seed, rand_device=rand_device)

        audio_num_samples = (int(pipe.audio_vae.sample_rate * num_frames / frame_rate) - 1) // int(pipe.audio_vae.hop_length) + 1
        audio_shape = (1, pipe.audio_vae.latent_dim, audio_num_samples)
        audio_noise = pipe.generate_noise(audio_shape, seed=seed, rand_device=rand_device)
        return {"video_noise": video_noise, "audio_noise": audio_noise}


class MovaAudioVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "video_noise", "tiled", "tile_size", "tile_stride"),
            output_params=("video_latents", "input_latents"),
            onload_model_names=("video_vae",)
        )

    def process(self, pipe: MovaAudioVideoPipeline, input_video, video_noise, tiled, tile_size, tile_stride):
        if input_video is None or not pipe.scheduler.training:
            return {"video_latents": video_noise}
        else:
            pipe.load_models_to_device(self.onload_model_names)
            input_video = pipe.preprocess_video(input_video)
            input_latents = pipe.video_vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"input_latents": input_latents}


class MovaAudioVideoUnit_InputAudioEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_audio", "audio_noise"),
            output_params=("audio_latents", "audio_input_latents"),
            onload_model_names=("audio_vae",)
        )

    def process(self, pipe: MovaAudioVideoPipeline, input_audio, audio_noise):
        if input_audio is None or not pipe.scheduler.training:
            return {"audio_latents": audio_noise}
        else:
            pipe.load_models_to_device(self.onload_model_names)
            input_audio, sample_rate = input_audio
            input_audio = convert_to_mono(input_audio)
            input_audio = resample_waveform(input_audio, sample_rate, pipe.audio_vae.sample_rate)
            input_audio = pipe.audio_vae.preprocess(input_audio.unsqueeze(0), pipe.audio_vae.sample_rate)
            z, _, _, _, _ = pipe.audio_vae.encode(input_audio)
            return {"audio_input_latents": z.mode()}


class MovaAudioVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )

    def encode_prompt(self, pipe: MovaAudioVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_mask=True,
            return_tensors="pt",
        )
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: MovaAudioVideoPipeline, prompt) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}


class MovaAudioVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("video_vae",)
        )

    def process(self, pipe: MovaAudioVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.video_dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)

        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]

        y = pipe.video_vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


class MovaAudioVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("use_unified_sequence_parallel",))

    def process(self, pipe: MovaAudioVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel") and pipe.use_unified_sequence_parallel:
            return {"use_unified_sequence_parallel": True}
        return {"use_unified_sequence_parallel": False}


def model_fn_mova_audio_video(
    video_dit: WanModel,
    audio_dit: MovaAudioDit,
    dual_tower_bridge: DualTowerConditionalBridge,
    video_latents: torch.Tensor = None,
    audio_latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    y: Optional[torch.Tensor] = None,
    frame_rate: Optional[int] = 24,
    use_unified_sequence_parallel: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    video_x, audio_x = video_latents, audio_latents
    # First-Last Frame
    if y is not None:
        video_x = torch.cat([video_x, y], dim=1)

    # Timestep
    video_t = video_dit.time_embedding(sinusoidal_embedding_1d(video_dit.freq_dim, timestep))
    video_t_mod = video_dit.time_projection(video_t).unflatten(1, (6, video_dit.dim))
    audio_t = audio_dit.time_embedding(sinusoidal_embedding_1d(audio_dit.freq_dim, timestep))
    audio_t_mod = audio_dit.time_projection(audio_t).unflatten(1, (6, audio_dit.dim))

    # Context
    video_context = video_dit.text_embedding(context)
    audio_context = audio_dit.text_embedding(context)

    # Patchify
    video_x = video_dit.patch_embedding(video_x)
    f_v, h, w = video_x.shape[2:]
    video_x = rearrange(video_x, 'b c f h w -> b (f h w) c').contiguous()
    seq_len_video = video_x.shape[1]

    audio_x = audio_dit.patch_embedding(audio_x)
    f_a = audio_x.shape[2]
    audio_x = rearrange(audio_x, 'b c f -> b f c').contiguous()
    seq_len_audio = audio_x.shape[1]

    # Freqs
    video_freqs = torch.cat([
        video_dit.freqs[0][:f_v].view(f_v, 1, 1, -1).expand(f_v, h, w, -1),
        video_dit.freqs[1][:h].view(1, h, 1, -1).expand(f_v, h, w, -1),
        video_dit.freqs[2][:w].view(1, 1, w, -1).expand(f_v, h, w, -1)
    ], dim=-1).reshape(f_v * h * w, 1, -1).to(video_x.device)
    audio_freqs = torch.cat([
        audio_dit.freqs[0][:f_a].view(f_a, -1).expand(f_a, -1),
        audio_dit.freqs[1][:f_a].view(f_a, -1).expand(f_a, -1),
        audio_dit.freqs[2][:f_a].view(f_a, -1).expand(f_a, -1),
    ], dim=-1).reshape(f_a, 1, -1).to(audio_x.device)

    video_rope, audio_rope = dual_tower_bridge.build_aligned_freqs(
        video_fps=frame_rate,
        grid_size=(f_v, h, w),
        audio_steps=audio_x.shape[1],
        device=video_x.device,
        dtype=video_x.dtype,
    )
    # usp func
    if use_unified_sequence_parallel:
        from ..utils.xfuser import get_current_chunk, gather_all_chunks
    else:
        get_current_chunk = lambda x, dim=1: x
        gather_all_chunks = lambda x, seq_len, dim=1: x
    # Forward blocks
    for block_id in range(len(audio_dit.blocks)):
        if dual_tower_bridge.should_interact(block_id, "a2v"):
            video_x, audio_x = dual_tower_bridge(
                block_id,
                video_x,
                audio_x,
                x_freqs=video_rope,
                y_freqs=audio_rope,
                condition_scale=1.0,
                video_grid_size=(f_v, h, w),
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        video_x = get_current_chunk(video_x, dim=1)
        video_x = gradient_checkpoint_forward(
            video_dit.blocks[block_id],
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            video_x, video_context, video_t_mod, video_freqs
        )
        video_x = gather_all_chunks(video_x, seq_len=seq_len_video, dim=1)
        audio_x = get_current_chunk(audio_x, dim=1)
        audio_x = gradient_checkpoint_forward(
            audio_dit.blocks[block_id],
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            audio_x, audio_context, audio_t_mod, audio_freqs
        )
        audio_x = gather_all_chunks(audio_x, seq_len=seq_len_audio, dim=1)

    video_x = get_current_chunk(video_x, dim=1)
    for block_id in range(len(audio_dit.blocks), len(video_dit.blocks)):
        video_x = gradient_checkpoint_forward(
            video_dit.blocks[block_id],
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            video_x, video_context, video_t_mod, video_freqs
        )
    video_x = gather_all_chunks(video_x, seq_len=seq_len_video, dim=1)

    # Head
    video_x = video_dit.head(video_x, video_t)
    video_x = video_dit.unpatchify(video_x, (f_v, h, w))

    audio_x = audio_dit.head(audio_x, audio_t)
    audio_x = audio_dit.unpatchify(audio_x, (f_a,))
    return video_x, audio_x
