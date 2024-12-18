from ..models import ModelManager, SD3TextEncoder1, HunyuanVideoVAEDecoder, HunyuanVideoVAEEncoder
from ..models.hunyuan_video_dit import HunyuanVideoDiT
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import HunyuanVideoPrompter
import torch
from transformers import LlamaModel
from einops import rearrange
import numpy as np
from PIL import Image



class HunyuanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=7.0, sigma_min=0.0, extra_one_step=True)
        self.prompter = HunyuanVideoPrompter()
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: LlamaModel = None
        self.dit: HunyuanVideoDiT = None
        self.vae_decoder: HunyuanVideoVAEDecoder = None
        self.vae_encoder: HunyuanVideoVAEEncoder = None
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'dit', 'vae_decoder', 'vae_encoder']
        self.vram_management = False


    def enable_vram_management(self):
        self.vram_management = True
        self.enable_cpu_offload()
        self.dit.enable_auto_offload(dtype=self.torch_dtype, device=self.device)


    def fetch_models(self, model_manager: ModelManager):
        self.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        self.text_encoder_2 = model_manager.fetch_model("hunyuan_video_text_encoder_2")
        self.dit = model_manager.fetch_model("hunyuan_video_dit")
        self.vae_decoder = model_manager.fetch_model("hunyuan_video_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("hunyuan_video_vae_encoder")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, enable_vram_management=True):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = HunyuanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        if enable_vram_management:
            pipe.enable_vram_management()
        return pipe


    def encode_prompt(self, prompt, positive=True, clip_sequence_length=77, llm_sequence_length=256):
        prompt_emb, pooled_prompt_emb, text_mask = self.prompter.encode_prompt(
            prompt, device=self.device, positive=positive, clip_sequence_length=clip_sequence_length, llm_sequence_length=llm_sequence_length
        )
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_mask": text_mask}


    def prepare_extra_input(self, latents=None, guidance=1.0):
        freqs_cos, freqs_sin = self.dit.prepare_freqs(latents)
        guidance = torch.Tensor([guidance] * latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
        return {"freqs_cos": freqs_cos, "freqs_sin": freqs_sin, "guidance": guidance}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def encode_video(self, frames):
        # frames : (B, C, T, H, W)
        latents = self.vae_encoder(frames)
        return latents
      
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        seed=None,
        height=720,
        width=1280,
        num_frames=129,
        embedded_guidance=6.0,
        cfg_scale=1.0,
        num_inference_steps=30,
        tile_size=(17, 30, 30),
        tile_stride=(12, 20, 20),
        progress_bar_cmd=lambda x: x,
        progress_bar_st=None,
    ):
        # Initialize noise
        latents = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
        
        # Encode prompts
        self.load_models_to_device(["text_encoder_1", "text_encoder_2"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Extra input
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Denoise
        self.load_models_to_device([] if self.vram_management else ["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)
            print(f"Step {progress_id + 1} / {len(self.scheduler.timesteps)}")

            # Inference
            with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                noise_pred_posi = self.dit(latents, timestep, **prompt_emb_posi, **extra_input)
                if cfg_scale != 1.0:
                    noise_pred_nega = self.dit(latents, timestep, **prompt_emb_nega, **extra_input)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
        
        # Tiler parameters
        tiler_kwargs = {"tile_size": tile_size, "tile_stride": tile_stride}

        # Decode
        self.load_models_to_device(['vae_decoder'])
        frames = self.vae_decoder.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames
