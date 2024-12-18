from ..models import ModelManager, SD3TextEncoder1, HunyuanVideoVAEDecoder
from ..models.hunyuan_video_dit import HunyuanVideoDiT
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import HunyuanVideoPrompter
import torch
from transformers import LlamaModel
from einops import rearrange
import numpy as np
from tqdm import tqdm
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
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'dit', 'vae_decoder']


    def fetch_models(self, model_manager: ModelManager):
        self.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        self.text_encoder_2 = model_manager.fetch_model("hunyuan_video_text_encoder_2")
        self.dit = model_manager.fetch_model("hunyuan_video_dit")
        self.vae_decoder = model_manager.fetch_model("hunyuan_video_vae_decoder")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, enable_vram_management=True):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = HunyuanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        # VRAM management is automatically enabled.
        if enable_vram_management:
            pipe.enable_cpu_offload()
            pipe.dit.enable_auto_offload(dtype=torch_dtype, device=device)
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
        progress_bar_cmd=lambda x: x,
        progress_bar_st=None,
    ):
        latents = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
        
        self.load_models_to_device(["text_encoder_1", "text_encoder_2"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        self.scheduler.set_timesteps(num_inference_steps)

        self.load_models_to_device([])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)
            with torch.autocast(device_type=self.device, dtype=self.torch_dtype):
                print(f"Step {progress_id + 1} / {len(self.scheduler.timesteps)}")

                noise_pred_posi = self.dit(latents, timestep, **prompt_emb_posi, **extra_input)
                if cfg_scale != 1.0:
                    noise_pred_nega = self.dit(latents, timestep, **prompt_emb_nega, **extra_input)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
        
        # Tiler parameters
        tiler_kwargs = dict(use_temporal_tiling=False, use_spatial_tiling=False, sample_ssize=256, sample_tsize=64)
        # decode
        self.load_models_to_device(['vae_decoder'])
        frames = self.vae_decoder.decode_video(latents, **tiler_kwargs)
        frames = self.tensor2video(frames[0])
        return frames
