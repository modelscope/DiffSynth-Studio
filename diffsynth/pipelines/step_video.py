from ..models import ModelManager
from ..models.hunyuan_dit_text_encoder import HunyuanDiTCLIPTextEncoder
from ..models.stepvideo_text_encoder import STEP1TextEncoder
from ..models.stepvideo_dit import StepVideoModel
from ..models.stepvideo_vae import StepVideoVAE
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import StepVideoPrompter
import torch
from einops import rearrange
import numpy as np
from PIL import Image
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from transformers.models.bert.modeling_bert import BertEmbeddings
from ..models.stepvideo_dit import RMSNorm
from ..models.stepvideo_vae import CausalConv, CausalConvAfterNorm, Upsample2D, BaseGroupNorm



class StepVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(sigma_min=0.0, extra_one_step=True, shift=13.0, reverse_sigmas=True, num_train_timesteps=1)
        self.prompter = StepVideoPrompter()
        self.text_encoder_1: HunyuanDiTCLIPTextEncoder = None
        self.text_encoder_2: STEP1TextEncoder = None
        self.dit: StepVideoModel = None
        self.vae: StepVideoVAE = None
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'dit', 'vae']


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder_1.parameters())).dtype
        enable_vram_management(
            self.text_encoder_1,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                BertEmbeddings: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=torch.float32,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.text_encoder_2.parameters())).dtype
        enable_vram_management(
            self.text_encoder_2,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                RMSNorm: AutoWrappedModule,
                torch.nn.Embedding: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                CausalConv: AutoWrappedModule,
                CausalConvAfterNorm: AutoWrappedModule,
                Upsample2D: AutoWrappedModule,
                BaseGroupNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        self.text_encoder_1 = model_manager.fetch_model("hunyuan_dit_clip_text_encoder")
        self.text_encoder_2 = model_manager.fetch_model("stepvideo_text_encoder_2")
        self.dit = model_manager.fetch_model("stepvideo_dit")
        self.vae = model_manager.fetch_model("stepvideo_vae")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = StepVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe


    def encode_prompt(self, prompt, positive=True):
        clip_embeds, llm_embeds, llm_mask = self.prompter.encode_prompt(prompt, device=self.device, positive=positive)
        clip_embeds = clip_embeds.to(dtype=self.torch_dtype, device=self.device)
        llm_embeds = llm_embeds.to(dtype=self.torch_dtype, device=self.device)
        llm_mask = llm_mask.to(dtype=self.torch_dtype, device=self.device)
        return {"encoder_hidden_states_2": clip_embeds, "encoder_hidden_states": llm_embeds, "encoder_attention_mask": llm_mask}


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
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=544,
        width=992,
        num_frames=204,
        cfg_scale=9.0,
        num_inference_steps=30,
        tiled=True,
        tile_size=(34, 34),
        tile_stride=(16, 16),
        smooth_scale=0.6,
        progress_bar_cmd=lambda x: x,
        progress_bar_st=None,
    ):
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Initialize noise
        latents = self.generate_noise((1, max(num_frames//17*3, 1), 64, height//16, width//16), seed=seed, device=rand_device, dtype=self.torch_dtype).to(self.device)
        
        # Encode prompts
        self.load_models_to_device(["text_encoder_1", "text_encoder_2"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Denoise
        self.load_models_to_device(["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            print(f"Step {progress_id + 1} / {len(self.scheduler.timesteps)}")

            # Inference
            noise_pred_posi = self.dit(latents, timestep=timestep, **prompt_emb_posi)
            if cfg_scale != 1.0:
                noise_pred_nega = self.dit(latents, timestep=timestep, **prompt_emb_nega)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        # Decode
        self.load_models_to_device(['vae'])
        frames = self.vae.decode(latents, device=self.device, smooth_scale=smooth_scale, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames
