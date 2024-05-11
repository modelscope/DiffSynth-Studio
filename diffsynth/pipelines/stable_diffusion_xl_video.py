from ..models import ModelManager, SDXLTextEncoder, SDXLTextEncoder2, SDXLUNet, SDXLVAEDecoder, SDXLVAEEncoder, SDXLMotionModel
from .dancer import lets_dance_xl
# TODO: SDXL ControlNet
from ..prompts import SDXLPrompter
from ..schedulers import EnhancedDDIMScheduler
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np


class SDXLVideoPipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16, use_animatediff=True):
        super().__init__()
        self.scheduler = EnhancedDDIMScheduler(beta_schedule="linear" if use_animatediff else "scaled_linear")
        self.prompter = SDXLPrompter()
        self.device = device
        self.torch_dtype = torch_dtype
        # models
        self.text_encoder: SDXLTextEncoder = None
        self.text_encoder_2: SDXLTextEncoder2 = None
        self.unet: SDXLUNet = None
        self.vae_decoder: SDXLVAEDecoder = None
        self.vae_encoder: SDXLVAEEncoder = None
        # TODO: SDXL ControlNet
        self.motion_modules: SDXLMotionModel = None
    
    
    def fetch_main_models(self, model_manager: ModelManager):
        self.text_encoder = model_manager.text_encoder
        self.text_encoder_2 = model_manager.text_encoder_2
        self.unet = model_manager.unet
        self.vae_decoder = model_manager.vae_decoder
        self.vae_encoder = model_manager.vae_encoder


    def fetch_controlnet_models(self, model_manager: ModelManager, **kwargs):
        # TODO: SDXL ControlNet
        pass


    def fetch_motion_modules(self, model_manager: ModelManager):
        if "motion_modules_xl" in model_manager.model:
            self.motion_modules = model_manager.motion_modules_xl


    def fetch_prompter(self, model_manager: ModelManager):
        self.prompter.load_from_model_manager(model_manager)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, controlnet_config_units = [], **kwargs):
        pipe = SDXLVideoPipeline(
            device=model_manager.device,
            torch_dtype=model_manager.torch_dtype,
            use_animatediff="motion_modules_xl" in model_manager.model
        )
        pipe.fetch_main_models(model_manager)
        pipe.fetch_motion_modules(model_manager)
        pipe.fetch_prompter(model_manager)
        pipe.fetch_controlnet_models(model_manager, controlnet_config_units=controlnet_config_units)
        return pipe
    

    def preprocess_image(self, image):
        image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
        return image
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        image = image.cpu().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image
    

    def decode_images(self, latents, tiled=False, tile_size=64, tile_stride=32):
        images = [
            self.decode_image(latents[frame_id: frame_id+1], tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            for frame_id in range(latents.shape[0])
        ]
        return images
    

    def encode_images(self, processed_images, tiled=False, tile_size=64, tile_stride=32):
        latents = []
        for image in processed_images:
            image = self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype)
            latent = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).cpu()
            latents.append(latent)
        latents = torch.concat(latents, dim=0)
        return latents
    

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        cfg_scale=7.5,
        clip_skip=1,
        clip_skip_2=2,
        num_frames=None,
        input_frames=None,
        controlnet_frames=None,
        denoising_strength=1.0,
        height=512,
        width=512,
        num_inference_steps=20,
        animatediff_batch_size = 16,
        animatediff_stride = 8,
        unet_batch_size = 1,
        controlnet_batch_size = 1,
        cross_frame_attention = False,
        smoother=None,
        smoother_progress_ids=[],
        vram_limit_level=0,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        if self.motion_modules is None:
            noise = torch.randn((1, 4, height//8, width//8), device="cpu", dtype=self.torch_dtype).repeat(num_frames, 1, 1, 1)
        else:
            noise = torch.randn((num_frames, 4, height//8, width//8), device="cuda", dtype=self.torch_dtype)
        if input_frames is None or denoising_strength == 1.0:
            latents = noise
        else:
            latents = self.encode_images(input_frames)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])

        # Encode prompts
        add_prompt_emb_posi, prompt_emb_posi = self.prompter.encode_prompt(
            self.text_encoder,
            self.text_encoder_2,
            prompt,
            clip_skip=clip_skip, clip_skip_2=clip_skip_2,
            device=self.device,
            positive=True,
        )
        if cfg_scale != 1.0:
            add_prompt_emb_nega, prompt_emb_nega = self.prompter.encode_prompt(
                self.text_encoder,
                self.text_encoder_2,
                negative_prompt,
                clip_skip=clip_skip, clip_skip_2=clip_skip_2,
                device=self.device,
                positive=False,
            )

        # Prepare positional id
        add_time_id = torch.tensor([height, width, 0, 0, height, width], device=self.device)
        
        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = torch.IntTensor((timestep,))[0].to(self.device)

            # Classifier-free guidance
            noise_pred_posi = lets_dance_xl(
                self.unet, motion_modules=self.motion_modules, controlnet=None,
                sample=latents, add_time_id=add_time_id, add_text_embeds=add_prompt_emb_posi,
                timestep=timestep, encoder_hidden_states=prompt_emb_posi, controlnet_frames=controlnet_frames,
                cross_frame_attention=cross_frame_attention,
                device=self.device, vram_limit_level=vram_limit_level
            )
            if cfg_scale != 1.0:
                noise_pred_nega = lets_dance_xl(
                    self.unet, motion_modules=self.motion_modules, controlnet=None,
                    sample=latents, add_time_id=add_time_id, add_text_embeds=add_prompt_emb_nega,
                    timestep=timestep, encoder_hidden_states=prompt_emb_nega, controlnet_frames=controlnet_frames,
                    cross_frame_attention=cross_frame_attention,
                    device=self.device, vram_limit_level=vram_limit_level
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            latents = self.scheduler.step(noise_pred, timestep, latents)
            
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        image = self.decode_images(latents.to(torch.float32))

        return image
