from ..models import ModelManager, SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3, SD3DiT, SD3VAEDecoder, SD3VAEEncoder
from ..prompts import SD3Prompter
from ..schedulers import FlowMatchScheduler
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np


class SD3ImagePipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.scheduler = FlowMatchScheduler()
        self.prompter = SD3Prompter()
        self.device = device
        self.torch_dtype = torch_dtype
        # models
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: SD3TextEncoder2 = None
        self.text_encoder_3: SD3TextEncoder3 = None
        self.dit: SD3DiT = None
        self.vae_decoder: SD3VAEDecoder = None
        self.vae_encoder: SD3VAEEncoder = None


    def fetch_main_models(self, model_manager: ModelManager):
        self.text_encoder_1 = model_manager.sd3_text_encoder_1
        self.text_encoder_2 = model_manager.sd3_text_encoder_2
        if "sd3_text_encoder_3" in model_manager.model:
            self.text_encoder_3 = model_manager.sd3_text_encoder_3
        self.dit = model_manager.sd3_dit
        self.vae_decoder = model_manager.sd3_vae_decoder
        self.vae_encoder = model_manager.sd3_vae_encoder


    def fetch_prompter(self, model_manager: ModelManager):
        self.prompter.load_from_model_manager(model_manager)


    @staticmethod
    def from_model_manager(model_manager: ModelManager):
        pipe = SD3ImagePipeline(
            device=model_manager.device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_main_models(model_manager)
        pipe.fetch_prompter(model_manager)
        return pipe
    

    def preprocess_image(self, image):
        image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
        return image
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        image = image.cpu().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))
        return image
    

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="bad quality, poor quality, doll, disfigured, jpg, toy, bad anatomy, missing limbs, missing fingers, 3d, cgi",
        cfg_scale=4.5,
        input_image=None,
        denoising_strength=1.0,
        height=1024,
        width=1024,
        num_inference_steps=20,
        tiled=False,
        tile_size=128,
        tile_stride=64,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        if input_image is not None:
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.torch_dtype)
            latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            noise = torch.randn((1, 16, height//8, width//8), device=self.device, dtype=self.torch_dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = torch.randn((1, 16, height//8, width//8), device=self.device, dtype=self.torch_dtype)

        # Encode prompts
        prompt_emb_posi, pooled_prompt_emb_posi = self.prompter.encode_prompt(
            self.text_encoder_1, self.text_encoder_2, self.text_encoder_3,
            prompt,
            device=self.device, positive=True
        )
        prompt_emb_nega, pooled_prompt_emb_nega = self.prompter.encode_prompt(
            self.text_encoder_1, self.text_encoder_2, self.text_encoder_3,
            negative_prompt,
            device=self.device, positive=False
        )
        
        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = torch.Tensor((timestep,)).to(self.device)

            # Classifier-free guidance
            noise_pred_posi = self.dit(
                latents, timestep, prompt_emb_posi, pooled_prompt_emb_posi,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            )
            noise_pred_nega = self.dit(
                latents, timestep, prompt_emb_nega, pooled_prompt_emb_nega,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            )
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

            # DDIM
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        image = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        return image
