from ..models import ModelManager, SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3, SD3DiT, SD3VAEDecoder, SD3VAEEncoder
from ..prompters import SD3Prompter
from ..schedulers import FlowMatchScheduler
from .base import BasePipeline
import torch
from tqdm import tqdm



class SD3ImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler()
        self.prompter = SD3Prompter()
        # models
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: SD3TextEncoder2 = None
        self.text_encoder_3: SD3TextEncoder3 = None
        self.dit: SD3DiT = None
        self.vae_decoder: SD3VAEDecoder = None
        self.vae_encoder: SD3VAEEncoder = None


    def denoising_model(self):
        return self.dit


    def fetch_models(self, model_manager: ModelManager, prompt_refiner_classes=[]):
        self.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        self.text_encoder_2 = model_manager.fetch_model("sd3_text_encoder_2")
        self.text_encoder_3 = model_manager.fetch_model("sd3_text_encoder_3")
        self.dit = model_manager.fetch_model("sd3_dit")
        self.vae_decoder = model_manager.fetch_model("sd3_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("sd3_vae_encoder")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2, self.text_encoder_3)
        self.prompter.load_prompt_refiners(model_manager, prompt_refiner_classes)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, prompt_refiner_classes=[]):
        pipe = SD3ImagePipeline(
            device=model_manager.device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_models(model_manager, prompt_refiner_classes)
        return pipe
    

    def encode_image(self, image, tiled=False, tile_size=64, tile_stride=32):
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        return image
    

    def encode_prompt(self, prompt, positive=True):
        prompt_emb, pooled_prompt_emb = self.prompter.encode_prompt(
            prompt, device=self.device, positive=positive
        )
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb}
    

    def prepare_extra_input(self, latents=None):
        return {}
    

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        cfg_scale=7.5,
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
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        if input_image is not None:
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.torch_dtype)
            latents = self.encode_image(image, **tiler_kwargs)
            noise = torch.randn((1, 16, height//8, width//8), device=self.device, dtype=self.torch_dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = torch.randn((1, 16, height//8, width//8), device=self.device, dtype=self.torch_dtype)

        # Encode prompts
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            noise_pred_posi = self.dit(
                latents, timestep=timestep, **prompt_emb_posi, **tiler_kwargs,
            )
            noise_pred_nega = self.dit(
                latents, timestep=timestep, **prompt_emb_nega, **tiler_kwargs,
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
