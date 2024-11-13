from ..models import ModelManager, SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3, SD3DiT, SD3VAEDecoder, SD3VAEEncoder
from ..prompters import SD3Prompter
from ..schedulers import FlowMatchScheduler
from .base import BasePipeline
import torch
from tqdm import tqdm



class SD3ImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=16, width_division_factor=16)
        self.scheduler = FlowMatchScheduler()
        self.prompter = SD3Prompter()
        # models
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: SD3TextEncoder2 = None
        self.text_encoder_3: SD3TextEncoder3 = None
        self.dit: SD3DiT = None
        self.vae_decoder: SD3VAEDecoder = None
        self.vae_encoder: SD3VAEEncoder = None
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'text_encoder_3', 'dit', 'vae_decoder', 'vae_encoder']


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
    def from_model_manager(model_manager: ModelManager, prompt_refiner_classes=[], device=None):
        pipe = SD3ImagePipeline(
            device=model_manager.device if device is None else device,
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
    

    def encode_prompt(self, prompt, positive=True, t5_sequence_length=77):
        prompt_emb, pooled_prompt_emb = self.prompter.encode_prompt(
            prompt, device=self.device, positive=positive, t5_sequence_length=t5_sequence_length
        )
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb}
    

    def prepare_extra_input(self, latents=None):
        return {}
    

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        local_prompts=[],
        masks=[],
        mask_scales=[],
        negative_prompt="",
        cfg_scale=7.5,
        input_image=None,
        denoising_strength=1.0,
        height=1024,
        width=1024,
        num_inference_steps=20,
        t5_sequence_length=77,
        tiled=False,
        tile_size=128,
        tile_stride=64,
        seed=None,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        height, width = self.check_resize_height_width(height, width)
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        if input_image is not None:
            self.load_models_to_device(['vae_encoder'])
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.torch_dtype)
            latents = self.encode_image(image, **tiler_kwargs)
            noise = self.generate_noise((1, 16, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = self.generate_noise((1, 16, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)

        # Encode prompts
        self.load_models_to_device(['text_encoder_1', 'text_encoder_2', 'text_encoder_3'])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True, t5_sequence_length=t5_sequence_length)
        prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False, t5_sequence_length=t5_sequence_length)
        prompt_emb_locals = [self.encode_prompt(prompt_local, t5_sequence_length=t5_sequence_length) for prompt_local in local_prompts]

        # Denoise
        self.load_models_to_device(['dit'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            inference_callback = lambda prompt_emb_posi: self.dit(
                latents, timestep=timestep, **prompt_emb_posi, **tiler_kwargs,
            )
            noise_pred_posi = self.control_noise_via_local_prompts(prompt_emb_posi, prompt_emb_locals, masks, mask_scales, inference_callback)
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
        self.load_models_to_device(['vae_decoder'])
        image = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        # offload all models
        self.load_models_to_device([])
        return image
