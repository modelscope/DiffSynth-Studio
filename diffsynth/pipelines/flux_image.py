from ..models import ModelManager, FluxDiT, FluxTextEncoder1, FluxTextEncoder2, FluxVAEDecoder, FluxVAEEncoder
from ..prompters import FluxPrompter
from ..schedulers import FlowMatchScheduler
from .base import BasePipeline
import torch
from tqdm import tqdm



class FluxImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler()
        self.prompter = FluxPrompter()
        # models
        self.text_encoder_1: FluxTextEncoder1 = None
        self.text_encoder_2: FluxTextEncoder2 = None
        self.dit: FluxDiT = None
        self.vae_decoder: FluxVAEDecoder = None
        self.vae_encoder: FluxVAEEncoder = None
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'dit', 'vae_decoder', 'vae_encoder']


    def denoising_model(self):
        return self.dit


    def fetch_models(self, model_manager: ModelManager, prompt_refiner_classes=[], prompt_extender_classes=[]):
        self.text_encoder_1 = model_manager.fetch_model("flux_text_encoder_1")
        self.text_encoder_2 = model_manager.fetch_model("flux_text_encoder_2")
        self.dit = model_manager.fetch_model("flux_dit")
        self.vae_decoder = model_manager.fetch_model("flux_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("flux_vae_encoder")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2)
        self.prompter.load_prompt_refiners(model_manager, prompt_refiner_classes)
        self.prompter.load_prompt_extenders(model_manager, prompt_extender_classes)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, prompt_refiner_classes=[], prompt_extender_classes=[], device=None):
        pipe = FluxImagePipeline(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_models(model_manager, prompt_refiner_classes,prompt_extender_classes)
        return pipe
    

    def encode_image(self, image, tiled=False, tile_size=64, tile_stride=32):
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        return image
    

    def encode_prompt(self, prompt, positive=True, t5_sequence_length=512):
        prompt_emb, pooled_prompt_emb, text_ids = self.prompter.encode_prompt(
            prompt, device=self.device, positive=positive, t5_sequence_length=t5_sequence_length
        )
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_ids": text_ids}
    

    def prepare_extra_input(self, latents=None, guidance=1.0):
        latent_image_ids = self.dit.prepare_image_ids(latents)
        guidance = torch.Tensor([guidance] * latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
        return {"image_ids": latent_image_ids, "guidance": guidance}
    

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        local_prompts= None,
        masks= None,        
        mask_scales= None,
        negative_prompt="",
        cfg_scale=1.0,
        embedded_guidance=3.5,
        input_image=None,
        denoising_strength=1.0,
        height=1024,
        width=1024,
        num_inference_steps=30,
        t5_sequence_length=512,
        tiled=False,
        tile_size=128,
        tile_stride=64,
        seed=None,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
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

        # Extend prompt
        self.load_models_to_device(['text_encoder_1', 'text_encoder_2'])
        prompt, local_prompts, masks, mask_scales = self.extend_prompt(prompt, local_prompts, masks, mask_scales)

        # Encode prompts
        prompt_emb_posi = self.encode_prompt(prompt, t5_sequence_length=t5_sequence_length)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False, t5_sequence_length=t5_sequence_length)
        prompt_emb_locals = [self.encode_prompt(prompt_local, t5_sequence_length=t5_sequence_length) for prompt_local in local_prompts]

        # Extra input
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        # Denoise
        self.load_models_to_device(['dit'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            inference_callback = lambda prompt_emb_posi: self.dit(
                latents, timestep=timestep, **prompt_emb_posi, **tiler_kwargs, **extra_input
            )
            noise_pred_posi = self.control_noise_via_local_prompts(prompt_emb_posi, prompt_emb_locals, masks, mask_scales, inference_callback)
            if cfg_scale != 1.0:
                noise_pred_nega = self.dit(
                    latents, timestep=timestep, **prompt_emb_nega, **tiler_kwargs, **extra_input
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Iterate
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        self.load_models_to_device(['vae_decoder'])
        image = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        # Offload all models
        self.load_models_to_device([])
        return image
