from ..models import ModelManager
from ..prompts import SDXLPrompter
from ..schedulers import EnhancedDDIMScheduler
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np


class SDXLPipeline(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.scheduler = EnhancedDDIMScheduler()
    
    def preprocess_image(self, image):
        image = torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)
        return image
    
    @torch.no_grad()
    def __call__(
        self,
        model_manager: ModelManager,
        prompter: SDXLPrompter,
        prompt,
        negative_prompt="",
        cfg_scale=7.5,
        clip_skip=1,
        clip_skip_2=2,
        init_image=None,
        denoising_strength=1.0,
        refining_strength=0.0,
        height=1024,
        width=1024,
        num_inference_steps=20,
        tiled=False,
        tile_size=64,
        tile_stride=32,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Encode prompts
        add_text_embeds, prompt_emb = prompter.encode_prompt(
            model_manager.text_encoder,
            model_manager.text_encoder_2,
            prompt,
            clip_skip=clip_skip, clip_skip_2=clip_skip_2,
            device=model_manager.device
        )
        if cfg_scale != 1.0:
            negative_add_text_embeds, negative_prompt_emb = prompter.encode_prompt(
                model_manager.text_encoder,
                model_manager.text_encoder_2,
                negative_prompt,
                clip_skip=clip_skip, clip_skip_2=clip_skip_2,
                device=model_manager.device
            )
        
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        if init_image is not None:
            image = self.preprocess_image(init_image).to(
                device=model_manager.device, dtype=model_manager.torch_type
            )
            latents = model_manager.vae_encoder(
                image.to(torch.float32),
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            )
            noise = torch.randn(
                (1, 4, height//8, width//8),
                device=model_manager.device, dtype=model_manager.torch_type
            )
            latents = self.scheduler.add_noise(
                latents.to(model_manager.torch_type),
                noise,
                timestep=self.scheduler.timesteps[0]
            )
        else:
            latents = torch.randn((1, 4, height//8, width//8), device=model_manager.device, dtype=model_manager.torch_type)
        
        # Prepare positional id
        add_time_id = torch.tensor([height, width, 0, 0, height, width], device=model_manager.device)
        
        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = torch.IntTensor((timestep,))[0].to(model_manager.device)

            # Classifier-free guidance
            if timestep >= 1000 * refining_strength:
                denoising_model = model_manager.unet
            else:
                denoising_model = model_manager.refiner

            if cfg_scale != 1.0:
                noise_pred_cond = denoising_model(
                    latents, timestep, prompt_emb,
                    add_time_id=add_time_id, add_text_embeds=add_text_embeds,
                    tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                )
                noise_pred_uncond = denoising_model(
                    latents, timestep, negative_prompt_emb,
                    add_time_id=add_time_id, add_text_embeds=negative_add_text_embeds,
                    tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                )
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = denoising_model(
                    latents, timestep, prompt_emb,
                    add_time_id=add_time_id, add_text_embeds=add_text_embeds,
                    tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                )

            latents = self.scheduler.step(noise_pred, timestep, latents)
            
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        latents = latents.to(torch.float32)
        image = model_manager.vae_decoder(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        image = image.cpu().permute(1, 2, 0).numpy()
        image = Image.fromarray(((image / 2 + 0.5).clip(0, 1) * 255).astype("uint8"))

        return image
