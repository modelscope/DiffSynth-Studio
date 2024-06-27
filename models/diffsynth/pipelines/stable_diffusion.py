from ..models import ModelManager, SDTextEncoder, SDUNet, SDVAEDecoder, SDVAEEncoder, SDIpAdapter, IpAdapterCLIPImageEmbedder
from ..controlnets import MultiControlNetManager, ControlNetUnit, ControlNetConfigUnit, Annotator
from ..prompts import SDPrompter
from ..schedulers import EnhancedDDIMScheduler
from .dancer import lets_dance
from typing import List
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np


class SDImagePipeline(torch.nn.Module):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__()
        self.scheduler = EnhancedDDIMScheduler()
        self.prompter = SDPrompter()
        self.device = device
        self.torch_dtype = torch_dtype
        # models
        self.text_encoder: SDTextEncoder = None
        self.unet: SDUNet = None
        self.vae_decoder: SDVAEDecoder = None
        self.vae_encoder: SDVAEEncoder = None
        self.controlnet: MultiControlNetManager = None
        self.ipadapter_image_encoder: IpAdapterCLIPImageEmbedder = None
        self.ipadapter: SDIpAdapter = None


    def fetch_main_models(self, model_manager: ModelManager):
        self.text_encoder = model_manager.text_encoder
        self.unet = model_manager.unet
        self.vae_decoder = model_manager.vae_decoder
        self.vae_encoder = model_manager.vae_encoder


    def fetch_controlnet_models(self, model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[]):
        controlnet_units = []
        for config in controlnet_config_units:
            controlnet_unit = ControlNetUnit(
                Annotator(config.processor_id),
                model_manager.get_model_with_model_path(config.model_path),
                config.scale
            )
            controlnet_units.append(controlnet_unit)
        self.controlnet = MultiControlNetManager(controlnet_units)

    
    def fetch_ipadapter(self, model_manager: ModelManager):
        if "ipadapter" in model_manager.model:
            self.ipadapter = model_manager.ipadapter
        if "ipadapter_image_encoder" in model_manager.model:
            self.ipadapter_image_encoder = model_manager.ipadapter_image_encoder


    def fetch_prompter(self, model_manager: ModelManager):
        self.prompter.load_from_model_manager(model_manager)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[]):
        pipe = SDImagePipeline(
            device=model_manager.device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_main_models(model_manager)
        pipe.fetch_prompter(model_manager)
        pipe.fetch_controlnet_models(model_manager, controlnet_config_units)
        pipe.fetch_ipadapter(model_manager)
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
        negative_prompt="",
        cfg_scale=7.5,
        clip_skip=1,
        input_image=None,
        ipadapter_images=None,
        ipadapter_scale=1.0,
        controlnet_image=None,
        denoising_strength=1.0,
        height=512,
        width=512,
        num_inference_steps=20,
        tiled=False,
        tile_size=64,
        tile_stride=32,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        if input_image is not None:
            image = self.preprocess_image(input_image).to(device=self.device, dtype=self.torch_dtype)
            latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            noise = torch.randn((1, 4, height//8, width//8), device=self.device, dtype=self.torch_dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = torch.randn((1, 4, height//8, width//8), device=self.device, dtype=self.torch_dtype)

        # Encode prompts
        prompt_emb_posi = self.prompter.encode_prompt(self.text_encoder, prompt, clip_skip=clip_skip, device=self.device, positive=True)
        prompt_emb_nega = self.prompter.encode_prompt(self.text_encoder, negative_prompt, clip_skip=clip_skip, device=self.device, positive=False)

        # IP-Adapter
        if ipadapter_images is not None:
            ipadapter_image_encoding = self.ipadapter_image_encoder(ipadapter_images)
            ipadapter_kwargs_list_posi = self.ipadapter(ipadapter_image_encoding, scale=ipadapter_scale)
            ipadapter_kwargs_list_nega = self.ipadapter(torch.zeros_like(ipadapter_image_encoding))
        else:
            ipadapter_kwargs_list_posi, ipadapter_kwargs_list_nega = {}, {}

        # Prepare ControlNets
        if controlnet_image is not None:
            controlnet_image = self.controlnet.process_image(controlnet_image).to(device=self.device, dtype=self.torch_dtype)
            controlnet_image = controlnet_image.unsqueeze(1)
        
        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = torch.IntTensor((timestep,))[0].to(self.device)

            # Classifier-free guidance
            noise_pred_posi = lets_dance(
                self.unet, motion_modules=None, controlnet=self.controlnet,
                sample=latents, timestep=timestep, encoder_hidden_states=prompt_emb_posi, controlnet_frames=controlnet_image,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
                ipadapter_kwargs_list=ipadapter_kwargs_list_posi,
                device=self.device, vram_limit_level=0
            )
            noise_pred_nega = lets_dance(
                self.unet, motion_modules=None, controlnet=self.controlnet,
                sample=latents, timestep=timestep, encoder_hidden_states=prompt_emb_nega, controlnet_frames=controlnet_image,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
                ipadapter_kwargs_list=ipadapter_kwargs_list_nega,
                device=self.device, vram_limit_level=0
            )
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

            # DDIM
            latents = self.scheduler.step(noise_pred, timestep, latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        image = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        return image
