from ..models import SDTextEncoder, SDUNet, SDVAEDecoder, SDVAEEncoder, SDIpAdapter, IpAdapterCLIPImageEmbedder
from ..models.model_manager import ModelManager
from ..controlnets import MultiControlNetManager, ControlNetUnit, ControlNetConfigUnit, Annotator
from ..prompters import SDPrompter
from ..schedulers import EnhancedDDIMScheduler
from .base import BasePipeline
from .dancer import lets_dance
from typing import List
import torch
from tqdm import tqdm



class SDImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = EnhancedDDIMScheduler()
        self.prompter = SDPrompter()
        # models
        self.text_encoder: SDTextEncoder = None
        self.unet: SDUNet = None
        self.vae_decoder: SDVAEDecoder = None
        self.vae_encoder: SDVAEEncoder = None
        self.controlnet: MultiControlNetManager = None
        self.ipadapter_image_encoder: IpAdapterCLIPImageEmbedder = None
        self.ipadapter: SDIpAdapter = None
        self.model_names = ['text_encoder', 'unet', 'vae_decoder', 'vae_encoder', 'controlnet', 'ipadapter_image_encoder', 'ipadapter']


    def denoising_model(self):
        return self.unet


    def fetch_models(self, model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[], prompt_refiner_classes=[]):
        # Main models
        self.text_encoder = model_manager.fetch_model("sd_text_encoder")
        self.unet = model_manager.fetch_model("sd_unet")
        self.vae_decoder = model_manager.fetch_model("sd_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("sd_vae_encoder")
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.load_prompt_refiners(model_manager, prompt_refiner_classes)

        # ControlNets
        controlnet_units = []
        for config in controlnet_config_units:
            controlnet_unit = ControlNetUnit(
                Annotator(config.processor_id, device=self.device),
                model_manager.fetch_model("sd_controlnet", config.model_path),
                config.scale
            )
            controlnet_units.append(controlnet_unit)
        self.controlnet = MultiControlNetManager(controlnet_units)

        # IP-Adapters
        self.ipadapter = model_manager.fetch_model("sd_ipadapter")
        self.ipadapter_image_encoder = model_manager.fetch_model("sd_ipadapter_clip_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[], prompt_refiner_classes=[], device=None):
        pipe = SDImagePipeline(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_models(model_manager, controlnet_config_units, prompt_refiner_classes=[])
        return pipe
    

    def encode_image(self, image, tiled=False, tile_size=64, tile_stride=32):
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        return image
    

    def encode_prompt(self, prompt, clip_skip=1, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, clip_skip=clip_skip, device=self.device, positive=positive)
        return {"encoder_hidden_states": prompt_emb}
    

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
            noise = self.generate_noise((1, 4, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = self.generate_noise((1, 4, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)

        # Encode prompts
        self.load_models_to_device(['text_encoder'])
        prompt_emb_posi = self.encode_prompt(prompt, clip_skip=clip_skip, positive=True)
        prompt_emb_nega = self.encode_prompt(negative_prompt, clip_skip=clip_skip, positive=False)
        prompt_emb_locals = [self.encode_prompt(prompt_local, clip_skip=clip_skip, positive=True) for prompt_local in local_prompts]

        # IP-Adapter
        if ipadapter_images is not None:
            self.load_models_to_device(['ipadapter_image_encoder'])
            ipadapter_image_encoding = self.ipadapter_image_encoder(ipadapter_images)
            self.load_models_to_device(['ipadapter'])
            ipadapter_kwargs_list_posi = {"ipadapter_kwargs_list": self.ipadapter(ipadapter_image_encoding, scale=ipadapter_scale)}
            ipadapter_kwargs_list_nega = {"ipadapter_kwargs_list": self.ipadapter(torch.zeros_like(ipadapter_image_encoding))}
        else:
            ipadapter_kwargs_list_posi, ipadapter_kwargs_list_nega = {"ipadapter_kwargs_list": {}}, {"ipadapter_kwargs_list": {}}

        # Prepare ControlNets
        if controlnet_image is not None:
            self.load_models_to_device(['controlnet'])
            controlnet_image = self.controlnet.process_image(controlnet_image).to(device=self.device, dtype=self.torch_dtype)
            controlnet_image = controlnet_image.unsqueeze(1)
            controlnet_kwargs = {"controlnet_frames": controlnet_image}
        else:
            controlnet_kwargs = {"controlnet_frames": None}
        
        # Denoise
        self.load_models_to_device(['controlnet', 'unet'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            inference_callback = lambda prompt_emb_posi: lets_dance(
                self.unet, motion_modules=None, controlnet=self.controlnet,
                sample=latents, timestep=timestep, 
                **prompt_emb_posi, **controlnet_kwargs, **tiler_kwargs, **ipadapter_kwargs_list_posi,
                device=self.device,
            )
            noise_pred_posi = self.control_noise_via_local_prompts(prompt_emb_posi, prompt_emb_locals, masks, mask_scales, inference_callback)
            noise_pred_nega = lets_dance(
                self.unet, motion_modules=None, controlnet=self.controlnet,
                sample=latents, timestep=timestep, **prompt_emb_nega, **controlnet_kwargs, **tiler_kwargs, **ipadapter_kwargs_list_nega,
                device=self.device,
            )
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)

            # DDIM
            latents = self.scheduler.step(noise_pred, timestep, latents)

            # UI
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))
        
        # Decode image
        self.load_models_to_device(['vae_decoder'])
        image = self.decode_image(latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        # offload all models
        self.load_models_to_device([])
        return image
