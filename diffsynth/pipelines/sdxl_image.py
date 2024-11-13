from ..models import SDXLTextEncoder, SDXLTextEncoder2, SDXLUNet, SDXLVAEDecoder, SDXLVAEEncoder, SDXLIpAdapter, IpAdapterXLCLIPImageEmbedder
from ..models.kolors_text_encoder import ChatGLMModel
from ..models.model_manager import ModelManager
from ..controlnets import MultiControlNetManager, ControlNetUnit, ControlNetConfigUnit, Annotator
from ..prompters import SDXLPrompter, KolorsPrompter
from ..schedulers import EnhancedDDIMScheduler
from .base import BasePipeline
from .dancer import lets_dance_xl
from typing import List
import torch
from tqdm import tqdm
from einops import repeat



class SDXLImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = EnhancedDDIMScheduler()
        self.prompter = SDXLPrompter()
        # models
        self.text_encoder: SDXLTextEncoder = None
        self.text_encoder_2: SDXLTextEncoder2 = None
        self.text_encoder_kolors: ChatGLMModel = None
        self.unet: SDXLUNet = None
        self.vae_decoder: SDXLVAEDecoder = None
        self.vae_encoder: SDXLVAEEncoder = None
        self.controlnet: MultiControlNetManager = None
        self.ipadapter_image_encoder: IpAdapterXLCLIPImageEmbedder = None
        self.ipadapter: SDXLIpAdapter = None
        self.model_names = ['text_encoder', 'text_encoder_2', 'text_encoder_kolors', 'unet', 'vae_decoder', 'vae_encoder', 'controlnet', 'ipadapter_image_encoder', 'ipadapter']


    def denoising_model(self):
        return self.unet


    def fetch_models(self, model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[], prompt_refiner_classes=[]):
        # Main models
        self.text_encoder = model_manager.fetch_model("sdxl_text_encoder")
        self.text_encoder_2 = model_manager.fetch_model("sdxl_text_encoder_2")
        self.text_encoder_kolors = model_manager.fetch_model("kolors_text_encoder")
        self.unet = model_manager.fetch_model("sdxl_unet")
        self.vae_decoder = model_manager.fetch_model("sdxl_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("sdxl_vae_encoder")

        # ControlNets
        controlnet_units = []
        for config in controlnet_config_units:
            controlnet_unit = ControlNetUnit(
                Annotator(config.processor_id, device=self.device),
                model_manager.fetch_model("sdxl_controlnet", config.model_path),
                config.scale
            )
            controlnet_units.append(controlnet_unit)
        self.controlnet = MultiControlNetManager(controlnet_units)

        # IP-Adapters
        self.ipadapter = model_manager.fetch_model("sdxl_ipadapter")
        self.ipadapter_image_encoder = model_manager.fetch_model("sdxl_ipadapter_clip_image_encoder")

        # Kolors
        if self.text_encoder_kolors is not None:
            print("Switch to Kolors. The prompter and scheduler will be replaced.")
            self.prompter = KolorsPrompter()
            self.prompter.fetch_models(self.text_encoder_kolors)
            self.scheduler = EnhancedDDIMScheduler(beta_end=0.014, num_train_timesteps=1100)
        else:
            self.prompter.fetch_models(self.text_encoder, self.text_encoder_2)
        self.prompter.load_prompt_refiners(model_manager, prompt_refiner_classes)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[], prompt_refiner_classes=[], device=None):
        pipe = SDXLImagePipeline(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_models(model_manager, controlnet_config_units, prompt_refiner_classes)
        return pipe
    

    def encode_image(self, image, tiled=False, tile_size=64, tile_stride=32):
        latents = self.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    

    def decode_image(self, latent, tiled=False, tile_size=64, tile_stride=32):
        image = self.vae_decoder(latent.to(self.device), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        return image
    

    def encode_prompt(self, prompt, clip_skip=1, clip_skip_2=2, positive=True):
        add_prompt_emb, prompt_emb = self.prompter.encode_prompt(
            prompt,
            clip_skip=clip_skip, clip_skip_2=clip_skip_2,
            device=self.device,
            positive=positive,
        )
        return {"encoder_hidden_states": prompt_emb, "add_text_embeds": add_prompt_emb}
    

    def prepare_extra_input(self, latents=None):
        height, width = latents.shape[2] * 8, latents.shape[3] * 8
        add_time_id = torch.tensor([height, width, 0, 0, height, width], device=self.device).repeat(latents.shape[0])
        return {"add_time_id": add_time_id}
    

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
        clip_skip_2=2,
        input_image=None,
        ipadapter_images=None,
        ipadapter_scale=1.0,
        ipadapter_use_instant_style=False,
        controlnet_image=None,
        denoising_strength=1.0,
        height=1024,
        width=1024,
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
        self.load_models_to_device(['text_encoder', 'text_encoder_2', 'text_encoder_kolors'])
        prompt_emb_posi = self.encode_prompt(prompt, clip_skip=clip_skip, clip_skip_2=clip_skip_2, positive=True)
        prompt_emb_nega = self.encode_prompt(negative_prompt, clip_skip=clip_skip, clip_skip_2=clip_skip_2, positive=False)
        prompt_emb_locals = [self.encode_prompt(prompt_local, clip_skip=clip_skip, clip_skip_2=clip_skip_2, positive=True) for prompt_local in local_prompts]

        # IP-Adapter
        if ipadapter_images is not None:
            if ipadapter_use_instant_style:
                self.ipadapter.set_less_adapter()
            else:
                self.ipadapter.set_full_adapter()
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

        # Prepare extra input
        extra_input = self.prepare_extra_input(latents)
        
        # Denoise
        self.load_models_to_device(['controlnet', 'unet'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            inference_callback = lambda prompt_emb_posi: lets_dance_xl(
                self.unet, motion_modules=None, controlnet=self.controlnet,
                sample=latents, timestep=timestep, **extra_input,
                **prompt_emb_posi, **controlnet_kwargs, **tiler_kwargs, **ipadapter_kwargs_list_posi,
                device=self.device,
            )
            noise_pred_posi = self.control_noise_via_local_prompts(prompt_emb_posi, prompt_emb_locals, masks, mask_scales, inference_callback)

            if cfg_scale != 1.0:
                noise_pred_nega = lets_dance_xl(
                    self.unet, motion_modules=None, controlnet=self.controlnet,
                    sample=latents, timestep=timestep, **extra_input,
                    **prompt_emb_nega, **controlnet_kwargs, **tiler_kwargs, **ipadapter_kwargs_list_nega,
                    device=self.device,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

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
