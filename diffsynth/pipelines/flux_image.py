from ..models import ModelManager, FluxDiT, FluxTextEncoder1, FluxTextEncoder2, FluxVAEDecoder, FluxVAEEncoder
from ..controlnets import FluxMultiControlNetManager, ControlNetUnit, ControlNetConfigUnit, Annotator
from ..prompters import FluxPrompter
from ..schedulers import FlowMatchScheduler
from .base import BasePipeline
from typing import List
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from ..models.tiler import FastTileWorker



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
        self.controlnet: FluxMultiControlNetManager = None
        self.model_names = ['text_encoder_1', 'text_encoder_2', 'dit', 'vae_decoder', 'vae_encoder', 'controlnet']


    def denoising_model(self):
        return self.dit


    def fetch_models(self, model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[], prompt_refiner_classes=[], prompt_extender_classes=[]):
        self.text_encoder_1 = model_manager.fetch_model("flux_text_encoder_1")
        self.text_encoder_2 = model_manager.fetch_model("flux_text_encoder_2")
        self.dit = model_manager.fetch_model("flux_dit")
        self.vae_decoder = model_manager.fetch_model("flux_vae_decoder")
        self.vae_encoder = model_manager.fetch_model("flux_vae_encoder")
        self.prompter.fetch_models(self.text_encoder_1, self.text_encoder_2)
        self.prompter.load_prompt_refiners(model_manager, prompt_refiner_classes)
        self.prompter.load_prompt_extenders(model_manager, prompt_extender_classes)

        # ControlNets
        controlnet_units = []
        for config in controlnet_config_units:
            controlnet_unit = ControlNetUnit(
                Annotator(config.processor_id, device=self.device, skip_processor=config.skip_processor),
                model_manager.fetch_model("flux_controlnet", config.model_path),
                config.scale
            )
            controlnet_units.append(controlnet_unit)
        self.controlnet = FluxMultiControlNetManager(controlnet_units)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[], prompt_refiner_classes=[], prompt_extender_classes=[], device=None):
        pipe = FluxImagePipeline(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype,
        )
        pipe.fetch_models(model_manager, controlnet_config_units, prompt_refiner_classes, prompt_extender_classes)
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
    

    def apply_controlnet_mask_on_latents(self, latents, mask):
        mask = (self.preprocess_image(mask) + 1) / 2
        mask = mask.mean(dim=1, keepdim=True)
        mask = mask.to(dtype=self.torch_dtype, device=self.device)
        mask = 1 - torch.nn.functional.interpolate(mask, size=latents.shape[-2:])
        latents = torch.concat([latents, mask], dim=1)
        return latents
    

    def apply_controlnet_mask_on_image(self, image, mask):
        mask = mask.resize(image.size)
        mask = self.preprocess_image(mask).mean(dim=[0, 1])
        image = np.array(image)
        image[mask > 0] = 0
        image = Image.fromarray(image)
        return image
    

    def prepare_controlnet_input(self, controlnet_image, controlnet_inpaint_mask, tiler_kwargs):
        if isinstance(controlnet_image, Image.Image):
            controlnet_image = [controlnet_image] * len(self.controlnet.processors)

        controlnet_frames = []
        for i in range(len(self.controlnet.processors)):
            # image annotator
            image = self.controlnet.process_image(controlnet_image[i], processor_id=i)[0]
            if controlnet_inpaint_mask is not None and self.controlnet.processors[i].processor_id == "inpaint":
                image = self.apply_controlnet_mask_on_image(image, controlnet_inpaint_mask)

            # image to tensor
            image = self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype)

            # vae encoder
            image = self.encode_image(image, **tiler_kwargs)
            if controlnet_inpaint_mask is not None and self.controlnet.processors[i].processor_id == "inpaint":
                image = self.apply_controlnet_mask_on_latents(image, controlnet_inpaint_mask)
            
            # store it
            controlnet_frames.append(image)
        return controlnet_frames
    

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        local_prompts=None,
        masks=None,        
        mask_scales=None,
        negative_prompt="",
        cfg_scale=1.0,
        embedded_guidance=3.5,
        input_image=None,
        controlnet_image=None,
        controlnet_inpaint_mask=None,
        enable_controlnet_on_negative=False,
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

        # Prepare ControlNets
        if controlnet_image is not None:
            self.load_models_to_device(['vae_encoder'])
            controlnet_kwargs = {"controlnet_frames": self.prepare_controlnet_input(controlnet_image, controlnet_inpaint_mask, tiler_kwargs)}
            if len(masks) > 0 and controlnet_inpaint_mask is not None:
                print("The controlnet_inpaint_mask will be overridden by masks.")
                local_controlnet_kwargs = [{"controlnet_frames": self.prepare_controlnet_input(controlnet_image, mask, tiler_kwargs)} for mask in masks]
            else:
                local_controlnet_kwargs = None
        else:
            controlnet_kwargs, local_controlnet_kwargs = {"controlnet_frames": None}, [{}] * len(masks)

        # Denoise
        self.load_models_to_device(['dit', 'controlnet'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            inference_callback = lambda prompt_emb_posi, controlnet_kwargs: lets_dance_flux(
                dit=self.dit, controlnet=self.controlnet,
                hidden_states=latents, timestep=timestep,
                **prompt_emb_posi, **tiler_kwargs, **extra_input, **controlnet_kwargs
            )
            noise_pred_posi = self.control_noise_via_local_prompts(
                prompt_emb_posi, prompt_emb_locals, masks, mask_scales, inference_callback,
                special_kwargs=controlnet_kwargs, special_local_kwargs_list=local_controlnet_kwargs
            )
            if cfg_scale != 1.0:
                negative_controlnet_kwargs = controlnet_kwargs if enable_controlnet_on_negative else {}
                noise_pred_nega = lets_dance_flux(
                    dit=self.dit, controlnet=self.controlnet,
                    hidden_states=latents, timestep=timestep,
                    **prompt_emb_nega, **tiler_kwargs, **extra_input, **negative_controlnet_kwargs,
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



def lets_dance_flux(
    dit: FluxDiT,
    controlnet: FluxMultiControlNetManager = None,
    hidden_states=None,
    timestep=None,
    prompt_emb=None,
    pooled_prompt_emb=None,
    guidance=None,
    text_ids=None,
    image_ids=None,
    controlnet_frames=None,
    tiled=False,
    tile_size=128,
    tile_stride=64,
    **kwargs
):
    if tiled:
        def flux_forward_fn(hl, hr, wl, wr):
            return lets_dance_flux(
                dit=dit,
                controlnet=controlnet,
                hidden_states=hidden_states[:, :, hl: hr, wl: wr],
                timestep=timestep,
                prompt_emb=prompt_emb,
                pooled_prompt_emb=pooled_prompt_emb,
                guidance=guidance,
                text_ids=text_ids,
                image_ids=None,
                controlnet_frames=[f[:, :, hl: hr, wl: wr] for f in controlnet_frames],
                tiled=False,
                **kwargs
            )
        return FastTileWorker().tiled_forward(
            flux_forward_fn,
            hidden_states,
            tile_size=tile_size,
            tile_stride=tile_stride,
            tile_device=hidden_states.device,
            tile_dtype=hidden_states.dtype
        )


    # ControlNet
    if controlnet is not None and controlnet_frames is not None:
        controlnet_extra_kwargs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "prompt_emb": prompt_emb,
            "pooled_prompt_emb": pooled_prompt_emb,
            "guidance": guidance,
            "text_ids": text_ids,
            "image_ids": image_ids,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        controlnet_res_stack, controlnet_single_res_stack = controlnet(
            controlnet_frames, **controlnet_extra_kwargs
        )

    if image_ids is None:
        image_ids = dit.prepare_image_ids(hidden_states)
    
    conditioning = dit.time_embedder(timestep, hidden_states.dtype) + dit.pooled_text_embedder(pooled_prompt_emb)
    if dit.guidance_embedder is not None:
        guidance = guidance * 1000
        conditioning = conditioning + dit.guidance_embedder(guidance, hidden_states.dtype)
    prompt_emb = dit.context_embedder(prompt_emb)
    image_rotary_emb = dit.pos_embedder(torch.cat((text_ids, image_ids), dim=1))

    height, width = hidden_states.shape[-2:]
    hidden_states = dit.patchify(hidden_states)
    hidden_states = dit.x_embedder(hidden_states)
    
    # Joint Blocks
    for block_id, block in enumerate(dit.blocks):
        hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)
        # ControlNet
        if controlnet is not None and controlnet_frames is not None:
            hidden_states = hidden_states + controlnet_res_stack[block_id]

    # Single Blocks
    hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
    for block_id, block in enumerate(dit.single_blocks):
        hidden_states, prompt_emb = block(hidden_states, prompt_emb, conditioning, image_rotary_emb)
        # ControlNet
        if controlnet is not None and controlnet_frames is not None:
            hidden_states[:, prompt_emb.shape[1]:] = hidden_states[:, prompt_emb.shape[1]:] + controlnet_single_res_stack[block_id]
    hidden_states = hidden_states[:, prompt_emb.shape[1]:]

    hidden_states = dit.final_norm_out(hidden_states, conditioning)
    hidden_states = dit.final_proj_out(hidden_states)
    hidden_states = dit.unpatchify(hidden_states, height, width)

    return hidden_states
