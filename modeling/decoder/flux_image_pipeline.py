from typing import List
from tqdm import tqdm
import torch
from diffsynth.models import ModelManager
from diffsynth.controlnets import ControlNetConfigUnit
from diffsynth.prompters.flux_prompter import FluxPrompter
from diffsynth.pipelines.flux_image import FluxImagePipeline, lets_dance_flux, TeaCache


class FluxPrompterAll2All(FluxPrompter):
    def encode_prompt(
        self,
        prompt,
        positive=True,
        device="cuda",
        t5_sequence_length=512,
        clip_only=False
    ):
        prompt = self.process_prompt(prompt, positive=positive)
        # CLIP
        pooled_prompt_emb = self.encode_prompt_using_clip(prompt, self.text_encoder_1, self.tokenizer_1, 77, device)
        if clip_only:
            return None, pooled_prompt_emb, None
        # T5
        prompt_emb = self.encode_prompt_using_t5(prompt, self.text_encoder_2, self.tokenizer_2, t5_sequence_length, device)
        # text_ids
        text_ids = torch.zeros(prompt_emb.shape[0], prompt_emb.shape[1], 3).to(device=device, dtype=prompt_emb.dtype)
        return prompt_emb, pooled_prompt_emb, text_ids


class FluxImagePipelineAll2All(FluxImagePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompter = FluxPrompterAll2All()

    def encode_prompt(self, prompt, positive=True, t5_sequence_length=512, clip_only=False):
        prompt_emb, pooled_prompt_emb, text_ids = self.prompter.encode_prompt(
            prompt, device=self.device, positive=positive, t5_sequence_length=t5_sequence_length, clip_only=clip_only
        )
        return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_ids": text_ids}


    @staticmethod
    def from_model_manager(model_manager: ModelManager, controlnet_config_units: List[ControlNetConfigUnit]=[], prompt_refiner_classes=[], prompt_extender_classes=[], device=None, torch_dtype=None):
        pipe = FluxImagePipelineAll2All(
            device=model_manager.device if device is None else device,
            torch_dtype=model_manager.torch_dtype if torch_dtype is None else torch_dtype,
        )
        pipe.fetch_models(model_manager, controlnet_config_units, prompt_refiner_classes, prompt_extender_classes)
        return pipe


    def prepare_prompts(self, prompt, image_embed, local_prompts, masks, mask_scales, t5_sequence_length, negative_prompt, cfg_scale):
        # Extend prompt
        self.load_models_to_device(['text_encoder_1', 'text_encoder_2'])
        prompt, local_prompts, masks, mask_scales = self.extend_prompt(prompt, local_prompts, masks, mask_scales)

        # Encode prompts
        if image_embed is not None:
            image_embed = image_embed.to(self.torch_dtype)
            prompt_emb_posi = self.encode_prompt("", positive=True, clip_only=True)
            if len(image_embed.size()) == 2:
                image_embed = image_embed.unsqueeze(0)
            prompt_emb_posi['prompt_emb'] = image_embed
            prompt_emb_posi['text_ids'] = torch.zeros(image_embed.shape[0], image_embed.shape[1], 3).to(device=self.device, dtype=self.torch_dtype)
        else:
            prompt_emb_posi = self.encode_prompt(prompt, t5_sequence_length=t5_sequence_length)
        prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False, t5_sequence_length=t5_sequence_length) if cfg_scale != 1.0 else None
        prompt_emb_locals = [self.encode_prompt(prompt_local, t5_sequence_length=t5_sequence_length) for prompt_local in local_prompts]
        return prompt_emb_posi, prompt_emb_nega, prompt_emb_locals


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt,
        negative_prompt="",
        cfg_scale=1.0,
        embedded_guidance=3.5,
        t5_sequence_length=512,
        # Image
        input_image=None,
        denoising_strength=1.0,
        height=1024,
        width=1024,
        seed=None,
        # image_embed
        image_embed=None,
        # Steps
        num_inference_steps=30,
        # local prompts
        local_prompts=(),
        masks=(),
        mask_scales=(),
        # ControlNet
        controlnet_image=None,
        controlnet_inpaint_mask=None,
        enable_controlnet_on_negative=False,
        # IP-Adapter
        ipadapter_images=None,
        ipadapter_scale=1.0,
        # EliGen
        eligen_entity_prompts=None,
        eligen_entity_masks=None,
        enable_eligen_on_negative=False,
        enable_eligen_inpaint=False,
        # TeaCache
        tea_cache_l1_thresh=None,
        # Tile
        tiled=False,
        tile_size=128,
        tile_stride=64,
        # Progress bar
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        height, width = self.check_resize_height_width(height, width)

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength)

        # Prepare latent tensors
        latents, input_latents = self.prepare_latents(input_image, height, width, seed, tiled, tile_size, tile_stride)

        # Prompt
        prompt_emb_posi, prompt_emb_nega, prompt_emb_locals = self.prepare_prompts(prompt, image_embed, local_prompts, masks, mask_scales, t5_sequence_length, negative_prompt, cfg_scale)

        # Extra input
        extra_input = self.prepare_extra_input(latents, guidance=embedded_guidance)

        # Entity control
        eligen_kwargs_posi, eligen_kwargs_nega, fg_mask, bg_mask = self.prepare_eligen(prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length, enable_eligen_inpaint, enable_eligen_on_negative, cfg_scale)

        # IP-Adapter
        ipadapter_kwargs_list_posi, ipadapter_kwargs_list_nega = self.prepare_ipadapter(ipadapter_images, ipadapter_scale)

        # ControlNets
        controlnet_kwargs_posi, controlnet_kwargs_nega, local_controlnet_kwargs = self.prepare_controlnet(controlnet_image, masks, controlnet_inpaint_mask, tiler_kwargs, enable_controlnet_on_negative)

        # TeaCache
        tea_cache_kwargs = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh) if tea_cache_l1_thresh is not None else None}

        # Denoise
        self.load_models_to_device(['dit', 'controlnet'])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Positive side
            inference_callback = lambda prompt_emb_posi, controlnet_kwargs: lets_dance_flux(
                dit=self.dit, controlnet=self.controlnet,
                hidden_states=latents, timestep=timestep,
                **prompt_emb_posi, **tiler_kwargs, **extra_input, **controlnet_kwargs, **ipadapter_kwargs_list_posi, **eligen_kwargs_posi, **tea_cache_kwargs,
            )
            noise_pred_posi = self.control_noise_via_local_prompts(
                prompt_emb_posi, prompt_emb_locals, masks, mask_scales, inference_callback,
                special_kwargs=controlnet_kwargs_posi, special_local_kwargs_list=local_controlnet_kwargs
            )

            # Inpaint
            if enable_eligen_inpaint:
                noise_pred_posi = self.inpaint_fusion(latents, input_latents, noise_pred_posi, fg_mask, bg_mask, progress_id)
            
            # Classifier-free guidance
            if cfg_scale != 1.0:
                # Negative side
                noise_pred_nega = lets_dance_flux(
                    dit=self.dit, controlnet=self.controlnet,
                    hidden_states=latents, timestep=timestep,
                    **prompt_emb_nega, **tiler_kwargs, **extra_input, **controlnet_kwargs_nega, **ipadapter_kwargs_list_nega, **eligen_kwargs_nega,
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
        image = self.decode_image(latents, **tiler_kwargs)

        # Offload all models
        self.load_models_to_device([])
        return image
