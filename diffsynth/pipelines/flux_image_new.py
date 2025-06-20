import torch, warnings, glob, os, types
import numpy as np
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union
from dataclasses import dataclass
from modelscope import snapshot_download
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal

from ..schedulers import FlowMatchScheduler
from ..prompters import FluxPrompter
from ..models import ModelManager, load_state_dict, SD3TextEncoder1, FluxTextEncoder2, FluxDiT, FluxVAEEncoder, FluxVAEDecoder
from ..models.tiler import FastTileWorker
from .wan_video_new import BasePipeline, ModelConfig, PipelineUnitRunner, PipelineUnit
from ..lora.flux_lora import FluxLoRALoader



class FluxImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler()
        self.prompter = FluxPrompter()
        self.text_encoder_1: SD3TextEncoder1 = None
        self.text_encoder_2: FluxTextEncoder2 = None
        self.dit: FluxDiT = None
        self.vae_decoder: FluxVAEDecoder = None
        self.vae_encoder: FluxVAEEncoder = None
        self.unit_runner = PipelineUnitRunner()
        self.in_iteration_models = ("dit", )
        self.units = [
            FluxImageUnit_ShapeChecker(),
            FluxImageUnit_NoiseInitializer(),
            FluxImageUnit_PromptEmbedder(),
            FluxImageUnit_InputImageEmbedder(),
            FluxImageUnit_ImageIDs(),
            FluxImageUnit_EmbeddedGuidanceEmbedder(),
            FluxImageUnit_IPAdapter(),
            FluxImageUnit_EntityControl(),
        ]
        self.model_fn = model_fn_flux_image
        
        
    def load_lora(self, module, path, alpha=1):
        loader = FluxLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)
    
    
    def training_loss(self, **inputs):
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)
        
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        
        noise_pred = self.model_fn(**inputs, timestep=timestep)
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss
    
    
    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        pass
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        local_model_path: str = "./models",
        skip_download: bool = False,
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(local_model_path, skip_download=skip_download)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Initialize pipeline
        pipe = FluxImagePipeline(device=device, torch_dtype=torch_dtype)
        pipe.text_encoder_1 = model_manager.fetch_model("sd3_text_encoder_1")
        pipe.text_encoder_2 = model_manager.fetch_model("flux_text_encoder_2")
        pipe.dit = model_manager.fetch_model("flux_dit")
        pipe.vae_decoder = model_manager.fetch_model("flux_vae_decoder")
        pipe.vae_encoder = model_manager.fetch_model("flux_vae_encoder")
        pipe.prompter.fetch_models(pipe.text_encoder_1, pipe.text_encoder_2)
        pipe.ipadapter = model_manager.fetch_model("flux_ipadapter")
        pipe.ipadapter_image_encoder = model_manager.fetch_model("siglip_vision_model")

        return pipe
    
    
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
        # Shape
        height=1024,
        width=1024,
        # Randomness
        seed=None,
        rand_device: Optional[str] = "cpu",
        # Scheduler
        sigma_shift=None,
        # Steps
        num_inference_steps=30,
        # local prompts
        multidiffusion_prompts=(),
        multidiffusion_masks=(),
        multidiffusion_scales=(),
        # ControlNet
        controlnet_inputs=None,
        # IP-Adapter
        ipadapter_images=None,
        ipadapter_scale=1.0,
        # EliGen
        eligen_entity_prompts=None,
        eligen_entity_masks=None,
        eligen_enable_on_negative=False,
        eligen_enable_inpaint=False,
        # InfiniteYou
        infinityou_id_image=None,
        infinityou_guidance=1.0,
        # Flex
        flex_inpaint_image=None,
        flex_inpaint_mask=None,
        flex_control_image=None,
        flex_control_strength=0.5,
        flex_control_stop=0.5,
        # Step1x
        step1x_reference_image=None,
        # TeaCache
        tea_cache_l1_thresh=None,
        # Tile
        tiled=False,
        tile_size=128,
        tile_stride=64,
        # Progress bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale, "embedded_guidance": embedded_guidance, "t5_sequence_length": t5_sequence_length,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "sigma_shift": sigma_shift, "num_inference_steps": num_inference_steps,
            "multidiffusion_prompts": multidiffusion_prompts, "multidiffusion_masks": multidiffusion_masks, "multidiffusion_scales": multidiffusion_scales,
            "controlnet_inputs": controlnet_inputs,
            "ipadapter_images": ipadapter_images, "ipadapter_scale": ipadapter_scale,
            "eligen_entity_prompts": eligen_entity_prompts, "eligen_entity_masks": eligen_entity_masks, "eligen_enable_on_negative": eligen_enable_on_negative, "eligen_enable_inpaint": eligen_enable_inpaint,
            "infinityou_id_image": infinityou_id_image, "infinityou_guidance": infinityou_guidance,
            "flex_inpaint_image": flex_inpaint_image, "flex_inpaint_mask": flex_inpaint_mask, "flex_control_image": flex_control_image, "flex_control_strength": flex_control_strength, "flex_control_stop": flex_control_stop,
            "step1x_reference_image": step1x_reference_image,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "progress_bar_cmd": progress_bar_cmd,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
        
        # Decode
        self.load_models_to_device(['vae_decoder'])
        image = self.vae_decoder(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image



class FluxImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width"))

    def process(self, pipe: FluxImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}



class FluxImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "seed", "rand_device"))

    def process(self, pipe: FluxImagePipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 16, height//8, width//8), seed=seed, rand_device=rand_device)
        return {"noise": noise}



class FluxImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: FluxImagePipeline, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae_encoder'])
        image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": None}



class FluxImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            input_params=("t5_sequence_length",),
            onload_model_names=("text_encoder_1", "text_encoder_2")
        )

    def process(self, pipe: FluxImagePipeline, prompt, t5_sequence_length, positive) -> dict:
        if pipe.text_encoder_1 is not None and pipe.text_encoder_2 is not None:
            prompt_emb, pooled_prompt_emb, text_ids = pipe.prompter.encode_prompt(
                prompt, device=pipe.device, positive=positive, t5_sequence_length=t5_sequence_length
            )
            return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_ids": text_ids}
        else:
            return {}


class FluxImageUnit_ImageIDs(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents",))

    def process(self, pipe: FluxImagePipeline, latents):
        latent_image_ids = pipe.dit.prepare_image_ids(latents)
        return {"image_ids": latent_image_ids}



class FluxImageUnit_EmbeddedGuidanceEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("embedded_guidance", "latents"))

    def process(self, pipe: FluxImagePipeline, embedded_guidance, latents):
        guidance = torch.Tensor([embedded_guidance] * latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
        return {"guidance": guidance}


class FluxImageUnit_IPAdapter(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("ipadapter_image_encoder", "ipadapter")
        )

    def process(self, pipe: FluxImagePipeline, inputs_shared, inputs_posi, inputs_nega):
        ipadapter_images, ipadapter_scale = inputs_shared.get("ipadapter_images", None), inputs_shared.get("ipadapter_scale", 1.0)
        if ipadapter_images is None:
            return inputs_shared, inputs_posi, inputs_nega

        pipe.load_models_to_device(self.onload_model_names)
        images = [image.convert("RGB").resize((384, 384), resample=3) for image in ipadapter_images]
        images = [pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype) for image in images]
        ipadapter_images = torch.cat(images, dim=0)
        ipadapter_image_encoding = pipe.ipadapter_image_encoder(ipadapter_images).pooler_output

        inputs_posi.update({"ipadapter_kwargs_list": pipe.ipadapter(ipadapter_image_encoding, scale=ipadapter_scale)})
        if inputs_shared.get("cfg_scale", 1.0) != 1.0:
            inputs_nega.update({"ipadapter_kwargs_list": pipe.ipadapter(torch.zeros_like(ipadapter_image_encoding))})
        return inputs_shared, inputs_posi, inputs_nega


class FluxImageUnit_EntityControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("text_encoder_1", "text_encoder_2")
        )

    def preprocess_masks(self, pipe, masks, height, width, dim):
        out_masks = []
        for mask in masks:
            mask = pipe.preprocess_image(mask.resize((width, height), resample=Image.NEAREST)).mean(dim=1, keepdim=True) > 0
            mask = mask.repeat(1, dim, 1, 1).to(device=pipe.device, dtype=pipe.torch_dtype)
            out_masks.append(mask)
        return out_masks

    def prepare_entity_inputs(self, pipe, entity_prompts, entity_masks, width, height, t5_sequence_length=512):
        entity_masks = self.preprocess_masks(pipe, entity_masks, height//8, width//8, 1)
        entity_masks = torch.cat(entity_masks, dim=0).unsqueeze(0) # b, n_mask, c, h, w

        prompt_emb, _, _ = pipe.prompter.encode_prompt(
            entity_prompts, device=pipe.device, t5_sequence_length=t5_sequence_length
        )
        return prompt_emb.unsqueeze(0), entity_masks

    def prepare_eligen(self, pipe, prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length, enable_eligen_on_negative, cfg_scale):
        entity_prompt_emb_posi, entity_masks_posi = self.prepare_entity_inputs(pipe, eligen_entity_prompts, eligen_entity_masks, width, height, t5_sequence_length)
        if enable_eligen_on_negative and cfg_scale != 1.0:
            entity_prompt_emb_nega = prompt_emb_nega['prompt_emb'].unsqueeze(1).repeat(1, entity_masks_posi.shape[1], 1, 1)
            entity_masks_nega = entity_masks_posi
        else:
            entity_prompt_emb_nega, entity_masks_nega = None, None
        eligen_kwargs_posi = {"entity_prompt_emb": entity_prompt_emb_posi, "entity_masks": entity_masks_posi}
        eligen_kwargs_nega = {"entity_prompt_emb": entity_prompt_emb_nega, "entity_masks": entity_masks_nega}
        return eligen_kwargs_posi, eligen_kwargs_nega

    def process(self, pipe: FluxImagePipeline, inputs_shared, inputs_posi, inputs_nega):
        eligen_entity_prompts, eligen_entity_masks = inputs_shared.get("eligen_entity_prompts", None), inputs_shared.get("eligen_entity_masks", None)
        if eligen_entity_prompts is None or eligen_entity_masks is None:
            return inputs_shared, inputs_posi, inputs_nega
        pipe.load_models_to_device(self.onload_model_names)
        eligen_kwargs_posi, eligen_kwargs_nega = self.prepare_eligen(pipe, inputs_nega,
            eligen_entity_prompts, eligen_entity_masks, inputs_shared["width"], inputs_shared["height"], 
            inputs_shared["t5_sequence_length"], inputs_shared["eligen_enable_on_negative"], inputs_shared["cfg_scale"])
        inputs_posi.update(eligen_kwargs_posi)
        if inputs_shared.get("cfg_scale", 1.0) != 1.0:
            inputs_nega.update(eligen_kwargs_nega)
        return inputs_shared, inputs_posi, inputs_nega


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

    def check(self, dit: FluxDiT, hidden_states, conditioning):
        inp = hidden_states.clone()
        temb_ = conditioning.clone()
        modulated_inp, _, _, _, _ = dit.blocks[0].norm1_a(inp, emb=temb_)
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else: 
            coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp 
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = hidden_states.clone()
        return not should_calc
    
    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states
    
    
def model_fn_flux_image(
    dit: FluxDiT,
    controlnet=None,
    step1x_connector=None,
    latents=None,
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
    entity_prompt_emb=None,
    entity_masks=None,
    ipadapter_kwargs_list={},
    id_emb=None,
    infinityou_guidance=None,
    flex_condition=None,
    flex_uncondition=None,
    flex_control_stop_timestep=None,
    step1x_llm_embedding=None,
    step1x_mask=None,
    step1x_reference_latents=None,
    tea_cache: TeaCache = None,
    **kwargs
):
    if tiled:
        def flux_forward_fn(hl, hr, wl, wr):
            tiled_controlnet_frames = [f[:, :, hl: hr, wl: wr] for f in controlnet_frames] if controlnet_frames is not None else None
            return model_fn_flux_image(
                dit=dit,
                controlnet=controlnet,
                latents=latents[:, :, hl: hr, wl: wr],
                timestep=timestep,
                prompt_emb=prompt_emb,
                pooled_prompt_emb=pooled_prompt_emb,
                guidance=guidance,
                text_ids=text_ids,
                image_ids=None,
                controlnet_frames=tiled_controlnet_frames,
                tiled=False,
                **kwargs
            )
        return FastTileWorker().tiled_forward(
            flux_forward_fn,
            latents,
            tile_size=tile_size,
            tile_stride=tile_stride,
            tile_device=latents.device,
            tile_dtype=latents.dtype
        )

    hidden_states = latents

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
        if id_emb is not None:
            controlnet_text_ids = torch.zeros(id_emb.shape[0], id_emb.shape[1], 3).to(device=hidden_states.device, dtype=hidden_states.dtype)
            controlnet_extra_kwargs.update({"prompt_emb": id_emb, 'text_ids': controlnet_text_ids, 'guidance': infinityou_guidance})
        controlnet_res_stack, controlnet_single_res_stack = controlnet(
            controlnet_frames, **controlnet_extra_kwargs
        )
        
    # Flex
    if flex_condition is not None:
        if timestep.tolist()[0] >= flex_control_stop_timestep:
            hidden_states = torch.concat([hidden_states, flex_condition], dim=1)
        else:
            hidden_states = torch.concat([hidden_states, flex_uncondition], dim=1)
            
    # Step1x
    if step1x_llm_embedding is not None:
        prompt_emb, pooled_prompt_emb = step1x_connector(step1x_llm_embedding, timestep / 1000, step1x_mask)
        text_ids = torch.zeros((1, prompt_emb.shape[1], 3), dtype=prompt_emb.dtype, device=prompt_emb.device)

    if image_ids is None:
        image_ids = dit.prepare_image_ids(hidden_states)
    
    conditioning = dit.time_embedder(timestep, hidden_states.dtype) + dit.pooled_text_embedder(pooled_prompt_emb)
    if dit.guidance_embedder is not None:
        guidance = guidance * 1000
        conditioning = conditioning + dit.guidance_embedder(guidance, hidden_states.dtype)

    height, width = hidden_states.shape[-2:]
    hidden_states = dit.patchify(hidden_states)
    
    # Step1x
    if step1x_reference_latents is not None:
        step1x_reference_image_ids = dit.prepare_image_ids(step1x_reference_latents)
        step1x_reference_latents = dit.patchify(step1x_reference_latents)
        image_ids = torch.concat([image_ids, step1x_reference_image_ids], dim=-2)
        hidden_states = torch.concat([hidden_states, step1x_reference_latents], dim=1)
        
    hidden_states = dit.x_embedder(hidden_states)

    if entity_prompt_emb is not None and entity_masks is not None:
        prompt_emb, image_rotary_emb, attention_mask = dit.process_entity_masks(hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids, image_ids)
    else:
        prompt_emb = dit.context_embedder(prompt_emb)
        image_rotary_emb = dit.pos_embedder(torch.cat((text_ids, image_ids), dim=1))
        attention_mask = None

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, hidden_states, conditioning)
    else:
        tea_cache_update = False

    if tea_cache_update:
        hidden_states = tea_cache.update(hidden_states)
    else:
        # Joint Blocks
        for block_id, block in enumerate(dit.blocks):
            hidden_states, prompt_emb = block(
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
                ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id, None)
            )
            # ControlNet
            if controlnet is not None and controlnet_frames is not None:
                hidden_states = hidden_states + controlnet_res_stack[block_id]

        # Single Blocks
        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        num_joint_blocks = len(dit.blocks)
        for block_id, block in enumerate(dit.single_blocks):
            hidden_states, prompt_emb = block(
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
                ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id + num_joint_blocks, None)
            )
            # ControlNet
            if controlnet is not None and controlnet_frames is not None:
                hidden_states[:, prompt_emb.shape[1]:] = hidden_states[:, prompt_emb.shape[1]:] + controlnet_single_res_stack[block_id]
        hidden_states = hidden_states[:, prompt_emb.shape[1]:]

        if tea_cache is not None:
            tea_cache.store(hidden_states)

    hidden_states = dit.final_norm_out(hidden_states, conditioning)
    hidden_states = dit.final_proj_out(hidden_states)
    
    # Step1x
    if step1x_reference_latents is not None:
        hidden_states = hidden_states[:, :hidden_states.shape[1] // 2]

    hidden_states = dit.unpatchify(hidden_states, height, width)

    return hidden_states
