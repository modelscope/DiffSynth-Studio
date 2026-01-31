import torch, math
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange, repeat
import numpy as np
from transformers import CLIPTokenizer, T5TokenizerFast

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward, load_state_dict
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput
from ..utils.lora.flux import FluxLoRALoader

from ..models.flux_dit import FluxDiT
from ..models.flux_text_encoder_clip import FluxTextEncoderClip
from ..models.flux_text_encoder_t5 import FluxTextEncoderT5
from ..models.flux_vae import FluxVAEEncoder, FluxVAEDecoder
from ..models.flux_value_control import MultiValueEncoder
from ..models.step1x_text_encoder import Step1xEditEmbedder
from ..core.vram.layers import AutoWrappedLinear

class MultiControlNet(torch.nn.Module):
    def __init__(self, models: list[torch.nn.Module]):
        super().__init__()
        if not isinstance(models, list):
            models = [models]
        self.models = torch.nn.ModuleList(models)
        
    def process_single_controlnet(self, controlnet_input: ControlNetInput, conditioning: torch.Tensor, **kwargs):
        model = self.models[controlnet_input.controlnet_id]
        res_stack, single_res_stack = model(
            controlnet_conditioning=conditioning,
            processor_id=controlnet_input.processor_id,
            **kwargs
        )
        res_stack = [res * controlnet_input.scale for res in res_stack]
        single_res_stack = [res * controlnet_input.scale for res in single_res_stack]
        return res_stack, single_res_stack

    def forward(self, conditionings: list[torch.Tensor], controlnet_inputs: list[ControlNetInput], progress_id, num_inference_steps, **kwargs):
        res_stack, single_res_stack = None, None
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            progress = (num_inference_steps - 1 - progress_id) / max(num_inference_steps - 1, 1)
            if progress > controlnet_input.start or progress < controlnet_input.end:
                continue
            res_stack_, single_res_stack_ = self.process_single_controlnet(controlnet_input, conditioning, **kwargs)
            if res_stack is None:
                res_stack = res_stack_
                single_res_stack = single_res_stack_
            else:
                res_stack = [i + j for i, j in zip(res_stack, res_stack_)]
                single_res_stack = [i + j for i, j in zip(single_res_stack, single_res_stack_)]
        return res_stack, single_res_stack


class FluxImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("FLUX.1")
        self.tokenizer_1: CLIPTokenizer = None
        self.tokenizer_2: T5TokenizerFast = None
        self.text_encoder_1: FluxTextEncoderClip = None
        self.text_encoder_2: FluxTextEncoderT5 = None
        self.dit: FluxDiT = None
        self.vae_decoder: FluxVAEDecoder = None
        self.vae_encoder: FluxVAEEncoder = None
        self.controlnet = None
        self.ipadapter = None
        self.ipadapter_image_encoder = None
        self.qwenvl = None
        self.step1x_connector = None
        self.nexus_gen = None
        self.nexus_gen_generation_adapter = None
        self.nexus_gen_editing_adapter = None
        self.value_controller = None
        self.infinityou_processor = None
        self.image_proj_model = None
        self.lora_patcher = None
        self.lora_encoder = None
        self.in_iteration_models = ("dit", "step1x_connector", "controlnet", "lora_patcher")
        self.units = [
            FluxImageUnit_ShapeChecker(),
            FluxImageUnit_NoiseInitializer(),
            FluxImageUnit_PromptEmbedder(),
            FluxImageUnit_InputImageEmbedder(),
            FluxImageUnit_ImageIDs(),
            FluxImageUnit_EmbeddedGuidanceEmbedder(),
            FluxImageUnit_Kontext(),
            FluxImageUnit_InfiniteYou(),
            FluxImageUnit_ControlNet(),
            FluxImageUnit_IPAdapter(),
            FluxImageUnit_EntityControl(),
            FluxImageUnit_NexusGen(),
            FluxImageUnit_TeaCache(),
            FluxImageUnit_Flex(),
            FluxImageUnit_Step1x(),
            FluxImageUnit_ValueControl(),
            FluxImageUnit_LoRAEncode(),
        ]
        self.model_fn = model_fn_flux_image
        self.lora_loader = FluxLoRALoader

    def enable_lora_merger(self):
        if not (hasattr(self.dit, "vram_management_enabled") and getattr(self.dit, "vram_management_enabled")):
            raise ValueError("DiT VRAM management is not enabled.")
        if self.lora_patcher is not None:
            for name, module in self.dit.named_modules():
                if isinstance(module, AutoWrappedLinear):
                    merger_name = name.replace(".", "___")
                    if merger_name in self.lora_patcher.model_dict:
                        module.lora_merger = self.lora_patcher.model_dict[merger_name]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_1_config: ModelConfig = ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="tokenizer/"),
        tokenizer_2_config: ModelConfig = ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="tokenizer_2/"),
        nexus_gen_processor_config: ModelConfig = ModelConfig(model_id="DiffSynth-Studio/Nexus-GenV2", origin_file_pattern="processor/"),
        step1x_processor_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen2.5-VL-7B-Instruct", origin_file_pattern=""),
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = FluxImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder_1 = model_pool.fetch_model("flux_text_encoder_clip")
        pipe.text_encoder_2 = model_pool.fetch_model("flux_text_encoder_t5")
        pipe.dit = model_pool.fetch_model("flux_dit")
        pipe.vae_encoder = model_pool.fetch_model("flux_vae_encoder")
        pipe.vae_decoder = model_pool.fetch_model("flux_vae_decoder")
        if tokenizer_1_config is not None:
            tokenizer_1_config.download_if_necessary()
            pipe.tokenizer_1 = CLIPTokenizer.from_pretrained(tokenizer_1_config.path)
        if tokenizer_2_config is not None:
            tokenizer_2_config.download_if_necessary()
            pipe.tokenizer_2 = T5TokenizerFast.from_pretrained(tokenizer_2_config.path)
        
        value_controllers = model_pool.fetch_model("flux_value_controller")
        if value_controllers is not None:
            pipe.value_controller = MultiValueEncoder(value_controllers)
            if hasattr(pipe.value_controller.encoders[0], "vram_management_enabled"):
                pipe.value_controller.vram_management_enabled = pipe.value_controller.encoders[0].vram_management_enabled
        controlnets = model_pool.fetch_model("flux_controlnet")
        if controlnets is not None: pipe.controlnet = MultiControlNet(controlnets)
        pipe.ipadapter = model_pool.fetch_model("flux_ipadapter")
        pipe.ipadapter_image_encoder = model_pool.fetch_model("siglip_vision_model")
        qwenvl = model_pool.fetch_model("qwen_image_text_encoder")
        if qwenvl is not None:
            from transformers import AutoProcessor
            step1x_processor_config.download_if_necessary()
            processor = AutoProcessor.from_pretrained(step1x_processor_config.path, min_pixels=256 * 28 * 28, max_pixels=324 * 28 * 28)
            pipe.qwenvl = Step1xEditEmbedder(qwenvl, processor)
        pipe.step1x_connector = model_pool.fetch_model("step1x_connector")
        pipe.image_proj_model = model_pool.fetch_model("infiniteyou_image_projector")
        if pipe.image_proj_model is not None:
            pipe.infinityou_processor = InfinitYou(device=device)
        pipe.lora_patcher = model_pool.fetch_model("flux_lora_patcher")
        pipe.lora_encoder = model_pool.fetch_model("flux_lora_encoder")
        pipe.nexus_gen = model_pool.fetch_model("nexus_gen_llm")
        pipe.nexus_gen_generation_adapter = model_pool.fetch_model("nexus_gen_generation_adapter")
        pipe.nexus_gen_editing_adapter = model_pool.fetch_model("nexus_gen_editing_adapter")
        if pipe.nexus_gen is not None:
            nexus_gen_processor_config.download_if_necessary()
            pipe.nexus_gen.load_processor(nexus_gen_processor_config.path)
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        embedded_guidance: float = 3.5,
        t5_sequence_length: int = 512,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Scheduler
        sigma_shift: float = None,
        # Steps
        num_inference_steps: int = 30,
        # local prompts
        multidiffusion_prompts=(),
        multidiffusion_masks=(),
        multidiffusion_scales=(),
        # Kontext
        kontext_images: Union[list[Image.Image], Image.Image] = None,
        # ControlNet
        controlnet_inputs: list[ControlNetInput] = None,
        # IP-Adapter
        ipadapter_images: Union[list[Image.Image], Image.Image] = None,
        ipadapter_scale: float = 1.0,
        # EliGen
        eligen_entity_prompts: list[str] = None,
        eligen_entity_masks: list[Image.Image] = None,
        eligen_enable_on_negative: bool = False,
        eligen_enable_inpaint: bool = False,
        # InfiniteYou
        infinityou_id_image: Image.Image = None,
        infinityou_guidance: float = 1.0,
        # Flex
        flex_inpaint_image: Image.Image = None,
        flex_inpaint_mask: Image.Image = None,
        flex_control_image: Image.Image = None,
        flex_control_strength: float = 0.5,
        flex_control_stop: float = 0.5,
        # Value Controller
        value_controller_inputs: Union[list[float], float] = None,
        # Step1x
        step1x_reference_image: Image.Image = None,
        # NexusGen
        nexus_gen_reference_image: Image.Image = None,
        # LoRA Encoder
        lora_encoder_inputs: Union[list[ModelConfig], ModelConfig, str] = None,
        lora_encoder_scale: float = 1.0,
        # TeaCache
        tea_cache_l1_thresh: float = None,
        # Tile
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        # Progress bar
        progress_bar_cmd = tqdm,
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
            "kontext_images": kontext_images,
            "controlnet_inputs": controlnet_inputs,
            "ipadapter_images": ipadapter_images, "ipadapter_scale": ipadapter_scale,
            "eligen_entity_prompts": eligen_entity_prompts, "eligen_entity_masks": eligen_entity_masks, "eligen_enable_on_negative": eligen_enable_on_negative, "eligen_enable_inpaint": eligen_enable_inpaint,
            "infinityou_id_image": infinityou_id_image, "infinityou_guidance": infinityou_guidance,
            "flex_inpaint_image": flex_inpaint_image, "flex_inpaint_mask": flex_inpaint_mask, "flex_control_image": flex_control_image, "flex_control_strength": flex_control_strength, "flex_control_stop": flex_control_stop,
            "value_controller_inputs": value_controller_inputs,
            "step1x_reference_image": step1x_reference_image,
            "nexus_gen_reference_image": nexus_gen_reference_image,
            "lora_encoder_inputs": lora_encoder_inputs, "lora_encoder_scale": lora_encoder_scale,
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
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        # Decode
        self.load_models_to_device(['vae_decoder'])
        image = self.vae_decoder(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


class FluxImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width"), output_params=("height", "width"))

    def process(self, pipe: FluxImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}



class FluxImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "seed", "rand_device"), output_params=("noise",))

    def process(self, pipe: FluxImagePipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 16, height//8, width//8), seed=seed, rand_device=rand_device)
        return {"noise": noise}



class FluxImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
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
            output_params=("prompt_emb", "pooled_prompt_emb", "text_ids"),
            onload_model_names=("text_encoder_1", "text_encoder_2")
        )
    
    def encode_prompt_using_clip(self, prompt, text_encoder, tokenizer, max_length, device):
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        ).input_ids.to(device)
        pooled_prompt_emb, _ = text_encoder(input_ids)
        return pooled_prompt_emb
    
    def encode_prompt_using_t5(self, prompt, text_encoder, tokenizer, max_length, device):
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(device)
        prompt_emb = text_encoder(input_ids)
        return prompt_emb

    def encode_prompt(
        self,
        tokenizer_1,
        tokenizer_2,
        text_encoder_1,
        text_encoder_2,
        prompt,
        positive=True,
        device=get_device_type(),
        t5_sequence_length=512,
    ):
        pooled_prompt_emb = self.encode_prompt_using_clip(prompt, text_encoder_1, tokenizer_1, 77, device)
        prompt_emb = self.encode_prompt_using_t5(prompt, text_encoder_2, tokenizer_2, t5_sequence_length, device)
        text_ids = torch.zeros(prompt_emb.shape[0], prompt_emb.shape[1], 3).to(device=device, dtype=prompt_emb.dtype)
        return prompt_emb, pooled_prompt_emb, text_ids

    def process(self, pipe: FluxImagePipeline, prompt, t5_sequence_length, positive) -> dict:
        if pipe.text_encoder_1 is not None and pipe.text_encoder_2 is not None:
            prompt_emb, pooled_prompt_emb, text_ids = self.encode_prompt(
                tokenizer_1=pipe.tokenizer_1, tokenizer_2=pipe.tokenizer_2,
                text_encoder_1=pipe.text_encoder_1, text_encoder_2=pipe.text_encoder_2,
                prompt=prompt, device=pipe.device, positive=positive, t5_sequence_length=t5_sequence_length,
            )
            return {"prompt_emb": prompt_emb, "pooled_prompt_emb": pooled_prompt_emb, "text_ids": text_ids}
        else:
            return {}


class FluxImageUnit_ImageIDs(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents",), output_params=("image_ids",))

    def process(self, pipe: FluxImagePipeline, latents):
        latent_image_ids = pipe.dit.prepare_image_ids(latents)
        return {"image_ids": latent_image_ids}



class FluxImageUnit_EmbeddedGuidanceEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("embedded_guidance", "latents"), output_params=("guidance",))

    def process(self, pipe: FluxImagePipeline, embedded_guidance, latents):
        guidance = torch.Tensor([embedded_guidance] * latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
        return {"guidance": guidance}



class FluxImageUnit_Kontext(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("kontext_images", "tiled", "tile_size", "tile_stride"),
            output_params=("kontext_latents", "kontext_image_ids"),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: FluxImagePipeline, kontext_images, tiled, tile_size, tile_stride):
        if kontext_images is None:
            return {}
        if not isinstance(kontext_images, list):
            kontext_images = [kontext_images]
            
        kontext_latents = []
        kontext_image_ids = []
        for kontext_image in kontext_images:
            kontext_image = pipe.preprocess_image(kontext_image)
            kontext_latent = pipe.vae_encoder(kontext_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            image_ids = pipe.dit.prepare_image_ids(kontext_latent)
            image_ids[..., 0] = 1
            kontext_image_ids.append(image_ids)
            kontext_latent = pipe.dit.patchify(kontext_latent)
            kontext_latents.append(kontext_latent)
        kontext_latents = torch.concat(kontext_latents, dim=1)
        kontext_image_ids = torch.concat(kontext_image_ids, dim=-2)
        return {"kontext_latents": kontext_latents, "kontext_image_ids": kontext_image_ids}



class FluxImageUnit_ControlNet(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("controlnet_inputs", "tiled", "tile_size", "tile_stride"),
            output_params=("controlnet_conditionings",),
            onload_model_names=("vae_encoder",)
        )
        
    def apply_controlnet_mask_on_latents(self, pipe, latents, mask):
        mask = (pipe.preprocess_image(mask) + 1) / 2
        mask = mask.mean(dim=1, keepdim=True)
        mask = 1 - torch.nn.functional.interpolate(mask, size=latents.shape[-2:])
        latents = torch.concat([latents, mask], dim=1)
        return latents
        
    def apply_controlnet_mask_on_image(self, pipe, image, mask):
        mask = mask.resize(image.size)
        mask = pipe.preprocess_image(mask).mean(dim=[0, 1]).cpu()
        image = np.array(image)
        image[mask > 0] = 0
        image = Image.fromarray(image)
        return image

    def process(self, pipe: FluxImagePipeline, controlnet_inputs: list[ControlNetInput], tiled, tile_size, tile_stride):
        if controlnet_inputs is None:
            return {}
        pipe.load_models_to_device(['vae_encoder'])
        conditionings = []
        for controlnet_input in controlnet_inputs:
            image = controlnet_input.image
            if controlnet_input.inpaint_mask is not None:
                image = self.apply_controlnet_mask_on_image(pipe, image, controlnet_input.inpaint_mask)

            image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
            image = pipe.vae_encoder(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            
            if controlnet_input.inpaint_mask is not None:
                image = self.apply_controlnet_mask_on_latents(pipe, image, controlnet_input.inpaint_mask)
            conditionings.append(image)
        return {"controlnet_conditionings": conditionings}



class FluxImageUnit_IPAdapter(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("ipadapter_images", "ipadapter_scale"),
            output_params=("ipadapter_kwargs_list",),
            onload_model_names=("ipadapter_image_encoder", "ipadapter")
        )

    def process(self, pipe: FluxImagePipeline, inputs_shared, inputs_posi, inputs_nega):
        ipadapter_images, ipadapter_scale = inputs_shared.get("ipadapter_images", None), inputs_shared.get("ipadapter_scale", 1.0)
        if ipadapter_images is None:
            return inputs_shared, inputs_posi, inputs_nega
        if not isinstance(ipadapter_images, list):
            ipadapter_images = [ipadapter_images]

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
            input_params=("eligen_entity_prompts", "eligen_entity_masks", "eligen_enable_on_negative", "width", "height", "t5_sequence_length", "cfg_scale"),
            output_params=("entity_prompt_emb", "entity_masks"),
            onload_model_names=("text_encoder_1", "text_encoder_2")
        )
        
    def encode_prompt_using_clip(self, prompt, text_encoder, tokenizer, max_length, device):
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        ).input_ids.to(device)
        pooled_prompt_emb, _ = text_encoder(input_ids)
        return pooled_prompt_emb
    
    def encode_prompt_using_t5(self, prompt, text_encoder, tokenizer, max_length, device):
        input_ids = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(device)
        prompt_emb = text_encoder(input_ids)
        return prompt_emb

    def encode_prompt(
        self,
        tokenizer_1,
        tokenizer_2,
        text_encoder_1,
        text_encoder_2,
        prompt,
        positive=True,
        device=get_device_type(),
        t5_sequence_length=512,
    ):
        pooled_prompt_emb = self.encode_prompt_using_clip(prompt, text_encoder_1, tokenizer_1, 77, device)
        prompt_emb = self.encode_prompt_using_t5(prompt, text_encoder_2, tokenizer_2, t5_sequence_length, device)
        text_ids = torch.zeros(prompt_emb.shape[0], prompt_emb.shape[1], 3).to(device=device, dtype=prompt_emb.dtype)
        return prompt_emb, pooled_prompt_emb, text_ids

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

        prompt_emb, _, _ = self.encode_prompt(
            tokenizer_1=pipe.tokenizer_1, tokenizer_2=pipe.tokenizer_2,
            text_encoder_1=pipe.text_encoder_1, text_encoder_2=pipe.text_encoder_2,
            prompt=entity_prompts, device=pipe.device, t5_sequence_length=t5_sequence_length,
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
        eligen_enable_on_negative = inputs_shared.get("eligen_enable_on_negative", False)
        eligen_kwargs_posi, eligen_kwargs_nega = self.prepare_eligen(pipe, inputs_nega,
            eligen_entity_prompts, eligen_entity_masks, inputs_shared["width"], inputs_shared["height"], 
            inputs_shared["t5_sequence_length"], eligen_enable_on_negative, inputs_shared["cfg_scale"])
        inputs_posi.update(eligen_kwargs_posi)
        if inputs_shared.get("cfg_scale", 1.0) != 1.0:
            inputs_nega.update(eligen_kwargs_nega)
        return inputs_shared, inputs_posi, inputs_nega


class FluxImageUnit_NexusGen(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("nexus_gen_reference_image", "prompt", "latents"),
            output_params=("prompt_emb", "text_ids"),
            onload_model_names=("nexus_gen", "nexus_gen_generation_adapter", "nexus_gen_editing_adapter"),
        )

    def process(self, pipe: FluxImagePipeline, inputs_shared, inputs_posi, inputs_nega):
        if pipe.nexus_gen is None:
            return inputs_shared, inputs_posi, inputs_nega
        pipe.load_models_to_device(self.onload_model_names)
        if inputs_shared.get("nexus_gen_reference_image", None) is None:
            assert pipe.nexus_gen_generation_adapter is not None, "NexusGen requires a generation adapter to be set."
            embed = pipe.nexus_gen(inputs_posi["prompt"])[0].unsqueeze(0)
            inputs_posi["prompt_emb"] = pipe.nexus_gen_generation_adapter(embed)
            inputs_posi['text_ids'] = torch.zeros(embed.shape[0], embed.shape[1], 3).to(device=pipe.device, dtype=pipe.torch_dtype)
        else:
            assert pipe.nexus_gen_editing_adapter is not None, "NexusGen requires an editing adapter to be set."
            embed, ref_embed, grids = pipe.nexus_gen(inputs_posi["prompt"], inputs_shared["nexus_gen_reference_image"])
            embeds_grid = grids[1:2].to(device=pipe.device, dtype=torch.long)
            ref_embeds_grid = grids[0:1].to(device=pipe.device, dtype=torch.long)

            inputs_posi["prompt_emb"] = pipe.nexus_gen_editing_adapter(embed.unsqueeze(0), embeds_grid, ref_embed.unsqueeze(0), ref_embeds_grid)
            inputs_posi["text_ids"] = self.get_editing_text_ids(
                inputs_shared["latents"],
                embeds_grid[0][1].item(), embeds_grid[0][2].item(),
                ref_embeds_grid[0][1].item(), ref_embeds_grid[0][2].item(),
                )
        return inputs_shared, inputs_posi, inputs_nega


    def get_editing_text_ids(self, latents, target_embed_height, target_embed_width, ref_embed_height, ref_embed_width):
        # prepare text ids for target and reference embeddings
        batch_size, height, width = latents.shape[0], target_embed_height, target_embed_width
        embed_ids = torch.zeros(height // 2, width // 2, 3)
        scale_factor_height, scale_factor_width = latents.shape[-2] / height, latents.shape[-1] / width
        embed_ids[..., 1] = embed_ids[..., 1] + torch.arange(height // 2)[:, None] * scale_factor_height
        embed_ids[..., 2] = embed_ids[..., 2] + torch.arange(width // 2)[None, :] * scale_factor_width
        embed_ids = embed_ids[None, :].repeat(batch_size, 1, 1, 1).reshape(batch_size, height // 2 * width // 2, 3)
        embed_text_ids = embed_ids.to(device=latents.device, dtype=latents.dtype)

        batch_size, height, width = latents.shape[0], ref_embed_height, ref_embed_width
        ref_embed_ids = torch.zeros(height // 2, width // 2, 3)
        scale_factor_height, scale_factor_width = latents.shape[-2] / height, latents.shape[-1] / width
        ref_embed_ids[..., 0] = ref_embed_ids[..., 0] + 1.0
        ref_embed_ids[..., 1] = ref_embed_ids[..., 1] + torch.arange(height // 2)[:, None] * scale_factor_height
        ref_embed_ids[..., 2] = ref_embed_ids[..., 2] + torch.arange(width // 2)[None, :] * scale_factor_width
        ref_embed_ids = ref_embed_ids[None, :].repeat(batch_size, 1, 1, 1).reshape(batch_size, height // 2 * width // 2, 3)
        ref_embed_text_ids = ref_embed_ids.to(device=latents.device, dtype=latents.dtype)

        text_ids = torch.cat([embed_text_ids, ref_embed_text_ids], dim=1)
        return text_ids


class FluxImageUnit_Step1x(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("step1x_reference_image", "prompt", "negative_prompt"),
            output_params=("step1x_llm_embedding", "step1x_mask", "step1x_reference_latents"),
            onload_model_names=("qwenvl","vae_encoder")
        )
    
    def process(self, pipe: FluxImagePipeline, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict):
        image = inputs_shared.get("step1x_reference_image",None)
        if image is None:
            return inputs_shared, inputs_posi, inputs_nega
        else:
            pipe.load_models_to_device(self.onload_model_names)
            prompt = inputs_posi["prompt"]
            nega_prompt = inputs_nega["negative_prompt"]
            captions = [prompt, nega_prompt]
            ref_images = [image, image]
            embs, masks = pipe.qwenvl(captions, ref_images)
            image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
            image = pipe.vae_encoder(image)
            inputs_posi.update({"step1x_llm_embedding": embs[0:1], "step1x_mask": masks[0:1], "step1x_reference_latents": image})
            if inputs_shared.get("cfg_scale", 1) != 1:
                inputs_nega.update({"step1x_llm_embedding": embs[1:2], "step1x_mask": masks[1:2], "step1x_reference_latents": image})
            return inputs_shared, inputs_posi, inputs_nega

            
class FluxImageUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("num_inference_steps","tea_cache_l1_thresh"), output_params=("tea_cache",))
    
    def process(self, pipe: FluxImagePipeline, num_inference_steps, tea_cache_l1_thresh):
        if tea_cache_l1_thresh is None:
            return {}
        else:
            return {"tea_cache": TeaCache(num_inference_steps=num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh)}

class FluxImageUnit_Flex(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("latents", "flex_inpaint_image", "flex_inpaint_mask", "flex_control_image", "flex_control_strength", "flex_control_stop", "tiled", "tile_size", "tile_stride"),
            output_params=("flex_condition", "flex_uncondition", "flex_control_stop_timestep"),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: FluxImagePipeline, latents, flex_inpaint_image, flex_inpaint_mask, flex_control_image, flex_control_strength, flex_control_stop, tiled, tile_size, tile_stride):
        if pipe.dit.input_dim == 196:
            if flex_control_stop is None:
                flex_control_stop = 1
            pipe.load_models_to_device(self.onload_model_names)
            if flex_inpaint_image is None:
                flex_inpaint_image = torch.zeros_like(latents)
            else:
                flex_inpaint_image = pipe.preprocess_image(flex_inpaint_image).to(device=pipe.device, dtype=pipe.torch_dtype)
                flex_inpaint_image = pipe.vae_encoder(flex_inpaint_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            if flex_inpaint_mask is None:
                flex_inpaint_mask = torch.ones_like(latents)[:, 0:1, :, :]
            else:
                flex_inpaint_mask = flex_inpaint_mask.resize((latents.shape[3], latents.shape[2]))
                flex_inpaint_mask = pipe.preprocess_image(flex_inpaint_mask).to(device=pipe.device, dtype=pipe.torch_dtype)
                flex_inpaint_mask = (flex_inpaint_mask[:, 0:1, :, :] + 1) / 2
            flex_inpaint_image = flex_inpaint_image * (1 - flex_inpaint_mask)
            if flex_control_image is None:
                flex_control_image = torch.zeros_like(latents)
            else:
                flex_control_image = pipe.preprocess_image(flex_control_image).to(device=pipe.device, dtype=pipe.torch_dtype)
                flex_control_image = pipe.vae_encoder(flex_control_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride) * flex_control_strength
            flex_condition = torch.concat([flex_inpaint_image, flex_inpaint_mask, flex_control_image], dim=1)
            flex_uncondition = torch.concat([flex_inpaint_image, flex_inpaint_mask, torch.zeros_like(flex_control_image)], dim=1)
            flex_control_stop_timestep = pipe.scheduler.timesteps[int(flex_control_stop * (len(pipe.scheduler.timesteps) - 1))]
            return {"flex_condition": flex_condition, "flex_uncondition": flex_uncondition, "flex_control_stop_timestep": flex_control_stop_timestep}
        else:
            return {}



class FluxImageUnit_InfiniteYou(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("infinityou_id_image", "infinityou_guidance"),
            output_params=("id_emb", "infinityou_guidance"),
            onload_model_names=("infinityou_processor",)
        )

    def process(self, pipe: FluxImagePipeline, infinityou_id_image, infinityou_guidance):
        pipe.load_models_to_device("infinityou_processor")
        if infinityou_id_image is not None:
            return pipe.infinityou_processor.prepare_infinite_you(pipe.image_proj_model, infinityou_id_image, infinityou_guidance, pipe.device)
        else:
            return {}



class FluxImageUnit_ValueControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt_emb": "prompt_emb", "text_ids": "text_ids"},
            input_params_nega={"prompt_emb": "prompt_emb", "text_ids": "text_ids"},
            input_params=("value_controller_inputs",),
            output_params=("prompt_emb", "text_ids"),
            onload_model_names=("value_controller",)
        )
        
    def add_to_text_embedding(self, prompt_emb, text_ids, value_emb):
        prompt_emb = torch.concat([prompt_emb, value_emb], dim=1)
        extra_text_ids = torch.zeros((value_emb.shape[0], value_emb.shape[1], 3), device=value_emb.device, dtype=value_emb.dtype)
        text_ids = torch.concat([text_ids, extra_text_ids], dim=1)
        return prompt_emb, text_ids

    def process(self, pipe: FluxImagePipeline, prompt_emb, text_ids, value_controller_inputs):
        if value_controller_inputs is None:
            return {}
        if not isinstance(value_controller_inputs, list):
            value_controller_inputs = [value_controller_inputs]
        value_controller_inputs = torch.tensor(value_controller_inputs).to(dtype=pipe.torch_dtype, device=pipe.device)
        pipe.load_models_to_device(["value_controller"])
        value_emb = pipe.value_controller(value_controller_inputs, pipe.torch_dtype)
        value_emb = value_emb.unsqueeze(0)
        prompt_emb, text_ids = self.add_to_text_embedding(prompt_emb, text_ids, value_emb)
        return {"prompt_emb": prompt_emb, "text_ids": text_ids}



class InfinitYou(torch.nn.Module):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__()
        from facexlib.recognition import init_recognition_model
        from insightface.app import FaceAnalysis
        self.device = device
        self.torch_dtype = torch_dtype
        insightface_root_path = 'models/ByteDance/InfiniteYou/supports/insightface'
        self.app_640 = FaceAnalysis(name='antelopev2', root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_640.prepare(ctx_id=0, det_size=(640, 640))
        self.app_320 = FaceAnalysis(name='antelopev2', root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_320.prepare(ctx_id=0, det_size=(320, 320))
        self.app_160 = FaceAnalysis(name='antelopev2', root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_160.prepare(ctx_id=0, det_size=(160, 160))
        self.arcface_model = init_recognition_model('arcface', device=self.device).to(torch_dtype)

    def _detect_face(self, id_image_cv2):
        face_info = self.app_640.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info
        face_info = self.app_320.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info
        face_info = self.app_160.get(id_image_cv2)
        return face_info

    def extract_arcface_bgr_embedding(self, in_image, landmark, device):
        from insightface.utils import face_align
        arc_face_image = face_align.norm_crop(in_image, landmark=np.array(landmark), image_size=112)
        arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0, 3, 1, 2) / 255.
        arc_face_image = 2 * arc_face_image - 1
        arc_face_image = arc_face_image.contiguous().to(device=device, dtype=self.torch_dtype)
        face_emb = self.arcface_model(arc_face_image)[0] # [512], normalized
        return face_emb

    def prepare_infinite_you(self, model, id_image, infinityou_guidance, device):
        import cv2
        if id_image is None:
            return {'id_emb': None}
        id_image_cv2 = cv2.cvtColor(np.array(id_image), cv2.COLOR_RGB2BGR)
        face_info = self._detect_face(id_image_cv2)
        if len(face_info) == 0:
            raise ValueError('No face detected in the input ID image')
        landmark = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]['kps'] # only use the maximum face
        id_emb = self.extract_arcface_bgr_embedding(id_image_cv2, landmark, device)
        id_emb = model(id_emb.unsqueeze(0).reshape([1, -1, 512]).to(dtype=self.torch_dtype))
        infinityou_guidance = torch.Tensor([infinityou_guidance]).to(device=device, dtype=self.torch_dtype)
        return {'id_emb': id_emb, 'infinityou_guidance': infinityou_guidance}



class FluxImageUnit_LoRAEncode(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("lora_encoder_inputs", "lora_encoder_scale"),
            output_params=("prompt_emb", "text_ids"),
            onload_model_names=("lora_encoder",)
        )
        
    def parse_lora_encoder_inputs(self, lora_encoder_inputs):
        if not isinstance(lora_encoder_inputs, list):
            lora_encoder_inputs = [lora_encoder_inputs]
        lora_configs = []
        for lora_encoder_input in lora_encoder_inputs:
            if isinstance(lora_encoder_input, str):
                lora_encoder_input = ModelConfig(path=lora_encoder_input)
            lora_encoder_input.download_if_necessary()
            lora_configs.append(lora_encoder_input)
        return lora_configs
        
    def load_lora(self, lora_config, dtype, device):
        loader = FluxLoRALoader(torch_dtype=dtype, device=device)
        lora = load_state_dict(lora_config.path, torch_dtype=dtype, device=device)
        lora = loader.convert_state_dict(lora)
        return lora
    
    def lora_embedding(self, pipe, lora_encoder_inputs):
        lora_emb = []
        for lora_config in self.parse_lora_encoder_inputs(lora_encoder_inputs):
            lora = self.load_lora(lora_config, pipe.torch_dtype, pipe.device)
            lora_emb.append(pipe.lora_encoder(lora))
        lora_emb = torch.concat(lora_emb, dim=1)
        return lora_emb
    
    def add_to_text_embedding(self, prompt_emb, text_ids, lora_emb):
        prompt_emb = torch.concat([prompt_emb, lora_emb], dim=1)
        extra_text_ids = torch.zeros((lora_emb.shape[0], lora_emb.shape[1], 3), device=lora_emb.device, dtype=lora_emb.dtype)
        text_ids = torch.concat([text_ids, extra_text_ids], dim=1)
        return prompt_emb, text_ids

    def process(self, pipe: FluxImagePipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("lora_encoder_inputs", None) is None:
            return inputs_shared, inputs_posi, inputs_nega
        
        # Encode
        pipe.load_models_to_device(["lora_encoder"])
        lora_encoder_inputs = inputs_shared["lora_encoder_inputs"]
        lora_emb = self.lora_embedding(pipe, lora_encoder_inputs)
        
        # Scale
        lora_encoder_scale = inputs_shared.get("lora_encoder_scale", None)
        if lora_encoder_scale is not None:
            lora_emb = lora_emb * lora_encoder_scale
        
        # Add to prompt embedding
        inputs_posi["prompt_emb"], inputs_posi["text_ids"] = self.add_to_text_embedding(
            inputs_posi["prompt_emb"], inputs_posi["text_ids"], lora_emb)
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


class FastTileWorker:
    def __init__(self):
        pass


    def build_mask(self, data, is_bound):
        _, _, H, W = data.shape
        h = repeat(torch.arange(H), "H -> H W", H=H, W=W)
        w = repeat(torch.arange(W), "W -> H W", H=H, W=W)
        border_width = (H + W) // 4
        pad = torch.ones_like(h) * border_width
        mask = torch.stack([
            pad if is_bound[0] else h + 1,
            pad if is_bound[1] else H - h,
            pad if is_bound[2] else w + 1,
            pad if is_bound[3] else W - w
        ]).min(dim=0).values
        mask = mask.clip(1, border_width)
        mask = (mask / border_width).to(dtype=data.dtype, device=data.device)
        mask = rearrange(mask, "H W -> 1 H W")
        return mask


    def tiled_forward(self, forward_fn, model_input, tile_size, tile_stride, tile_device="cpu", tile_dtype=torch.float32, border_width=None):
        # Prepare
        B, C, H, W = model_input.shape
        border_width = int(tile_stride*0.5) if border_width is None else border_width
        weight = torch.zeros((1, 1, H, W), dtype=tile_dtype, device=tile_device)
        values = torch.zeros((B, C, H, W), dtype=tile_dtype, device=tile_device)

        # Split tasks
        tasks = []
        for h in range(0, H, tile_stride):
            for w in range(0, W, tile_stride):
                if (h-tile_stride >= 0 and h-tile_stride+tile_size >= H) or (w-tile_stride >= 0 and w-tile_stride+tile_size >= W):
                    continue
                h_, w_ = h + tile_size, w + tile_size
                if h_ > H: h, h_ = H - tile_size, H
                if w_ > W: w, w_ = W - tile_size, W
                tasks.append((h, h_, w, w_))
        
        # Run
        for hl, hr, wl, wr in tasks:
            # Forward
            hidden_states_batch = forward_fn(hl, hr, wl, wr).to(dtype=tile_dtype, device=tile_device)

            mask = self.build_mask(hidden_states_batch, is_bound=(hl==0, hr>=H, wl==0, wr>=W))
            values[:, :, hl:hr, wl:wr] += hidden_states_batch * mask
            weight[:, :, hl:hr, wl:wr] += mask
        values /= weight
        return values

    
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
    kontext_latents=None,
    kontext_image_ids=None,
    controlnet_inputs=None,
    controlnet_conditionings=None,
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
    progress_id=0,
    num_inference_steps=1,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs
):
    if tiled:
        def flux_forward_fn(hl, hr, wl, wr):
            tiled_controlnet_conditionings = [f[:, :, hl: hr, wl: wr] for f in controlnet_conditionings] if controlnet_conditionings is not None else None
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
                controlnet_inputs=controlnet_inputs,
                controlnet_conditionings=tiled_controlnet_conditionings,
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
    if controlnet is not None and controlnet_conditionings is not None:
        controlnet_extra_kwargs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "prompt_emb": prompt_emb,
            "pooled_prompt_emb": pooled_prompt_emb,
            "guidance": guidance,
            "text_ids": text_ids,
            "image_ids": image_ids,
            "controlnet_inputs": controlnet_inputs,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "progress_id": progress_id,
            "num_inference_steps": num_inference_steps,
        }
        if id_emb is not None:
            controlnet_text_ids = torch.zeros(id_emb.shape[0], id_emb.shape[1], 3).to(device=hidden_states.device, dtype=hidden_states.dtype)
            controlnet_extra_kwargs.update({"prompt_emb": id_emb, 'text_ids': controlnet_text_ids, 'guidance': infinityou_guidance})
        controlnet_res_stack, controlnet_single_res_stack = controlnet(
            controlnet_conditionings, **controlnet_extra_kwargs
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
    
    # Kontext
    if kontext_latents is not None:
        image_ids = torch.concat([image_ids, kontext_image_ids], dim=-2)
        hidden_states = torch.concat([hidden_states, kontext_latents], dim=1)
    
    # Step1x
    if step1x_reference_latents is not None:
        step1x_reference_image_ids = dit.prepare_image_ids(step1x_reference_latents)
        step1x_reference_latents = dit.patchify(step1x_reference_latents)
        image_ids = torch.concat([image_ids, step1x_reference_image_ids], dim=-2)
        hidden_states = torch.concat([hidden_states, step1x_reference_latents], dim=1)
        
    hidden_states = dit.x_embedder(hidden_states)

    # EliGen
    if entity_prompt_emb is not None and entity_masks is not None:
        prompt_emb, image_rotary_emb, attention_mask = dit.process_entity_masks(hidden_states, prompt_emb, entity_prompt_emb, entity_masks, text_ids, image_ids, latents.shape[1])
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
            hidden_states, prompt_emb = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
                ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id, None),
            )
            # ControlNet
            if controlnet is not None and controlnet_conditionings is not None and controlnet_res_stack is not None:
                if kontext_latents is None:
                    hidden_states = hidden_states + controlnet_res_stack[block_id]
                else:
                    hidden_states[:, :-kontext_latents.shape[1]] = hidden_states[:, :-kontext_latents.shape[1]] + controlnet_res_stack[block_id]

        # Single Blocks
        hidden_states = torch.cat([prompt_emb, hidden_states], dim=1)
        num_joint_blocks = len(dit.blocks)
        for block_id, block in enumerate(dit.single_blocks):
            hidden_states, prompt_emb = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                hidden_states,
                prompt_emb,
                conditioning,
                image_rotary_emb,
                attention_mask,
                ipadapter_kwargs_list=ipadapter_kwargs_list.get(block_id + num_joint_blocks, None),
            )
            # ControlNet
            if controlnet is not None and controlnet_conditionings is not None and controlnet_single_res_stack is not None:
                if kontext_latents is None:
                    hidden_states[:, prompt_emb.shape[1]:] = hidden_states[:, prompt_emb.shape[1]:] + controlnet_single_res_stack[block_id]
                else:
                    hidden_states[:, prompt_emb.shape[1]:-kontext_latents.shape[1]] = hidden_states[:, prompt_emb.shape[1]:-kontext_latents.shape[1]] + controlnet_single_res_stack[block_id]
        hidden_states = hidden_states[:, prompt_emb.shape[1]:]

        if tea_cache is not None:
            tea_cache.store(hidden_states)

    hidden_states = dit.final_norm_out(hidden_states, conditioning)
    hidden_states = dit.final_proj_out(hidden_states)
    
    # Step1x
    if step1x_reference_latents is not None:
        hidden_states = hidden_states[:, :hidden_states.shape[1] // 2]
    
    # Kontext
    if kontext_latents is not None:
        hidden_states = hidden_states[:, :-kontext_latents.shape[1]]

    hidden_states = dit.unpatchify(hidden_states, height, width)

    return hidden_states
