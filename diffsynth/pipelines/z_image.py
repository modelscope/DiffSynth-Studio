import torch, math
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import numpy as np
from typing import Union, List, Optional, Tuple

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput
from ..utils.lora import merge_lora

from transformers import AutoTokenizer
from ..models.z_image_text_encoder import ZImageTextEncoder
from ..models.z_image_dit import ZImageDiT
from ..models.flux_vae import FluxVAEEncoder, FluxVAEDecoder
from ..models.siglip2_image_encoder import Siglip2ImageEncoder
from ..models.dinov3_image_encoder import DINOv3ImageEncoder
from ..models.z_image_image2lora import ZImageImage2LoRAModel


class ZImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("Z-Image")
        self.text_encoder: ZImageTextEncoder = None
        self.dit: ZImageDiT = None
        self.vae_encoder: FluxVAEEncoder = None
        self.vae_decoder: FluxVAEDecoder = None
        self.siglip2_image_encoder: Siglip2ImageEncoder = None
        self.dinov3_image_encoder: DINOv3ImageEncoder = None
        self.image2lora_style: ZImageImage2LoRAModel = None
        self.tokenizer: AutoTokenizer = None
        self.in_iteration_models = ("dit",)
        self.units = [
            ZImageUnit_ShapeChecker(),
            ZImageUnit_PromptEmbedder(),
            ZImageUnit_NoiseInitializer(),
            ZImageUnit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_z_image
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = ZImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("z_image_text_encoder")
        pipe.dit = model_pool.fetch_model("z_image_dit")
        pipe.vae_encoder = model_pool.fetch_model("flux_vae_encoder")
        pipe.vae_decoder = model_pool.fetch_model("flux_vae_decoder")
        pipe.siglip2_image_encoder = model_pool.fetch_model("siglip2_image_encoder")
        pipe.dinov3_image_encoder = model_pool.fetch_model("dinov3_image_encoder")
        pipe.image2lora_style = model_pool.fetch_model("z_image_image2lora_style")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        
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
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 8,
        # Image to LoRA
        image2lora_images: List[Image.Image] = None,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength)
        
        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "image2lora_images": image2lora_images,
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
        image = self.vae_decoder(inputs_shared["latents"])
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


class ZImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: ZImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class ZImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_embeds",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(
        self,
        pipe,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:
        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = pipe.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    def process(self, pipe: ZImagePipeline, prompt):
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds = self.encode_prompt(pipe, prompt, pipe.device)
        return {"prompt_embeds": prompt_embeds}


class ZImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: ZImagePipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}


class ZImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: ZImagePipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(input_image)
        input_latents = pipe.vae_encoder(image)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


class ZImageUnit_Image2LoRAEncode(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("image2lora_images",),
            output_params=("image2lora_x", "image2lora_residual", "image2lora_residual_highres"),
            onload_model_names=("siglip2_image_encoder", "dinov3_image_encoder",),
        )
        from ..core.data.operators import ImageCropAndResize
        self.processor_lowres = ImageCropAndResize(height=28*8, width=28*8)
        self.processor_highres = ImageCropAndResize(height=1024, width=1024)

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def encode_prompt_edit(self, pipe: ZImagePipeline, prompt, edit_image):
        prompt = [prompt]
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
        prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
        return prompt_embeds.view(1, -1)
    
    def encode_images_using_siglip2(self, pipe: ZImagePipeline, images: list[Image.Image]):
        pipe.load_models_to_device(["siglip2_image_encoder"])
        embs = []
        for image in images:
            image = self.processor_highres(image)
            embs.append(pipe.siglip2_image_encoder(image).to(pipe.torch_dtype))
        embs = torch.stack(embs)
        return embs
    
    def encode_images_using_dinov3(self, pipe: ZImagePipeline, images: list[Image.Image]):
        pipe.load_models_to_device(["dinov3_image_encoder"])
        embs = []
        for image in images:
            image = self.processor_highres(image)
            embs.append(pipe.dinov3_image_encoder(image).to(pipe.torch_dtype))
        embs = torch.stack(embs)
        return embs
    
    def encode_images_using_qwenvl(self, pipe: ZImagePipeline, images: list[Image.Image], highres=False):
        pipe.load_models_to_device(["text_encoder"])
        embs = []
        for image in images:
            image = self.processor_highres(image) if highres else self.processor_lowres(image)
            embs.append(self.encode_prompt_edit(pipe, prompt="", edit_image=image))
        embs = torch.stack(embs)
        return embs

    def encode_images(self, pipe: ZImagePipeline, images: list[Image.Image]):
        if images is None:
            return {}
        if not isinstance(images, list):
            images = [images]
        embs_siglip2 = self.encode_images_using_siglip2(pipe, images)
        embs_dinov3 = self.encode_images_using_dinov3(pipe, images)
        x = torch.concat([embs_siglip2, embs_dinov3], dim=-1)
        residual = None
        residual_highres = None
        return x, residual, residual_highres

    def process(self, pipe: ZImagePipeline, image2lora_images):
        if image2lora_images is None:
            return {}
        x, residual, residual_highres = self.encode_images(pipe, image2lora_images)
        return {"image2lora_x": x, "image2lora_residual": residual, "image2lora_residual_highres": residual_highres}


class ZImageUnit_Image2LoRADecode(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("image2lora_x", "image2lora_residual", "image2lora_residual_highres"),
            output_params=("lora",),
            onload_model_names=("image2lora_style",),
        )
    
    def process(self, pipe: ZImagePipeline, image2lora_x, image2lora_residual, image2lora_residual_highres):
        if image2lora_x is None:
            return {}
        loras = []
        if pipe.image2lora_style is not None:
            pipe.load_models_to_device(["image2lora_style"])
            for x in image2lora_x:
                loras.append(pipe.image2lora_style(x=x, residual=None))
        lora = merge_lora(loras, alpha=1 / len(image2lora_x))
        return {"lora": lora}


class ZImageUnit_Image2LoRATraining(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("lora",),
        )
    
    def process(self, pipe: ZImagePipeline, lora):
        if lora is None:
            return {}
        pipe.clear_lora()
        pipe.load_lora(pipe.dit, state_dict=lora)
        return {}


class ZImageUnit_DelUnusedParams(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
    
    def process(self, pipe: ZImagePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not pipe.scheduler.training:
            return inputs_shared, inputs_posi, inputs_nega
        if "input_image" in inputs_shared: inputs_shared.pop("input_image")
        if "image2lora_images" in inputs_shared: inputs_shared.pop("image2lora_images")
        if "noise" in inputs_shared: inputs_shared.pop("noise")
        if "latents" in inputs_shared: inputs_shared.pop("latents")
        return inputs_shared, inputs_posi, inputs_nega

def model_fn_z_image(
    dit: ZImageDiT,
    latents=None,
    timestep=None,
    prompt_embeds=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    latents = [rearrange(latents, "B C H W -> C B H W")]
    timestep = (1000 - timestep) / 1000
    model_output = dit(
        latents,
        timestep,
        prompt_embeds,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )[0][0]
    model_output = -model_output
    model_output = rearrange(model_output, "C B H W -> B C H W")
    return model_output
