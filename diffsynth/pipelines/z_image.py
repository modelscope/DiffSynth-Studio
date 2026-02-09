import torch, math, warnings
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import numpy as np
from typing import Union, List, Optional, Tuple, Iterable, Dict

from ..core.device.npu_compatible_device import get_device_type, IS_NPU_AVAILABLE
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..core.data.operators import ImageCropAndResize
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput
from ..utils.lora import merge_lora

from transformers import AutoTokenizer
from ..models.z_image_text_encoder import ZImageTextEncoder
from ..models.z_image_dit import ZImageDiT
from ..models.flux_vae import FluxVAEEncoder, FluxVAEDecoder
from ..models.siglip2_image_encoder import Siglip2ImageEncoder428M
from ..models.z_image_controlnet import ZImageControlNet
from ..models.siglip2_image_encoder import Siglip2ImageEncoder
from ..models.dinov3_image_encoder import DINOv3ImageEncoder
from ..models.z_image_image2lora import ZImageImage2LoRAModel


class ZImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("Z-Image")
        self.text_encoder: ZImageTextEncoder = None
        self.dit: ZImageDiT = None
        self.vae_encoder: FluxVAEEncoder = None
        self.vae_decoder: FluxVAEDecoder = None
        self.image_encoder: Siglip2ImageEncoder428M = None
        self.controlnet: ZImageControlNet = None
        self.siglip2_image_encoder: Siglip2ImageEncoder = None
        self.dinov3_image_encoder: DINOv3ImageEncoder = None
        self.image2lora_style: ZImageImage2LoRAModel = None
        self.tokenizer: AutoTokenizer = None
        self.in_iteration_models = ("dit", "controlnet")
        self.units = [
            ZImageUnit_ShapeChecker(),
            ZImageUnit_PromptEmbedder(),
            ZImageUnit_NoiseInitializer(),
            ZImageUnit_InputImageEmbedder(),
            ZImageUnit_EditImageAutoResize(),
            ZImageUnit_EditImageEmbedderVAE(),
            ZImageUnit_EditImageEmbedderSiglip(),
            ZImageUnit_PAIControlNet(),
        ]
        self.model_fn = model_fn_z_image
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
        enable_npu_patch: bool = True,
    ):
        # Initialize pipeline
        pipe = ZImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("z_image_text_encoder")
        pipe.dit = model_pool.fetch_model("z_image_dit")
        pipe.vae_encoder = model_pool.fetch_model("flux_vae_encoder")
        pipe.vae_decoder = model_pool.fetch_model("flux_vae_decoder")
        pipe.image_encoder = model_pool.fetch_model("siglip_vision_model_428m")
        pipe.controlnet = model_pool.fetch_model("z_image_controlnet")
        pipe.siglip2_image_encoder = model_pool.fetch_model("siglip2_image_encoder")
        pipe.dinov3_image_encoder = model_pool.fetch_model("dinov3_image_encoder")
        pipe.image2lora_style = model_pool.fetch_model("z_image_image2lora_style")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        # NPU patch
        apply_npu_patch(enable_npu_patch)
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
        # Edit
        edit_image: Image.Image = None,
        edit_image_auto_resize: bool = True,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 8,
        sigma_shift: float = None,
        # ControlNet
        controlnet_inputs: List[ControlNetInput] = None,
        # Image to LoRA
        image2lora_images: List[Image.Image] = None,
        positive_only_lora: Dict[str, torch.Tensor] = None,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
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
            "edit_image": edit_image, "edit_image_auto_resize": edit_image_auto_resize,
            "controlnet_inputs": controlnet_inputs,
            "image2lora_images": image2lora_images, "positive_only_lora": positive_only_lora,
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
            input_params=("edit_image",),
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
    
    def encode_prompt_omni(
        self,
        pipe,
        prompt: Union[str, List[str]],
        edit_image=None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:
        if isinstance(prompt, str):
            prompt = [prompt]

        if edit_image is None:
            num_condition_images = 0
        elif isinstance(edit_image, list):
            num_condition_images = len(edit_image)
        else:
            num_condition_images = 1

        for i, prompt_item in enumerate(prompt):
            if num_condition_images == 0:
                prompt[i] = ["<|im_start|>user\n" + prompt_item + "<|im_end|>\n<|im_start|>assistant\n"]
            elif num_condition_images > 0:
                prompt_list = ["<|im_start|>user\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|vision_start|>"] * (num_condition_images - 1)
                prompt_list += ["<|vision_end|>" + prompt_item + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|im_end|>"]
                prompt[i] = prompt_list

        flattened_prompt = []
        prompt_list_lengths = []

        for i in range(len(prompt)):
            prompt_list_lengths.append(len(prompt[i]))
            flattened_prompt.extend(prompt[i])

        text_inputs = pipe.tokenizer(
            flattened_prompt,
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
        start_idx = 0
        for i in range(len(prompt_list_lengths)):
            batch_embeddings = []
            end_idx = start_idx + prompt_list_lengths[i]
            for j in range(start_idx, end_idx):
                batch_embeddings.append(prompt_embeds[j][prompt_masks[j]])
            embeddings_list.append(batch_embeddings)
            start_idx = end_idx

        return embeddings_list

    def process(self, pipe: ZImagePipeline, prompt, edit_image):
        pipe.load_models_to_device(self.onload_model_names)
        if hasattr(pipe, "dit") and pipe.dit.siglip_embedder is not None:
            # Z-Image-Turbo and Z-Image-Omni-Base use different prompt encoding methods.
            # We determine which encoding method to use based on the model architecture.
            # If you are using two-stage split training,
            # please use `--offload_models` instead of skipping the DiT model loading.
            prompt_embeds = self.encode_prompt_omni(pipe, prompt, edit_image, pipe.device)
        else:
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


class ZImageUnit_EditImageAutoResize(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "edit_image_auto_resize"),
            output_params=("edit_image",),
        )

    def process(self, pipe: ZImagePipeline, edit_image, edit_image_auto_resize):
        if edit_image is None:
            return {}
        if edit_image_auto_resize is None or not edit_image_auto_resize:
            return {}
        operator = ImageCropAndResize(max_pixels=1024*1024, height_division_factor=16, width_division_factor=16)
        if not isinstance(edit_image, list):
            edit_image = [edit_image]
        edit_image = [operator(i) for i in edit_image]
        return {"edit_image": edit_image}


class ZImageUnit_EditImageEmbedderSiglip(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image",),
            output_params=("image_embeds",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: ZImagePipeline, edit_image):
        if edit_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if not isinstance(edit_image, list):
            edit_image = [edit_image]
        image_emb = []
        for image_ in edit_image:
            image_emb.append(pipe.image_encoder(image_, device=pipe.device))
        return {"image_embeds": image_emb}


class ZImageUnit_EditImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image",),
            output_params=("image_latents",),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: ZImagePipeline, edit_image):
        if edit_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if not isinstance(edit_image, list):
            edit_image = [edit_image]
        image_latents = []
        for image_ in edit_image:
            image_ = pipe.preprocess_image(image_)
            image_latents.append(pipe.vae_encoder(image_))
        return {"image_latents": image_latents}


class ZImageUnit_PAIControlNet(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("controlnet_inputs", "height", "width"),
            output_params=("control_context", "control_scale"),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: ZImagePipeline, controlnet_inputs: List[ControlNetInput], height, width):
        if controlnet_inputs is None:
            return {}
        if len(controlnet_inputs) != 1:
            print("Z-Image ControlNet doesn't support multi-ControlNet. Only one image will be used.")
        controlnet_input = controlnet_inputs[0]
        pipe.load_models_to_device(self.onload_model_names)

        control_image = controlnet_input.image
        if control_image is not None:
            control_image = pipe.preprocess_image(control_image)
            control_latents = pipe.vae_encoder(control_image)
        else:
            control_latents = torch.ones((1, 16, height // 8, width // 8), dtype=pipe.torch_dtype, device=pipe.device) * -1
        
        inpaint_mask = controlnet_input.inpaint_mask
        if inpaint_mask is not None:
            inpaint_mask = pipe.preprocess_image(inpaint_mask, min_value=0, max_value=1)
            inpaint_image = controlnet_input.inpaint_image
            inpaint_image = pipe.preprocess_image(inpaint_image)
            inpaint_image = inpaint_image * (inpaint_mask < 0.5)
            inpaint_mask = torch.nn.functional.interpolate(1 - inpaint_mask, (height // 8, width // 8), mode='nearest')[:, :1]
        else:
            inpaint_mask = torch.zeros((1, 1, height // 8, width // 8), dtype=pipe.torch_dtype, device=pipe.device)
            inpaint_image = torch.zeros((1, 3, height, width), dtype=pipe.torch_dtype, device=pipe.device)
        inpaint_latent = pipe.vae_encoder(inpaint_image)

        control_context = torch.concat([control_latents, inpaint_mask, inpaint_latent], dim=1)
        control_context = rearrange(control_context, "B C H W -> B C 1 H W")
        return {"control_context": control_context, "control_scale": controlnet_input.scale}


def model_fn_z_image(
    dit: ZImageDiT,
    controlnet: ZImageControlNet = None,
    latents=None,
    timestep=None,
    prompt_embeds=None,
    image_embeds=None,
    image_latents=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    # Due to the complex and verbose codebase of Z-Image,
    # we are temporarily using this inelegant structure.
    # We will refactor this part in the future (if time permits).
    if dit.siglip_embedder is None:
        return model_fn_z_image_turbo(
            dit,
            controlnet=controlnet,
            latents=latents,
            timestep=timestep,
            prompt_embeds=prompt_embeds,
            image_embeds=image_embeds,
            image_latents=image_latents,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **kwargs,
        )
    latents = [rearrange(latents, "B C H W -> C B H W")]
    if dit.siglip_embedder is not None:
        if image_latents is not None:
            image_latents = [rearrange(image_latent, "B C H W -> C B H W") for image_latent in image_latents]
            latents = [image_latents + latents]
            image_noise_mask = [[0] * len(image_latents) + [1]]
        else:
            latents = [latents]
            image_noise_mask = [[1]]
        image_embeds = [image_embeds]
    else:
        image_noise_mask = None
    timestep = (1000 - timestep) / 1000
    model_output = dit(
        latents,
        timestep,
        prompt_embeds,
        siglip_feats=image_embeds,
        image_noise_mask=image_noise_mask,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )[0]
    model_output = -model_output
    model_output = rearrange(model_output, "C B H W -> B C H W")
    return model_output


class ZImageUnit_Image2LoRAEncode(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("image2lora_images",),
            output_params=("image2lora_x",),
            onload_model_names=("siglip2_image_encoder", "dinov3_image_encoder",),
        )
        from ..core.data.operators import ImageCropAndResize
        self.processor_highres = ImageCropAndResize(height=1024, width=1024)
    
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

    def encode_images(self, pipe: ZImagePipeline, images: list[Image.Image]):
        if images is None:
            return {}
        if not isinstance(images, list):
            images = [images]
        embs_siglip2 = self.encode_images_using_siglip2(pipe, images)
        embs_dinov3 = self.encode_images_using_dinov3(pipe, images)
        x = torch.concat([embs_siglip2, embs_dinov3], dim=-1)
        return x

    def process(self, pipe: ZImagePipeline, image2lora_images):
        if image2lora_images is None:
            return {}
        x = self.encode_images(pipe, image2lora_images)
        return {"image2lora_x": x}


class ZImageUnit_Image2LoRADecode(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("image2lora_x",),
            output_params=("lora",),
            onload_model_names=("image2lora_style",),
        )
    
    def process(self, pipe: ZImagePipeline, image2lora_x):
        if image2lora_x is None:
            return {}
        loras = []
        if pipe.image2lora_style is not None:
            pipe.load_models_to_device(["image2lora_style"])
            for x in image2lora_x:
                loras.append(pipe.image2lora_style(x=x, residual=None))
        lora = merge_lora(loras, alpha=1 / len(image2lora_x))
        return {"lora": lora}


def model_fn_z_image_turbo(
    dit: ZImageDiT,
    controlnet: ZImageControlNet = None,
    latents=None,
    timestep=None,
    prompt_embeds=None,
    image_embeds=None,
    image_latents=None,
    control_context=None,
    control_scale=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    while isinstance(prompt_embeds, list):
        prompt_embeds = prompt_embeds[0]
    while isinstance(latents, list):
        latents = latents[0]
    while isinstance(image_embeds, list):
        image_embeds = image_embeds[0]

    # Timestep
    timestep = 1000 - timestep
    t_noisy = dit.t_embedder(timestep)
    t_clean = dit.t_embedder(torch.ones_like(timestep) * 1000)

    # Patchify
    latents = rearrange(latents, "B C H W -> C B H W")
    x, cap_feats, patch_metadata = dit.patchify_and_embed([latents], [prompt_embeds])
    x = x[0]
    cap_feats = cap_feats[0]

    # Noise refine
    x = dit.all_x_embedder["2-1"](x)
    x[torch.cat(patch_metadata.get("x_pad_mask"))] = dit.x_pad_token.to(dtype=x.dtype, device=x.device)
    x_freqs_cis = dit.rope_embedder(torch.cat(patch_metadata.get("x_pos_ids"), dim=0))
    x = rearrange(x, "L C -> 1 L C")
    x_freqs_cis = rearrange(x_freqs_cis, "L C -> 1 L C")

    if control_context is not None:
        kwargs = dict(attn_mask=None, freqs_cis=x_freqs_cis, adaln_input=t_noisy)
        refiner_hints, control_context, control_context_item_seqlens = controlnet.forward_refiner(
            dit, x, [cap_feats], control_context, kwargs, t=t_noisy, patch_size=2, f_patch_size=1,
            use_gradient_checkpointing=use_gradient_checkpointing, use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
    
    for layer_id, layer in enumerate(dit.noise_refiner):
        x = gradient_checkpoint_forward(
            layer,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            x=x,
            attn_mask=None,
            freqs_cis=x_freqs_cis,
            adaln_input=t_noisy,
        )
        if control_context is not None:
            x = x + refiner_hints[layer_id] * control_scale

    # Prompt refine
    cap_feats = dit.cap_embedder(cap_feats)
    cap_feats[torch.cat(patch_metadata.get("cap_pad_mask"))] = dit.cap_pad_token.to(dtype=x.dtype, device=x.device)
    cap_freqs_cis = dit.rope_embedder(torch.cat(patch_metadata.get("cap_pos_ids"), dim=0))
    cap_feats = rearrange(cap_feats, "L C -> 1 L C")
    cap_freqs_cis = rearrange(cap_freqs_cis, "L C -> 1 L C")
    
    for layer in dit.context_refiner:
        cap_feats = gradient_checkpoint_forward(
            layer,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            x=cap_feats,
            attn_mask=None,
            freqs_cis=cap_freqs_cis,
        )

    # Unified
    unified = torch.cat([x, cap_feats], dim=1)
    unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=1)

    if control_context is not None:
        kwargs = dict(attn_mask=None, freqs_cis=unified_freqs_cis, adaln_input=t_noisy)
        hints = controlnet.forward_layers(
            unified, cap_feats, control_context, control_context_item_seqlens, kwargs,
            use_gradient_checkpointing=use_gradient_checkpointing, use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    for layer_id, layer in enumerate(dit.layers):
        unified = gradient_checkpoint_forward(
            layer,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            x=unified,
            attn_mask=None,
            freqs_cis=unified_freqs_cis,
            adaln_input=t_noisy,
        )
        if control_context is not None:
            if layer_id in controlnet.control_layers_mapping:
                unified = unified + hints[controlnet.control_layers_mapping[layer_id]] * control_scale
    
    # Output
    unified = dit.all_final_layer["2-1"](unified, t_noisy)
    x = dit.unpatchify([unified[0]], patch_metadata.get("x_size"))[0]
    x = rearrange(x, "C B H W -> B C H W")
    x = -x
    return x


def apply_npu_patch(enable_npu_patch: bool=True):
    if IS_NPU_AVAILABLE and enable_npu_patch:
        from ..models.general_modules import RMSNorm
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        from ..models.z_image_dit import Attention
        from ..core.npu_patch.npu_fused_operator import (
            rms_norm_forward_npu, 
            rms_norm_forward_transformers_npu,
            rotary_emb_Zimage_npu
        )
        warnings.warn("Replacing RMSNorm and Rope with NPU fusion operators to improve the performance of the model on NPU.Set enable_npu_patch=False to disable this feature.")
        RMSNorm.forward = rms_norm_forward_npu
        Qwen3RMSNorm.forward = rms_norm_forward_transformers_npu
        Attention.apply_rotary_emb = rotary_emb_Zimage_npu
