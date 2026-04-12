"""
ERNIE-Image Text-to-Image Pipeline for DiffSynth-Studio.

Architecture: SharedAdaLN DiT + RoPE 3D + Joint Image-Text Attention.
"""

import torch, json
import numpy as np
from PIL import Image
from typing import Union, List, Optional
from tqdm import tqdm
from transformers import AutoTokenizer

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.ernie_image_text_encoder import ErnieImageTextEncoder
from ..models.ernie_image_dit import ErnieImageDiT
from ..models.ernie_image_pe import ErnieImagePE
from ..models.flux2_vae import Flux2VAE


# ============================================================
# ErnieImagePipeline
# ============================================================

class ErnieImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("ERNIE-Image")
        self.text_encoder: ErnieImageTextEncoder = None
        self.dit: ErnieImageDiT = None
        self.vae: Flux2VAE = None
        self.tokenizer: AutoTokenizer = None
        self.pe: ErnieImagePE = None
        self.pe_tokenizer: AutoTokenizer = None

        self.in_iteration_models = ("dit",)
        self.units = [
            ErnieImageUnit_ShapeChecker(),
            ErnieImageUnit_PromptEnhancer(),
            ErnieImageUnit_PromptEmbedder(),
            ErnieImageUnit_NoiseInitializer(),
            ErnieImageUnit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_ernie_image
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="tokenizer/"),
        pe_tokenizer_config: ModelConfig = ModelConfig(model_id="baidu/ERNIE-Image", origin_file_pattern="pe/"),
        vram_limit: float = None,
    ):
        pipe = ErnieImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)

        pipe.text_encoder = model_pool.fetch_model("ernie_image_text_encoder")
        pipe.dit = model_pool.fetch_model("ernie_image_dit")
        pipe.vae = model_pool.fetch_model("flux2_vae")
        pipe.pe = model_pool.fetch_model("ernie_image_pe")

        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)

        if pe_tokenizer_config is not None:
            pe_tokenizer_config.download_if_necessary()
            pipe.pe_tokenizer = AutoTokenizer.from_pretrained(pe_tokenizer_config.path)

        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        # PE (Prompt Enhancement)
        use_pe: bool = False,
        pe_temperature: float = 0.6,
        pe_top_p: float = 0.95,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cuda",
        # Steps
        num_inference_steps: int = 50,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps)

        # Parameters
        inputs_posi = {"prompt": prompt, "use_pe": use_pe, "pe_temperature": pe_temperature, "pe_top_p": pe_top_p}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "height": height, "width": width, "seed": seed,
            "cfg_scale": cfg_scale, "num_inference_steps": num_inference_steps,
            "rand_device": rand_device,
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
        self.load_models_to_device(['vae'])
        latents = inputs_shared["latents"]
        # VAE decode handles BN unnormalization and unpatchify internally (Flux2VAE.decode L2105-2110)
        image = self.vae.decode(latents)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        # Return revised prompt if PE was used
        revised_prompt = inputs_posi.get("revised_prompt", None)
        if use_pe and revised_prompt is not None:
            return image, revised_prompt
        return image


# ============================================================
# PipelineUnit Classes
# ============================================================

class ErnieImageUnit_ShapeChecker(PipelineUnit):
    """Size validation for height and width."""
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: ErnieImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class ErnieImageUnit_PromptEnhancer(PipelineUnit):
    """Prompt Enhancement using PE model to rewrite/enhance prompts.

    PE enhancement is only applied to positive prompts; negative prompts pass through.
    """
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={
                "use_pe": "use_pe",
                "prompt": "prompt",
                "pe_temperature": "pe_temperature",
                "pe_top_p": "pe_top_p",
            },
            input_params_nega={
                "use_pe": "use_pe",
                "prompt": "negative_prompt",
            },
            input_params=("height", "width"),
            output_params=("prompt", "revised_prompt"),
            onload_model_names=("pe",)
        )

    def enhance_prompt(self, pipe: ErnieImagePipeline, prompt, height, width, temperature, top_p, system_prompt=None):
        """Enhance a prompt using the PE model."""
        if pipe.pe is None or pipe.pe_tokenizer is None:
            return prompt

        # Build user message as JSON carrying prompt text and target resolution
        user_content = json.dumps(
            {"prompt": prompt, "width": width, "height": height},
            ensure_ascii=False,
        )
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        # apply_chat_template picks up the chat_template.jinja loaded with pe_tokenizer
        input_text = pipe.pe_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,  # "Output:" is already in the user block
        )
        inputs = pipe.pe_tokenizer(input_text, return_tensors="pt").to(pipe.device)
        output_ids = pipe.pe.generate(
            **inputs,
            max_new_tokens=pipe.pe_tokenizer.model_max_length,
            do_sample=temperature != 1.0 or top_p != 1.0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pipe.pe_tokenizer.pad_token_id,
            eos_token_id=pipe.pe_tokenizer.eos_token_id,
        )
        # Decode only newly generated tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return pipe.pe_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def process(self, pipe: ErnieImagePipeline, use_pe=False, prompt="", height=1024, width=1024,
                pe_temperature=1.0, pe_top_p=1.0):
        """PipelineUnitRunner calls process() twice with seperate_cfg:
        - Positive: reads from input_params_posi + input_params → use_pe has value, height/width have values
        - Negative: reads from input_params_nega → use_pe is None → pass through directly

        PE enhancement is only applied to positive prompts.
        """
        if use_pe and pipe.pe is not None and pipe.pe_tokenizer is not None:
            # Positive prompt: enhance with PE using the resolution passed to this unit
            pipe.load_models_to_device(self.onload_model_names)
            enhanced = self.enhance_prompt(pipe, prompt, height, width, pe_temperature, pe_top_p)
            return {"prompt": enhanced, "revised_prompt": enhanced}
        # Negative prompt or PE not enabled: pass through
        return {"prompt": prompt}


class ErnieImageUnit_PromptEmbedder(PipelineUnit):
    """Text condition encoding via text_encoder with padding to uniform length.

    Processes positive/negative prompts separately, encodes via text_encoder,
    then pads to uniform length.
    """
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_embeds", "prompt_embeds_mask"),
            onload_model_names=("text_encoder",)
        )

    def encode_prompt(self, pipe: ErnieImagePipeline, prompt):
        if isinstance(prompt, str):
            prompt = [prompt]

        text_hiddens = []
        text_lens_list = []
        for p in prompt:
            ids = pipe.tokenizer(
                p,
                add_special_tokens=True,
                truncation=True,
                padding=False,
            )["input_ids"]

            if len(ids) == 0:
                if pipe.tokenizer.bos_token_id is not None:
                    ids = [pipe.tokenizer.bos_token_id]
                else:
                    ids = [0]

            input_ids = torch.tensor([ids], device=pipe.device)
            outputs = pipe.text_encoder(
                input_ids=input_ids,
            )
            # Text encoder returns tuple of (hidden_states_tuple,) where each layer's hidden state is included
            all_hidden_states = outputs[0]
            hidden = all_hidden_states[-2][0]  # [T, H] - second to last layer
            text_hiddens.append(hidden)
            text_lens_list.append(hidden.shape[0])

        # Pad to uniform length
        if len(text_hiddens) == 0:
            text_in_dim = pipe.text_encoder.config.hidden_size if hasattr(pipe.text_encoder, 'config') else 3072
            return {
                "prompt_embeds": torch.zeros((0, 0, text_in_dim), device=pipe.device, dtype=pipe.torch_dtype),
                "prompt_embeds_mask": torch.zeros((0,), device=pipe.device, dtype=torch.long),
            }

        normalized = [th.to(pipe.device).to(pipe.torch_dtype) for th in text_hiddens]
        text_lens = torch.tensor([t.shape[0] for t in normalized], device=pipe.device, dtype=torch.long)
        Tmax = int(text_lens.max().item())
        text_in_dim = normalized[0].shape[1]
        text_bth = torch.zeros((len(normalized), Tmax, text_in_dim), device=pipe.device, dtype=pipe.torch_dtype)
        for i, t in enumerate(normalized):
            text_bth[i, :t.shape[0], :] = t

        return {"prompt_embeds": text_bth, "prompt_embeds_mask": text_lens}

    def process(self, pipe: ErnieImagePipeline, prompt):
        pipe.load_models_to_device(self.onload_model_names)
        if pipe.text_encoder is not None:
            return self.encode_prompt(pipe, prompt)
        return {}


class ErnieImageUnit_NoiseInitializer(PipelineUnit):
    """Initial noise generation using pipeline generate_noise."""
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: ErnieImagePipeline, height, width, seed, rand_device):
        latent_h = height // pipe.height_division_factor
        latent_w = width // pipe.width_division_factor
        latent_channels = pipe.dit.in_channels

        # Use pipeline device if rand_device is not specified
        if rand_device is None:
            rand_device = str(pipe.device)

        noise = pipe.generate_noise(
            (1, latent_channels, latent_h, latent_w),
            seed=seed,
            rand_device=rand_device,
            rand_torch_dtype=pipe.torch_dtype,
        )
        return {"noise": noise}


class ErnieImageUnit_InputImageEmbedder(PipelineUnit):
    """Input image embedding via VAE encoding.

    Inference mode (training=False):
    - input_image is None (pure T2I): latents = noise
    - input_image is not None: VAE encode input image → add_noise → latents (I2I path)

    Training mode (training=True):
    - input_image provided by get_pipeline_inputs into inputs_shared
    - Returns latents = noise, input_latents = VAE encoded input image
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: ErnieImagePipeline, input_image, noise):
        if input_image is None:
            # T2I path: use noise directly as initial latents
            return {"latents": noise, "input_latents": None}

        # I2I path: VAE encode input image
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae.encode(image)

        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            # In inference mode, add noise to encoded latents
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}


# ============================================================
# model_fn
# ============================================================

def model_fn_ernie_image(
    dit: ErnieImageDiT,
    latents=None,
    timestep=None,
    prompt_embeds=None,
    prompt_embeds_mask=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    output = dit(
        hidden_states=latents,
        timestep=timestep,
        text_bth=prompt_embeds,
        text_lens=prompt_embeds_mask,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return output
