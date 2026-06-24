import torch, math
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange, repeat

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.krea2_dit import SingleStreamDiT
from ..models.krea2_text_encoder import Krea2TextEncoder
from ..models.qwen_image_vae import QwenImageVAE
from ..utils.lora.krea2 import Krea2LoRALoader


class Krea2Pipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("Krea-2")
        self.text_encoder: Krea2TextEncoder = None
        self.dit: SingleStreamDiT = None
        self.vae: QwenImageVAE = None
        self.tokenizer = None
        self.in_iteration_models = ("dit",)
        self.units = [
            Krea2Unit_ShapeChecker(),
            Krea2Unit_NoiseInitializer(),
            Krea2Unit_PromptEmbedder(),
            Krea2Unit_InputImageEmbedder(),
            Krea2Unit_PromptEmbPreCompute(),
        ]
        self.model_fn = model_fn_krea2
        self.compilable_models = ["dit"]
        self.lora_loader = Krea2LoRALoader

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen3-VL-4B-Instruct", origin_file_pattern=""),
        vram_limit: float = None,
    ):
        pipe = Krea2Pipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.text_encoder = model_pool.fetch_model("krea2_text_encoder")
        pipe.dit = model_pool.fetch_model("krea2_dit")
        pipe.vae = model_pool.fetch_model("qwen_image_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import AutoTokenizer
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path, max_length=512)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        cfg_scale: float = 3.5,
        height: int = 1024,
        width: int = 1024,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 52,
        mu: float = None,
        context_pre_compute: bool = True,
        progress_bar_cmd = tqdm,
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, dynamic_shift_len=(height // 16) * (width // 16), mu=mu)

        inputs_posi = {"prompt": prompt, "context_pre_compute": context_pre_compute}
        inputs_nega = {"negative_prompt": negative_prompt, "context_pre_compute": context_pre_compute}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

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

        self.load_models_to_device(['vae'])
        image = self.vae.decode(inputs_shared["latents"])
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])
        return image


class Krea2Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: Krea2Pipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class Krea2Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: Krea2Pipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}


class Krea2Unit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=(),
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )

    def encode_prompt(self, pipe: Krea2Pipeline, prompt):
        select_layers = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
        max_length = 512
        prefix_idx = 34
        suffix_start_idx = 5
        prompt_template_prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        prompt_template_suffix = "<|im_end|>\n<|im_start|>assistant\n"

        prompt = [prompt_template_prefix + p for p in prompt]
        suffix_text = [prompt_template_suffix] * len(prompt)
        suffix_inputs = pipe.processor(text=suffix_text, return_tensors="pt").to(pipe.device)
        suffix_ids = suffix_inputs["input_ids"]
        suffix_mask = suffix_inputs["attention_mask"].bool()

        inputs = pipe.tokenizer(
            prompt, truncation=True,
            padding="max_length",
            max_length=max_length + prefix_idx - suffix_start_idx,
            return_tensors="pt",
        ).to(pipe.device)
        input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)

        pipe.load_models_to_device(["text_encoder"])
        hidden_states = pipe.text_encoder(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)

        hiddens = torch.stack([hidden_states[i] for i in select_layers], dim=2)
        hiddens = hiddens[:, prefix_idx:]
        mask = mask[:, prefix_idx:]
        return hiddens, mask

    def process(self, pipe: Krea2Pipeline, prompt) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        if pipe.text_encoder is not None:
            prompt = [prompt]
            hiddens, mask = self.encode_prompt(pipe, prompt)
            hiddens = hiddens.to(dtype=pipe.torch_dtype)
            return {"prompt_emb": hiddens, "prompt_emb_mask": mask}
        else:
            return {}


class Krea2Unit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: Krea2Pipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(input_image)
        input_latents = pipe.vae.encode(image)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


class Krea2Unit_PromptEmbPreCompute(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt_emb": "prompt_emb", "prompt_emb_mask": "prompt_emb_mask", "context_pre_compute": "context_pre_compute"},
            input_params_nega={"prompt_emb": "prompt_emb", "prompt_emb_mask": "prompt_emb_mask", "context_pre_compute": "context_pre_compute"},
            input_params=("use_gradient_checkpointing", "use_gradient_checkpointing_offload"),
            output_params=("prompt_emb", "context_pre_compute"),
            onload_model_names=("dit",)
        )

    def process(self, pipe: Krea2Pipeline, prompt_emb, prompt_emb_mask, context_pre_compute, use_gradient_checkpointing, use_gradient_checkpointing_offload) -> dict:
        if context_pre_compute is None or context_pre_compute == False:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if use_gradient_checkpointing is None: use_gradient_checkpointing = False
        if use_gradient_checkpointing_offload is None: use_gradient_checkpointing_offload = False
        mask = prompt_emb_mask.unsqueeze(1).unsqueeze(2) * prompt_emb_mask.unsqueeze(1).unsqueeze(3)
        prompt_emb = pipe.dit.txtfusion(
            prompt_emb, mask=mask,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        prompt_emb = pipe.dit.txtmlp(prompt_emb)
        return {"prompt_emb": prompt_emb, "context_pre_compute": True}


def _krea2_prepare(img, txtlen, patch, txtmask):
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask, imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask


def model_fn_krea2(
    dit: SingleStreamDiT = None,
    latents=None,
    timestep=None,
    prompt_emb=None,
    prompt_emb_mask=None,
    height=None,
    width=None,
    context_pre_compute=False,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs
):
    patch = dit.config.patch
    timestep = timestep

    txtlen = prompt_emb.shape[1]
    img, pos, mask = _krea2_prepare(latents, txtlen, patch, prompt_emb_mask)

    output = dit(
        img=img, context=prompt_emb, t=timestep, pos=pos, mask=mask,
        context_pre_compute=context_pre_compute,
        use_gradient_checkpointing=use_gradient_checkpointing, use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )

    h_ = height // (8 * patch)
    w_ = width // (8 * patch)
    output = rearrange(
        output,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch, pw=patch, h=h_, w=w_,
    )
    return output
