# HiUni Pipeline for DiffSynth-Studio: text-to-image, instruction-based editing and image understanding.

import math, os, random
import torch
import torch.nn.functional as F
from typing import Union
from tqdm import tqdm
from PIL import Image

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.hiuni_dit import HiUniModel, HIUNI_TIME_TOKEN_ID


HIUNI_REPO = "ernie-research/HiUni"


class HiUniPipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=32, width_division_factor=32)
        self.scheduler = FlowMatchScheduler("HiUni")
        self.dit: HiUniModel = None
        self.tokenizer = None
        self.processor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            HiUniUnit_EditImageEmbedder(),
            HiUniUnit_InputImageEmbedder(),
            HiUniUnit_ShapeChecker(),
            HiUniUnit_PromptTokenizer(),
            HiUniUnit_UnconditionalEditBranch(),
            HiUniUnit_NoiseInitializer(),
        ]
        self.model_fn = model_fn_hiuni
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id=HIUNI_REPO, origin_file_pattern="tokenizer*.json"),
        processor_config: ModelConfig = ModelConfig(model_id=HIUNI_REPO, origin_file_pattern="preprocessor_config.json"),
        vram_limit: float = None,
        parallel_config=None,
    ):
        # Initialize pipeline
        pipe = HiUniPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit, parallel_config=parallel_config)

        # Fetch models
        pipe.dit = model_pool.fetch_model("hiuni_dit")
        if tokenizer_config is not None:
            # The tokenizer is a standard Qwen2 tokenizer; loading its class directly skips the checkpoint's
            # `auto_map` custom code, which `AutoTokenizer` would otherwise try to import.
            from transformers import Qwen2Tokenizer
            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(pipe.config_directory(tokenizer_config))
        if processor_config is not None:
            # The slow Qwen2-VL image processor matches the training preprocessing numerically.
            from transformers import Qwen2VLImageProcessor
            pipe.processor = Qwen2VLImageProcessor.from_pretrained(pipe.config_directory(processor_config))

        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @staticmethod
    def config_directory(config: ModelConfig):
        # The tokenizer and the image processor load from a directory; a file pattern resolves to its directory.
        config.download_if_necessary()
        path = config.path[0] if isinstance(config.path, list) else config.path
        return path if os.path.isdir(path) else os.path.dirname(path)

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        cfg_scale: float = 4.0,
        # Shape
        height: int = None,
        width: int = None,
        # Randomness
        seed: int = 42,
        # Steps
        num_inference_steps: int = 25,
        shift: float = 4.0,
        # Image editing
        edit_image: Image.Image = None,
        image_cfg_scale: float = 1.0,
        resize_to_edit_image: bool = True,
        min_pixels: int = 262144,
        max_pixels: int = 262144,
        # Progress bar
        progress_bar_cmd=tqdm,
    ):
        # The reference-free branch only guides editing, and that guidance needs the negative branch.
        if edit_image is not None and image_cfg_scale != 1 and cfg_scale == 1:
            raise ValueError("image_cfg_scale != 1 requires cfg_scale != 1")

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, shift=shift)

        # Parameters
        inputs_posi = {"prompt": prompt, "is_positive": True}
        inputs_nega = {"is_positive": False}
        inputs_shared = {
            "cfg_scale": cfg_scale, "image_cfg_scale": image_cfg_scale,
            "height": height, "width": width,
            "seed": seed, "num_inference_steps": num_inference_steps,
            "edit_image": edit_image, "resize_to_edit_image": resize_to_edit_image,
            "min_pixels": min_pixels, "max_pixels": max_pixels,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # The model embeds the timestep on the [0, 1000] scale, and the step length is a sigma.
            timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, sigma=self.scheduler.sigmas[progress_id].to(self.device),
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

        # Decode
        size = edit_image.size if edit_image is not None and resize_to_edit_image else None
        image = self.decode_latents(inputs_shared["latents"], inputs_shared["height"], inputs_shared["width"], size)
        self.load_models_to_device([])
        return image

    def cfg_guided_model_fn(self, model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega, **inputs_others):
        uncond_edit_inputs = inputs_shared.get("uncond_edit_inputs")
        if uncond_edit_inputs is None:
            return super().cfg_guided_model_fn(model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega, **inputs_others)
        # Editing with image_cfg_scale != 1 adds a third, reference-free branch, which no framework helper covers
        # and which cannot be split across CFG ranks: guided = full + cfg_scale * (posi - nega) + image_cfg_scale * (nega - full).
        posi, nega, full = (model_fn(**branch, **inputs_shared, **inputs_others) for branch in (inputs_posi, inputs_nega, uncond_edit_inputs))
        return full + cfg_scale * (posi - nega) + inputs_shared["image_cfg_scale"] * (nega - full)

    def decode_latents(self, latents, height, width, size=None):
        # Raster pixel patches [4*h*w, 768] in [-1, 1] -> PIL. The default output keeps the reference resolution,
        # so the resize happens in float space before quantisation, as in the reference sampler.
        if not torch.isfinite(latents).all():
            raise RuntimeError("Non-finite pixels produced during denoising")
        h, w = height // 32, width // 32
        image = latents.float().reshape(2 * h, 2 * w, 3, 16, 16).permute(2, 0, 3, 1, 4).reshape(3, height, width)
        image = image.mul(0.5).add(0.5).clamp(0, 1)
        if size is not None and (size[1], size[0]) != (height, width):
            image = F.interpolate(image[None], size=(size[1], size[0]), mode="bicubic", align_corners=False, antialias=True)[0]
        return Image.fromarray(image.clamp(0, 1).mul(255).add(0.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy())

    @torch.no_grad()
    def chat(
        self,
        prompt: str,
        images: Union[Image.Image, list[Image.Image]] = None,
        system: str = "",
        think_mode: str = "think",
        max_new_tokens: int = 1024,
        max_length: int = 9216,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 42,
        min_pixels: int = 262144,
        max_pixels: int = 4194304,
        return_raw: bool = False,
    ):
        # Image understanding and text chat. The answer is everything after the last </think>, or the full
        # decoded text with return_raw=True.
        inputs_shared = {
            "prompt": prompt, "images": images, "system": system, "think_mode": think_mode,
            "max_new_tokens": max_new_tokens, "max_length": max_length,
            "temperature": temperature, "top_k": top_k, "top_p": top_p, "seed": seed,
            "min_pixels": min_pixels, "max_pixels": max_pixels,
        }
        inputs_shared, _, _ = self.unit_runner(HiUniUnit_ChatGenerator(), self, inputs_shared, {}, {})
        raw = inputs_shared["text"]
        return raw if return_raw else raw.rsplit("</think>", 1)[-1].strip()


class HiUniUnit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "height", "width", "min_pixels", "max_pixels"),
            output_params=("edit_reference", "height", "width"),
            onload_model_names=("dit",),
        )

    def process(self, pipe: HiUniPipeline, edit_image, height, width, min_pixels, max_pixels):
        # The clean reference goes through the ViT once; its raster patches feed the HiGate at every step.
        if edit_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        kwargs = {key: int(value) for key, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)) if value}
        inputs = pipe.processor(images=[edit_image.convert("RGB")], return_tensors="pt", **kwargs)
        grid = inputs["image_grid_thw"].long()
        _, h, w = grid[0].tolist()
        pixels = inputs["pixel_values"].to(pipe.device)
        features = pipe.dit.visual(pixels, grid.to(pipe.device))
        raster = pixels.float().reshape(h // 2, w // 2, 2, 2, 3, 2, 16, 16)[:, :, :, :, :, 0].permute(0, 2, 1, 3, 4, 5, 6).reshape(h * w, 768)
        edit_reference = {"features": features, "raster": raster, "height": h // 2, "width": w // 2}
        # By default the output keeps the reference resolution.
        return {"edit_reference": edit_reference, "height": height or h * 16, "width": width or w * 16}


class HiUniUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "height", "width"),
            output_params=("input_latents", "height", "width"),
        )

    @staticmethod
    def encode_image(pipe: HiUniPipeline, image, height, width):
        # Pixels in [-1, 1] in the raster patch layout of the diffusion head, the inverse of `decode_latents`.
        pixels = pipe.preprocess_image(image.resize((width, height), Image.BICUBIC), torch_dtype=torch.float32, pattern="C H W")
        h, w = height // 32, width // 32
        return pixels.reshape(3, 2 * h, 16, 2 * w, 16).permute(1, 3, 0, 2, 4).reshape(-1, 768)

    def process(self, pipe: HiUniPipeline, input_image, height, width):
        # Training only: the target image becomes the raster patches that the flow matching loss noises.
        if input_image is None:
            return {"input_latents": None}
        height, width = pipe.check_resize_height_width(height or input_image.size[1], width or input_image.size[0])
        return {"input_latents": self.encode_image(pipe, input_image, height, width), "height": height, "width": width}


class HiUniUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: HiUniPipeline, height, width):
        height, width = pipe.check_resize_height_width(height or 1024, width or 1024)
        return {"height": height, "width": width}


class HiUniUnit_PromptTokenizer(PipelineUnit):
    # Instruction prefixes seen in training; T2I picks one with random.Random(seed).
    _INSTRUCTIONS = [
        "请根据以下描述生成一张图片：", "根据下面的描述生成图片：", "生成图片：", "请画一张图：", "帮我生成一张图片：", "根据描述创建图像：",
        "请根据文字描述绘制图片：", "按照以下内容生成图片：", "生成一张符合描述的图片：", "请为以下内容创作一张图片：", "根据提示生成图像：", "图片生成：",
        "Generate an image based on the following description: ", "Create an image: ", "Please generate an image: ", "Draw an image based on: ",
        "Create a picture: ", "Generate: ", "Please draw: ", "Create an image from the description: ", "Generate a picture: ",
        "Please create an image based on: ",
    ]
    _IMG_PAD, _BOI, _EOI, _TIME = 151655, 151652, 151653, HIUNI_TIME_TOKEN_ID
    _IM_START, _IM_END, _NL, _USER, _ASSISTANT = 151644, 151645, 198, 872, 77091

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params=("height", "width", "seed", "edit_reference"),
            input_params_posi={"prompt": "prompt", "is_positive": "is_positive"},
            input_params_nega={"is_positive": "is_positive"},
            output_params=("input_ids", "position_ids", "attention_mask"),
        )

    @staticmethod
    def build_positions(ids, merged_grids):
        # Training M-RoPE layout [3, L]: linear text positions, a (t, h, w) ramp per contiguous image pad block.
        ids = torch.as_tensor(ids)
        positions = torch.zeros(3, ids.numel(), dtype=torch.long)
        grids, pos, index = iter(merged_grids), 0, 0
        while index < ids.numel():
            if ids[index] != HiUniUnit_PromptTokenizer._IMG_PAD:
                positions[:, index] = pos
                pos, index = pos + 1, index + 1
                continue
            t, h, w = next(grids)
            grid = torch.stack(torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij"))
            positions[:, index:index + t * h * w] = pos + grid.reshape(3, -1)
            pos, index = pos + max(t, h, w), index + t * h * w
        return positions

    @staticmethod
    def build_startend(ids):
        # Per-column (first visible row, first hidden row): causal, image blocks bidirectional, noisy blocks closed after EOI.
        cls, ids = HiUniUnit_PromptTokenizer, torch.as_tensor(ids)
        length = ids.numel()
        ute, lts = torch.arange(length), torch.full((length,), length)
        for start, end in zip(ids.eq(cls._BOI).nonzero(as_tuple=True)[0].tolist(), ids.eq(cls._EOI).nonzero(as_tuple=True)[0].tolist()):
            if start >= end:
                continue
            times = ids[start + 1:end].eq(cls._TIME).nonzero(as_tuple=True)[0]
            patch_start = start + 2 + int(times[0]) if times.numel() else start + 1
            ute[patch_start:end] = patch_start
            if times.numel():
                lts[start:end + 1] = lts[start:end + 1].clamp(max=end + 1)
        return ute, lts

    @staticmethod
    def build_attention_mask(ute, lts, dtype, device):
        rows = torch.arange(ute.numel(), device=device)[:, None]
        allowed = (ute.to(device)[None, :] <= rows) & (rows < lts.to(device)[None, :])
        return torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill_(~allowed, torch.finfo(dtype).min)[None, None]

    @staticmethod
    def t2i_layout(ids, height, width, dtype, device):
        # T2I layout: one image block at fixed position `start` with row/column ramps, causal elsewhere.
        pad_index = ids.eq(HiUniUnit_PromptTokenizer._IMG_PAD).nonzero(as_tuple=True)[0]
        start, end, length = int(pad_index[0]), int(pad_index[-1]) + 1, ids.numel()
        positions = torch.arange(length).view(1, 1, -1).repeat(3, 1, 1)
        positions[0, 0, start:end] = start
        positions[1, 0, start:end] = start + torch.arange(height).repeat_interleave(width)
        positions[2, 0, start:end] = start + torch.arange(width).repeat(height)
        positions[:, 0, end:] = start + max(height, width) + torch.arange(length - end)
        allowed = torch.ones(length, length, dtype=torch.bool).tril()
        allowed[start:end, start:end] = True
        mask = torch.zeros(length, length, dtype=dtype).masked_fill_(~allowed, torch.finfo(dtype).min)
        return positions.to(device), mask[None, None].to(device)

    @staticmethod
    def build_edit_inputs(pipe: HiUniPipeline, prompt, edit_reference, height, width):
        # Editing branch sequence: user turn [BOI, reference pads, EOI] + instruction, assistant turn
        # [BOI, TIME, pads, EOI, EOS]. Without a reference (the unconditional branch) only the image block is kept.
        cls, h, w = HiUniUnit_PromptTokenizer, height // 32, width // 32
        image = [cls._BOI, cls._TIME] + [cls._IMG_PAD] * (h * w) + [cls._EOI, cls._IM_END]
        if edit_reference is None:
            ids, grids = image, [[1, h, w]]
        else:
            user = [cls._BOI] + [cls._IMG_PAD] * (edit_reference["height"] * edit_reference["width"]) + [cls._EOI]
            user += pipe.tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
            ids = [cls._IM_START, cls._USER, cls._NL] + user + [cls._IM_END, cls._NL]
            ids = ids + [cls._IM_START, cls._ASSISTANT, cls._NL] + image
            grids = [[1, edit_reference["height"], edit_reference["width"]], [1, h, w]]
        mask = cls.build_attention_mask(*cls.build_startend(ids), pipe.torch_dtype, pipe.device)
        positions = cls.build_positions(ids, grids)[:, None].to(pipe.device)
        return {"input_ids": torch.tensor([ids], device=pipe.device), "position_ids": positions, "attention_mask": mask}

    def process(self, pipe: HiUniPipeline, is_positive, height, width, seed, edit_reference=None, prompt=None):
        if edit_reference is not None:
            # Editing: the negative branch keeps the reference and drops the instruction.
            return self.build_edit_inputs(pipe, prompt if is_positive else None, edit_reference, height, width)
        # T2I: the positive branch is the chat prompt with a training instruction prefix, the negative branch
        # the image block alone.
        h, w = height // 32, width // 32
        ids = [self._BOI, self._TIME] + [self._IMG_PAD] * (h * w) + [self._EOI, self._IM_END]
        if is_positive:
            encode = lambda text: pipe.tokenizer.encode(text, add_special_tokens=False)
            instruction = random.Random(seed).choice(self._INSTRUCTIONS)
            prefix = [self._IM_START] + encode("user") + encode("\n") + encode(instruction) + encode(prompt) + [self._IM_END] + encode("\n")
            ids = prefix + [self._IM_START] + encode("assistant") + encode("\n") + ids
        ids = torch.tensor(ids)
        position_ids, attention_mask = self.t2i_layout(ids, h, w, pipe.torch_dtype, pipe.device)
        return {"input_ids": ids[None].to(pipe.device), "position_ids": position_ids, "attention_mask": attention_mask}


class HiUniUnit_UnconditionalEditBranch(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_reference", "image_cfg_scale", "height", "width"),
            output_params=("uncond_edit_inputs",),
        )

    def process(self, pipe: HiUniPipeline, edit_reference, image_cfg_scale, height, width):
        if edit_reference is None or image_cfg_scale == 1:
            return {}
        return {"uncond_edit_inputs": HiUniUnit_PromptTokenizer.build_edit_inputs(pipe, None, None, height, width)}


class HiUniUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed"),
            output_params=("latents", "noise_scale"),
        )

    def process(self, pipe: HiUniPipeline, height, width, seed):
        # Raster pixel patches [4*h*w, 768] in float32; the noise is scaled by sqrt(tokens / 256) as in training.
        tokens = 4 * (height // 32) * (width // 32)
        noise_scale = math.sqrt(tokens / 256)
        generator = None if seed is None else torch.Generator(pipe.device).manual_seed(seed)
        noise = torch.randn(tokens, 768, generator=generator, device=pipe.device, dtype=torch.float32)
        return {"latents": noise * noise_scale, "noise_scale": noise_scale}


class HiUniUnit_ChatGenerator(PipelineUnit):
    _THINK_PREFILLS = {"think": "<think>\n", "empty": "<think>\n\n</think>\n\n", "none": ""}

    def __init__(self):
        super().__init__(
            input_params=("prompt", "images", "system", "think_mode", "max_new_tokens", "max_length",
                          "temperature", "top_k", "top_p", "seed", "min_pixels", "max_pixels"),
            output_params=("text",),
            onload_model_names=("dit",),
        )

    @staticmethod
    def sample_token(logits, temperature, top_k, top_p, generator):
        # Greedy for temperature <= 0, otherwise top-k / top-p sampling.
        if temperature <= 0:
            return int(logits.argmax())
        logits = logits.float() / temperature
        if top_k > 0:
            logits = logits.masked_fill(logits < torch.topk(logits, min(top_k, logits.numel())).values[-1], float("-inf"))
        if 0 < top_p < 1:
            sorted_logits, order = torch.sort(logits, descending=True)
            probs = sorted_logits.softmax(-1)
            logits[order[probs.cumsum(-1) - probs > top_p]] = float("-inf")
        return int(torch.multinomial(logits.softmax(-1), 1, generator=generator))

    def encode_images(self, pipe: HiUniPipeline, images, min_pixels, max_pixels):
        kwargs = {key: int(value) for key, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)) if value}
        inputs = pipe.processor(images=[image.convert("RGB") for image in images], return_tensors="pt", **kwargs)
        grid = inputs["image_grid_thw"].long()
        features = pipe.dit.visual(inputs["pixel_values"].to(pipe.device), grid.to(pipe.device))
        return features, [[t, h // 2, w // 2] for t, h, w in grid.tolist()]

    def encode_prompt(self, pipe: HiUniPipeline, prompt, images, system, think_mode):
        prefill = self._THINK_PREFILLS[think_mode]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content if images else prompt}]
        text = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=False)
        return prefill, pipe.tokenizer.encode(text + prefill, add_special_tokens=False)

    def generate(self, pipe: HiUniPipeline, ids, positions, ute, features, grids,
                 max_new_tokens, max_length, temperature, top_k, top_p, seed):
        # Like training, there is no KV cache: each new token recomputes the whole sequence.
        generator = torch.Generator(pipe.device).manual_seed(seed) if temperature > 0 else None
        eos_id = pipe.tokenizer.eos_token_id
        stop_ids = {HiUniUnit_PromptTokenizer._IM_END} | ({int(eos_id)} if eos_id is not None else set())
        new_tokens = []
        while len(new_tokens) < max_new_tokens and ids.numel() < max_length:
            mask = HiUniUnit_PromptTokenizer.build_attention_mask(ute, torch.full_like(ute, ute.numel()), pipe.torch_dtype, pipe.device)
            logits = pipe.dit.text_logits(ids[None].to(pipe.device), positions[:, None].to(pipe.device), mask, features, len(grids))[0]
            token = self.sample_token(logits, temperature, top_k, top_p, generator)
            if token in stop_ids:
                break
            new_tokens.append(token)
            positions = torch.cat([positions, positions[:, -1:] + 1], dim=1)
            ute = torch.cat([ute, ute.new_tensor([ids.numel()])])
            ids = torch.cat([ids, ids.new_tensor([token])])
        return new_tokens

    @staticmethod
    def expand_image_pads(ids, grids):
        # Each <|image_pad|> becomes a time token (embedding t=0) followed by one pad per merged ViT token.
        tokenizer, counts, expanded = HiUniUnit_PromptTokenizer, iter([t * h * w for t, h, w in grids]), []
        for source in ids:
            if source == tokenizer._IMG_PAD:
                expanded += [tokenizer._TIME] + [tokenizer._IMG_PAD] * next(counts)
            else:
                expanded.append(source)
        return expanded

    def process(self, pipe: HiUniPipeline, prompt, images, system, think_mode, max_new_tokens, max_length,
                temperature, top_k, top_p, seed, min_pixels, max_pixels):
        images = [images] if isinstance(images, Image.Image) else list(images or [])
        prefill, ids = self.encode_prompt(pipe, prompt, images, system, think_mode)
        pipe.load_models_to_device(self.onload_model_names)
        features, grids = None, []
        if images:
            features, grids = self.encode_images(pipe, images, min_pixels, max_pixels)
            ids = self.expand_image_pads(ids, grids)
        positions = HiUniUnit_PromptTokenizer.build_positions(ids, grids)
        ids = torch.tensor(ids)
        # Causal attention; image patch columns are visible from the first patch row of their block.
        ute = torch.arange(ids.numel())
        if images:
            tokenizer = HiUniUnit_PromptTokenizer
            for start, end in zip(ids.eq(tokenizer._BOI).nonzero(as_tuple=True)[0].tolist(), ids.eq(tokenizer._EOI).nonzero(as_tuple=True)[0].tolist()):
                ute[start + 2:end] = start + 2
        new_tokens = self.generate(
            pipe, ids, positions, ute, features, grids, max_new_tokens, max_length, temperature, top_k, top_p, seed,
        )
        return {"text": prefill + pipe.tokenizer.decode(new_tokens, skip_special_tokens=False)}


def model_fn_hiuni(
    dit: HiUniModel,
    latents=None,
    timestep=None,
    sigma=None,
    input_ids=None,
    position_ids=None,
    attention_mask=None,
    height=None,
    width=None,
    edit_reference=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    # The head predicts clean pixels x0; the sampler needs the flow velocity in the raster patch layout of `latents`.
    ref = edit_reference or {}
    x0 = dit(
        input_ids, position_ids, attention_mask, latents, timestep.reshape(1), height // 32, width // 32,
        ref_features=ref.get("features"), ref_raster=ref.get("raster"), ref_height=ref.get("height"), ref_width=ref.get("width"),
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return (latents - x0) / (timestep.reshape(1)[0].float() / 1000 if sigma is None else sigma)
