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


IMG_PAD_ID, BOI_ID, EOI_ID, TIME_TOKEN_ID = 151655, 151652, 151653, HIUNI_TIME_TOKEN_ID
IM_START_ID, IM_END_ID, NL_ID, USER_ID, ASSISTANT_ID = 151644, 151645, 198, 872, 77091
# Instruction prefixes seen in training; T2I picks one with random.Random(seed).
HIUNI_INSTRUCTIONS = [
    "请根据以下描述生成一张图片：", "根据下面的描述生成图片：", "生成图片：", "请画一张图：", "帮我生成一张图片：", "根据描述创建图像：",
    "请根据文字描述绘制图片：", "按照以下内容生成图片：", "生成一张符合描述的图片：", "请为以下内容创作一张图片：", "根据提示生成图像：", "图片生成：",
    "Generate an image based on the following description: ", "Create an image: ", "Please generate an image: ", "Draw an image based on: ",
    "Create a picture: ", "Generate: ", "Please draw: ", "Create an image from the description: ", "Generate a picture: ",
    "Please create an image based on: ",
]
THINK_PREFILLS = {"think": "<think>\n", "empty": "<think>\n\n</think>\n\n", "none": ""}
HIUNI_REPO = "ernie-research/HiUni"


def resolve_directory(config: ModelConfig):
    # Tokenizer and image processor load from a directory; a file pattern resolves to the directory holding the files.
    config.download_if_necessary()
    path = config.path[0] if isinstance(config.path, list) else config.path
    return path if os.path.isdir(path) else os.path.dirname(path)


def build_positions(ids, merged_grids):
    # Training M-RoPE layout [3, L]: linear text positions, a (t, h, w) ramp per contiguous image pad block.
    positions = torch.zeros(3, len(ids), dtype=torch.long)
    grids, pos, index = iter(merged_grids), 0, 0
    while index < len(ids):
        if ids[index] != IMG_PAD_ID:
            positions[:, index] = pos
            pos, index = pos + 1, index + 1
            continue
        t, h, w = next(grids)
        grid = torch.stack(torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij"))
        positions[:, index:index + t * h * w] = pos + grid.reshape(3, -1)
        pos, index = pos + max(t, h, w), index + t * h * w
    return positions


def build_startend(ids):
    # Per-column (first visible row, first hidden row): causal, image blocks bidirectional, noisy blocks closed after EOI.
    ids = torch.as_tensor(ids)
    length = ids.numel()
    ute, lts = torch.arange(length), torch.full((length,), length)
    for start, end in zip(ids.eq(BOI_ID).nonzero(as_tuple=True)[0].tolist(), ids.eq(EOI_ID).nonzero(as_tuple=True)[0].tolist()):
        if start >= end:
            continue
        times = ids[start + 1:end].eq(TIME_TOKEN_ID).nonzero(as_tuple=True)[0]
        patch_start = start + 2 + int(times[0]) if times.numel() else start + 1
        ute[patch_start:end] = patch_start
        if times.numel():
            lts[start:end + 1] = lts[start:end + 1].clamp(max=end + 1)
    return ute, lts


def build_attention_mask(ute, lts, dtype, device):
    rows = torch.arange(ute.numel(), device=device)[:, None]
    allowed = (ute.to(device)[None, :] <= rows) & (rows < lts.to(device)[None, :])
    return torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill_(~allowed, torch.finfo(dtype).min)[None, None]


def t2i_layout(ids, height, width, dtype, device):
    # T2I layout: one image block at fixed position `start` with row/column ramps, causal elsewhere.
    pad_index = ids.eq(IMG_PAD_ID).nonzero(as_tuple=True)[0]
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
            HiUniUnit_ShapeChecker(),
            HiUniUnit_PromptTokenizer(),
            HiUniUnit_UnconditionalEditBranch(),
            HiUniUnit_NoiseInitializer(),
        ]
        self.model_fn = model_fn_hiuni

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id=HIUNI_REPO, origin_file_pattern="tokenizer*.json", download_source="huggingface",
        ),
        processor_config: ModelConfig = ModelConfig(
            model_id=HIUNI_REPO, origin_file_pattern="preprocessor_config.json", download_source="huggingface",
        ),
        vram_limit: float = None,
    ):
        from transformers import AutoTokenizer, Qwen2VLImageProcessor
        pipe = HiUniPipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.dit = model_pool.fetch_model("hiuni_dit")
        if tokenizer_config is not None:
            pipe.tokenizer = AutoTokenizer.from_pretrained(resolve_directory(tokenizer_config))
        if processor_config is not None:
            # The slow Qwen2-VL image processor matches the training preprocessing numerically.
            pipe.processor = Qwen2VLImageProcessor.from_pretrained(resolve_directory(processor_config))
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        cfg_scale: float = 4.0,
        height: int = None,
        width: int = None,
        seed: int = 42,
        num_inference_steps: int = 25,
        shift: float = 4.0,
        edit_image: Image.Image = None,
        image_cfg_scale: float = 1.0,
        resize_to_edit_image: bool = True,
        min_pixels: int = 262144,
        max_pixels: int = 262144,
        progress_bar_cmd=tqdm,
    ):
        # T2I runs [chat prompt, image-only] CFG. Editing runs [reference + instruction, reference only] and, when
        # image_cfg_scale != 1, an unconditional branch: full + cfg * (posi - nega) + image_cfg * (nega - full).
        if edit_image is not None and image_cfg_scale != 1 and cfg_scale == 1:
            raise ValueError("image_cfg_scale != 1 requires cfg_scale != 1")
        self.scheduler.set_timesteps(num_inference_steps, shift=shift)

        inputs_posi = {"prompt": prompt, "branch": "posi"}
        inputs_nega = {"branch": "nega"}
        inputs_shared = {
            "cfg_scale": cfg_scale, "image_cfg_scale": image_cfg_scale,
            "height": height, "width": width, "seed": seed,
            "edit_image": edit_image, "min_pixels": min_pixels, "max_pixels": max_pixels,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        inputs_full = inputs_shared.pop("inputs_full", None)
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Keep the timestep in float32: the model embeds it on the [0, 1000] scale.
            timestep = timestep.reshape(1).to(dtype=torch.float32, device=self.device)
            # CFG combines the x0 predictions, as in the reference sampler, before converting them to a flow velocity.
            if inputs_full is None:
                x0 = self.cfg_guided_model_fn(self.model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega, **models, timestep=timestep)
            else:
                branches = (inputs_posi, inputs_nega, inputs_full)
                posi, nega, full = (self.model_fn(**inputs, **inputs_shared, **models, timestep=timestep) for inputs in branches)
                x0 = full + cfg_scale * (posi - nega) + image_cfg_scale * (nega - full)
            # Divide by a device tensor: CUDA divides by a CPU scalar through its reciprocal, which drifts by one ulp.
            noise_pred = (inputs_shared["latents"] - x0) / self.scheduler.sigmas[progress_id].to(self.device)
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        if not torch.isfinite(inputs_shared["latents"]).all():
            raise RuntimeError("Non-finite pixels produced during denoising")
        return self.decode_image(inputs_shared["latents"], inputs_shared["height"], inputs_shared["width"],
                                 edit_image.size if edit_image is not None and resize_to_edit_image else None)

    def decode_image(self, latents, height, width, size=None):
        # Raster patches [4*h*w, 768] in [-1, 1] -> PIL, optionally resized (bicubic + antialias) to `size` = (W, H).
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
        # Image understanding and text chat. think_mode: "think" prefills "<think>\n", "empty" an empty think block,
        # "none" nothing. Returns the answer after </think>, or the full decoded text with return_raw=True.
        # Like training, there is no KV cache: each new token recomputes the whole sequence.
        images = [images] if isinstance(images, Image.Image) else list(images or [])
        prefill = THINK_PREFILLS[think_mode]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content if images else prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=False)
        ids = self.tokenizer.encode(text + prefill, add_special_tokens=False)
        self.load_models_to_device(self.in_iteration_models)
        features, grids = None, []
        if images:
            kwargs = {key: int(value) for key, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)) if value}
            inputs = self.processor(images=[image.convert("RGB") for image in images], return_tensors="pt", **kwargs)
            grid = inputs["image_grid_thw"].long()
            features = self.dit.visual(inputs["pixel_values"].to(self.device), grid.to(self.device))
            grids = [[t, h // 2, w // 2] for t, h, w in grid.tolist()]
            # Each <|image_pad|> becomes a time token (embedding t=0) followed by one pad per merged ViT token.
            counts = iter([t * h * w for t, h, w in grids])
            ids = [token for source in ids for token in ([TIME_TOKEN_ID] + [IMG_PAD_ID] * next(counts) if source == IMG_PAD_ID else [source])]
        positions = build_positions(ids, grids)
        ids = torch.tensor(ids)
        # Causal attention; image patch columns are visible from the first patch row of their block.
        ute = torch.arange(ids.numel())
        if images:
            for start, end in zip(ids.eq(BOI_ID).nonzero(as_tuple=True)[0].tolist(), ids.eq(EOI_ID).nonzero(as_tuple=True)[0].tolist()):
                ute[start + 2:end] = start + 2
        generator = torch.Generator(self.device).manual_seed(seed) if temperature > 0 else None
        eos_id = self.tokenizer.eos_token_id
        stop_ids = {IM_END_ID} | ({int(eos_id)} if eos_id is not None else set())
        new_tokens = []
        while len(new_tokens) < max_new_tokens and ids.numel() < max_length:
            mask = build_attention_mask(ute, torch.full_like(ute, ute.numel()), self.torch_dtype, self.device)
            logits = self.dit.text_logits(ids[None].to(self.device), positions[:, None].to(self.device), mask, features, len(grids))[0]
            token = sample_token(logits, temperature, top_k, top_p, generator)
            if token in stop_ids:
                break
            new_tokens.append(token)
            positions = torch.cat([positions, positions[:, -1:] + 1], dim=1)
            ute = torch.cat([ute, ute.new_tensor([ids.numel()])])
            ids = torch.cat([ids, ids.new_tensor([token])])
        raw = prefill + self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        if return_raw:
            return raw
        return (raw.split("</think>", 1)[1] if "</think>" in raw else raw).strip()


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


def build_edit_inputs(pipe, prompt, edit_reference, height, width):
    # Editing branch sequence: user turn [BOI, reference pads, EOI] + instruction, assistant turn [BOI, TIME, pads, EOI, EOS].
    # Without a reference (unconditional branch) the sequence is the assistant image block alone.
    h, w = height // 32, width // 32
    image = [BOI_ID, TIME_TOKEN_ID] + [IMG_PAD_ID] * (h * w) + [EOI_ID, IM_END_ID]
    if edit_reference is None:
        ids, grids = image, [[1, h, w]]
    else:
        user = [BOI_ID] + [IMG_PAD_ID] * (edit_reference["height"] * edit_reference["width"]) + [EOI_ID]
        user += pipe.tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
        ids = [IM_START_ID, USER_ID, NL_ID] + user + [IM_END_ID, NL_ID, IM_START_ID, ASSISTANT_ID, NL_ID] + image
        grids = [[1, edit_reference["height"], edit_reference["width"]], [1, h, w]]
    mask = build_attention_mask(*build_startend(ids), pipe.torch_dtype, pipe.device)
    position_ids = build_positions(ids, grids)[:, None].to(pipe.device)
    return {"input_ids": torch.tensor([ids], device=pipe.device), "position_ids": position_ids, "attention_mask": mask}


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


class HiUniUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width"), output_params=("height", "width"))

    def process(self, pipe: HiUniPipeline, height, width):
        height, width = pipe.check_resize_height_width(height or 1024, width or 1024)
        return {"height": height, "width": width}


class HiUniUnit_PromptTokenizer(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params=("height", "width", "seed", "edit_reference"),
            input_params_posi={"prompt": "prompt", "branch": "branch"},
            input_params_nega={"branch": "branch"},
            output_params=("input_ids", "position_ids", "attention_mask"),
        )

    def process(self, pipe: HiUniPipeline, branch, height, width, seed, edit_reference=None, prompt=None):
        if edit_reference is not None:
            # Editing: the negative branch keeps the reference and drops the instruction.
            return build_edit_inputs(pipe, prompt if branch == "posi" else None, edit_reference, height, width)
        # T2I: the positive branch is the chat prompt with a training instruction prefix, the negative branch the image block alone.
        h, w = height // 32, width // 32
        ids = [BOI_ID, TIME_TOKEN_ID] + [IMG_PAD_ID] * (h * w) + [EOI_ID, IM_END_ID]
        if branch == "posi":
            encode = lambda text: pipe.tokenizer.encode(text, add_special_tokens=False)
            instruction = random.Random(seed).choice(HIUNI_INSTRUCTIONS)
            prefix = [IM_START_ID] + encode("user") + encode("\n") + encode(instruction) + encode(prompt) + [IM_END_ID] + encode("\n")
            ids = prefix + [IM_START_ID] + encode("assistant") + encode("\n") + ids
        ids = torch.tensor(ids)
        position_ids, attention_mask = t2i_layout(ids, h, w, pipe.torch_dtype, pipe.device)
        return {"input_ids": ids[None].to(pipe.device), "position_ids": position_ids, "attention_mask": attention_mask}


class HiUniUnit_UnconditionalEditBranch(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("edit_reference", "image_cfg_scale", "height", "width"), output_params=("inputs_full",))

    def process(self, pipe: HiUniPipeline, edit_reference, image_cfg_scale, height, width):
        if edit_reference is None or image_cfg_scale == 1:
            return {}
        return {"inputs_full": build_edit_inputs(pipe, None, None, height, width)}


class HiUniUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "seed"), output_params=("latents",))

    def process(self, pipe: HiUniPipeline, height, width, seed):
        # Raster pixel patches [4*h*w, 768] in float32; the noise is scaled by sqrt(tokens / 256) as in training.
        tokens = 4 * (height // 32) * (width // 32)
        generator = None if seed is None else torch.Generator(pipe.device).manual_seed(seed)
        noise = torch.randn(tokens, 768, generator=generator, device=pipe.device, dtype=torch.float32)
        return {"latents": noise * math.sqrt(tokens / 256)}


def model_fn_hiuni(dit: HiUniModel, latents, timestep, input_ids, position_ids, attention_mask, height, width, edit_reference=None, **kwargs):
    # The head predicts clean pixels x0 in the raster patch layout of `latents`.
    ref = edit_reference or {}
    return dit(
        input_ids, position_ids, attention_mask, latents, timestep, height // 32, width // 32,
        ref_features=ref.get("features"), ref_raster=ref.get("raster"), ref_height=ref.get("height"), ref_width=ref.get("width"),
    )
