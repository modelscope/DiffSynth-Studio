import torch, math
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm

from ..core import ModelConfig
from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit
from ..models.ming_image_dit import MingImageDiT
from ..models.qwen_image_vae import QwenImageVAE
from ..models.ming_image_text_encoder import MingImageTextEncoder
from ..models.ming_image_connector import MingImageConnector
from ..models.ming_image_condition_proj import MingImageConditionProj


class MingImagePipeline(BasePipeline):
    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=16, width_division_factor=16)
        self.scheduler = FlowMatchScheduler("Ming-Image")
        self.text_encoder: MingImageTextEncoder = None
        self.dit: MingImageDiT = None
        self.vae: QwenImageVAE = None
        self.connector: MingImageConnector = None
        self.condition_proj: MingImageConditionProj = None
        self.in_iteration_models = ("dit",)
        self.units = [
            MingImageUnit_ShapeChecker(),
            MingImageUnit_EditImageEmbedder(),
            MingImageUnit_PromptEmbedder(),
            MingImageUnit_NoiseInitializer(),
            MingImageUnit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_ming_image
        self.compilable_models = ["dit"]

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        vram_limit: float = None,
        parallel_config=None,
    ):
        pipe = MingImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit, parallel_config=parallel_config)
        pipe.text_encoder = model_pool.fetch_model("ming_image_text_encoder")
        pipe.dit = model_pool.fetch_model("ming_image_dit")
        pipe.vae = model_pool.fetch_model("qwen_image_vae")
        pipe.connector = model_pool.fetch_model("ming_image_connector")
        pipe.condition_proj = model_pool.fetch_model("ming_image_condition_proj")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import PreTrainedTokenizerFast
            pipe.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_config.path)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        cfg_scale: float = 1.0,
        height: int = 1024,
        width: int = 1024,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 12,
        edit_image = None,
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Parameters
        inputs_posi = {"prompt": prompt, "positive": True}
        inputs_nega = {"positive": False}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "edit_image": edit_image, "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id,
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)

        # Decode
        self.load_models_to_device(["vae"])
        output = self.vae.decode_ming(inputs_shared["latents"].to(dtype=self.torch_dtype), tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        image = self.vae_output_to_image(output)
        self.load_models_to_device([])
        return image


class MingImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class MingImageUnit_EditImageEmbedder(PipelineUnit):
    _OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self):
        super().__init__(
            input_params=("edit_image", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("ref_latents", "pixel_values", "image_grid_thw"),
            onload_model_names=("vae",),
        )

    def _smart_resize(self, height, width, factor, min_pixels, max_pixels):
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    def _vit_preprocess(self, image):
        img = image.convert("RGB")
        w, h = img.size
        rh, rw = self._smart_resize(h, w, 28, 4 * 28 * 28, 1024 * 28 * 28)
        img = img.resize((rw, rh))
        h2, w2 = img.size[1], img.size[0]
        rh2, rw2 = self._smart_resize(h2, w2, 28, 451584, 451584)
        img = img.resize((rw2, rh2), Image.BICUBIC)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - np.array(self._OPENAI_MEAN, dtype=np.float32)) / np.array(self._OPENAI_STD, dtype=np.float32)
        patches = np.tile(arr.transpose(2, 0, 1)[None], (2, 1, 1, 1))
        grid_t, grid_h, grid_w = 1, rh2 // 14, rw2 // 14
        patches = patches.reshape(grid_t, 2, 3, grid_h // 2, 2, 14, grid_w // 2, 2, 14)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flat = patches.reshape(grid_t * grid_h * grid_w, 3 * 2 * 14 * 14)
        return torch.from_numpy(flat.copy()), torch.tensor([[grid_t, grid_h, grid_w]])

    def process(self, pipe, edit_image, height, width, tiled, tile_size, tile_stride):
        if edit_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        ref = edit_image.convert("RGBA")
        ref = torchvision.transforms.functional.resize(ref, [height, width], interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        ref_t = ((torchvision.transforms.functional.to_tensor(ref) - 0.5) * 2.0).unsqueeze(0).to(device=pipe.device, dtype=pipe.torch_dtype)
        ref_latents = pipe.vae.encode_ming(ref_t, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pixel_values, grid_thw = self._vit_preprocess(edit_image)
        return {"ref_latents": ref_latents, "pixel_values": pixel_values.to(pipe.device), "image_grid_thw": grid_thw.to(pipe.device)}


class MingImageUnit_PromptEmbedder(PipelineUnit):
    _SYSTEM_PROMPT = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"
    _USER_PREFIX = "<role>HUMAN</role>"
    _ASSISTANT_PREFIX = "<role>ASSISTANT</role>"

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"positive": "positive"},
            input_params=("pixel_values", "image_grid_thw"),
            output_params=("prompt_embeds", "prompt_embeds_2"),
            onload_model_names=("text_encoder", "connector", "condition_proj"),
        )
        self._neg_shapes = None  # set by the positive branch; the negative branch is only valid after a positive call (unit_runner runs nega after posi, only when cfg_scale != 1)

    def process(self, pipe, prompt=None, positive=True, pixel_values=None, image_grid_thw=None):
        if not positive:
            if self._neg_shapes is None:
                raise RuntimeError("MingImageUnit_PromptEmbedder: negative branch requires a prior positive call to provide embeds shapes.")
            shape, shape_2 = self._neg_shapes
            return {"prompt_embeds": torch.zeros(shape, device=pipe.device, dtype=pipe.torch_dtype),
                    "prompt_embeds_2": torch.zeros(shape_2, device=pipe.device, dtype=pipe.torch_dtype)}
        pipe.load_models_to_device(self.onload_model_names)
        embeds, embeds_2 = self.encode_prompt(pipe, prompt, pixel_values, image_grid_thw)
        self._neg_shapes = (embeds.shape, embeds_2.shape)
        return {"prompt_embeds": embeds, "prompt_embeds_2": embeds_2}

    @staticmethod
    def _find_runs(lst):
        result, i, n = [], 0, len(lst)
        while i < n:
            if lst[i] == 1:
                result.append(i)
                while i < n and lst[i] == 1:
                    i += 1
            else:
                i += 1
        return result

    @staticmethod
    def _merge_runs(lst, n):
        result, i = [], 0
        while i < len(lst):
            if lst[i] == 0:
                result.append(0)
                i += 1
            else:
                count = 0
                while i < len(lst) and lst[i] == 1:
                    count += 1
                    i += 1
                result.extend([1] * (count // n))
        return result

    def encode_prompt(self, pipe, prompt, pixel_values=None, image_grid_thw=None):
        tokenizer = pipe.tokenizer
        image_placeholder = "<IMAGE>" if pixel_values is not None else ""
        text = self._SYSTEM_PROMPT + tokenizer.eos_token + self._USER_PREFIX + image_placeholder + prompt + tokenizer.eos_token + self._ASSISTANT_PREFIX
        if pixel_values is not None:
            n_tok = int(torch.prod(image_grid_thw, dim=1)[0]) // 4
            text = text.replace("<IMAGE>", "<image>" + "<imagePatch>" * n_tok + "</image>" + "\n", 1)
        enc = tokenizer([text], return_tensors="pt", add_special_tokens=False)
        input_ids = enc.input_ids.to(pipe.device)
        attention_mask = enc.attention_mask.to(pipe.device)
        proj = pipe.condition_proj
        llm_cfg = pipe.text_encoder.model.config
        patch_id = llm_cfg.image_patch_token
        orig_len = input_ids.shape[1]
        extra_ids, gen_mask, labels = [], [], []
        for scale in proj.img_gen_scales:
            extra_ids += [patch_id + 1] + [patch_id] * (scale * scale) + [patch_id + 2]
            gen_mask += [0] + [1] * (scale * scale) + [0]
            labels += [1] * (scale * scale + 2)
        input_ids = torch.cat([input_ids, torch.tensor([extra_ids], device=pipe.device, dtype=input_ids.dtype)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(1, len(extra_ids), device=pipe.device, dtype=attention_mask.dtype)], dim=1)
        gen_mask = torch.tensor([[0] * orig_len + gen_mask], device=pipe.device)
        labels = torch.tensor([[-100] * (input_ids.shape[1] - len(extra_ids)) + labels], device=pipe.device)
        query_embeds = torch.cat([proj.query_tokens_dict[f"{s}x{s}"] for s in proj.img_gen_scales], dim=0)
        image_embeds = None
        if pixel_values is not None:
            image_embeds = pipe.text_encoder.extract_image_feature(pixel_values.to(dtype=pipe.torch_dtype), grid_thw=image_grid_thw)
            query_embeds = query_embeds.to(image_embeds.dtype).to(image_embeds.device)
        if image_embeds is not None:
            ids_list = input_ids.view(-1).tolist()
            gm_list = gen_mask.view(-1).tolist()
            is_patch = [1 if i == patch_id else 0 for i in ids_list]
            idx_start = self._find_runs(is_patch)
            isgen = self._merge_runs([gm_list[i] for i in idx_start], len(proj.img_gen_scales))
            new_thw_list, new_embeds_list, cum, cnt = [], [], 0, 0
            for g in isgen:
                if g:
                    for sc in proj.img_gen_scales:
                        new_thw_list.append([1, 2, sc * sc * 2])
                    new_embeds_list.append(query_embeds)
                else:
                    thw = image_grid_thw[cnt].tolist()
                    n_tok = (thw[1] // 2) * (thw[2] // 2)
                    new_embeds_list.append(image_embeds[cum:cum + n_tok, :])
                    new_thw_list.append(thw)
                    cnt += 1
                    cum += n_tok
            new_thw = torch.tensor(new_thw_list, device=pipe.device, dtype=input_ids.dtype)
            query_embeds = torch.cat(new_embeds_list, dim=0).to(pipe.device)
        else:
            new_thw = torch.tensor([[1, 2, s * s * 2] for s in proj.img_gen_scales], device=pipe.device, dtype=input_ids.dtype)
        words_embeds, image_mask, _ = pipe.text_encoder.model.model.prompt_wrap_navit(
            input_ids=input_ids.clip(0, pipe.text_encoder.model.get_input_embeddings().weight.shape[0] - 1),
            config=llm_cfg, query_embeds_image=query_embeds,
        )
        out = pipe.text_encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=words_embeds, image_grid_thw=new_thw, image_mask=image_mask)
        hidden_states = out.hidden_states
        use_input_mask = (labels < 0).int() * attention_mask
        max_valid = use_input_mask.cumsum(-1).argmax(-1).max().item() + 1
        use_input_mask = use_input_mask[:, :max_valid]
        directvlm = torch.cat([
            hidden_states[i][:, :max_valid, :] * use_input_mask.unsqueeze(-1)
            for i in proj.selected_hidden_states_layers
        ], dim=-1)
        embeds_2 = proj.proj_directvlm(directvlm.detach())
        hs = hidden_states[-1]
        gm = gen_mask.unsqueeze(-1).expand(hs.shape).bool()
        hs_gen = torch.masked_select(hs, gm).view(hs.shape[0], -1, hs.shape[-1])
        scale_embeds = proj.proj_in(hs_gen)
        scale_embeds = pipe.connector(
            inputs_embeds=scale_embeds,
            attention_mask=torch.ones(scale_embeds.shape[0], 1, scale_embeds.shape[1], scale_embeds.shape[1], device=scale_embeds.device),
        )
        embeds = proj.proj_out(scale_embeds)
        return embeds, embeds_2


class MingImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 16, height // 8, width // 8), seed=seed, rand_device=rand_device, rand_torch_dtype=torch.float32, torch_dtype=torch.float32)
        return {"noise": noise}


class MingImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(self, pipe, input_image, noise, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.scheduler.training:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(self.onload_model_names)
        image = input_image.convert("RGBA").resize((width, height), Image.BILINEAR)
        image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
        input_latents = pipe.vae.encode_ming(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return {"latents": noise, "input_latents": input_latents}


def model_fn_ming_image(
    dit,
    latents,
    timestep,
    prompt_embeds,
    prompt_embeds_2,
    ref_latents=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    t = (1000 - timestep) / 1000
    x = list(latents.to(dtype=prompt_embeds.dtype).unsqueeze(2).unbind(dim=0))
    cap_feats = list(prompt_embeds.unbind(dim=0))
    cap_feats_2 = list(prompt_embeds_2.unbind(dim=0))
    ref_x = list(ref_latents.unsqueeze(2).unbind(dim=0)) if ref_latents is not None else None
    output = dit(
        x, t, cap_feats, ref_x=ref_x, cap_feats_2=cap_feats_2, return_dict=False,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )[0]
    output = torch.stack([o[:, :1, :, :].float() for o in output], dim=0).squeeze(2)
    return -output
