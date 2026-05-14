# HiDream-O1-Image Pipeline for DiffSynth-Studio.

import torch, math
from typing import Optional, Union
from tqdm import tqdm
from PIL import Image

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.hidream_o1_image_dit import HiDreamO1ImageModel
from ..models.hidream_common import (
    add_special_tokens, get_rope_index_fix_point, patchify, unpatchify, PATCH_SIZE,
    resize_pilimage, calculate_dimensions,
)

class HiDreamO1ImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=PATCH_SIZE, width_division_factor=PATCH_SIZE,
        )
        self.scheduler = FlowMatchScheduler("HiDream-O1-Image")
        self.dit: HiDreamO1ImageModel = None
        self.processor = None

        self.in_iteration_models = ("dit",)
        self.units = [
            HiDreamO1ImageUnit_ShapeChecker(),
            HiDreamO1ImageUnit_RefImageEmbedder(),
            HiDreamO1ImageUnit_PromptTokenizer(),
            HiDreamO1ImageUnit_NoiseInitializer(),
            HiDreamO1ImageUnit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_hidream_o1_image

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        processor_config: ModelConfig = ModelConfig(model_id="HiDream-ai/HiDream-O1-Image-Dev", origin_file_pattern="./"),
        vram_limit: float = None,
    ):
        pipe = HiDreamO1ImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        pipe.dit = model_pool.fetch_model("hidream_o1_image_dit")
        if processor_config is not None:
            from transformers import AutoProcessor
            processor_config.download_if_necessary()
            pipe.processor = AutoProcessor.from_pretrained(processor_config.path)
            add_special_tokens(pipe.processor.tokenizer)
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = " ",
        cfg_scale: float = 4.0,
        height: int = 2048,
        width: int = 2048,
        seed: int = None,
        rand_device: str = "cpu",
        num_inference_steps: int = 50,
        model_type: str = "full",
        shift: float = 3.0,
        noise_scale: float = 8.0,
        edit_image: Union[Image.Image, list[Image.Image]] = None,
        keep_original_aspect: bool = True,
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, shift=shift, special_case=model_type)

        # Parameters
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "noise_scale": noise_scale,
            "edit_image": edit_image, "keep_original_aspect": keep_original_aspect,
        }

        # Units
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

        image = self.vae_output_to_image(inputs_shared["latents"])
        return image

class HiDreamO1ImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image",),
            output_params=("input_latents",),
        )

    def process(self, pipe: HiDreamO1ImagePipeline, input_image):
        if input_image is None or not pipe.scheduler.training:
            return {}
        img_tensor = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        return {"input_latents": img_tensor}

class HiDreamO1ImageUnit_NoiseInitializer(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "noise_scale"),
            output_params=("latents",),
        )

    def prepare_inputs(self, inputs_shared, inputs_posi, inputs_nega):
        return inputs_shared, inputs_posi, inputs_nega

    def process(self, pipe: HiDreamO1ImagePipeline, height=None, width=None, seed=None, rand_device=None, noise_scale=None):
        noise = pipe.generate_noise((1, 3, height, width), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        noise = noise_scale * noise
        return {"latents": noise}


class HiDreamO1ImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: HiDreamO1ImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class HiDreamO1ImageUnit_RefImageEmbedder(PipelineUnit):

    def __init__(self):
        super().__init__(
            input_params=("edit_image", "height", "width", "keep_original_aspect"),
            output_params=("ref_for_prompt_tokenizer", "height", "width", "ref_patches"),
        )

    def get_sizes(self, edit_image, height, width):
        K = len(edit_image)
        if K == 1: max_size = max(height, width)
        elif K == 2: max_size = max(height, width) * 48 // 64
        elif K <= 4: max_size = max(height, width) // 2
        elif K <= 8: max_size = max(height, width) * 24 // 64
        else: max_size = max(height, width) // 4

        CONDITION_IMAGE_SIZE = 384
        if K <= 4: cond_img_size = CONDITION_IMAGE_SIZE
        elif K <= 8: cond_img_size = CONDITION_IMAGE_SIZE * 48 // 64
        else: cond_img_size = CONDITION_IMAGE_SIZE // 2
        return max_size, cond_img_size


    def process(self, pipe: HiDreamO1ImagePipeline, edit_image, height, width, keep_original_aspect):
        if edit_image is None:
            return {}
        if isinstance(edit_image, Image.Image):
            edit_image = [edit_image]
        if keep_original_aspect and len(edit_image) == 1:
            edit_image = [resize_pilimage(pil, 2048) for pil in edit_image]
            width, height = edit_image[0].size

        max_size, cond_img_size = self.get_sizes(edit_image, height, width)
        ref_image_tensors, ref_pils_vlm = [], []
        image_grid_thw_tgt = torch.tensor([1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64).unsqueeze(0)
        image_grid_thw_ref = torch.zeros((len(edit_image), 3), dtype=torch.int64)
        for i, pil in enumerate(edit_image):
            pil_r = pil if keep_original_aspect and len(edit_image) == 1 else resize_pilimage(pil, max_size)
            cond_w, cond_h = calculate_dimensions(cond_img_size, pil_r.width / pil_r.height)
            ref_pils_vlm.append(pil_r.resize((cond_w, cond_h), resample=Image.LANCZOS))
            image_grid_thw_ref[i] = torch.tensor([1, pil_r.height // PATCH_SIZE, pil_r.width // PATCH_SIZE], dtype=torch.int64)
            x = pipe.preprocess_image(pil_r)
            x = patchify(x)
            ref_image_tensors.append(x)

        ref_image_lens = [img.shape[1] for img in ref_image_tensors]
        ref_patches = torch.cat(ref_image_tensors, dim=1).to(pipe.device, pipe.torch_dtype)

        return {
            "ref_for_prompt_tokenizer": {
                "ref_image_lens": ref_image_lens,
                "ref_pils_vlm": ref_pils_vlm,
                "image_grid_thw_tgt": image_grid_thw_tgt,
                "image_grid_thw_ref": image_grid_thw_ref,
                "tgt_image_len": (height // PATCH_SIZE) * (width // PATCH_SIZE),
            },
            "height": height,
            "width": width,
            "ref_patches": ref_patches,
        }


class HiDreamO1ImageUnit_PromptTokenizer(PipelineUnit):

    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params=("height", "width", "ref_for_prompt_tokenizer"),
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("input_ids", "position_ids", "token_types", "vinput_mask",
                           "pixel_values", "image_grid_thw"),
        )

    def process(self, pipe: HiDreamO1ImagePipeline, prompt, height, width, ref_for_prompt_tokenizer=None):
        # T2I path
        if ref_for_prompt_tokenizer is None:
            return self.build_text_sample(
                prompt=prompt,
                height=height, width=width,
                tokenizer=pipe.processor.tokenizer, processor=pipe.processor,
                model_config=pipe.dit.config, device=pipe.device,
            )

        # I2I path
        return self.build_i2i_sample(
            prompt=prompt,
            height=height, width=width,
            ref_for_prompt_tokenizer=ref_for_prompt_tokenizer,
            tokenizer=pipe.processor.tokenizer, processor=pipe.processor,
            model_config=pipe.dit.config, device=pipe.device,
            torch_dtype=pipe.torch_dtype,
        )

    def build_text_sample(self, prompt, height, width, tokenizer, processor, model_config, device):
        TIMESTEP_TOKEN_NUM = 1
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

        boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
        tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

        messages = [{"role": "user", "content": prompt}]
        template_caption = (
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            + boi_token
            + tms_token * TIMESTEP_TOKEN_NUM
        )

        input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

        image_grid_thw = torch.tensor(
            [1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64
        ).unsqueeze(0)

        vision_tokens = torch.zeros((1, image_len), dtype=input_ids.dtype) + image_token_id
        vision_tokens[0, 0] = vision_start_token_id
        input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

        position_ids, _ = get_rope_index_fix_point(
            spatial_merge_size=1,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            input_ids=input_ids_pad,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=None,
            skip_vision_start_token=[1],
        )

        txt_seq_len = input_ids.shape[-1]
        all_seq_len = position_ids.shape[-1]

        token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
        bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
        token_types[0, bgn: bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
        token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3

        vinput_mask = (token_types == 1)
        token_types_bin = (token_types > 0).to(token_types.dtype)

        return {
            'input_ids': input_ids.to(device),
            'position_ids': position_ids.to(device),
            'token_types': token_types_bin.to(device),
            'vinput_mask': vinput_mask.to(device),
        }

    def build_i2i_sample(self, prompt, height, width, ref_for_prompt_tokenizer,
                         tokenizer, processor, model_config, device, torch_dtype):
        TIMESTEP_TOKEN_NUM = 1
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        spatial_merge_size = model_config.vision_config.spatial_merge_size

        ref_pils_vlm = ref_for_prompt_tokenizer["ref_pils_vlm"]
        image_grid_thw_tgt = ref_for_prompt_tokenizer["image_grid_thw_tgt"]
        ref_image_lens = ref_for_prompt_tokenizer["ref_image_lens"]
        image_grid_thw_ref = ref_for_prompt_tokenizer["image_grid_thw_ref"]
        tgt_image_len = ref_for_prompt_tokenizer["tgt_image_len"]
        K = len(ref_pils_vlm)

        # processor call
        boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
        tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

        content = [{"type": "image"} for _ in range(K)]
        content.append({"type": "text", "text": prompt})
        template_caption = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        proc = processor(text=[template_caption], images=ref_pils_vlm, padding="longest", return_tensors="pt")
        input_ids_2 = tokenizer.encode(boi_token + tms_token * TIMESTEP_TOKEN_NUM, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([proc.input_ids, input_ids_2], dim=-1)

        # image_grid_thw combine
        igthw_cond = proc.image_grid_thw.clone()
        for i in range(K):
            igthw_cond[i, 1] //= spatial_merge_size
            igthw_cond[i, 2] //= spatial_merge_size
        igthw_all = torch.cat([igthw_cond, image_grid_thw_tgt, image_grid_thw_ref], dim=0)

        # vision tokens
        vision_tokens_list = []
        vt_tgt = torch.full((1, tgt_image_len), image_token_id, dtype=input_ids.dtype)
        vt_tgt[0, 0] = vision_start_token_id
        vision_tokens_list.append(vt_tgt)
        for rl in ref_image_lens:
            vt_ref = torch.full((1, rl), image_token_id, dtype=input_ids.dtype)
            vt_ref[0, 0] = vision_start_token_id
            vision_tokens_list.append(vt_ref)
        vision_tokens = torch.cat(vision_tokens_list, dim=1)
        input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

        # position_ids
        position_ids, _ = get_rope_index_fix_point(
            1, image_token_id, video_token_id, vision_start_token_id,
            input_ids=input_ids_pad, image_grid_thw=igthw_all,
            video_grid_thw=None, attention_mask=None,
            skip_vision_start_token=[0] * K + [1] + [1] * K,
        )
        txt_seq_len = input_ids.shape[-1]
        all_seq_len = position_ids.shape[-1]

        # token_types / vinput_mask
        token_types_raw = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
        bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
        end = bgn + tgt_image_len + TIMESTEP_TOKEN_NUM
        token_types_raw[0, bgn:end] = 1  # target
        token_types_raw[0, end: end + sum(ref_image_lens)] = 2  # ref
        token_types_raw[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3  # TMS

        vinput_mask = torch.logical_or(token_types_raw == 1, token_types_raw == 2)
        token_types_bin = (token_types_raw > 0).to(token_types_raw.dtype)

        return {
            'input_ids': input_ids.to(device),
            'position_ids': position_ids.to(device),
            'token_types': token_types_bin.to(device),
            'vinput_mask': vinput_mask.to(device),
            'pixel_values': proc.pixel_values.to(device, torch_dtype),
            'image_grid_thw': proc.image_grid_thw.to(device),
        }


def model_fn_hidream_o1_image(
    dit,
    latents,
    timestep,
    input_ids,
    position_ids,
    token_types,
    vinput_mask,
    pixel_values=None,
    image_grid_thw=None,
    ref_patches=None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):

    b, c, h, w = latents.shape
    x = patchify(latents)
    img_seq_len = x.shape[1]
    if ref_patches is not None:
        x = torch.cat([x, ref_patches], dim=1)
    timestep = timestep / 1000.

    outputs = dit(
        input_ids=input_ids,
        position_ids=position_ids,
        vinputs=x,
        timestep=(1 - timestep).reshape(-1),
        token_types=token_types,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )

    x_pred = outputs.x_pred[0, vinput_mask[0]][:img_seq_len].unsqueeze(0)
    x_pred = unpatchify(x_pred, h, w)
    v_pred = (latents - x_pred) / timestep
    return v_pred
