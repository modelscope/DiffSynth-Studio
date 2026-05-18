import math
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor

ImageInput = Union[Image.Image, list[Image.Image], tuple[Image.Image, ...]]

HPSV3_INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best. 

**Visual Quality:**  
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible. 

**Text Alignment:**  
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}


"""

HPSV3_PROMPT_WITH_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

HPSV3_PROMPT_WITHOUT_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: 
"""

def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

def _round_by_factor(number, factor):
    return round(number / factor) * factor

def _ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor

def _floor_by_factor(number, factor):
    return math.floor(number / factor) * factor

def _smart_resize(height, width, factor=28, min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Image aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}.")
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def _find_checkpoint(path):
    path = Path(path)
    if path.is_file():
        return path
    for name in ("HPSv3.safetensors", "*.safetensors", "*.bin", "*.pt", "*.pth"):
        candidate = path / name
        if candidate.exists():
            return candidate
        matches = sorted(path.rglob(name))
        if matches:
            return matches[0]
    return None

class HPSv3RewardModelMixin:
    def init_reward_head(
        self,
        output_dim=2,
        reward_token="special",
        special_token_ids=None,
        rm_head_type="ranknet",
        rm_head_kwargs=None,
    ):
        self.output_dim = output_dim
        self.reward_token = "special" if special_token_ids is not None else reward_token
        self.special_token_ids = special_token_ids
        hidden_size = getattr(self.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.config, "text_config"):
            hidden_size = self.config.text_config.hidden_size
        if rm_head_type == "ranknet":
            rm_head_kwargs = {} if rm_head_kwargs is None else rm_head_kwargs
            hidden = rm_head_kwargs.get("hidden_size", 1024)
            dropout = rm_head_kwargs.get("dropout", 0.05)
            self.rm_head = nn.Sequential(
                nn.Linear(hidden_size, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 16),
                nn.ReLU(),
                nn.Linear(16, output_dim),
            )
        else:
            self.rm_head = nn.Linear(hidden_size, output_dim, bias=False)
        self.rm_head.to(torch.float32)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            mm_token_type_ids=mm_token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        logits = self.rm_head(hidden_states.to(next(self.rm_head.parameters()).dtype))

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1

        if self.reward_token == "last":
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack([logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled_logits = logits[special_token_mask, ...].view(batch_size, -1)
        else:
            raise ValueError(f"Invalid HPSv3 reward token mode: {self.reward_token}")
        return {"logits": pooled_logits}

def _create_reward_model_class():
    from transformers import Qwen2VLForConditionalGeneration

    class HPSv3Qwen2VLRewardModel(HPSv3RewardModelMixin, Qwen2VLForConditionalGeneration):
        def __init__(
            self,
            config,
            output_dim=2,
            reward_token="special",
            special_token_ids=None,
            rm_head_type="ranknet",
            rm_head_kwargs=None,
        ):
            super().__init__(config)
            self.init_reward_head(
                output_dim=output_dim,
                reward_token=reward_token,
                special_token_ids=special_token_ids,
                rm_head_type=rm_head_type,
                rm_head_kwargs=rm_head_kwargs,
            )

    return HPSv3Qwen2VLRewardModel

class HPSv3Model(torch.nn.Module):
    def __init__(
        self,
        model,
        processor,
        use_special_tokens=True,
        max_pixels=256 * 28 * 28,
        min_pixels=256 * 28 * 28,
        score_index=0,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.use_special_tokens = use_special_tokens
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.score_index = score_index

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        base_model_path: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cpu",
        output_dim: int = 2,
        score_index: int = 0,
        use_special_tokens: bool = True,
        reward_token: str = "special",
        rm_head_type: str = "ranknet",
        rm_head_kwargs: dict = None,
        max_pixels: int = 256 * 28 * 28,
        min_pixels: int = 256 * 28 * 28,
        model_kwargs: dict = None,
        processor_kwargs: dict = None,
    ):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        processor_kwargs = {} if processor_kwargs is None else processor_kwargs
        model_path = Path(model_path)
        base_model_path = base_model_path or str(model_path)
        checkpoint_path = _find_checkpoint(model_path)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Cannot find an HPSv3 checkpoint under {model_path}.")

        processor = AutoProcessor.from_pretrained(base_model_path, padding_side="right", **processor_kwargs)
        special_token_ids = None
        if use_special_tokens:
            special_tokens = ["<|Reward|>"]
            processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

        reward_model_class = _create_reward_model_class()
        model = reward_model_class.from_pretrained(
            base_model_path,
            output_dim=output_dim,
            reward_token=reward_token,
            special_token_ids=special_token_ids,
            torch_dtype=torch_dtype,
            attn_implementation=model_kwargs.pop("attn_implementation", "sdpa"),
            **model_kwargs,
        )
        if use_special_tokens:
            model.resize_token_embeddings(len(processor.tokenizer))
        state_dict = cls._load_checkpoint(checkpoint_path)
        state_dict = cls._prepare_state_dict(state_dict, model.state_dict())
        model.load_state_dict(state_dict, strict=True)
        model.config.tokenizer_padding_side = processor.tokenizer.padding_side
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.rm_head.to(torch.float32)
        model = model.to(device).eval()
        return cls(
            model=model,
            processor=processor,
            use_special_tokens=use_special_tokens,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            score_index=score_index,
        )

    @staticmethod
    def _load_checkpoint(checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.suffix == ".safetensors":
            import safetensors.torch

            state_dict = safetensors.torch.load_file(str(checkpoint_path), device="cpu")
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
        return {key[len("module.") :] if key.startswith("module.") else key: value for key, value in state_dict.items()}

    @staticmethod
    def _prepare_state_dict(state_dict, target_state_dict):
        target_keys = set(target_state_dict.keys())
        converted = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith("visual.") and f"model.{key}" in target_keys:
                new_key = f"model.{key}"
            elif key.startswith("model.visual.") and key[len("model.") :] in target_keys:
                new_key = key[len("model.") :]
            elif key.startswith("model.") and not key.startswith("model.language_model."):
                suffix = key[len("model.") :]
                if f"model.language_model.{suffix}" in target_keys:
                    new_key = f"model.language_model.{suffix}"
            elif key.startswith("model.language_model."):
                suffix = key[len("model.language_model.") :]
                if f"model.{suffix}" in target_keys:
                    new_key = f"model.{suffix}"
            elif key.startswith("lm_head.") and f"model.{key}" in target_keys:
                new_key = f"model.{key}"
            elif key.startswith("model.lm_head.") and key[len("model.") :] in target_keys:
                new_key = key[len("model.") :]
            converted[new_key] = value
        return converted

    @property
    def device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _normalize_inputs(self, prompts, images):
        images = _as_list(images)
        prompts = _as_list(prompts)
        if len(prompts) == 1 and len(images) > 1:
            prompts = prompts * len(images)
        if len(images) == 1 and len(prompts) > 1:
            images = images * len(prompts)
        if len(prompts) != len(images):
            raise ValueError(f"Expected the same number of prompts and images, got {len(prompts)} and {len(images)}.")
        return prompts, images

    def _prepare_images(self, images):
        prepared = []
        for image in images:
            image = image.convert("RGB")
            height, width = image.height, image.width
            resized_height, resized_width = _smart_resize(
                height,
                width,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            prepared.append(image.resize((resized_width, resized_height), Image.BICUBIC))
        return prepared

    def _messages(self, prompts, images):
        suffix = HPSV3_PROMPT_WITH_SPECIAL_TOKEN if self.use_special_tokens else HPSV3_PROMPT_WITHOUT_SPECIAL_TOKEN
        messages = []
        for prompt, image in zip(prompts, images):
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": HPSV3_INSTRUCTION.format(text_prompt=prompt) + suffix},
                        ],
                    }
                ]
            )
        return messages

    def _prepare_batch(self, prompts, images):
        images = self._prepare_images(images)
        messages = self._messages(prompts, images)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=text, images=images, padding=True, return_tensors="pt")
        return batch.to(self.device)

    @torch.inference_mode()
    def forward(self, prompts: Union[str, list[str]], images):
        prompts, images = self._normalize_inputs(prompts, images)
        batch = self._prepare_batch(prompts, images)
        rewards = self.model(return_dict=True, **batch)["logits"]
        if rewards.ndim == 2:
            return rewards[:, self.score_index]
        return rewards
