import math
from typing import Optional, Union
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration

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
            rm_head_kwargs = rm_head_kwargs or {}
            hidden = rm_head_kwargs.get("hidden_size", 1024)
            dropout = rm_head_kwargs.get("dropout", 0.05)
            self.rm_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, output_dim),
            )
        else:
            self.rm_head = torch.nn.Linear(hidden_size, output_dim, bias=False)
            
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
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs.pop("logits_to_keep", None)
        mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)
        
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
            return_dict=return_dict,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        logits = self.rm_head(hidden_states.to(next(self.rm_head.parameters()).dtype))

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None and hasattr(self.config, "text_config"):
            pad_token_id = getattr(self.config.text_config, "pad_token_id", None)
            
        if pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
            
        if pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
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
            if self.special_token_ids is None:
                raise ValueError("HPSv3 reward_token='special' requires special_token_ids.")
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled_logits = logits[special_token_mask, ...].view(batch_size, -1)
        else:
            raise ValueError(f"Invalid HPSv3 reward token mode: {self.reward_token}")
            
        return {"logits": pooled_logits}


class HPSv3Qwen2VLRewardModel(HPSv3RewardModelMixin, Qwen2VLForConditionalGeneration):
    def __init__(
        self,
        config=None,
        vocab_size=None,
        output_dim=2,
        reward_token="special",
        special_token_ids=None,
        rm_head_type="ranknet",
        rm_head_kwargs=None,
    ):
        if config is None:
            config = self.default_config(vocab_size or 151658)
        elif vocab_size is not None and hasattr(config, "text_config"):
            config.text_config.vocab_size = vocab_size
            
        super().__init__(config)
        self.init_reward_head(
            output_dim=output_dim,
            reward_token=reward_token,
            special_token_ids=special_token_ids,
            rm_head_type=rm_head_type,
            rm_head_kwargs=rm_head_kwargs,
        )

    @staticmethod
    def default_config(vocab_size=151658):
        from transformers import Qwen2VLConfig

        return Qwen2VLConfig(
            text_config={
                "vocab_size": vocab_size,
                "hidden_size": 3584,
                "intermediate_size": 18944,
                "num_hidden_layers": 28,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "hidden_act": "silu",
                "max_position_embeddings": 32768,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-6,
                "use_cache": True,
                "use_sliding_window": False,
                "sliding_window": 32768,
                "max_window_layers": 28,
                "attention_dropout": 0.0,
                "rope_parameters": {
                    "rope_type": "default",
                    "type": "mrope",
                    "mrope_section": [16, 24, 24],
                    "rope_theta": 1000000.0,
                },
                "bos_token_id": 151643,
                "eos_token_id": 151645,
            },
            vision_config={
                "depth": 32,
                "embed_dim": 1280,
                "hidden_size": 3584,
                "mlp_ratio": 4,
                "num_heads": 16,
                "in_channels": 3,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            image_token_id=151655,
            video_token_id=151656,
            vision_start_token_id=151652,
            vision_end_token_id=151653,
            tie_word_embeddings=False,
        )


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

    @property
    def device(self):
        return next(self.parameters(), torch.tensor([])).device

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

    @torch.no_grad()
    def forward(self, prompts: Union[str, list[str]], images):
        prompts, images = self._normalize_inputs(prompts, images)
        batch = self._prepare_batch(prompts, images)
        
        rewards = self.model(return_dict=True, **batch)["logits"]
        if rewards.ndim == 2:
            return rewards[:, self.score_index]
            
        return rewards