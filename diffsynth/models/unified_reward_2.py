import re
from typing import Union
import torch
from PIL import Image

ImageInput = Union[Image.Image, list[Image.Image], tuple[Image.Image, ...]]


class UnifiedReward2Qwen35ForConditionalGeneration(torch.nn.Module):
    def __init__(self, variant: str = "qwen35_9b"):
        super().__init__()
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        if variant != "qwen35_9b":
            raise ValueError(f"Unsupported UnifiedReward-2 variant: {variant}")
        hidden_size = 4096
        intermediate_size = 12288
        num_hidden_layers = 32
        num_attention_heads = 16
        linear_num_value_heads = 32

        config = Qwen3_5Config(
            bos_token_id=None,
            eos_token_id=248046,
            hidden_size=hidden_size,
            image_token_id=248056,
            model_type="qwen3_5",
            pad_token_id=248044,
            text_config={
                "attention_bias": False,
                "attention_dropout": 0.0,
                "attn_output_gate": True,
                "bos_token_id": None,
                "eos_token_id": 248044,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_act": "silu",
                "hidden_size": hidden_size,
                "initializer_range": 0.02,
                "intermediate_size": intermediate_size,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": linear_num_value_heads,
                "linear_value_head_dim": 128,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262144,
                "mlp_only_layers": [],
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": False,
                "num_attention_heads": num_attention_heads,
                "num_hidden_layers": num_hidden_layers,
                "num_key_value_heads": 4,
                "pad_token_id": None,
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "mrope_interleaved": True,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default",
                },
                "tie_word_embeddings": False,
                "use_cache": True,
                "vocab_size": 248320,
            },
            tie_word_embeddings=False,
            use_cache=True,
            video_token_id=248057,
            vision_config={
                "deepstack_visual_indexes": [],
                "depth": 27,
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "in_channels": 3,
                "initializer_range": 0.02,
                "intermediate_size": 4304,
                "model_type": "qwen3_5",
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": hidden_size,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            vision_end_token_id=248054,
            vision_start_token_id=248053,
        )
        config.text_config.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * (
            num_hidden_layers // 4
        )
        self.model = Qwen3_5ForConditionalGeneration(config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _coerce_float(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+", value)
        if match:
            return float(match.group())
    return None


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


class UnifiedReward2Model(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, processor, max_new_tokens: int = 1024):
        super().__init__()
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    @property
    def device(self):
        return next(self.parameters(), torch.tensor([])).device

    @property
    def dtype(self):
        return next(self.parameters(), torch.tensor(0.0)).dtype

    def _build_prompt(self, prompt: str | None):
        prompt = prompt or ""
        return (
            "You are presented with a generated image and its associated text caption. "
            "Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n"
            "Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
            "- Alignment Score: How well the image matches the caption in terms of content.\n"
            "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
            "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
            "Output your evaluation using the format below:\n\n"
            "Alignment Score (1-5): X\n"
            "Coherence Score (1-5): Y\n"
            "Style Score (1-5): Z\n\n"
            "Do not include explanations, analysis, bullet points, or any text outside the requested output format.\n\n"
            "Your task is provided as follows:\n"
            f"Text Caption: [{prompt}]"
        )

    def _build_messages(self, prompt: str | None, image: Image.Image):
        content = [
            {"type": "image", "image": image.convert("RGB")},
            {"type": "text", "text": self._build_prompt(prompt)},
        ]
        return [{"role": "user", "content": content}]

    def _processor_inputs(self, text, images):
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        inputs = inputs.to(self.device)
        if self.dtype != torch.float32:
            inputs = {
                name: value.to(dtype=self.dtype) if torch.is_tensor(value) and torch.is_floating_point(value) else value
                for name, value in inputs.items()
            }
        return inputs, input_ids

    def _decode_sample(self, prompt: str | None, image: Image.Image):
        image = image.convert("RGB")
        messages = self._build_messages(prompt, image)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs, input_ids = self._processor_inputs(text, [image])
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=getattr(self.processor.tokenizer, "pad_token_id", None),
                eos_token_id=getattr(self.processor.tokenizer, "eos_token_id", None),
            )
        generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
        return self.processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    @staticmethod
    def _extract_labeled_number(text: str, label: str):
        text = text.replace("*", "")
        pattern = rf"{re.escape(label)}\s*(?:\([^)]+\))?\s*[::]\s*([-+]?\d*\.?\d+)"
        match = re.search(pattern, text, flags=re.I)
        return _coerce_float(match.group(1)) if match else None

    def _parse_output(self, text: str):
        alignment = self._extract_labeled_number(text, "Alignment Score")
        coherence = self._extract_labeled_number(text, "Coherence Score")
        style = self._extract_labeled_number(text, "Style Score")
        raw_score = [score for score in (alignment, coherence, style) if score is not None]
        return {
            "alignment": alignment,
            "coherence": coherence,
            "style": style,
            "score": _mean(raw_score),
        }

    @staticmethod
    def _primary_score(parsed: dict):
        return parsed.get("score")

    @torch.no_grad()
    def forward(self, prompt: str | list[str] | None, images: ImageInput):
        prompts = _as_list(prompt if prompt is not None else "")
        images = _as_list(images)
        if len(prompts) == 1 and len(images) > 1:
            prompts = prompts * len(images)
        if len(images) == 1 and len(prompts) > 1:
            images = images * len(prompts)
        if len(prompts) != len(images):
            raise ValueError(f"Expected the same number of prompts and images, got {len(prompts)} and {len(images)}.")
        outputs = []
        for single_prompt, single_image in zip(prompts, images):
            raw_text = self._decode_sample(single_prompt, single_image)
            outputs.append(self._parse_output(raw_text))
        return outputs
