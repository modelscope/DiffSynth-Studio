import json
import re
from typing import Union

import torch
from PIL import Image

ImageInput = Union[Image.Image, list[Image.Image], tuple[Image.Image, ...]]


class UnifiedRewardQwen3VLForConditionalGeneration(torch.nn.Module):
    def __init__(self, variant: str = "qwen3vl"):
        super().__init__()
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        config = Qwen3VLConfig(
            text_config={
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 4096,
                "initializer_range": 0.02,
                "intermediate_size": 12288,
                "max_position_embeddings": 262144,
                "model_type": "qwen3_vl_text",
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-6,
                "rope_scaling": {
                    "mrope_interleaved": True,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default",
                },
                "rope_theta": 5000000,
                "use_cache": True,
                "vocab_size": 151936,
            },
            vision_config={
                "deepstack_visual_indexes": [8, 16, 24],
                "depth": 27,
                "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152,
                "in_channels": 3,
                "initializer_range": 0.02,
                "intermediate_size": 4304,
                "model_type": "qwen3_vl",
                "num_heads": 16,
                "num_position_embeddings": 2304,
                "out_hidden_size": 4096,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            image_token_id=151655,
            video_token_id=151656,
            vision_start_token_id=151652,
            vision_end_token_id=151653,
            tie_word_embeddings=False,
        )
        self.model = Qwen3VLForConditionalGeneration(config)

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


class UnifiedRewardEditModel(torch.nn.Module):
    SUPPORTED_TASKS = {"edit_pointwise_score", "edit_pairwise_rank", "edit_pairwise_score"}
    DEFAULT_TASK = "edit_pointwise_score"

    def __init__(self, model: torch.nn.Module, processor, task: str = DEFAULT_TASK, max_new_tokens: int = 256):
        super().__init__()
        self._validate_task(task)
        self.model = model
        self.processor = processor
        self.task = task
        self.max_new_tokens = max_new_tokens

    @classmethod
    def _validate_task(cls, task: str):
        if task not in cls.SUPPORTED_TASKS:
            supported = ", ".join(sorted(cls.SUPPORTED_TASKS))
            raise ValueError(f"Unsupported UnifiedReward task: {task}. Supported tasks: {supported}.")

    @property
    def device(self):
        return next(self.parameters(), torch.tensor([])).device

    @property
    def dtype(self):
        return next(self.parameters(), torch.tensor(0.0)).dtype

    def _build_edit_pointwise_score_prompt(self, instruction: str):
        return (
            "You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.\n"
            "All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.\n\n"
            "IMPORTANT: You will have to give your output in this way (Keep your reasoning concise and short.):\n"
            "{\n\n\"reasoning\" : \"...\",\n\"score\" : [...],\n}\n\n"
            "RULES:\n\n"
            "Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.\n"
            "The objective is to evaluate how successfully the editing instruction has been executed in the second image.\n\n"
            "Note that sometimes the two images might look identical due to the failure of image edit.\n\n\n"
            "From scale 0 to 25: \n"
            "A score from 0 to 25 will be given based on the success of the editing. "
            "(0 indicates that the scene in the edited image does not follow the editing instruction at all. "
            "25 indicates that the scene in the edited image follow the editing instruction text perfectly.)\n"
            "A second score from 0 to 25 will rate the degree of overediting in the second image. "
            "(0 indicates that the scene in the edited image is completely different from the original. "
            "25 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)\n"
            "Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.\n\n"
            f"Editing instruction:{instruction}\n"
        )

    def _build_edit_pairwise_rank_prompt(self, instruction: str):
        return (
            "You are tasked with comparing two edited images and determining which one is better based on the given criteria.\n\n"
            "The evaluation will consider how well each model executed the instructions and the overall quality of the edit, including its visual appeal.\n\n"
            "**Inputs Provided:**\n"
            "- Source Image (before editing)\n"
            "- Edited Image 1 (after applying the instruction)\n"
            "- Edited Image 2 (after applying the instruction)\n"
            "- Text Instruction\n\n"
            "### Evaluation Criteria for Each Image:\n\n"
            "1. **Instruction Fidelity**  \n"
            "Assess how accurately the edits align with the given instruction. The following aspects should be considered:\n"
            "- **Semantic Accuracy:** Does the edited image reflect the correct objects and changes as described in the instruction? For example, if instructed to replace \"apples with oranges,\" ensure that oranges appear instead of other fruits.\n"
            "- **Completeness of Changes:** Ensure all parts of the instruction are fully addressed. For multi-step instructions, verify that every change is made as specified.\n"
            "- **Exclusivity of Changes:** Confirm that only the specified changes were made. Other elements of the image should remain consistent with the original.\n\n"
            "2. **Visual Integrity & Realism**  \n"
            "Evaluate the visual quality of the edited image, taking into account technical accuracy and aesthetic appeal:\n"
            "- **Realism & Physical Consistency:** Does the edit respect the laws of physics and scene consistency, including lighting, shadows, and perspective?\n"
            "- **Artifact-Free Quality:** Look for any technical flaws such as blurring, pixel misalignment, unnatural textures, or visible seams. The image should be clean and free from distractions.\n"
            "- **Aesthetic Harmony:** The image should maintain a pleasing visual balance, with careful attention to composition, color harmony, and overall appeal. The changes should enhance the image rather than detract from it.\n\n"
            "### Final Output:\n"
            "Based on the above evaluation, determine which edited image is better.\n\n"
            f"Text instruction - {instruction}\n"
        )

    def _build_edit_pairwise_score_prompt(self, instruction: str):
        return (
            "You are tasked with assigning scores to two edited images, comparing each with the original source image. \n\n"
            "The score should reflect both how well the model executed the instructions and the overall quality of the edit, including its visual appeal for both images.\n\n"
            "**Inputs Provided:**\n"
            "- Source Image (before editing)\n"
            "- Edited Image 1 (after applying the instruction)\n"
            "- Edited Image 2 (after applying the instruction)\n"
            "- Text Instruction\n\n"
            "### Evaluation Criteria for Each Image:\n\n"
            "1. **Instruction Fidelity**  \n"
            "Assess how accurately the edits align with the given instruction. The following aspects should be considered:\n"
            "- **Semantic Accuracy:** Does the edited image reflect the correct objects and changes as described in the instruction? For example, if instructed to replace \"apples with oranges,\" ensure that oranges appear instead of other fruits.\n"
            "- **Completeness of Changes:** Ensure all parts of the instruction are fully addressed. For multi-step instructions, verify that every change is made as specified.\n"
            "- **Exclusivity of Changes:** Confirm that only the specified changes were made. Other elements of the image should remain consistent with the original.\n\n"
            "2. **Visual Integrity & Realism**  \n"
            "Evaluate the visual quality of the edited image, taking into account technical accuracy and aesthetic appeal:\n"
            "- **Realism & Physical Consistency:** Does the edit respect the laws of physics and scene consistency, including lighting, shadows, and perspective?\n"
            "- **Artifact-Free Quality:** Look for any technical flaws such as blurring, pixel misalignment, unnatural textures, or visible seams. The image should be clean and free from distractions.\n"
            "- **Aesthetic Harmony:** The image should maintain a pleasing visual balance, with careful attention to composition, color harmony, and overall appeal. The changes should enhance the image rather than detract from it.\n\n"
            "### Scoring Guidelines:\n"
            "- The score can range from **positive to negative** based on how well the edit follows the instruction and maintains visual quality.\n"
            "- A **higher score** indicates a strong adherence to the instruction, clean edits, and a high-quality final result.\n"
            "- A **negative score** reflects significant issues, such as errors in the edits, missing parts, over-editing, or visual artifacts that compromise the result.\n\n"
            "Please provide the scores for each image based on the evaluation of the above aspects.\n\n"
            f"Text instruction - {instruction}\n"
        )

    def _build_prompt(self, prompt: str | None, task: str):
        self._validate_task(task)
        instruction = prompt or ""
        if task == "edit_pointwise_score":
            return self._build_edit_pointwise_score_prompt(instruction)
        if task == "edit_pairwise_rank":
            return self._build_edit_pairwise_rank_prompt(instruction)
        return self._build_edit_pairwise_score_prompt(instruction)

    def _build_messages(self, prompt: str | None, images, task: str):
        images = [image.convert("RGB") for image in _as_list(images)]
        expected_images = 2 if task == "edit_pointwise_score" else 3
        if len(images) != expected_images:
            raise ValueError(
                f"UnifiedReward {task} expects exactly {expected_images} images. "
                "Use [source_image, edited_image] for edit_pointwise_score and "
                "[source_image, edited_image_1, edited_image_2] for pairwise tasks."
            )
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": self._build_prompt(prompt, task)})
        return [{"role": "user", "content": content}]

    def _processor_inputs(self, text, images):
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"]
        inputs = inputs.to(self.device)

        if self.dtype != torch.float32:
            inputs = {
                name: (
                    value.to(dtype=self.dtype)
                    if torch.is_tensor(value) and torch.is_floating_point(value)
                    else value
                )
                for name, value in inputs.items()
            }
        return inputs, input_ids

    def _decode_sample(self, prompt: str | None, images, task: str):
        messages = self._build_messages(prompt, images, task)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs, input_ids = self._processor_inputs(text, [image.convert("RGB") for image in _as_list(images)])

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=getattr(self.processor.tokenizer, "pad_token_id", None),
                eos_token_id=getattr(self.processor.tokenizer, "eos_token_id", None),
            )
        generated_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    @staticmethod
    def _extract_first_json(text: str):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        payload = match.group(0)
        for loader in (json.loads, lambda s: json.loads(s.replace("'", '"'))):
            try:
                return loader(payload)
            except Exception:
                pass
        return None

    @staticmethod
    def _extract_score_list(text: str):
        match = re.search(r'"?score"?\s*:\s*\[([^\]]+)\]', text, flags=re.I)
        if not match:
            return []
        scores = [_coerce_float(value) for value in re.findall(r"[-+]?\d*\.?\d+", match.group(1))]
        return [score for score in scores if score is not None]

    @staticmethod
    def _extract_pair_scores(text: str):
        patterns = [
            r"Image\s*1[^-+\d]*([-+]?\d*\.?\d+).*?Image\s*2[^-+\d]*([-+]?\d*\.?\d+)",
            r"Edited\s*Image\s*1[^-+\d]*([-+]?\d*\.?\d+).*?Edited\s*Image\s*2[^-+\d]*([-+]?\d*\.?\d+)",
            r"score(?:s)?[^-+\d]*([-+]?\d*\.?\d+)[^\n\r-+\d]+([-+]?\d*\.?\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I | re.S)
            if match:
                return [_coerce_float(match.group(1)), _coerce_float(match.group(2))]
        return []

    @staticmethod
    def _parse_rank(text: str):
        if re.search(r"both images? (are )?(equally good|equal|tie)", text, flags=re.I):
            return "Edited image 1 and 2 are equally good"
        if re.search(r"((edited )?image\s*1|first image).{0,24}\b(better|best|wins?|preferred|superior)\b", text, flags=re.I | re.S):
            return "Edited image 1"
        if re.search(r"((edited )?image\s*2|second image).{0,24}\b(better|best|wins?|preferred|superior)\b", text, flags=re.I | re.S):
            return "Edited image 2"
        return None

    def _parse_pointwise_score(self, text: str):
        parsed = self._extract_first_json(text)
        scores = []
        reasoning = None
        if isinstance(parsed, dict):
            if isinstance(parsed.get("score"), list):
                scores = [_coerce_float(value) for value in parsed["score"]]
                scores = [value for value in scores if value is not None]
            reasoning = parsed.get("reasoning")
        if not scores:
            scores = self._extract_score_list(text)
        editing_success = scores[0] if len(scores) > 0 else None
        overediting = scores[1] if len(scores) > 1 else None
        return {
            "editing_success": editing_success,
            "overediting": overediting,
            "score": _mean([editing_success, overediting]),
            "reasoning": reasoning,
        }

    def _parse_pairwise_score(self, text: str):
        scores = self._extract_score_list(text)
        if len(scores) < 2:
            scores = self._extract_pair_scores(text)
        scores = [score for score in scores if score is not None]
        return {
            "image_1_score": scores[0] if len(scores) > 0 else None,
            "image_2_score": scores[1] if len(scores) > 1 else None,
        }

    def _parse_pairwise_rank(self, text: str):
        return {
            "winner": self._parse_rank(text),
        }

    def _parse_output(self, text: str, task: str):
        self._validate_task(task)
        if task == "edit_pointwise_score":
            return self._parse_pointwise_score(text)
        if task == "edit_pairwise_score":
            return self._parse_pairwise_score(text)
        return self._parse_pairwise_rank(text)

    @staticmethod
    def _primary_score(parsed: dict, task: str):
        if task == "edit_pairwise_rank":
            return int(parsed.get("winner") or 0)
        if task == "edit_pairwise_score":
            return [parsed.get("image_1_score"), parsed.get("image_2_score")]
        return parsed.get("score")

    @torch.no_grad()
    def forward(self, prompt: str | list[str] | None, images, task: str | None = None):
        task = task or self.task
        self._validate_task(task)
        prompts = _as_list(prompt if prompt is not None else "")
        images = _as_list(images)

        if images and isinstance(images[0], Image.Image):
            images = [images]

        if len(prompts) == 1 and len(images) > 1:
            prompts = prompts * len(images)
        if len(images) == 1 and len(prompts) > 1:
            images = images * len(prompts)
        if len(prompts) != len(images):
            raise ValueError(f"Expected the same number of prompts and images, got {len(prompts)} and {len(images)}.")

        outputs = []
        for single_prompt, single_images in zip(prompts, images):
            raw_text = self._decode_sample(single_prompt, single_images, task)
            outputs.append(self._parse_output(raw_text, task))
        return outputs
        
