import json
import re
from collections import defaultdict
from typing import Union

import torch
from PIL import Image

ImageInput = Union[Image.Image, list[Image.Image], tuple[Image.Image, ...]]


class QwenImageBenchQwen35ForConditionalGeneration(torch.nn.Module):
    def __init__(self, variant: str = "qwen35"):
        super().__init__()
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        config = Qwen3_5Config(
            bos_token_id=None,
            eos_token_id=248046,
            hidden_size=5120,
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
                "hidden_size": 5120,
                "initializer_range": 0.02,
                "intermediate_size": 17408,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262144,
                "mlp_only_layers": [],
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": False,
                "num_attention_heads": 24,
                "num_hidden_layers": 64,
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
                "out_hidden_size": 5120,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            vision_end_token_id=248054,
            vision_start_token_id=248053,
        )
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16
        config.text_config.layer_types = layer_types
        self.model = Qwen3_5ForConditionalGeneration(config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


QUALITY_CHECKLIST = """## Realism
- Physical Logic: Does the image adhere to real-world physical laws (e.g., gravity, reflection, shadow direction, object stability)?
- Material Texture: Do the surface materials of objects (such as skin, fabric, metal, wood) exhibit realistic texture and material properties?
## Detail
- Noise: Is the image rich in detail without excessive noise or unnatural smoothing?
- Edge Clarity: Are the outlines and edges of objects sharp, well-defined, and free from blurring or aliasing?
- Naturalness: Does the image appear natural and free from the artificial "plastic" or "greasy" look commonly associated with AI-generated images?
## Resolution
- Resolution: Is the overall image resolution high-definition, free from visible pixelation or compression artifacts?"""


AESTHETICS_CHECKLIST = """## Composition
- Composition: Is the composition of the image balanced, visually guided, and aesthetically pleasing?
## Color Harmony
- Color Harmony: Is the overall color palette harmonious, cohesive, and appropriate for the mood of the image?
## Lighting
- Lighting & Atmosphere: Does the lighting and shadow atmosphere of the image (such as contrast between light and dark, and the overall lighting atmosphere) match the scene setting of the prompt?
## Anatomical Portraiture
- Anatomical Fidelity: Are the facial feature proportions, skeletal structure, and limb articulation anatomically correct and consistent with human biology? Does the facial skin exhibit realistic micro-level textures such as pores and fine lines?
## Emotional Expression
- Emotional Expression: Does the image's overall aesthetic tone effectively convey the intended emotion and mood described in the prompt?
## Style Control
- Style Control: Does the image accurately capture and represent the specific artistic style requested in the prompt (e.g., Van Gogh's brushwork, Cyberpunk aesthetic)?"""

ALIGNMENT_CHECKLIST = """## Attributes
- Quantity: Does the number of objects in the image match the quantity specified in the prompt?
- Facial Expression: Does the facial expression of the person or animal accurately reflect the emotional state specified in the prompt?
- Material Properties: Do the materials of objects in the image match the material descriptions in the prompt?
- Color: Do the colors of objects in the image match the color specifications in the prompt?
- Shape: Do the shapes of objects in the image match the shape descriptions in the prompt?
- Size: Do the sizes of objects in the image match the size specifications in the prompt?
## Actions
- Contact Interaction: If the prompt involves physical contact between subjects, is the contact interaction depicted naturally and realistically?
- Non-contact Interaction: If the prompt involves non-contact relationships between subjects, is the spatial and social relationship depicted naturally and logically?
- Full-body Action: Does the overall posture and body action of the subject (person or animal) accurately perform the activity described in the prompt?
## Layout
- 2D Space: Are the relative positions of objects on the 2D plane (e.g., left/right, top/bottom, foreground/background) consistent with the prompt's spatial instructions?
- 3D Space: Does the layout, occlusion, and relative position of objects in 3D space conform to the prompt requirements or spatial logic?
## Relations
- Composition Relationship: Does the image successfully integrate multiple elements into a visually coherent and logically consistent whole?
- Difference/Similarity: Are the specified differences or similarities in shape, color, or material between objects accurately represented?
- Containment: Are the containment or enclosure relationships between objects correctly depicted?
## Scene
- Real-world Scene: Does the scene type and environmental setting (e.g., office, forest, street) match the location described in the prompt?
- Virtual Scene: Are the elements within a fictional or fantasy scene internally consistent and logically coherent?"""

REAL_WORLD_FIDELITY_CHECKLIST = """## Fairness
- Social Bias: Does the image avoid reinforcing social biases by automatically associating specific genders with particular professions or settings?
- Cultural Fairness: Is the image free from stereotypical portrayals based on region, race, or cultural background?
## Safety & Compliance
- Safety & Compliance: Is the image safe and compliant, effectively avoiding prohibited content such as pornography, violence, or hate symbols?
## World Knowledge
- Animals: Are real-world animals depicted with anatomically accurate features and realistic biological details?
- Objects: Are the typical appearance, structure, brand logo, or iconic characteristics of real-world items accurately reproduced?
- Information Visualization: Does the image accurately and clearly translate abstract or scientific concepts from the prompt into an effective and understandable visual form?
- Temporal Characteristics: Does the image accurately reflect the iconic elements of a specific historical period (e.g., technology, clothing, architecture, lifestyle of that era)?
- Cultural Elements: Are the cultural elements (such as symbols, traditional clothing, rituals, and customs) accurately depicted and consistent with real-world cultural practices?"""


CREATIVE_GENERATION_CHECKLIST = """## Imagination
- Imagination: Does the image demonstrate creative originality and imaginative thinking when combining novel or surreal elements?
## Feature Matching
- Feature Matching: Are the multi-element fusion regions in the image visually seamless, without abrupt breaks, harsh edges, or logical contradictions?
## Logical Resolution
- Logical Resolution: Does the image accurately depict causal relationships between events (e.g., breaking glass → shards flying, rain → wet surfaces)?
## Text Rendering
- Text Accuracy: If the image contains text, is the text clear, legible, and free from garbled characters, misspellings, or typographical errors?
- Text Layout: Is the text layout (e.g., centering, alignment, line spacing, margins) in the image visually appealing and professionally structured?
- Font: Does the font style used in the image match the font type specified in the prompt (e.g., SimSun, Heiti, handwritten, serif)?
- Cross-lingual Generation: Does the image correctly follow the translation instructions in the prompt, producing accurate text in the target language?
## Design Applications
- Graphic Design: Does the graphic design (e.g., advertisement, poster) exhibit a clear information hierarchy, effective visual guidance, and professional layout?
- Product Design: Does the product design in the image demonstrate reasonable industrial design logic (e.g., ergonomic grip, logical interface placement, structural integrity)?
- Spatial Design: Does the interior or architectural space conform to the principles of perspective, proportion, and building design standards?
- Fashion Styling: Does the clothing cut and silhouette match the style described in the prompt (e.g., Hanfu, cyberpunk, haute couture)? Does the makeup style (e.g., smoky eyes, nude makeup, theatrical look) suit the occasion and character setting?
- Game Design: Do the game props and UI elements have practical in-game usability (e.g., icon recognizability, interactive affordances, clear feedback cues)?
- Art Design: Does the image successfully demonstrate the specific artistic design style required by the prompt (e.g., unique brushstrokes, distinctive color scheme, coherent artistic language)?
## Visual Storytelling
- Cinematic Style: Does the image reproduce the signature visual language of the specific director referenced in the prompt (e.g., Wes Anderson's symmetrical composition, Wong Kar-wai's warm color palette)?
- Camera / Lens Style: Does the image reflect the characteristic imaging effects of the specific photographic equipment or lens referenced in the prompt (e.g., film grain, bokeh, digital sharpening)?
- Storyboard Creation: Does the image's scene composition follow the panel layout requirements outlined in the prompt (e.g., three-panel, four-panel, split-screen)?
- Shot Sizes: Does the image meet the framing and shot size requirements specified in the prompt (e.g., close-up, medium shot, wide shot)?
- Composition: Does the image follow the specific composition rules required by the prompt (e.g., rule of thirds, golden ratio, leading lines)?
- Angles: Does the camera angle comply with the prompt's specification (e.g., bird's-eye view, low angle, Dutch angle)?
- Comic Creation: Does the image conform to the comic style required by the prompt (e.g., American comics, Japanese manga, European BD)?"""

DIM_TO_CHECKLIST = {
    "Quality": QUALITY_CHECKLIST,
    "Aesthetics": AESTHETICS_CHECKLIST,
    "Alignment": ALIGNMENT_CHECKLIST,
    "Real-world Fidelity": REAL_WORLD_FIDELITY_CHECKLIST,
    "Creative Generation": CREATIVE_GENERATION_CHECKLIST,
}
LEVEL1_DIMS = list(DIM_TO_CHECKLIST)

SYSTEM_PROMPT = (
    "You are an expert evaluator for text-to-image (T2I) generation quality. "
    "Given an image and the text prompt used to generate it, you evaluate the image "
    "on specific quality criteria using a structured checklist."
)

USER_PROMPT_HEADER = """# Text Prompt Used to Generate the Image
{prompt}

# Generated Image
"""

USER_PROMPT_BODY = """

# Evaluation Dimension
{level1_dim}

# Scoring Rules
- **0 (Fail)**: Clear defect present. Would noticeably reduce image quality.
- **1 (Pass)**: No defect. Meets baseline expectations.
- **2 (Excel)**: Exceptionally executed. Only when concrete excellence is observable.
- **N/A**: This criterion does not apply to this image/prompt.

# Evaluation Checklist
{format_checklist}

# Output Format
Respond with a valid JSON object only (no markdown code blocks):
{{
  "{{level2_dim}}": {{
    "{{level3_dim}}": {{"score": 0|1|2}},
    "{{level3_dim}}": {{"score": "N/A"}}
  }}
}}"""

SCORE_MAP = {0: 0.0, 1: 60.0, 2: 100.0}

CHECKLIST_L3_TO_L2 = {
    "Quality": {
        "Physical Logic": "Realism", "Material Texture": "Realism",
        "Noise": "Detail", "Edge Clarity": "Detail", "Naturalness": "Detail",
        "Resolution": "Resolution",
    },
    "Aesthetics": {
        "Composition": "Composition", "Color Harmony": "Color Harmony",
        "Lighting & Atmosphere": "Lighting",
        "Anatomical Fidelity": "Anatomical Portraiture",
        "Emotional Expression": "Emotional Expression",
        "Style Control": "Style Control",
    },
    "Alignment": {
        "Quantity": "Attributes", "Facial Expression": "Attributes",
        "Material Properties": "Attributes", "Color": "Attributes",
        "Shape": "Attributes", "Size": "Attributes",
        "Contact Interaction": "Actions", "Non-contact Interaction": "Actions",
        "Full-body Action": "Actions",
        "2D Space": "Layout", "3D Space": "Layout",
        "Composition Relationship": "Relations", "Difference/Similarity": "Relations",
        "Containment": "Relations",
        "Real-world Scene": "Scene", "Virtual Scene": "Scene",
    },
    "Real-world Fidelity": {
        "Social Bias": "Fairness", "Cultural Fairness": "Fairness",
        "Safety & Compliance": "Safety & Compliance",
        "Animals": "World Knowledge", "Objects": "World Knowledge",
        "Information Visualization": "World Knowledge",
        "Temporal Characteristics": "World Knowledge",
        "Cultural Elements": "World Knowledge",
    },
    "Creative Generation": {
        "Imagination": "Imagination",
        "Feature Matching": "Feature Matching",
        "Logical Resolution": "Logical Resolution",
        "Text Accuracy": "Text Rendering", "Text Layout": "Text Rendering",
        "Font": "Text Rendering", "Cross-lingual Generation": "Text Rendering",
        "Graphic Design": "Design Applications", "Product Design": "Design Applications",
        "Spatial Design": "Design Applications", "Fashion Styling": "Design Applications",
        "Game Design": "Design Applications", "Art Design": "Design Applications",
        "Cinematic Style": "Visual Storytelling", "Camera / Lens Style": "Visual Storytelling",
        "Storyboard Creation": "Visual Storytelling", "Shot Sizes": "Visual Storytelling",
        "Composition": "Visual Storytelling", "Angles": "Visual Storytelling",
        "Comic Creation": "Visual Storytelling",
    },
}

L3_RENAME = {
    "Creative Generation": {"Feature Mapping": "Feature Matching"},
}


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _mean_non_none(values):
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def parse_dims_by_level1(dims_en_str):
    """
    Parse dims_en string, group by level-1 dimension.
    Input:  "Quality / Realism / Physical Logic; Aesthetics / Color Harmony / Color Harmony"
    Output: {"Quality": [("Realism", "Physical Logic")], "Aesthetics": [("Color Harmony", "Color Harmony")]}
    """
    result = defaultdict(list)
    parts = [p.strip() for p in dims_en_str.split(';')]
    for p in parts:
        levels = [l.strip() for l in p.split('/')]
        if len(levels) >= 3:
            result[levels[0]].append((levels[1], levels[2]))
        elif len(levels) == 2:
            result[levels[0]].append((levels[1], levels[1]))
    return dict(result)


def extract_json_from_response(response_text: str):
    text = response_text or ""
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


def map_score(raw_score):
    """Map raw score to final score: 0→0, 1→60, 2→100, 'N/A'→None."""
    if isinstance(raw_score, str) and raw_score.upper() == "N/A":
        return None
    try:
        return SCORE_MAP[int(raw_score)]
    except (KeyError, ValueError, TypeError):
        return None


def fix_score_json(score_json, l1_dim):
    """Fix flat structure, L3 misplacement, and L3 typos based on checklists.py hierarchy."""
    if not score_json:
        return score_json

    mapping = CHECKLIST_L3_TO_L2.get(l1_dim, {})
    rename = L3_RENAME.get(l1_dim, {})

    first_val = next(iter(score_json.values()), None)
    if isinstance(first_val, dict) and "score" in first_val:
        result = {}
        for l3_name, score_obj in score_json.items():
            l3_name = rename.get(l3_name, l3_name)
            l2_name = mapping.get(l3_name, l3_name)
            result.setdefault(l2_name, {})[l3_name] = score_obj
        return result

    result = {}
    for l2_key, l3_dict in score_json.items():
        if not isinstance(l3_dict, dict):
            continue
        for l3_name, score_obj in l3_dict.items():
            l3_name = rename.get(l3_name, l3_name)
            correct_l2 = mapping.get(l3_name, l2_key)
            result.setdefault(correct_l2, {})[l3_name] = score_obj
    return result



def compute_dimension_score(score_json):
    """
    Compute aggregated score for a single level-1 dimension.

    Input: {"Realism": {"Physical Logic": {"score": 0}, "Material Properties": {"score": 1}}, ...}
    Output: {
        "level1_score": float | None,
        "level2_scores": {"Realism": float | None, ...},
        "level3_scores": {"Realism": {"Physical Logic": 0.0, ...}, ...}
    }
    """
    level2_scores = {}
    level3_scores = {}

    for level2_name, level3_dict in score_json.items():
        level3_scores[level2_name] = {}
        level3_mapped = []

        for level3_name, score_obj in level3_dict.items():
            raw = score_obj.get("score") if isinstance(score_obj, dict) else score_obj
            mapped = map_score(raw)
            level3_scores[level2_name][level3_name] = mapped
            if mapped is not None:
                level3_mapped.append(mapped)

        level2_scores[level2_name] = _mean_non_none(level3_mapped)

    level1_score = _mean_non_none(list(level2_scores.values()))

    return {
        "level1_score": level1_score,
        "level2_scores": level2_scores,
        "level3_scores": level3_scores,
    }


def aggregate_total_score(dim_results):
    """
    Aggregate across all level-1 dimensions to total score.

    Input: {"Quality": {"level1_score": 60.0, ...}, "Aesthetics": {"level1_score": 80.0, ...}, ...}
    Output: float | None
    """
    level1_scores = [
        r["level1_score"] for r in dim_results.values()
        if r is not None and r.get("level1_score") is not None
    ]
    return _mean_non_none(level1_scores)


class QwenImageBenchModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, processor, max_new_tokens: int = 4096, resize_long_edge: int = 1024):
        super().__init__()
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.resize_long_edge = resize_long_edge

    @property
    def device(self):
        return next(self.parameters(), torch.tensor([])).device

    @property
    def dtype(self):
        return next(self.parameters(), torch.tensor(0.0)).dtype

    def _prepare_image(self, image: Image.Image):
        image = image.convert("RGB")
        if self.resize_long_edge and max(image.size) > self.resize_long_edge:
            image = image.resize((self.resize_long_edge, self.resize_long_edge), Image.LANCZOS)
        return image

    @staticmethod
    def _validate_dimension(level1_dim: str):
        if level1_dim not in DIM_TO_CHECKLIST:
            supported = ", ".join(LEVEL1_DIMS)
            raise ValueError(f"Unsupported Qwen-Image-Bench dimension: {level1_dim}. Supported dimensions: {supported}.")

    @classmethod
    def _dims_to_level1(cls, dimensions=None):
        if dimensions is None:
            return LEVEL1_DIMS
        if isinstance(dimensions, dict):
            dimensions = list(dimensions.keys())
        elif isinstance(dimensions, str):
            if "/" in dimensions or ";" in dimensions:
                dimensions = list(parse_dims_by_level1(dimensions).keys())
            elif "," in dimensions:
                dimensions = [dim.strip() for dim in dimensions.split(",") if dim.strip()]
            else:
                dimensions = [dimensions]
        else:
            dimensions = list(dimensions)
        if not dimensions:
            dimensions = LEVEL1_DIMS
        for level1_dim in dimensions:
            cls._validate_dimension(level1_dim)
        return dimensions

    @classmethod
    def _normalize_dimensions(cls, dimensions, batch_size: int):
        if isinstance(dimensions, (list, tuple)) and dimensions and not all(isinstance(dim, str) for dim in dimensions):
            if len(dimensions) != batch_size:
                raise ValueError(f"Expected {batch_size} dimension sets, got {len(dimensions)}.")
            return [cls._dims_to_level1(dim_set) for dim_set in dimensions]
        dims = cls._dims_to_level1(dimensions)
        return [dims for _ in range(batch_size)]

    def _build_messages(self, prompt: str, image: Image.Image, level1_dim: str):
        self._validate_dimension(level1_dim)
        checklist = DIM_TO_CHECKLIST[level1_dim]
        user_content = [
            {"type": "text", "text": USER_PROMPT_HEADER.format(prompt=prompt or "")},
            {"type": "image", "image": image},
            {"type": "text", "text": USER_PROMPT_BODY.format(level1_dim=level1_dim, format_checklist=checklist)},
        ]
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _apply_chat_template(self, messages):
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _processor_inputs(self, text, image):
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        inputs = inputs.to(self.device)
        if self.dtype != torch.float32:
            inputs = {
                name: value.to(dtype=self.dtype) if torch.is_tensor(value) and torch.is_floating_point(value) else value
                for name, value in inputs.items()
            }
        return inputs, input_ids

    def _decode_dimension(self, prompt: str, image: Image.Image, level1_dim: str):
        messages = self._build_messages(prompt, image, level1_dim)
        text = self._apply_chat_template(messages)
        inputs, input_ids = self._processor_inputs(text, image)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                repetition_penalty=1.05,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=getattr(self.processor.tokenizer, "pad_token_id", None),
                eos_token_id=getattr(self.processor.tokenizer, "eos_token_id", None),
            )
        generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]
        return self.processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    @staticmethod
    def _parse_dimension_output(raw_text: str, level1_dim: str):
        score_json = extract_json_from_response(raw_text)
        if score_json is None:
            return None
        score_json = fix_score_json(score_json, level1_dim)
        return compute_dimension_score(score_json), score_json

    def _evaluate_sample(self, prompt: str, image: Image.Image, dimensions):
        image = self._prepare_image(image)
        raw_outputs = {}
        raw_scores = {}
        dimension_scores = {}
        parse_failures = []
        for level1_dim in dimensions:
            raw_text = self._decode_dimension(prompt, image, level1_dim)
            raw_outputs[level1_dim] = raw_text
            parsed = self._parse_dimension_output(raw_text, level1_dim)
            if parsed is None:
                raw_scores[level1_dim] = None
                dimension_scores[level1_dim] = None
                parse_failures.append(level1_dim)
                continue
            dimension_score, fixed_score_json = parsed
            raw_scores[level1_dim] = fixed_score_json
            dimension_scores[level1_dim] = dimension_score
        total_score = aggregate_total_score(dimension_scores)
        return {
            "total_score": total_score,
            "level1_scores": {
                dim: data.get("level1_score") if data is not None else None for dim, data in dimension_scores.items()
            },
            "level2_scores": {
                dim: data.get("level2_scores", {}) if data is not None else {} for dim, data in dimension_scores.items()
            },
            "level3_scores": {
                dim: data.get("level3_scores", {}) if data is not None else {} for dim, data in dimension_scores.items()
            },
            "raw_scores": raw_scores,
            "raw_outputs": raw_outputs,
            "parse_failures": parse_failures,
        }

    def _normalize_inputs(self, prompts, images):
        prompts = _as_list(prompts if prompts is not None else "")
        images = _as_list(images)
        if len(prompts) == 1 and len(images) > 1:
            prompts = prompts * len(images)
        if len(images) == 1 and len(prompts) > 1:
            images = images * len(prompts)
        if len(prompts) != len(images):
            raise ValueError(f"Expected the same number of prompts and images, got {len(prompts)} and {len(images)}.")
        return prompts, images

    @staticmethod
    def _primary_score(parsed: dict):
        score = parsed.get("total_score")
        return float(score) if score is not None else 0.0

    @torch.no_grad()
    def forward(self, prompt: str | list[str] | None, images: ImageInput, dimensions=None):
        prompts, images = self._normalize_inputs(prompt, images)
        dimension_sets = self._normalize_dimensions(dimensions, len(prompts))
        outputs = []
        for single_prompt, single_image, single_dimensions in zip(prompts, images, dimension_sets):
            outputs.append(self._evaluate_sample(single_prompt, single_image, single_dimensions))
        return outputs
