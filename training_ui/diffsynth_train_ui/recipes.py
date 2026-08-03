from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


MODEL_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "model_configs"


@dataclass
class ModelPath:
    model_id: str = ""
    file_pattern: str = ""
    local_path: str = ""
    fp8: bool = False


@dataclass
class ModelRecipe:
    name: str
    label: str
    train_script: str
    config_path: str
    generation_type: str = "image"
    family: str = ""
    source_script: str = ""
    dataset_kind: str = "image"
    lora_base_model: str = "dit"
    remove_prefix: str = "pipe.dit."
    default_lora_target: str = ""
    default_model_paths: List[ModelPath] = field(default_factory=list)
    extra_defaults: Dict[str, Any] = field(default_factory=dict)
    default_data_file_keys: str = "image,video,audio,edit_image"
    default_resolution_mode: Optional[str] = None
    default_max_pixels: Optional[int] = None
    default_height: Optional[int] = None
    default_width: Optional[int] = None
    default_num_frames: Optional[int] = None
    default_extra_inputs: str = ""
    default_find_unused_parameters: bool = False
    default_dataset_repeat: int = 1
    default_lr: float = 1e-4
    default_epochs: int = 1
    default_lora_rank: int = 32
    default_dataset_num_workers: Optional[int] = 0
    default_use_gradient_checkpointing: bool = False
    default_optimizer: str = "torch.optim.AdamW"
    default_gradient_accumulation: int = 1
    default_enable_custom_lora_target: bool = False
    default_sample_prompts: List[str] = field(default_factory=list)
    sampling: Dict[str, Any] = field(default_factory=dict)
    disable_sections: List[str] = field(default_factory=list)
    additional_sections: List[str] = field(default_factory=list)
    training_stages: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    dataset_repeat_stage_index: Optional[int] = None
    editable_stage_parameters: List[Dict[str, Any]] = field(default_factory=list)
    sort_order: int = 0


def _load_recipe(path: Path) -> ModelRecipe:
    data = json.loads(path.read_text(encoding="utf-8"))
    family = str(data["family"])
    name = str(data["name"])
    defaults = data.get("defaults")
    training_args = data.get("training_args")
    stages = data.get("stages")
    sampling = data.get("sampling") or {}
    train_script = str(data["train_script"])
    source_script = str(data["source_script"])
    model_paths = [ModelPath(**item) for item in defaults["model_paths"]]
    generation_type = str(data.get("generation_type", "image"))
    managed_training_args = {
        "data_file_keys",
        "lora_base_model",
        "remove_prefix_in_ckpt",
        "use_gradient_checkpointing",
    }
    runtime_stages = [{**training_args, **stage} for stage in stages]
    dataset_repeat_stage_index = None
    if len(runtime_stages) > 1 and str(runtime_stages[0].get("task", "")).endswith(":data_process"):
        dataset_repeat_stage_index = 1
    default_dataset_repeat = defaults["dataset_repeat"]
    if dataset_repeat_stage_index is not None:
        default_dataset_repeat = runtime_stages[dataset_repeat_stage_index].get(
            "dataset_repeat", default_dataset_repeat
        )
    editable_stage_parameters = [
        {
            key: stage[key]
            for key in ("max_timestep_boundary", "min_timestep_boundary")
            if key in stage
        }
        for stage in runtime_stages
    ]
    return ModelRecipe(
        name=name,
        label=name,
        train_script=train_script,
        config_path=str(path.relative_to(MODEL_CONFIG_ROOT.parents[0])),
        generation_type=generation_type,
        family=family,
        source_script=source_script,
        dataset_kind={"image": "image", "video": "video", "audio": "audio"}[generation_type],
        lora_base_model=str(training_args.get("lora_base_model", "dit")),
        remove_prefix=str(training_args.get("remove_prefix_in_ckpt", "pipe.dit.")),
        default_lora_target=str(defaults["lora_target_modules"]),
        default_model_paths=model_paths,
        extra_defaults={
            key: value
            for key, value in training_args.items()
            if key not in managed_training_args
        },
        default_data_file_keys=(
            str(training_args["data_file_keys"])
            if training_args.get("data_file_keys")
            else None
        ),
        default_resolution_mode=(
            str(defaults["resolution_mode"])
            if defaults.get("resolution_mode") is not None else None
        ),
        default_max_pixels=(
            int(defaults["max_pixels"])
            if defaults.get("max_pixels") is not None else None
        ),
        default_height=(
            int(defaults["height"])
            if defaults.get("height") is not None else None
        ),
        default_width=(
            int(defaults["width"])
            if defaults.get("width") is not None else None
        ),
        default_num_frames=(
            int(defaults["num_frames"])
            if defaults.get("num_frames") is not None else None
        ),
        default_extra_inputs=str(defaults["extra_inputs"] or ""),
        default_find_unused_parameters=bool(defaults["find_unused_parameters"]),
        default_dataset_repeat=int(default_dataset_repeat),
        default_lr=float(defaults["learning_rate"]),
        default_epochs=int(defaults["num_epochs"]),
        default_lora_rank=int(defaults["lora_rank"]),
        default_dataset_num_workers=(
            int(defaults["dataset_num_workers"])
            if defaults["dataset_num_workers"] is not None else None
        ),
        default_use_gradient_checkpointing=bool(training_args.get("use_gradient_checkpointing", False)),
        default_optimizer=str(defaults["optimizer"]),
        default_gradient_accumulation=int(defaults["gradient_accumulation"]),
        default_enable_custom_lora_target=bool(defaults["enable_custom_lora_target"]),
        default_sample_prompts=[str(item) for item in sampling["sample_prompts"]],
        sampling=dict(sampling),
        disable_sections=["resolution"] if generation_type == "audio" else [],
        additional_sections=["num_frames"] if defaults.get("num_frames") is not None else [],
        training_stages=runtime_stages,
        dataset_repeat_stage_index=dataset_repeat_stage_index,
        editable_stage_parameters=editable_stage_parameters,
        sort_order=int(data.get("sort_order", 0)),
    )


def _load_all() -> Dict[str, ModelRecipe]:
    recipes = [_load_recipe(path) for path in MODEL_CONFIG_ROOT.glob("*/*/default.json")]
    recipes.sort(key=lambda item: (item.sort_order, item.name))
    result: Dict[str, ModelRecipe] = {}
    for recipe in recipes:
        if recipe.name in result:
            raise ValueError(f"Duplicate model config name: {recipe.name}")
        result[recipe.name] = recipe
    if not result:
        raise RuntimeError(f"No model configs found under {MODEL_CONFIG_ROOT}")
    return result


MODEL_RECIPES = _load_all()


def list_recipes() -> List[str]:
    return list(MODEL_RECIPES.keys())


def get_recipe(name: str) -> ModelRecipe:
    if name not in MODEL_RECIPES:
        raise KeyError(f"Unknown model type: {name}")
    return MODEL_RECIPES[name]


def get_default_config_path(name: str) -> Path:
    recipe = get_recipe(name)
    return MODEL_CONFIG_ROOT.parents[0] / recipe.config_path
