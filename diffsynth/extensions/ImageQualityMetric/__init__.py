from modelscope import snapshot_download
from typing_extensions import Literal, TypeAlias
import os
from diffsynth.extensions.ImageQualityMetric.aesthetic import AestheticScore
from diffsynth.extensions.ImageQualityMetric.imagereward import ImageRewardScore
from diffsynth.extensions.ImageQualityMetric.pickscore import PickScore
from diffsynth.extensions.ImageQualityMetric.clip import CLIPScore
from diffsynth.extensions.ImageQualityMetric.hps import HPScore_v2
from diffsynth.extensions.ImageQualityMetric.mps import MPScore


preference_model_id: TypeAlias = Literal[
    "ImageReward",
    "Aesthetic",
    "PickScore",
    "CLIP",
    "HPSv2",
    "HPSv2.1",
    "MPS",
]
model_dict = {
    "ImageReward": {
        "model_id": "DiffSynth-Studio/QualityMetric_reward_pretrained",
        "allow_file_pattern": [
            "ImageReward/ImageReward.safetensors",
            "ImageReward/med_config.json",
            "bert-base-uncased/config.json",
            "bert-base-uncased/model.safetensors",
            "bert-base-uncased/tokenizer.json",
            "bert-base-uncased/tokenizer_config.json",
            "bert-base-uncased/vocab.txt",
        ],
        "load_path": {
            "imagereward": "ImageReward/ImageReward.safetensors",
            "med_config": "ImageReward/med_config.json",
            "bert_model_path": "bert-base-uncased",
        },
        "model_class": ImageRewardScore
    },
    "Aesthetic": {
        "model_id": "DiffSynth-Studio/QualityMetric_reward_pretrained",
        "allow_file_pattern": [
            "aesthetic-predictor/sac+logos+ava1-l14-linearMSE.safetensors",
            "clip-vit-large-patch14/config.json",
            "clip-vit-large-patch14/merges.txt",
            "clip-vit-large-patch14/model.safetensors",
            "clip-vit-large-patch14/preprocessor_config.json",
            "clip-vit-large-patch14/special_tokens_map.json",
            "clip-vit-large-patch14/tokenizer.json",
            "clip-vit-large-patch14/tokenizer_config.json",
            "clip-vit-large-patch14/vocab.json",
        ],
        "load_path": {
            "aesthetic_predictor": "aesthetic-predictor/sac+logos+ava1-l14-linearMSE.safetensors",
            "clip-large": "clip-vit-large-patch14",
        },
        "model_class": AestheticScore
    },
    "PickScore": {
        "model_id": "DiffSynth-Studio/QualityMetric_reward_pretrained",
        "allow_file_pattern": [
            "PickScore_v1/*",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/config.json",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/merges.txt",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/preprocessor_config.json",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/special_tokens_map.json",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/tokenizer.json",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/tokenizer_config.json",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/vocab.json",
        ],
        "load_path": {
            "pickscore": "PickScore_v1",
            "clip": "CLIP-ViT-H-14-laion2B-s32B-b79K",
        },
        "model_class": PickScore
    },
    "CLIP": {
        "model_id": "DiffSynth-Studio/QualityMetric_reward_pretrained",
        "allow_file_pattern": [
            "CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
            "bpe_simple_vocab_16e6.txt.gz",
        ],
        "load_path": {
            "open_clip": "CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
            "open_clip_bpe": "bpe_simple_vocab_16e6.txt.gz",
        },
        "model_class": CLIPScore
    },
    "HPSv2": {
        "model_id": "DiffSynth-Studio/QualityMetric_reward_pretrained",
        "allow_file_pattern": [
            "HPS_v2/HPS_v2_compressed.safetensors",
            "bpe_simple_vocab_16e6.txt.gz",
        ],
        "load_path": {
            "hpsv2": "HPS_v2/HPS_v2_compressed.safetensors",
            "open_clip_bpe": "bpe_simple_vocab_16e6.txt.gz",
        },
        "model_class": HPScore_v2,
        "extra_kwargs": {"model_version": "v2"}
    },
    "HPSv2.1": {
        "model_id": "DiffSynth-Studio/QualityMetric_reward_pretrained",
        "allow_file_pattern": [
            "HPS_v2/HPS_v2.1_compressed.safetensors",
            "bpe_simple_vocab_16e6.txt.gz",
        ],
        "load_path": {
            "hpsv2.1": "HPS_v2/HPS_v2.1_compressed.safetensors",
            "open_clip_bpe": "bpe_simple_vocab_16e6.txt.gz",
        },
        "model_class": HPScore_v2,
        "extra_kwargs": {"model_version": "v21"}
    },
    "MPS": {
        "model_id": "DiffSynth-Studio/QualityMetric_reward_pretrained",
        "allow_file_pattern": [
            "MPS_overall_checkpoint/MPS_overall_checkpoint_diffsynth.safetensors",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/config.json",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/merges.txt",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/preprocessor_config.json",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/special_tokens_map.json",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/tokenizer.json",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/tokenizer_config.json",
            "CLIP-ViT-H-14-laion2B-s32B-b79K/vocab.json",
        ],
        "load_path": {
            "mps": "MPS_overall_checkpoint/MPS_overall_checkpoint_diffsynth.safetensors",
            "clip": "CLIP-ViT-H-14-laion2B-s32B-b79K",
        },
        "model_class": MPScore
    },
}


def download_preference_model(model_name: preference_model_id, cache_dir="models"):
    metadata = model_dict[model_name]
    snapshot_download(model_id=metadata["model_id"], allow_file_pattern=metadata["allow_file_pattern"], cache_dir=cache_dir)
    load_path = metadata["load_path"]
    load_path = {key: os.path.join(cache_dir, metadata["model_id"], path) for key, path in load_path.items()}
    return load_path


def load_preference_model(model_name: preference_model_id, device = "cuda", path = None):
    model_class = model_dict[model_name]["model_class"]
    extra_kwargs = model_dict[model_name].get("extra_kwargs", {})
    preference_model = model_class(device=device, path=path, **extra_kwargs)
    return preference_model
