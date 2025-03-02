import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
model_path = os.path.join(project_root, 'models', 'QualityMetric')


def get_model_path(model_name):
    return os.path.join(model_path, model_name)


MODEL_PATHS = {
    "aesthetic_predictor": get_model_path("aesthetic-predictor/sac+logos+ava1-l14-linearMSE.safetensors"),
    "open_clip": get_model_path("CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"),
    "hpsv2": get_model_path("HPS_v2/HPS_v2_compressed.safetensors"),
    "hpsv2.1": get_model_path("HPS_v2/HPS_v2.1_compressed.safetensors"),
    "imagereward": get_model_path("ImageReward/ImageReward.safetensors"),
    "med_config": get_model_path("ImageReward/med_config.json"),
    "clip": get_model_path("CLIP-ViT-H-14-laion2B-s32B-b79K"),
    "clip-large": get_model_path("clip-vit-large-patch14"),
    "mps": get_model_path("MPS_overall_checkpoint/MPS_overall_checkpoint_diffsynth.safetensors"),
    "pickscore": get_model_path("PickScore_v1")
}