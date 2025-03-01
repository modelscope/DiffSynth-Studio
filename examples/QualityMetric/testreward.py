import os
import torch
from PIL import Image
from diffsynth.extensions.QualityMetric.imagereward import ImageRewardScore
from diffsynth.extensions.QualityMetric.pickscore import PickScore
from diffsynth.extensions.QualityMetric.aesthetic import AestheticScore
from diffsynth.extensions.QualityMetric.clip import CLIPScore
from diffsynth.extensions.QualityMetric.hps import HPScore_v2
from diffsynth.extensions.QualityMetric.mps import MPScore

# download model from modelscope
from modelscope.hub.snapshot_download import snapshot_download

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
model_folder = os.path.join(project_root, 'models', 'QualityMetric')

# download HPS_v2 to your folder
# model_id = "DiffSynth-Studio/QualityMetric_reward_pretrained"
# downloaded_path = snapshot_download(
#     model_id,
#     cache_dir=os.path.join(model_folder, 'HPS_v2'), 
#     allow_patterns=["HPS_v2/*"], 
# )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_path(model_folder, model_name):
    return os.path.join(model_folder, model_name)

# your model path
model_path = {
    "aesthetic_predictor": get_model_path(model_folder, "aesthetic-predictor/sac+logos+ava1-l14-linearMSE.safetensors"),
    "open_clip": get_model_path(model_folder, "CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"),
    "hpsv2": get_model_path(model_folder, "HPS_v2/HPS_v2_compressed.safetensors"),
    "hpsv2.1": get_model_path(model_folder, "HPS_v2/HPS_v2.1_compressed.safetensors"),
    "imagereward": get_model_path(model_folder, "ImageReward/ImageReward.safetensors"),
    "med_config": get_model_path(model_folder, "ImageReward/med_config.json"),
    "clip": get_model_path(model_folder, "CLIP-ViT-H-14-laion2B-s32B-b79K"),
    "clip-large": get_model_path(model_folder, "clip-vit-large-patch14"),
    "mps": get_model_path(model_folder, "MPS_overall_checkpoint/MPS_overall_checkpoint_diffsynth.safetensors"),
    "pickscore": get_model_path(model_folder, "PickScore_v1")
}

# load reward models
mps_score = MPScore(device,path = model_path)
image_reward = ImageRewardScore(device, path = model_path)
aesthetic_score = AestheticScore(device, path = model_path)
pick_score = PickScore(device, path = model_path)
clip_score = CLIPScore(device, path = model_path)
hps_score = HPScore_v2(device, path = model_path, model_version = 'v2')
hps2_score = HPScore_v2(device, path = model_path, model_version = 'v21')

prompt = "a painting of an ocean with clouds and birds, day time, low depth field effect"
img_prefix = "images"
generations = [f"{pic_id}.webp" for pic_id in range(1, 5)]

img_list = [Image.open(os.path.join(img_prefix, img)) for img in generations]
#img_list = [os.path.join(img_prefix, img) for img in generations]

imre_scores = image_reward.score(img_list, prompt)
print("ImageReward:", imre_scores)

aes_scores = aesthetic_score.score(img_list)
print("Aesthetic", aes_scores)

p_scores = pick_score.score(img_list, prompt)
print("PickScore:", p_scores)

c_scores = clip_score.score(img_list, prompt)
print("CLIPScore:", c_scores)

h_scores = hps_score.score(img_list,prompt)
print("HPScorev2:", h_scores)

h2_scores = hps2_score.score(img_list,prompt)
print("HPScorev21:", h2_scores)

m_scores = mps_score.score(img_list, prompt)
print("MPS_score:", m_scores)