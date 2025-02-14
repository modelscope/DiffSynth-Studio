import os
import torch
from PIL import Image
from diffsynth.extensions.QualityMetric.imagereward import ImageRewardScore
from diffsynth.extensions.QualityMetric.pickscore import PickScore
from diffsynth.extensions.QualityMetric.aesthetic import AestheticScore
from diffsynth.extensions.QualityMetric.clip import CLIPScore
from diffsynth.extensions.QualityMetric.hps import HPScore_v2
from diffsynth.extensions.QualityMetric.mps import MPScore


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load reward models
mps_score = MPScore(device)
image_reward = ImageRewardScore(device)
aesthetic_score = AestheticScore(device)
pick_score = PickScore(device)
clip_score = CLIPScore(device)
hps_score = HPScore_v2(device, model_version = 'v2')
hps2_score = HPScore_v2(device, model_version = 'v21')

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