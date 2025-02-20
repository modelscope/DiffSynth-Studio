import torch
from diffsynth.prompters import WanXPrompter
from diffsynth.models.wanx_text_encoder import WanXTextEncoder

prompter = WanXPrompter('models/WanX/google/umt5-xxl')
text_encoder = WanXTextEncoder()
text_encoder.load_state_dict(torch.load('models/WanX/models_t5_umt5-xxl-enc-bf16.pth', map_location='cpu'))
text_encoder = text_encoder.eval().requires_grad_(False).to(dtype=torch.bfloat16, device='cuda')

prompter.fetch_models(text_encoder)

prompt = '维京战士双手挥舞着大斧，对抗猛犸象，黄昏，雪地中，漫天飞雪'
neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

prompt_emb = prompter.encode_prompt(prompt)
neg_prompt_emb = prompter.encode_prompt(neg_prompt)
print(prompt_emb[0]) # torch.Size([31, 4096])
print(neg_prompt_emb[0]) # torch.Size([126, 4096])
